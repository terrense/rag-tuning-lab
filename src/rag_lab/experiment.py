"""
================================================================================
experiment.py —— 实验运行器 + 追踪器（项目的“真实数据”来源）
--------------------------------------------------------------------------------
把整个评测集跑一遍，测出真实数字（Recall@k、MRR、nDCG@k、延迟），并把每一次运行
追加到日志，让项目积累“可量化的历史”，而不是停留在“感觉变好了”：

    experiments/runs.jsonl      每次运行一条 JSON（提交到 git）
    experiments/LEADERBOARD.md  自动重生成的排行榜，按 pipeline Recall@5 排序

每条运行记录：时间戳、git sha、标签、关键参数（chunk大小/重叠、prepend_title、
融合权重、candidate_k、rerank模式、embedding模型）、语料规模、各阶段平均指标。

对比 4 个检索阶段（就是经典的 BM25/向量/混合/混合+精排 那张表）：
  bm25_only      ：只用 BM25
  vector_only    ：只用向量
  hybrid         ：RRF 融合，不精排
  hybrid_rerank  ：融合 + cross-encoder（配置里的完整链路）
================================================================================
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.loaders import load_eval_queries
from rag_lab.metrics import DEFAULT_KS, bootstrap_ci, mean_metrics, rank_metrics
from rag_lab.pipeline import query_config

EXP_DIR = Path("experiments")
RUNS_FILE = EXP_DIR / "runs.jsonl"
LEADERBOARD = EXP_DIR / "LEADERBOARD.md"


def _git_sha() -> str:
    """取当前 git 短 sha，记进实验记录，保证“这次结果对应哪版代码”可追溯。"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def _ids(hits) -> list[str]:
    """从命中列表抽出 source_id 列表（评测按 source_id 判对错）。"""
    return [h.source_id for h in hits]


def _percentile(values: list[float], pct: float) -> float:
    """求百分位数（p50=中位数，p90=90分位），用于延迟统计。"""
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100) * (len(s) - 1))))
    return s[idx]


def run_eval(cfg: dict, ks: tuple[int, ...] = DEFAULT_KS, with_hybrid_norerank: bool = True) -> dict:
    """把评测集跑一遍，返回各阶段平均指标 + 延迟统计。"""
    queries = load_eval_queries(get_path(cfg, "eval_queries"))

    # 为了能算到 @max(ks)（默认 @10），临时把检索深度顶到至少 maxk。
    maxk = max(ks)
    cfg_main = copy.deepcopy(cfg)
    cfg_main["retrieval"]["top_k"] = max(int(cfg_main["retrieval"].get("top_k", 5)), maxk)
    cfg_main["retrieval"]["candidate_k"] = max(
        int(cfg_main["retrieval"].get("candidate_k", 12)), maxk
    )

    # 三个阶段的逐题指标先攒着，最后求平均
    stage_rows: dict[str, list[dict]] = {
        "bm25_only": [],
        "vector_only": [],
        "hybrid_rerank": [],
    }
    latencies: list[float] = []
    per_query: list[dict] = []

    for item in queries:
        question = str(item["question"])
        expected = list(item.get("expected_source_ids", []))

        # 计时跑一次完整检索
        t0 = time.perf_counter()
        result = query_config(cfg_main, question)
        latencies.append((time.perf_counter() - t0) * 1000)   # 毫秒

        # 一次查询里就同时拿到了三路结果，分别算指标（省事）：
        bm25_m = rank_metrics(_ids(result["bm25_hits"]), expected, ks)   # 只看 BM25 这一路
        vec_m = rank_metrics(_ids(result["vector_hits"]), expected, ks)  # 只看向量这一路
        pipe_m = rank_metrics(_ids(result["hits"]), expected, ks)        # 完整链路最终结果
        stage_rows["bm25_only"].append(bm25_m)
        stage_rows["vector_only"].append(vec_m)
        stage_rows["hybrid_rerank"].append(pipe_m)
        per_query.append(                                     # 逐题明细，方便事后排查 miss
            {
                "id": item.get("id"),
                "expected": expected,
                "pipeline_first_rank": pipe_m["first_rank"],
                "pipeline_mrr": pipe_m["mrr"],
                "top_sources": _ids(result["hits"])[:5],
                # 逐题×逐阶段的完整指标：compare_runs 做配对显著性检验的原料。
                # 没有它就只能比均值，说不清 0.70 vs 0.65 是提升还是噪声。
                "metrics": {"bm25_only": bm25_m, "vector_only": vec_m,
                            "hybrid_rerank": pipe_m},
            }
        )

    # 额外再跑一遍“关掉 rerank”的，补出 hybrid(无精排) 这一行，凑齐 4 阶段对比表。
    if with_hybrid_norerank and str(cfg["rerank"].get("mode", "none")) != "none":
        cfg_nr = copy.deepcopy(cfg_main)
        cfg_nr["rerank"]["mode"] = "none"
        rows = []
        for idx, item in enumerate(queries):
            result = query_config(cfg_nr, str(item["question"]))
            m = rank_metrics(_ids(result["hits"]), item.get("expected_source_ids", []), ks)
            rows.append(m)
            per_query[idx]["metrics"]["hybrid"] = m
        stage_rows["hybrid"] = rows

    # 每个阶段把逐题指标求平均
    stages = {name: mean_metrics(rows, ks) for name, rows in stage_rows.items() if rows}
    return {
        "n_queries": len(queries),
        "stages": stages,
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p90": _percentile(latencies, 90),
            "mean": sum(latencies) / (len(latencies) or 1),
        },
        "per_query": per_query,
    }


def build_record(cfg: dict, label: str, eval_out: dict, corpus: dict | None) -> dict:
    """把一次运行打包成一条可追踪的记录（含时间、sha、参数快照、指标）。"""
    r = cfg["retrieval"]
    c = cfg["chunking"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "label": label,
        "config_path": cfg.get("_config_path"),
        # 评测集标识：不同评测集上的数字不可互比，排行榜按这列分组看
        "eval_set": Path(get_path(cfg, "eval_queries")).stem.replace("eval_queries_", ""),
        "params": {                                   # 关键旋钮快照，便于复现/对比
            "embedding_model": cfg["embedding"].get("model", cfg["embedding"].get("provider")),
            "chunk_size": c.get("chunk_size"),
            "chunk_overlap": c.get("chunk_overlap"),
            "prepend_title": c.get("prepend_title", False),
            "hybrid": r.get("hybrid"),
            "vector_weight": r.get("vector_weight"),
            "bm25_weight": r.get("bm25_weight"),
            "candidate_k": r.get("candidate_k"),
            "top_k": r.get("top_k"),
            "rerank_mode": cfg["rerank"].get("mode"),
            "rerank_model": cfg["rerank"].get("model", ""),
        },
        "corpus": corpus or {},
        "n_queries": eval_out["n_queries"],
        "latency_ms": eval_out["latency_ms"],
        "stages": eval_out["stages"],
        "per_query": eval_out["per_query"],
    }


def append_run(record: dict) -> None:
    """把记录追加到 runs.jsonl（每行一条 JSON）。"""
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def regenerate_leaderboard() -> None:
    """读全部历史运行，按 pipeline Recall@5 降序，重写成 Markdown 排行榜。"""
    if not RUNS_FILE.exists():
        return
    runs = [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    def pipe_recall5(run: dict) -> float:             # 排序键：完整链路的 Recall@5
        return float(run.get("stages", {}).get("hybrid_rerank", {}).get("recall@5", 0.0))

    runs_sorted = sorted(runs, key=pipe_recall5, reverse=True)
    lines = [
        "# Experiment Leaderboard",
        "",
        "Auto-generated by `python -m rag_lab.experiment`. Sorted by pipeline Recall@5.",
        "Pipeline = the configured retrieval stack (hybrid + rerank unless noted).",
        "95% CI = bootstrap over per-query recall@5 (runs logged before per-query",
        "metrics existed show `-`). Only compare rows with the same eval set!",
        "",
        "| Label | eval | When (UTC) | sha | chunk/ovlp | cand_k | bm25_w | rerank | Recall@5 | 95% CI | MRR | nDCG@5 | p50 ms | N |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs_sorted:
        p = run.get("params", {})
        s = run.get("stages", {}).get("hybrid_rerank", {})
        lat = run.get("latency_ms", {})
        # 逐题 recall@5 → bootstrap 95% CI（老记录没存逐题指标 → 显示 -）
        vals = [q["metrics"]["hybrid_rerank"].get("recall@5")
                for q in run.get("per_query", []) if q.get("metrics")]
        if vals and all(v is not None for v in vals):
            _, lo, hi = bootstrap_ci([float(v) for v in vals])
            ci = f"[{lo:.2f},{hi:.2f}]"
        else:
            ci = "-"
        lines.append(
            "| {label} | {ev} | {ts} | {sha} | {cs}/{ov} | {ck} | {bw} | {rr} | {r5:.3f} | {ci} | {mrr:.3f} | {n5:.3f} | {p50:.0f} | {n} |".format(
                label=run.get("label", ""),
                ev=run.get("eval_set", "v1"),
                ts=run.get("timestamp", "")[:16].replace("T", " "),
                sha=run.get("git_sha", ""),
                cs=p.get("chunk_size"),
                ov=p.get("chunk_overlap"),
                ck=p.get("candidate_k"),
                bw=p.get("bm25_weight"),
                rr=p.get("rerank_mode"),
                r5=float(s.get("recall@5", 0.0)),
                ci=ci,
                mrr=float(s.get("mrr", 0.0)),
                n5=float(s.get("ndcg@5", 0.0)),
                p50=float(lat.get("p50", 0.0)),
                n=run.get("n_queries", 0),
            )
        )
    LEADERBOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(record: dict) -> None:
    """终端打印本次运行摘要：参数 + 延迟 + 4 阶段对比表。"""
    print(f"\nExperiment: {record['label']}  (sha={record['git_sha']}, n={record['n_queries']})")
    p = record["params"]
    print(f"  params: chunk={p['chunk_size']}/{p['chunk_overlap']} prepend_title={p['prepend_title']} "
          f"hybrid={p['hybrid']} cand_k={p['candidate_k']} rerank={p['rerank_mode']}")
    lat = record["latency_ms"]
    print(f"  latency: p50={lat['p50']:.0f}ms p90={lat['p90']:.0f}ms mean={lat['mean']:.0f}ms")
    print("  stage comparison (averaged over eval set):")
    order = ["bm25_only", "vector_only", "hybrid", "hybrid_rerank"]   # 表格行的固定顺序
    header = f"    {'stage':14s} {'Recall@1':>9} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>7} {'nDCG@5':>8}"
    print(header)
    for name in order:
        s = record["stages"].get(name)
        if not s:
            continue
        print(f"    {name:14s} {s.get('recall@1',0):>9.3f} {s.get('recall@5',0):>9.3f} "
              f"{s.get('recall@10',0):>10.3f} {s.get('mrr',0):>7.3f} {s.get('ndcg@5',0):>8.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run + record a tracked RAG experiment.")
    parser.add_argument("--config", default="configs/diseases.yaml")
    parser.add_argument("--label", default="", help="Name this run for the leaderboard.")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--ingest", action="store_true", help="Rebuild the index before evaluating.")
    parser.add_argument("--no-record", action="store_true", help="Print only, do not log the run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:                              # 命令行覆盖
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    # 没给 --label 就用“配置名:rerank模式:pt=...”自动起个名
    label = args.label or f"{Path(args.config).stem}:{cfg['rerank'].get('mode')}:pt={cfg['chunking'].get('prepend_title', False)}"

    corpus = None
    if args.ingest:                                    # 需要时先重建库（chunking/语料变了才需要）
        from rag_lab.pipeline import ingest_config

        info = ingest_config(cfg)
        corpus = {"docs": info["docs"], "chunks": info["chunks"], "store_count": info["store_count"]}
        print(f"Re-ingested: {corpus}")

    eval_out = run_eval(cfg)                            # ★ 跑评测
    record = build_record(cfg, label, eval_out, corpus)
    _print_summary(record)                             # 打印到终端

    if not args.no_record:                             # 默认会落盘记录 + 更新排行榜
        append_run(record)
        regenerate_leaderboard()
        print(f"\nLogged to {RUNS_FILE}  |  leaderboard -> {LEADERBOARD}")


if __name__ == "__main__":
    main()
