"""Experiment runner + tracker.

Runs the eval-query set through the pipeline, measures real numbers (Recall@k,
MRR, nDCG@k, latency) at several retrieval stages, and APPENDS every run to a
tracked log so the project accumulates quantified history instead of vibes:

    experiments/runs.jsonl      one JSON object per run (committed)
    experiments/LEADERBOARD.md  regenerated table, sorted by pipeline Recall@5

Each run records: timestamp, git sha, label, the knobs that matter (chunk size,
overlap, prepend_title, hybrid weights, candidate_k, rerank mode, embedding
model), corpus size, and per-stage aggregate metrics.

Stages compared (the classic BM25 / vector / hybrid / hybrid+rerank table):
- bm25_only      : raw BM25 channel
- vector_only    : raw dense channel
- hybrid         : RRF fusion, no rerank
- hybrid_rerank  : RRF fusion + cross-encoder (the configured pipeline)
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
from rag_lab.metrics import DEFAULT_KS, mean_metrics, rank_metrics
from rag_lab.pipeline import query_config

EXP_DIR = Path("experiments")
RUNS_FILE = EXP_DIR / "runs.jsonl"
LEADERBOARD = EXP_DIR / "LEADERBOARD.md"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "nogit"


def _ids(hits) -> list[str]:
    return [h.source_id for h in hits]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((pct / 100) * (len(s) - 1))))
    return s[idx]


def run_eval(cfg: dict, ks: tuple[int, ...] = DEFAULT_KS, with_hybrid_norerank: bool = True) -> dict:
    """Run the eval set once; return per-stage aggregate metrics + latency."""
    queries = load_eval_queries(get_path(cfg, "eval_queries"))

    # Make sure we retrieve deep enough to score @max(ks).
    maxk = max(ks)
    cfg_main = copy.deepcopy(cfg)
    cfg_main["retrieval"]["top_k"] = max(int(cfg_main["retrieval"].get("top_k", 5)), maxk)
    cfg_main["retrieval"]["candidate_k"] = max(
        int(cfg_main["retrieval"].get("candidate_k", 12)), maxk
    )

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

        t0 = time.perf_counter()
        result = query_config(cfg_main, question)
        latencies.append((time.perf_counter() - t0) * 1000)

        bm25_m = rank_metrics(_ids(result["bm25_hits"]), expected, ks)
        vec_m = rank_metrics(_ids(result["vector_hits"]), expected, ks)
        pipe_m = rank_metrics(_ids(result["hits"]), expected, ks)
        stage_rows["bm25_only"].append(bm25_m)
        stage_rows["vector_only"].append(vec_m)
        stage_rows["hybrid_rerank"].append(pipe_m)
        per_query.append(
            {
                "id": item.get("id"),
                "expected": expected,
                "pipeline_first_rank": pipe_m["first_rank"],
                "pipeline_mrr": pipe_m["mrr"],
                "top_sources": _ids(result["hits"])[:5],
            }
        )

    # Optional extra pass: hybrid fusion WITHOUT rerank, to complete the table.
    if with_hybrid_norerank and str(cfg["rerank"].get("mode", "none")) != "none":
        cfg_nr = copy.deepcopy(cfg_main)
        cfg_nr["rerank"]["mode"] = "none"
        rows = []
        for item in queries:
            result = query_config(cfg_nr, str(item["question"]))
            rows.append(rank_metrics(_ids(result["hits"]), item.get("expected_source_ids", []), ks))
        stage_rows["hybrid"] = rows

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
    r = cfg["retrieval"]
    c = cfg["chunking"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "label": label,
        "config_path": cfg.get("_config_path"),
        "params": {
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
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def regenerate_leaderboard() -> None:
    if not RUNS_FILE.exists():
        return
    runs = [json.loads(line) for line in RUNS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    def pipe_recall5(run: dict) -> float:
        return float(run.get("stages", {}).get("hybrid_rerank", {}).get("recall@5", 0.0))

    runs_sorted = sorted(runs, key=pipe_recall5, reverse=True)
    lines = [
        "# Experiment Leaderboard",
        "",
        "Auto-generated by `python -m rag_lab.experiment`. Sorted by pipeline Recall@5.",
        "Pipeline = the configured retrieval stack (hybrid + rerank unless noted).",
        "",
        "| Label | When (UTC) | sha | chunk/ovlp | prepend | cand_k | bm25_w | rerank | Recall@5 | MRR | nDCG@5 | p50 ms | N |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in runs_sorted:
        p = run.get("params", {})
        s = run.get("stages", {}).get("hybrid_rerank", {})
        lat = run.get("latency_ms", {})
        lines.append(
            "| {label} | {ts} | {sha} | {cs}/{ov} | {pp} | {ck} | {bw} | {rr} | {r5:.3f} | {mrr:.3f} | {n5:.3f} | {p50:.0f} | {n} |".format(
                label=run.get("label", ""),
                ts=run.get("timestamp", "")[:16].replace("T", " "),
                sha=run.get("git_sha", ""),
                cs=p.get("chunk_size"),
                ov=p.get("chunk_overlap"),
                pp="Y" if p.get("prepend_title") else "n",
                ck=p.get("candidate_k"),
                bw=p.get("bm25_weight"),
                rr=p.get("rerank_mode"),
                r5=float(s.get("recall@5", 0.0)),
                mrr=float(s.get("mrr", 0.0)),
                n5=float(s.get("ndcg@5", 0.0)),
                p50=float(lat.get("p50", 0.0)),
                n=run.get("n_queries", 0),
            )
        )
    LEADERBOARD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(record: dict) -> None:
    print(f"\nExperiment: {record['label']}  (sha={record['git_sha']}, n={record['n_queries']})")
    p = record["params"]
    print(f"  params: chunk={p['chunk_size']}/{p['chunk_overlap']} prepend_title={p['prepend_title']} "
          f"hybrid={p['hybrid']} cand_k={p['candidate_k']} rerank={p['rerank_mode']}")
    lat = record["latency_ms"]
    print(f"  latency: p50={lat['p50']:.0f}ms p90={lat['p90']:.0f}ms mean={lat['mean']:.0f}ms")
    print("  stage comparison (averaged over eval set):")
    order = ["bm25_only", "vector_only", "hybrid", "hybrid_rerank"]
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
    for item in args.set:
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    label = args.label or f"{Path(args.config).stem}:{cfg['rerank'].get('mode')}:pt={cfg['chunking'].get('prepend_title', False)}"

    corpus = None
    if args.ingest:
        from rag_lab.pipeline import ingest_config

        info = ingest_config(cfg)
        corpus = {"docs": info["docs"], "chunks": info["chunks"], "store_count": info["store_count"]}
        print(f"Re-ingested: {corpus}")

    eval_out = run_eval(cfg)
    record = build_record(cfg, label, eval_out, corpus)
    _print_summary(record)

    if not args.no_record:
        append_run(record)
        regenerate_leaderboard()
        print(f"\nLogged to {RUNS_FILE}  |  leaderboard -> {LEADERBOARD}")


if __name__ == "__main__":
    main()
