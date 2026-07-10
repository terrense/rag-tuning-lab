"""
================================================================================
gen_eval.py —— P0-E0.7：生成端评测（检索指标好 ≠ 答案好，这里补后半段）
--------------------------------------------------------------------------------
两类指标，从便宜到贵：

  A) 程序可算的引用指标（每题零成本）：
     citation_valid    答案里的 [n] 是否都是合法编号（幻觉引用=编造不存在的资料号）
     citation_precision 引用的资料里有多大比例真的是标准答案文档
     gold_cited        标准答案文档被检索到时，模型有没有真的引用它
     abstain           资料不足时有没有诚实说"无法确定"（而不是编）

  B) LLM-as-judge（每题一次 judge 调用）：
     faithfulness 1-5  答案的每个论断是否都有资料支撑（抓幻觉）
     relevance    1-5  是否正面回答了问题
     裁判走 llm.roles.judge（默认 deepseek-pro）≠ 生成模型（默认 minimax），
     避免"自己评自己"的自偏袒。E6 做生成模型 A/B 时换 llm.roles.generate 即可。

结果追加到 experiments/gen_runs.jsonl，可与检索 LEADERBOARD 并排讲故事。

用法：
    python -m rag_lab.gen_eval --config configs/diseases.yaml --limit 10 --label gen-smoke
    python -m rag_lab.gen_eval --config configs/diseases.yaml \
        --set llm.roles.generate=deepseek-pro --label gen-deepseek   # E6 的 A/B
================================================================================
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.generate import build_context, generate_answer
from rag_lab.llm import chat, resolve_role
from rag_lab.loaders import load_eval_queries
from rag_lab.metrics import bootstrap_ci
from rag_lab.pipeline import query_config

GEN_RUNS = Path("experiments/gen_runs.jsonl")

_CITE_RE = re.compile(r"\[(\d+)\]")
_ABSTAIN_MARKERS = ("无法确定", "资料不足", "现有资料无法", "没有足够的资料")

_JUDGE_SYS = (
    "你是严格的RAG答案质检裁判。给你【问题】【编号资料】【待评答案】，"
    "请只依据资料评判答案（不要用你自己的医学知识补位）：\n"
    "- faithfulness (1-5)：答案中的每个事实性论断是否都能在资料中找到依据。"
    "5=全部有依据；3=多数有依据但有少量资料外内容；1=大量编造。\n"
    "- relevance (1-5)：是否正面、完整地回答了问题。5=完整正面；3=只答了一半；1=答非所问。\n"
    "- unsupported: 列出资料中找不到依据的论断（没有则空数组）。\n"
    '只输出 JSON：{"faithfulness": n, "relevance": n, "unsupported": ["..."]}'
)


def _parse_json_obj(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return {}
    try:
        out = json.loads(text[s : e + 1])
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def citation_metrics(answer: str, sources: list[dict], expected: list[str]) -> dict:
    """程序算引用指标。sources 是 generate 时的编号→source_id 对照表。"""
    cited_nums = [int(n) for n in _CITE_RE.findall(answer)]
    valid_ids = {s["n"]: s["source_id"] for s in sources}
    valid = [n for n in cited_nums if n in valid_ids]
    cited_sources = {valid_ids[n] for n in valid}
    expected_set = set(expected)
    retrieved_gold = any(s["source_id"] in expected_set for s in sources)
    abstained = any(m in answer for m in _ABSTAIN_MARKERS)
    return {
        "n_citations": len(cited_nums),
        # 编号全合法=1.0；出现编造的资料号按比例扣
        "citation_valid": (len(valid) / len(cited_nums)) if cited_nums else 0.0,
        # 引用的资料里，多大比例真是标准答案文档
        "citation_precision": (len(cited_sources & expected_set) / len(cited_sources))
                              if cited_sources else 0.0,
        # 标准答案文档在候选里时，有没有被引用到（引用召回）
        "gold_cited": float(bool(cited_sources & expected_set)) if retrieved_gold else None,
        "retrieved_gold": retrieved_gold,
        "abstained": abstained,
        # 检索没找到金标文档时，诚实弃答才是对的；找到了还弃答是错的
        "abstain_correct": (abstained == (not retrieved_gold)),
    }


def judge_answer(cfg: dict, question: str, context: str, answer: str) -> dict:
    user = f"【问题】{question}\n\n【编号资料】\n{context}\n\n【待评答案】\n{answer}"
    messages = [{"role": "system", "content": _JUDGE_SYS},
                {"role": "user", "content": user}]
    # unsupported 列表长时 JSON 可能被 max_tokens 截断 → 解析失败会把好答案
    # 误判成 0 分，比裁判本身不准还糟。所以给足 token，失败再重试一次。
    out = chat(cfg, messages, role="judge", max_tokens=1500, temperature=0.0)
    obj = _parse_json_obj(out["text"])
    if not obj:
        print(f"  [judge] parse failed, retrying. head={out['text'][:100]!r}")
        out = chat(cfg, messages, role="judge", max_tokens=2500, temperature=0.0)
        obj = _parse_json_obj(out["text"])
    return {
        "faithfulness": float(obj.get("faithfulness") or 0),
        "relevance": float(obj.get("relevance") or 0),
        "unsupported": obj.get("unsupported") or [],
        "judge_parse_ok": bool(obj),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the generation stage (citations + judge).")
    ap.add_argument("--config", default="configs/diseases.yaml")
    ap.add_argument("--label", default="gen-run")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 题（控成本），0=全部")
    ap.add_argument("--no-judge", action="store_true", help="只算程序指标，不调裁判")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw))

    queries = load_eval_queries(get_path(cfg, "eval_queries"))
    if args.limit:
        queries = queries[: args.limit]
    gen_alias = resolve_role(cfg, "generate")
    judge_alias = resolve_role(cfg, "judge")
    if not args.no_judge and gen_alias == judge_alias:
        print(f"warning: generator == judge ({gen_alias}) — 自评有自偏袒，建议换 llm.roles.judge")

    rows: list[dict] = []
    for i, item in enumerate(queries, 1):
        q = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
        t0 = time.perf_counter()
        result = query_config(cfg, q)
        answer_out = generate_answer(cfg, q, result["hits"])
        latency_ms = (time.perf_counter() - t0) * 1000
        answer, sources = answer_out["answer"], answer_out["sources"]

        row = {"id": item.get("id"), "latency_ms": latency_ms}
        row.update(citation_metrics(answer, sources, expected))
        if not args.no_judge:
            # 裁判必须看"答案实际生成时用的那份上下文"——parent 模式下是父文档，
            # 否则裁判把父文档里的正确信息误判成"无依据"，不公平地压低 parent 分。
            from rag_lab.generate import _build_messages
            gen_msgs, _, _ = _build_messages(cfg, q, result["hits"])
            uc = gen_msgs[1]["content"]
            judge_ctx = uc if isinstance(uc, str) else uc[0].get("text", "")
            try:                                    # 单次裁判失败(限流/超时)不该拖垮全局
                row.update(judge_answer(cfg, q, judge_ctx, answer))
            except Exception as exc:
                print(f"  [judge] call failed ({type(exc).__name__}), skipping this row")
                row.update({"faithfulness": 0.0, "relevance": 0.0, "judge_parse_ok": False})
        rows.append(row)
        print(f"[{i}/{len(queries)}] {item.get('id')}: cite_p={row['citation_precision']:.2f} "
              + (f"faith={row.get('faithfulness', 0):.0f} rel={row.get('relevance', 0):.0f}"
                 if not args.no_judge else "") )

    # 裁判解析失败的行（judge 返回空/坏 JSON）是"测量失败"，不是"质量 0 分"——
    # 算 faithfulness/relevance 均值时必须剔除，否则 API 抖动会污染结论。
    judged = [r for r in rows if r.get("judge_parse_ok")]
    n_judge_fail = sum(1 for r in rows if "judge_parse_ok" in r and not r["judge_parse_ok"])

    def _mean(key: str, source: list[dict]) -> float:
        vals = [float(r[key]) for r in source if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {k: _mean(k, rows) for k in
               ["citation_valid", "citation_precision", "gold_cited", "abstain_correct"]}
    if not args.no_judge:
        summary["faithfulness"] = _mean("faithfulness", judged)
        summary["relevance"] = _mean("relevance", judged)
        summary["judge_ok"] = len(judged)
        summary["judge_failed"] = n_judge_fail
    faith_vals = [r["faithfulness"] for r in judged if "faithfulness" in r]
    if faith_vals:
        _, lo, hi = bootstrap_ci(faith_vals)
        summary["faithfulness_ci"] = [round(lo, 2), round(hi, 2)]

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": args.label,
        "eval_set": Path(get_path(cfg, "eval_queries")).stem.replace("eval_queries_", ""),
        "generator": gen_alias,
        "judge": None if args.no_judge else judge_alias,
        "n": len(rows),
        "summary": summary,
        "per_query": rows,
    }
    GEN_RUNS.parent.mkdir(parents=True, exist_ok=True)
    with GEN_RUNS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n=== {args.label}  gen={gen_alias} judge={record['judge']} n={len(rows)} ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v if isinstance(v, list) else f'{v:.3f}'}")
    print(f"logged -> {GEN_RUNS}")


if __name__ == "__main__":
    main()
