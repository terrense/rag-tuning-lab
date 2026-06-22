"""
================================================================================
eval.py —— 批量评测（简版）
--------------------------------------------------------------------------------
把题库里每道题都跑一遍检索，打印每题的 hit/first_rank/MRR + 总体命中率/平均MRR。
这是“用数字代替感觉”的最小闭环：--set 改个参数 → 重跑 → 看总体指标变化。
（要更全面的多阶段指标 + 历史追踪，用 rag_lab.experiment。）
================================================================================
"""

from __future__ import annotations

import argparse
import json

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.loaders import load_eval_queries
from rag_lab.pipeline import evaluate_hits, query_config


def run_eval(cfg: dict) -> dict:
    """跑完整个题库，返回每题结果 + 汇总（命中率、平均 MRR）。"""
    queries = load_eval_queries(get_path(cfg, "eval_queries"))
    rows = []
    for item in queries:
        question = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
        result = query_config(cfg, question)          # 跑检索
        ev = evaluate_hits(result["hits"], expected)  # 算这题的指标
        rows.append(
            {
                "id": item.get("id"),
                "hit": ev["hit"],
                "first_rank": ev["first_rank"],
                "mrr": ev["mrr"],
                "expected": expected,
                "top_sources": [hit.source_id for hit in result["hits"]],
            }
        )
    n = len(rows) or 1
    hit_rate = sum(1 for r in rows if r["hit"]) / n    # 命中率 = 命中题数 / 总题数
    mean_mrr = sum(r["mrr"] for r in rows) / n         # 平均 MRR
    return {"rows": rows, "hit_rate": hit_rate, "mean_mrr": mean_mrr, "count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate the eval query set.")
    parser.add_argument("--config", default="configs/diseases.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:                              # 命令行覆盖配置
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    summary = run_eval(cfg)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # 美化打印：每题一行，未命中的额外显示它实际召回了啥（方便排查）
    print(f"Eval over {summary['count']} queries  "
          f"(config={args.config}, rerank={cfg['rerank'].get('mode')}, "
          f"prepend_title={cfg['chunking'].get('prepend_title', False)})")
    print("-" * 72)
    for r in summary["rows"]:
        mark = "HIT " if r["hit"] else "MISS"
        print(f"  [{mark}] {str(r['id']):16s} first_rank={str(r['first_rank']):>4} "
              f"mrr={r['mrr']:.3f}  expected={r['expected']}")
        if not r["hit"]:
            print(f"         got: {r['top_sources']}")
    print("-" * 72)
    print(f"  hit_rate = {summary['hit_rate']:.3f}    mean_mrr = {summary['mean_mrr']:.3f}")


if __name__ == "__main__":
    main()
