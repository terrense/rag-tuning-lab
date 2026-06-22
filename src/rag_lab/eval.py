"""Batch evaluation over an eval-query set.

Runs every query in `paths.eval_queries` through the retrieval pipeline and
reports per-query hit/first_rank/MRR plus aggregate hit-rate and mean MRR.
This is the "use numbers, not vibes" loop: change a knob with --set, re-run,
compare the aggregate.
"""

from __future__ import annotations

import argparse
import json

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.loaders import load_eval_queries
from rag_lab.pipeline import evaluate_hits, query_config


def run_eval(cfg: dict) -> dict:
    queries = load_eval_queries(get_path(cfg, "eval_queries"))
    rows = []
    for item in queries:
        question = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
        result = query_config(cfg, question)
        ev = evaluate_hits(result["hits"], expected)
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
    hit_rate = sum(1 for r in rows if r["hit"]) / n
    mean_mrr = sum(r["mrr"] for r in rows) / n
    return {"rows": rows, "hit_rate": hit_rate, "mean_mrr": mean_mrr, "count": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-evaluate the eval query set.")
    parser.add_argument("--config", default="configs/diseases.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    summary = run_eval(cfg)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

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
