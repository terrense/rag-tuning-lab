from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.formatting import print_query_result
from rag_lab.loaders import find_query, load_eval_queries
from rag_lab.pipeline import evaluate_hits, query_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one RAG retrieval query.")
    parser.add_argument("--config", default="configs/chroma.yaml")
    parser.add_argument("--query", default="")
    parser.add_argument("--query-id", default="")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value, for example --set rerank.mode=none",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    expected_source_ids: list[str] = []
    query = args.query
    if args.query_id:
        item = find_query(load_eval_queries(get_path(cfg, "eval_queries")), args.query_id)
        query = str(item["question"])
        expected_source_ids = list(item.get("expected_source_ids", []))
    if not query:
        raise SystemExit("Pass --query or --query-id.")

    result = query_config(cfg, query)
    if expected_source_ids:
        result["eval"] = evaluate_hits(result["hits"], expected_source_ids)
        result["expected_source_ids"] = expected_source_ids

    if args.json:
        payload = dict(result)
        payload["hits"] = [asdict(hit) for hit in result["hits"]]
        payload["vector_hits"] = [asdict(hit) for hit in result["vector_hits"]]
        payload["bm25_hits"] = [asdict(hit) for hit in result["bm25_hits"]]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print_query_result(result, snippet_chars=int(cfg["display"].get("snippet_chars", 220)))
        if "eval" in result:
            ev = result["eval"]
            print(
                "\nEval"
                f"  expected={expected_source_ids} hit={ev['hit']} "
                f"first_rank={ev['first_rank']} mrr={ev['mrr']:.3f}"
            )


if __name__ == "__main__":
    main()
