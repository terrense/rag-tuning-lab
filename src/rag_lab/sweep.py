from __future__ import annotations

import argparse
import copy
import itertools
import json
from typing import Any

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.loaders import find_query, load_eval_queries
from rag_lab.pipeline import evaluate_hits, ingest_config, query_config


def _parse_vary(items: list[str]) -> list[tuple[str, list[Any]]]:
    parsed: list[tuple[str, list[Any]]] = []
    for item in items:
        key, raw_values = item.split("=", 1)
        parsed.append((key, [parse_value(value) for value in raw_values.split(",")]))
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep several RAG parameters.")
    parser.add_argument("--config", default="configs/chroma.yaml")
    parser.add_argument("--query-id", default="q_rerank")
    parser.add_argument("--query", default="")
    parser.add_argument(
        "--vary",
        action="append",
        default=[],
        help="Parameter grid, for example --vary retrieval.candidate_k=6,12",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    query = args.query
    expected: list[str] = []
    if args.query_id:
        item = find_query(load_eval_queries(get_path(base_cfg, "eval_queries")), args.query_id)
        query = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
    if not query:
        raise SystemExit("Pass --query or --query-id.")

    variations = _parse_vary(args.vary) or [
        ("chunking.chunk_size", [240, 360, 520]),
        ("retrieval.candidate_k", [6, 12]),
        ("rerank.mode", ["none", "bm25"]),
    ]
    keys = [key for key, _ in variations]
    rows = []

    for values in itertools.product(*[vals for _, vals in variations]):
        cfg = copy.deepcopy(base_cfg)
        for key, value in zip(keys, values):
            set_dotted(cfg, key, value)
        cfg["vector_store"]["reset_on_ingest"] = True
        ingest_config(cfg)
        result = query_config(cfg, query)
        ev = evaluate_hits(result["hits"], expected) if expected else {}
        row = {
            **{key: value for key, value in zip(keys, values)},
            "top_sources": [hit.source_id for hit in result["hits"]],
            "hit": ev.get("hit"),
            "first_rank": ev.get("first_rank"),
            "mrr": ev.get("mrr"),
        }
        rows.append(row)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print(f"Query: {query}")
    print(f"Expected: {expected}")
    for idx, row in enumerate(rows, start=1):
        params = " ".join(f"{key}={row[key]}" for key in keys)
        print(
            f"[{idx:02d}] {params} | hit={row['hit']} "
            f"first_rank={row['first_rank']} mrr={row['mrr']} "
            f"top={row['top_sources']}"
        )


if __name__ == "__main__":
    main()
