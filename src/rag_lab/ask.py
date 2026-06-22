"""L1 CLI: retrieve + generate a grounded, cited answer (RAG end-to-end)."""

from __future__ import annotations

import argparse

from rag_lab.config import get_path, load_config, parse_value, set_dotted
from rag_lab.generate import generate_answer
from rag_lab.loaders import find_query, load_eval_queries
from rag_lab.pipeline import query_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question; RAG retrieves then generates.")
    parser.add_argument("--config", default="configs/diseases.yaml")
    parser.add_argument("--query", default="")
    parser.add_argument("--query-id", default="")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    query = args.query
    if args.query_id:
        item = find_query(load_eval_queries(get_path(cfg, "eval_queries")), args.query_id)
        query = str(item["question"])
    if not query:
        raise SystemExit("Pass --query or --query-id.")

    result = query_config(cfg, query)
    gen = generate_answer(cfg, query, result["hits"])

    print(f"问题：{query}\n")
    print("回答：")
    print(gen["answer"])
    print("\n引用来源：")
    for s in gen["sources"]:
        print(f"  [{s['n']}] {s['title']}  ({s['source_id']})")
    if gen.get("raw_usage"):
        print(f"\n(model={gen['model']}, usage={gen['raw_usage']})")


if __name__ == "__main__":
    main()
