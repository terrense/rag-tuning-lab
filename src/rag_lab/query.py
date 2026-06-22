"""
================================================================================
query.py —— 命令行入口：跑一次检索（只检索，不生成答案）
--------------------------------------------------------------------------------
用法：
  python -m rag_lab.query --config configs/diseases.yaml --query "苯中毒的症状？"
  python -m rag_lab.query --config configs/diseases.yaml --query-id d_benzene
能用 --query 直接传问题，或用 --query-id 引用题库里的题（这样还会自动算评测指标）。
--json 输出机器可读的 JSON；否则美化打印。
================================================================================
"""

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
    parser.add_argument("--query", default="")          # 直接给问题
    parser.add_argument("--query-id", default="")       # 或引用题库里的题 id
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a config value, for example --set rerank.mode=none",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:                               # 命令行覆盖配置
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))

    # 决定问题文本；用 --query-id 时还会取出标准答案，方便顺带评测
    expected_source_ids: list[str] = []
    query = args.query
    if args.query_id:
        item = find_query(load_eval_queries(get_path(cfg, "eval_queries")), args.query_id)
        query = str(item["question"])
        expected_source_ids = list(item.get("expected_source_ids", []))
    if not query:
        raise SystemExit("Pass --query or --query-id.")

    result = query_config(cfg, query)                   # ★ 跑检索
    if expected_source_ids:                             # 有标准答案就算 hit/first_rank/mrr
        result["eval"] = evaluate_hits(result["hits"], expected_source_ids)
        result["expected_source_ids"] = expected_source_ids

    if args.json:
        # 输出 JSON：dataclass 转成 dict
        payload = dict(result)
        payload["hits"] = [asdict(hit) for hit in result["hits"]]
        payload["vector_hits"] = [asdict(hit) for hit in result["vector_hits"]]
        payload["bm25_hits"] = [asdict(hit) for hit in result["bm25_hits"]]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        # 美化打印
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
