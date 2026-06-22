"""
================================================================================
sweep.py —— 命令行入口：参数网格扫描
--------------------------------------------------------------------------------
对一个问题，自动遍历多组参数组合（每组都重新建库 + 查询 + 评测），对比哪组最好。
用法示例：
  python -m rag_lab.sweep --config configs/diseases.yaml --query-id d_benzene \
      --vary chunking.chunk_size=240,360,520 --vary rerank.mode=none,cross_encoder
注意：每个组合都会重新 ingest，组合多会很慢。这是“单题”快速试参；
要系统化看全套指标用 rag_lab.experiment。
================================================================================
"""

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
    """把 --vary key=v1,v2,v3 解析成 (key, [v1,v2,v3])。"""
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
        action="append",       # 可多个 --vary，构成多维网格
        default=[],
        help="Parameter grid, for example --vary retrieval.candidate_k=6,12",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    # 确定问题（直接给 或 从题库取，后者带标准答案）
    query = args.query
    expected: list[str] = []
    if args.query_id:
        item = find_query(load_eval_queries(get_path(base_cfg, "eval_queries")), args.query_id)
        query = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
    if not query:
        raise SystemExit("Pass --query or --query-id.")

    # 没指定 --vary 时给一组默认网格
    variations = _parse_vary(args.vary) or [
        ("chunking.chunk_size", [240, 360, 520]),
        ("retrieval.candidate_k", [6, 12]),
        ("rerank.mode", ["none", "bm25"]),
    ]
    keys = [key for key, _ in variations]
    rows = []

    # itertools.product 生成所有参数组合的笛卡尔积
    for values in itertools.product(*[vals for _, vals in variations]):
        cfg = copy.deepcopy(base_cfg)
        for key, value in zip(keys, values):           # 把这组参数写进 cfg
            set_dotted(cfg, key, value)
        cfg["vector_store"]["reset_on_ingest"] = True   # 每组都重建库
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

    # 美化打印每个组合的结果
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
