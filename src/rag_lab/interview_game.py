"""
================================================================================
interview_game.py —— 面试练习模式
--------------------------------------------------------------------------------
用法：python -m rag_lab.interview_game --config configs/play.yaml --rounds 5
从题库随机抽题，先让你自己想答案（回车揭晓），再展示检索链路命中情况。
重点不是背答案，而是观察“检索在哪成功、在哪失败”。
================================================================================
"""

from __future__ import annotations

import argparse
import random

from rag_lab.config import get_path, load_config
from rag_lab.loaders import load_eval_queries
from rag_lab.pipeline import evaluate_hits, query_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice RAG interview questions with retrieval traces.")
    parser.add_argument("--config", default="configs/chroma.yaml")
    parser.add_argument("--rounds", type=int, default=3)     # 练几轮
    args = parser.parse_args()

    cfg = load_config(args.config)
    queries = load_eval_queries(get_path(cfg, "eval_queries"))
    random.shuffle(queries)                                  # 打乱顺序
    for idx, item in enumerate(queries[: args.rounds], start=1):
        question = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
        print(f"\nRound {idx}")
        print(f"Interviewer: {question}")
        print(f"Angle: {item.get('interview_angle', '')}")    # 这题考察什么
        input("Think through your answer, then press Enter to reveal retrieval trace...")
        # 揭晓：跑检索 + 评测，看看正确资料排在第几
        result = query_config(cfg, question)
        ev = evaluate_hits(result["hits"], expected)
        print(f"Expected source ids: {expected}")
        print(f"Hit={ev['hit']} first_rank={ev['first_rank']} mrr={ev['mrr']:.3f}")
        for rank, hit in enumerate(result["hits"], start=1):
            print(f"  [{rank}] {hit.source_id} | {hit.title} | score={hit.score:.4f}")


if __name__ == "__main__":
    main()
