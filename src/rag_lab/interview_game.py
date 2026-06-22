from __future__ import annotations

import argparse
import random

from rag_lab.config import get_path, load_config
from rag_lab.loaders import load_eval_queries
from rag_lab.pipeline import evaluate_hits, query_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Practice RAG interview questions with retrieval traces.")
    parser.add_argument("--config", default="configs/chroma.yaml")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    queries = load_eval_queries(get_path(cfg, "eval_queries"))
    random.shuffle(queries)
    for idx, item in enumerate(queries[: args.rounds], start=1):
        question = str(item["question"])
        expected = list(item.get("expected_source_ids", []))
        print(f"\nRound {idx}")
        print(f"Interviewer: {question}")
        print(f"Angle: {item.get('interview_angle', '')}")
        input("Think through your answer, then press Enter to reveal retrieval trace...")
        result = query_config(cfg, question)
        ev = evaluate_hits(result["hits"], expected)
        print(f"Expected source ids: {expected}")
        print(f"Hit={ev['hit']} first_rank={ev['first_rank']} mrr={ev['mrr']:.3f}")
        for rank, hit in enumerate(result["hits"], start=1):
            print(f"  [{rank}] {hit.source_id} | {hit.title} | score={hit.score:.4f}")


if __name__ == "__main__":
    main()
