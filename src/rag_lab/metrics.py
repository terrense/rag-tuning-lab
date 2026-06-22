"""Retrieval metrics for the eval harness.

All functions take a *ranked* list of source ids (best first) and a set/list of
expected (relevant) source ids, and return standard IR metrics. We report at
several cut-offs so we can fill in a Recall@5 / Recall@10 style table.
"""

from __future__ import annotations

import math
from typing import Iterable

DEFAULT_KS = (1, 3, 5, 10)


def _first_rank(ranked: list[str], expected: set[str]) -> int | None:
    for idx, sid in enumerate(ranked, start=1):
        if sid in expected:
            return idx
    return None


def hit_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    return 1.0 if any(sid in expected for sid in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Fraction of relevant docs found in the top-k (== hit@k when 1 relevant)."""
    if not expected:
        return 0.0
    found = len({sid for sid in ranked[:k] if sid in expected})
    return found / len(expected)


def mrr(ranked: list[str], expected: set[str]) -> float:
    fr = _first_rank(ranked, expected)
    return 0.0 if fr is None else 1.0 / fr


def ndcg_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    dcg = 0.0
    for idx, sid in enumerate(ranked[:k], start=1):
        if sid in expected:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def rank_metrics(
    ranked: list[str], expected: Iterable[str], ks: tuple[int, ...] = DEFAULT_KS
) -> dict[str, float | int | None]:
    expected_set = set(expected)
    out: dict[str, float | int | None] = {
        "first_rank": _first_rank(ranked, expected_set),
        "mrr": mrr(ranked, expected_set),
    }
    for k in ks:
        out[f"hit@{k}"] = hit_at_k(ranked, expected_set, k)
        out[f"recall@{k}"] = recall_at_k(ranked, expected_set, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, expected_set, k)
    return out


def mean_metrics(rows: list[dict], ks: tuple[int, ...] = DEFAULT_KS) -> dict[str, float]:
    """Average the per-query metric dicts produced by rank_metrics."""
    n = len(rows) or 1
    keys = ["mrr"] + [f"{m}@{k}" for k in ks for m in ("hit", "recall", "ndcg")]
    agg: dict[str, float] = {}
    for key in keys:
        agg[key] = sum(float(r.get(key) or 0.0) for r in rows) / n
    return agg
