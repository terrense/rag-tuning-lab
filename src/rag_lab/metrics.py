"""
================================================================================
metrics.py —— 检索评测指标（信息检索 IR 标准指标）
--------------------------------------------------------------------------------
每个函数都吃两样东西：
  - ranked   ：检索返回的 source_id 列表，按相关性从高到低排好
  - expected ：标准答案的 source_id 集合（哪些文档算“对”）
返回标准指标。我们在多个截断 k（1/3/5/10）上报告，好填出 Recall@5/@10 那种表。
================================================================================
"""

from __future__ import annotations

import math
from typing import Iterable

DEFAULT_KS = (1, 3, 5, 10)   # 默认在这些 k 上算指标


def _first_rank(ranked: list[str], expected: set[str]) -> int | None:
    """第一个“对”的文档排在第几名（1 起）；一个都没有返回 None。"""
    for idx, sid in enumerate(ranked, start=1):
        if sid in expected:
            return idx
    return None


def hit_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Hit@k：前 k 名里只要有一个对的，就算命中(1.0)，否则 0.0。"""
    return 1.0 if any(sid in expected for sid in ranked[:k]) else 0.0


def recall_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """Recall@k：前 k 名里找到了多少比例的“对”文档（只有 1 个标准答案时 == hit@k）。"""
    if not expected:
        return 0.0
    found = len({sid for sid in ranked[:k] if sid in expected})
    return found / len(expected)


def mrr(ranked: list[str], expected: set[str]) -> float:
    """MRR：第一个对的文档排名的倒数。排第 1 = 1.0，排第 2 = 0.5……没命中 = 0。"""
    fr = _first_rank(ranked, expected)
    return 0.0 if fr is None else 1.0 / fr


def ndcg_at_k(ranked: list[str], expected: set[str], k: int) -> float:
    """nDCG@k（二元相关性版）：在 Recall 基础上额外奖励“排得更靠前”。

    DCG：对的文档在第 idx 名贡献 1/log2(idx+1)，越靠前贡献越大。
    IDCG：理想情况（所有对的都排最前）的 DCG。nDCG = DCG / IDCG，归一到 [0,1]。
    """
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
    """一次性算出一道题的全部指标：first_rank、mrr，以及各 k 的 hit/recall/ndcg。"""
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
    """把多道题的指标 dict 求平均（评测集的总体表现）。"""
    n = len(rows) or 1
    keys = ["mrr"] + [f"{m}@{k}" for k in ks for m in ("hit", "recall", "ndcg")]
    agg: dict[str, float] = {}
    for key in keys:
        agg[key] = sum(float(r.get(key) or 0.0) for r in rows) / n
    return agg
