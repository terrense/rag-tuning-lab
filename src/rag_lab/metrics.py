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
import random as _random
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


# ============================================================================
# 统计显著性（P0-E0.5）：让 LEADERBOARD 上的差异"可信"，而不是噪声
# ----------------------------------------------------------------------------
# 为什么需要：N=10 的评测集上 Recall 0.70 vs 0.60 只差"一道题"，完全可能是
# 运气。两件工具：
#   bootstrap_ci             单个 run 的指标该报成 0.70 ± 多少（95% 置信区间）
#   paired_permutation_test  两个 run 的差异是真提升还是噪声（p 值）
# 都是无分布假设的重采样方法——评测指标（0/1 的 hit、截断的 mrr）远不是正态
# 分布，t 检验的前提不成立，所以用重采样而不是查表。
# ============================================================================
def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """逐题指标 → (均值, CI下界, CI上界)。

    做法：把 N 道题的成绩当作总体的样本，有放回地重抽 N 道题、算均值，
    重复 n_boot 次，取 [2.5%, 97.5%] 分位数。区间宽度直观反映"评测集
    有多小"——10 题的区间宽得吓人，这正是要扩评测集的证据。
    """
    if not values:
        return 0.0, 0.0, 0.0
    rng = _random.Random(seed)
    n = len(values)
    mean = sum(values) / n
    boots = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return mean, lo, hi


def paired_permutation_test(
    a: list[float], b: list[float], n_perm: int = 10000, seed: int = 0
) -> dict[str, float]:
    """配对置换检验：A、B 两套配置在同一批题上的逐题成绩，差异显著吗？

    配对的意义：同一道题两边都答了，比较的是"逐题差值"，题目难度本身的
    方差被消掉——比把两组当独立样本灵敏得多。
    零假设：A 和 B 没差别 → 每道题的差值正负号可以随便翻。
    做法：随机翻转差值符号 n_perm 次，看"真实平均差值"在这个零分布里有
    多极端。双尾 p < 0.05 → 差异大概率不是噪声。
    """
    assert len(a) == len(b), "paired test needs the same queries on both sides"
    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / (len(diffs) or 1)
    if all(d == 0 for d in diffs):
        return {"mean_diff": 0.0, "p_value": 1.0}
    rng = _random.Random(seed)
    hits = 0
    for _ in range(n_perm):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if abs(s) >= abs(observed) - 1e-12:
            hits += 1
    return {"mean_diff": observed, "p_value": (hits + 1) / (n_perm + 1)}
