"""
================================================================================
bm25.py —— BM25 关键词检索
--------------------------------------------------------------------------------
BM25 是经典的“关键词匹配”打分算法：看查询里的词在文档中出现的频率、
以及这个词稀不稀有，据此打分。本质是“字面命中”，不懂同义词，但对专有名词
（病名、药名、科室名）非常准——这正好补向量检索的短板。

中文处理：tokenize() 把每个汉字当一个 token（英文/数字按整词），
所以“苯中毒”会被切成 苯/中/毒 三个 token。
================================================================================
"""

from __future__ import annotations

import math
import re
from collections import Counter

from rag_lab.models import Chunk, SearchHit


# 每个汉字单独成 token；连续的英文/数字/常见符号合成一个 token。
# [一-鿿] 是 CJK 汉字的 Unicode 范围。
TOKEN_RE = re.compile(r"[一-鿿]|[a-zA-Z0-9_./:+-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _minmax(scores: list[float]) -> list[float]:
    """把一组分数归一化到 [0,1]，方便和别的分数比较/融合。"""
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


class BM25Index:
    """对一批 chunk 建 BM25 索引，支持按查询检索。"""
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [tokenize(chunk.text) for chunk in chunks]   # 每个块先分好词
        self._ranker = None
        try:
            from rank_bm25 import BM25Okapi
            self._ranker = BM25Okapi(self.tokens)                  # 用 rank_bm25 库建索引
        except Exception:
            self._ranker = None                                    # 库不可用就走下面的兜底打分

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        # 给每个块算一个 BM25 分
        if self._ranker is not None:
            raw_scores = [float(score) for score in self._ranker.get_scores(query_tokens)]
        else:
            raw_scores = self._fallback_scores(query_tokens)
        # 按分数从高到低排，只保留分数 > 0 的，取前 top_k
        ranked = sorted(enumerate(raw_scores), key=lambda item: item[1], reverse=True)
        ranked = [item for item in ranked if item[1] > 0][:top_k]
        # 把这 top_k 的原始分归一化到 [0,1]
        norm_lookup = {
            idx: score for (idx, _), score in zip(ranked, _minmax([score for _, score in ranked]))
        }
        # 包成 SearchHit 返回
        hits: list[SearchHit] = []
        for rank, (idx, raw_score) in enumerate(ranked, start=1):
            chunk = self.chunks[idx]
            score = norm_lookup.get(idx, 0.0)
            hits.append(
                SearchHit(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=score,
                    bm25_score=score,
                    rank_details={"bm25_rank": rank, "bm25_raw": raw_score},  # 留原始分供调试
                )
            )
        return hits

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
        """rank_bm25 不可用时的简化打分：词频对数加权求和（不是真 BM25，仅兜底）。"""
        query_counts = Counter(query_tokens)
        scores: list[float] = []
        for doc_tokens in self.tokens:
            doc_counts = Counter(doc_tokens)
            score = 0.0
            for token, q_count in query_counts.items():
                if token in doc_counts:
                    score += (1.0 + math.log1p(doc_counts[token])) * q_count
            scores.append(score)
        return scores
