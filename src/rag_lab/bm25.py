from __future__ import annotations

import math
import re
from collections import Counter

from rag_lab.models import Chunk, SearchHit


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_./:+-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if math.isclose(low, high):
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self._ranker = None
        try:
            from rank_bm25 import BM25Okapi

            self._ranker = BM25Okapi(self.tokens)
        except Exception:
            self._ranker = None

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []
        if self._ranker is not None:
            raw_scores = [float(score) for score in self._ranker.get_scores(query_tokens)]
        else:
            raw_scores = self._fallback_scores(query_tokens)
        ranked = sorted(enumerate(raw_scores), key=lambda item: item[1], reverse=True)
        ranked = [item for item in ranked if item[1] > 0][:top_k]
        norm_lookup = {
            idx: score for (idx, _), score in zip(ranked, _minmax([score for _, score in ranked]))
        }
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
                    rank_details={"bm25_rank": rank, "bm25_raw": raw_score},
                )
            )
        return hits

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
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
