from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class SearchHit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    rerank_score: float | None = None
    rank_details: dict[str, Any] = field(default_factory=dict)

    @property
    def source_id(self) -> str:
        return str(self.metadata.get("source_id", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))
