"""
================================================================================
models.py —— 两个贯穿全项目的数据结构
--------------------------------------------------------------------------------
  - Chunk     ：一个文本块（建库的最小单位）
  - SearchHit ：一次检索命中（带各种分数，在检索链路里被不断丰富）
用 @dataclass 自动生成 __init__ 等样板代码。
================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    """一个文本块。id 形如 'disease_00042::chunk_001'。"""
    id: str
    text: str                       # 块的正文（embedding/BM25 都基于它）
    metadata: dict[str, Any]        # 附带信息：source_id/title/科室/症状... 用于过滤和展示


@dataclass
class SearchHit:
    """一次检索命中。粗筛、融合、精排各阶段会往里填不同的分数字段。"""
    id: str
    text: str
    metadata: dict[str, Any]
    score: float                              # 当前主分数（融合后/精排后会被更新）
    vector_score: float | None = None         # 向量这一路的分
    bm25_score: float | None = None           # BM25 这一路的分
    rerank_score: float | None = None         # 精排分
    rank_details: dict[str, Any] = field(default_factory=dict)  # 各阶段排名/距离等调试信息

    @property
    def source_id(self) -> str:
        """这个块属于哪个原始文档（评测判命中、按文档去重都靠它）。"""
        return str(self.metadata.get("source_id", ""))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", ""))
