"""
================================================================================
embeddings.py —— 把“文字”变成“向量”
--------------------------------------------------------------------------------
向量 = 一串数字（如 384 个浮点数），意思相近的文字向量也相近。
建库时把每个 chunk 变成向量存进库；查询时把问题也变成向量去比对。

提供两种实现：
  - SentenceTransformerEmbedder：用神经网络模型（懂语义，质量高），项目默认
  - HashingEmbedder：纯哈希/字符 n-gram（不懂语义，但零依赖、超快），做兜底/对照
================================================================================
"""

from __future__ import annotations

import os
from typing import Protocol

# 限制底层数学库的线程数，避免在多核机器上过度并行导致反而变慢/不稳定。
# 必须在 import numpy 之前设置才生效。
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    """“嵌入器”的接口约定：有个 dimension（维度），有个 embed(文本列表)->向量列表。"""
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    """不依赖神经网络的兜底实现：把字符 n-gram 哈希到固定维度。快，但不懂语义。"""
    def __init__(self, cfg: dict):
        emb_cfg = cfg["embedding"]
        self.dimension = int(emb_cfg.get("dimension", 768))
        analyzer = str(emb_cfg.get("analyzer", "char_wb"))    # 按字符窗口取 n-gram
        ngram_min = int(emb_cfg.get("ngram_min", 2))
        ngram_max = int(emb_cfg.get("ngram_max", 4))
        self._normalize = bool(emb_cfg.get("normalize", True))
        self.vectorizer = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            n_features=self.dimension,                        # 哈希到这么多维
            alternate_sign=False,
            norm="l2" if self._normalize else None,
            lowercase=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.transform(texts)             # 稀疏矩阵
        if self._normalize:
            matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.astype(np.float32).toarray().tolist()   # 转成普通 list 返回


class SentenceTransformerEmbedder:
    """项目默认：用 sentence-transformers 神经模型，懂语义，质量高（但加载慢）。"""
    def __init__(self, cfg: dict):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install -r requirements-transformers.txt"
            ) from exc
        emb_cfg = cfg["embedding"]
        self.model_name = str(emb_cfg["model"])
        self._normalize = bool(emb_cfg.get("normalize", True))
        # device=None 让 sentence-transformers 自动选（有 CUDA 就用 GPU）；
        # 也可在配置里写 embedding.device: cuda / cpu 强制指定。
        device = emb_cfg.get("device") or None
        self.model = SentenceTransformer(self.model_name, device=device)     # ← 这一步从磁盘/网络加载模型，耗时
        # 不同版本接口名不一样，兼容一下，拿到向量维度
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        self.dimension = int(dim) if dim else int(emb_cfg.get("dimension", 768))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(                          # 批量编码成向量
            texts,
            normalize_embeddings=self._normalize,             # 归一化，配合 cosine
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


# 加载一个 SentenceTransformer 要好几秒；按模型名缓存，避免每次查询都重新加载。
# （这是我们把查询从 26 秒降到 10 秒的优化之一）
_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder(cfg: dict) -> Embedder:
    """工厂函数：根据配置 embedding.provider 返回对应的嵌入器（带缓存）。"""
    provider = str(cfg["embedding"].get("provider", "hashing")).lower()
    if provider in {"sentence-transformers", "sentence_transformers", "st"}:
        key = str(cfg["embedding"]["model"])
        if key not in _EMBEDDER_CACHE:                        # 同一个模型只加载一次
            _EMBEDDER_CACHE[key] = SentenceTransformerEmbedder(cfg)
        return _EMBEDDER_CACHE[key]
    if provider == "hashing":
        return HashingEmbedder(cfg)                           # 兜底实现很轻，不缓存也行
    raise ValueError(f"Unsupported embedding.provider: {provider}")
