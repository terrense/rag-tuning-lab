from __future__ import annotations

import os
from typing import Protocol

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class HashingEmbedder:
    def __init__(self, cfg: dict):
        emb_cfg = cfg["embedding"]
        self.dimension = int(emb_cfg.get("dimension", 768))
        analyzer = str(emb_cfg.get("analyzer", "char_wb"))
        ngram_min = int(emb_cfg.get("ngram_min", 2))
        ngram_max = int(emb_cfg.get("ngram_max", 4))
        self._normalize = bool(emb_cfg.get("normalize", True))
        self.vectorizer = HashingVectorizer(
            analyzer=analyzer,
            ngram_range=(ngram_min, ngram_max),
            n_features=self.dimension,
            alternate_sign=False,
            norm="l2" if self._normalize else None,
            lowercase=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.transform(texts)
        if self._normalize:
            matrix = normalize(matrix, norm="l2", copy=False)
        return matrix.astype(np.float32).toarray().tolist()


class SentenceTransformerEmbedder:
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
        self.model = SentenceTransformer(self.model_name)
        if hasattr(self.model, "get_embedding_dimension"):
            dim = self.model.get_embedding_dimension()
        else:
            dim = self.model.get_sentence_embedding_dimension()
        self.dimension = int(dim) if dim else int(emb_cfg.get("dimension", 768))

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


def get_embedder(cfg: dict) -> Embedder:
    provider = str(cfg["embedding"].get("provider", "hashing")).lower()
    if provider in {"sentence-transformers", "sentence_transformers", "st"}:
        return SentenceTransformerEmbedder(cfg)
    if provider == "hashing":
        return HashingEmbedder(cfg)
    raise ValueError(f"Unsupported embedding.provider: {provider}")
