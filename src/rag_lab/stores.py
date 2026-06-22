"""
================================================================================
stores.py —— 向量库读写（Chroma / Milvus）
--------------------------------------------------------------------------------
向量库负责：存 (chunk 文本 + 向量 + metadata)，并支持“给一个查询向量，返回最近的 N 个”。
本文件把两种向量库包成同一套接口：upsert(写) / search(查) / count(计数)，
上层 pipeline 不用关心底层是 Chroma 还是 Milvus。
================================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag_lab.models import Chunk, SearchHit


def _batch(items: list[Any], size: int = 128):
    """把列表切成每 size 个一组，分批写入（避免一次塞太多）。"""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class ChromaStore:
    """Chroma 向量库（本地文件持久化，开箱即用，项目默认）。"""
    def __init__(self, cfg: dict, dimension: int | None = None, reset: bool = False):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. Run: pip install -r requirements.txt") from exc
        path = Path(cfg["paths"].get("chroma_dir", "storage/chroma"))
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path))   # 数据落盘到这个目录
        self.name = str(cfg["vector_store"]["collection"])
        if reset:                                                 # 建库时清空旧集合
            try:
                self.client.delete_collection(self.name)
            except Exception:
                pass
        metric = str(cfg["vector_store"].get("metric", "cosine")).lower()
        self.collection = self.client.get_or_create_collection(
            name=self.name,
            metadata={"hnsw:space": metric},                      # 距离度量（cosine）
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """写入/更新：把 (chunk, 向量) 成对存进去，分批。"""
        records = list(zip(chunks, embeddings))
        for group in _batch(records):
            self.collection.upsert(
                ids=[chunk.id for chunk, _ in group],
                documents=[chunk.text for chunk, _ in group],
                metadatas=[chunk.metadata for chunk, _ in group],
                embeddings=[embedding for _, embedding in group],
            )

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        """给查询向量，返回最近的 top_k 个块（已转成 SearchHit）。"""
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        # Chroma 返回的是“批量”结构，这里只查了一条，取 [0]
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[SearchHit] = []
        for rank, (hit_id, text, metadata, distance) in enumerate(
            zip(ids, docs, metadatas, distances), start=1
        ):
            score = 1.0 - float(distance)            # cosine 距离 → 相似度分数（越大越像）
            hits.append(
                SearchHit(
                    id=str(hit_id),
                    text=str(text),
                    metadata=dict(metadata or {}),
                    score=score,
                    vector_score=score,
                    rank_details={"vector_rank": rank, "distance": float(distance)},
                )
            )
        return hits

    def count(self) -> int:
        return int(self.collection.count())


class MilvusStore:
    """Milvus 向量库（支持 Milvus Lite 本地 .db 文件，或连接 Docker 的 Milvus 服务）。"""
    def __init__(self, cfg: dict, dimension: int | None = None, reset: bool = False):
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:
            raise RuntimeError("pymilvus is not installed. Run: pip install -r requirements.txt") from exc
        uri = str(cfg["vector_store"].get("uri", "http://localhost:19530"))
        token = str(cfg["vector_store"].get("token", ""))
        self.is_local = "://" not in uri and uri.endswith(".db")   # 是否 Milvus Lite 本地文件
        self.reset = reset
        if "://" not in uri and uri.endswith(".db"):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
        self.client = MilvusClient(uri=uri, token=token or None)
        self.name = str(cfg["vector_store"]["collection"])
        self.metric = str(cfg["vector_store"].get("metric", "COSINE")).upper()
        if reset and self.client.has_collection(self.name):
            self.client.drop_collection(self.name)
        if not self.client.has_collection(self.name):
            if dimension is None:
                raise ValueError("Milvus collection creation requires embedding dimension.")
            # 建集合：主键 id(字符串) + 向量字段 + 动态字段（存 metadata）
            self.client.create_collection(
                collection_name=self.name,
                dimension=int(dimension),
                primary_field_name="id",
                id_type=DataType.VARCHAR,
                vector_field_name="vector",
                metric_type=self.metric,
                auto_id=False,
                max_length=256,
                enable_dynamic_field=True,
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """写入：每行 = id + 向量 + 文本 + 展开的 metadata。"""
        rows = []
        for chunk, embedding in zip(chunks, embeddings):
            row = {
                "id": chunk.id,
                "vector": embedding,
                "text": chunk.text,
                **chunk.metadata,
            }
            rows.append(row)
        for group in _batch(rows, size=128):
            if self.is_local or self.reset:
                self.client.insert(collection_name=self.name, data=group)
            else:
                self.client.upsert(collection_name=self.name, data=group)
        if not self.is_local:
            self.client.flush(collection_name=self.name)

    def search(self, query_embedding: list[float], top_k: int) -> list[SearchHit]:
        try:
            self.client.load_collection(collection_name=self.name)   # 查询前要先把集合加载到内存
        except Exception as exc:
            if "already" not in str(exc).lower():
                raise
        results = self.client.search(
            collection_name=self.name,
            data=[query_embedding],
            limit=top_k,
            output_fields=["text", "source_id", "title", "tags", "chunk_index"],
            search_params={"metric_type": self.metric, "params": {}},
        )
        # 兼容不同 pymilvus 版本的返回结构（对象 vs 字典）
        hits: list[SearchHit] = []
        for rank, item in enumerate(results[0], start=1):
            if hasattr(item, "entity"):
                entity = dict(item.entity)
                hit_id = str(item.id)
                distance = float(item.distance)
            else:
                entity = dict(item.get("entity", {}))
                hit_id = str(item.get("id"))
                distance = float(item.get("distance", item.get("score", 0.0)))
            score = self._distance_to_score(distance)
            text = str(entity.pop("text", ""))
            hits.append(
                SearchHit(
                    id=hit_id,
                    text=text,
                    metadata=entity,
                    score=score,
                    vector_score=score,
                    rank_details={"vector_rank": rank, "distance": distance},
                )
            )
        return hits

    def count(self) -> int:
        stats = self.client.get_collection_stats(collection_name=self.name)
        return int(stats.get("row_count", 0))

    def _distance_to_score(self, distance: float) -> float:
        """把不同度量的“距离”统一换算成“越大越相关”的分数。"""
        if self.metric == "COSINE":
            return 1.0 - distance
        if self.metric == "L2":
            return -distance
        return distance


def get_store(cfg: dict, dimension: int | None = None, reset: bool = False):
    """工厂函数：根据 vector_store.type 返回 Chroma 或 Milvus 的封装。"""
    store_type = str(cfg["vector_store"].get("type", "chroma")).lower()
    if store_type == "chroma":
        return ChromaStore(cfg, dimension=dimension, reset=reset)
    if store_type == "milvus":
        return MilvusStore(cfg, dimension=dimension, reset=reset)
    raise ValueError(f"Unsupported vector_store.type: {store_type}")
