from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from rag_lab.bm25 import BM25Index
from rag_lab.chunking import make_chunks
from rag_lab.config import get_path
from rag_lab.embeddings import get_embedder
from rag_lab.loaders import load_chunks, load_corpus, save_chunks
from rag_lab.models import Chunk, SearchHit
from rag_lab.rerankers import rerank_hits
from rag_lab.stores import get_store

# Cache loaded chunks + the BM25 index per chunks-cache file (keyed by mtime),
# so batch eval / repeated queries don't reload 27k chunks and rebuild BM25
# every single call. Invalidated automatically when the cache file changes.
_RETRIEVAL_CACHE: dict[str, tuple[float, list[Chunk], dict[str, Chunk], BM25Index]] = {}


def _get_retrieval_assets(chunks_path: Path) -> tuple[list[Chunk], dict[str, Chunk], BM25Index]:
    key = str(chunks_path)
    mtime = os.path.getmtime(chunks_path)
    cached = _RETRIEVAL_CACHE.get(key)
    if cached is None or cached[0] != mtime:
        chunks = load_chunks(chunks_path)
        lookup = {chunk.id: chunk for chunk in chunks}
        _RETRIEVAL_CACHE[key] = (mtime, chunks, lookup, BM25Index(chunks))
    _, chunks, lookup, bm25 = _RETRIEVAL_CACHE[key]
    return chunks, lookup, bm25


def ingest_config(cfg: dict[str, Any]) -> dict[str, Any]:
    docs, source_counts = load_corpus(cfg)
    chunks = make_chunks(docs, cfg)
    embedder = get_embedder(cfg)
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    if not embeddings:
        raise RuntimeError("No chunks were generated from the corpus.")
    dimension = len(embeddings[0])
    reset = bool(cfg["vector_store"].get("reset_on_ingest", True))
    store = get_store(cfg, dimension=dimension, reset=reset)
    store.upsert(chunks, embeddings)
    save_chunks(get_path(cfg, "chunks_cache"), chunks)
    return {
        "docs": len(docs),
        "source_counts": source_counts,
        "chunks": len(chunks),
        "dimension": dimension,
        "store_count": store.count(),
        "collection": cfg["vector_store"]["collection"],
        "store_type": cfg["vector_store"]["type"],
        "chunks_cache": str(get_path(cfg, "chunks_cache")),
    }


def query_config(cfg: dict[str, Any], query: str) -> dict[str, Any]:
    chunks_path = get_path(cfg, "chunks_cache")
    if not Path(chunks_path).exists():
        raise FileNotFoundError(f"Chunks cache not found: {chunks_path}. Run ingest first.")
    chunks, chunk_lookup, bm25_index = _get_retrieval_assets(chunks_path)
    embedder = get_embedder(cfg)
    query_embedding = embedder.embed([query])[0]
    store = get_store(cfg, dimension=len(query_embedding), reset=False)

    retrieval_cfg = cfg["retrieval"]
    candidate_k = int(retrieval_cfg.get("candidate_k", 12))
    vector_hits = store.search(query_embedding, top_k=candidate_k)
    bm25_hits = bm25_index.search(query, top_k=candidate_k)
    if bool(retrieval_cfg.get("hybrid", True)):
        candidates = combine_hits(
            vector_hits=vector_hits,
            bm25_hits=bm25_hits,
            chunk_lookup=chunk_lookup,
            vector_weight=float(retrieval_cfg.get("vector_weight", 0.7)),
            bm25_weight=float(retrieval_cfg.get("bm25_weight", 0.3)),
            rrf_k=float(retrieval_cfg.get("rrf_k", 60)),
        )
    else:
        candidates = copy.deepcopy(vector_hits)

    top_k = int(retrieval_cfg.get("top_k", 5))
    candidates = sorted(candidates, key=lambda hit: hit.score, reverse=True)[:candidate_k]
    final_hits = rerank_hits(query, candidates, cfg)[:top_k]
    return {
        "query": query,
        "hits": final_hits,
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "candidate_count": len(candidates),
        "config": {
            "store": cfg["vector_store"]["type"],
            "collection": cfg["vector_store"]["collection"],
            "chunk_size": cfg["chunking"]["chunk_size"],
            "chunk_overlap": cfg["chunking"]["chunk_overlap"],
            "candidate_k": candidate_k,
            "top_k": top_k,
            "hybrid": retrieval_cfg.get("hybrid", True),
            "rerank": cfg["rerank"].get("mode", "none"),
        },
    }


def combine_hits(
    vector_hits: list[SearchHit],
    bm25_hits: list[SearchHit],
    chunk_lookup: dict[str, Chunk],
    vector_weight: float,
    bm25_weight: float,
    rrf_k: float,
) -> list[SearchHit]:
    combined: dict[str, SearchHit] = {}

    def add(hit: SearchHit, channel: str, rank: int, weight: float) -> None:
        if hit.id in combined:
            target = combined[hit.id]
        else:
            chunk = chunk_lookup.get(hit.id)
            target = copy.deepcopy(hit)
            if chunk is not None:
                target.text = chunk.text
                target.metadata = chunk.metadata
            target.score = 0.0
            combined[hit.id] = target
        target.score += weight / (rrf_k + rank)
        if channel == "vector":
            target.vector_score = hit.vector_score if hit.vector_score is not None else hit.score
            target.rank_details["vector_rank"] = rank
        if channel == "bm25":
            target.bm25_score = hit.bm25_score if hit.bm25_score is not None else hit.score
            target.rank_details["bm25_rank"] = rank

    for rank, hit in enumerate(vector_hits, start=1):
        add(hit, "vector", rank, vector_weight)
    for rank, hit in enumerate(bm25_hits, start=1):
        add(hit, "bm25", rank, bm25_weight)
    return list(combined.values())


def evaluate_hits(hits: list[SearchHit], expected_source_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_source_ids)
    ranks = [idx for idx, hit in enumerate(hits, start=1) if hit.source_id in expected]
    first_rank = ranks[0] if ranks else None
    return {
        "hit": bool(ranks),
        "first_rank": first_rank,
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,
        "matched_source_ids": [hit.source_id for hit in hits if hit.source_id in expected],
    }
