from __future__ import annotations

from rag_lab.bm25 import BM25Index, tokenize
from rag_lab.models import Chunk, SearchHit


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    doc_tokens = set(tokenize(text))
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def rerank_hits(query: str, hits: list[SearchHit], cfg: dict) -> list[SearchHit]:
    rerank_cfg = cfg["rerank"]
    mode = str(rerank_cfg.get("mode", "none")).lower()
    top_k = int(rerank_cfg.get("top_k", cfg["retrieval"].get("top_k", 5)))
    if mode == "none" or not hits:
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    weight = float(rerank_cfg.get("weight", 0.45))
    base_scores = _normalize([hit.score for hit in hits])

    if mode == "bm25":
        chunks = [Chunk(id=hit.id, text=hit.text, metadata=hit.metadata) for hit in hits]
        bm25_hits = BM25Index(chunks).search(query, top_k=len(hits))
        bm25_lookup = {hit.id: hit.bm25_score or 0.0 for hit in bm25_hits}
        rerank_scores = [bm25_lookup.get(hit.id, 0.0) for hit in hits]
    elif mode in {"overlap", "keyword"}:
        rerank_scores = [_overlap_score(query, hit.text) for hit in hits]
        rerank_scores = _normalize(rerank_scores)
    elif mode in {"cross_encoder", "cross-encoder"}:
        rerank_scores = _cross_encoder_scores(query, hits, cfg)
        rerank_scores = _normalize(rerank_scores)
    else:
        raise ValueError(f"Unsupported rerank.mode: {mode}")

    updated: list[SearchHit] = []
    for hit, base_score, rerank_score in zip(hits, base_scores, rerank_scores):
        final_score = (1.0 - weight) * base_score + weight * rerank_score
        hit.score = final_score
        hit.rerank_score = rerank_score
        hit.rank_details["rerank_mode"] = mode
        updated.append(hit)
    return sorted(updated, key=lambda hit: hit.score, reverse=True)[:top_k]


# Cache CrossEncoder models by name — loading one per query is the main cost.
_CE_CACHE: dict = {}


def _get_cross_encoder(model_name: str):
    if model_name not in _CE_CACHE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "cross_encoder rerank needs sentence-transformers. Run: pip install -r requirements-transformers.txt"
            ) from exc
        _CE_CACHE[model_name] = CrossEncoder(model_name)
    return _CE_CACHE[model_name]


def _cross_encoder_scores(query: str, hits: list[SearchHit], cfg: dict) -> list[float]:
    model_name = str(cfg["rerank"].get("model") or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    model = _get_cross_encoder(model_name)
    pairs = [(query, hit.text) for hit in hits]
    return [float(score) for score in model.predict(pairs)]
