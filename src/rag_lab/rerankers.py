"""
================================================================================
rerankers.py —— 精排：把粗筛出的候选重新打分排序
--------------------------------------------------------------------------------
粗筛（向量+BM25）快但糙；精排用更准的方法把候选重排，把最相关的顶上来。
支持 4 种模式（配置 rerank.mode）：
  - none          ：不精排，直接按粗筛分排序
  - bm25          ：用 BM25 重新算一遍当精排分
  - overlap       ：词面重叠率当精排分（最朴素）
  - cross_encoder ：★最准★ 把(问题,文档)拼一起喂模型直接判相关性（也最慢）

最终分 = (1-weight)×粗筛分 + weight×精排分。weight 越大越信任精排。
================================================================================
"""

from __future__ import annotations

from rag_lab.bm25 import BM25Index, tokenize
from rag_lab.models import Chunk, SearchHit


def _normalize(values: list[float]) -> list[float]:
    """min-max 归一化到 [0,1]，让“粗筛分”和“精排分”能在同一尺度上加权融合。"""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:                       # 全相等，避免除零，统一给 1.0
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _overlap_score(query: str, text: str) -> float:
    """词面重叠率：问题的词里，有多大比例出现在文档里。最朴素的相关性度量。"""
    query_tokens = set(tokenize(query))
    doc_tokens = set(tokenize(text))
    if not query_tokens or not doc_tokens:
        return 0.0
    return len(query_tokens & doc_tokens) / len(query_tokens)


def rerank_hits(query: str, hits: list[SearchHit], cfg: dict) -> list[SearchHit]:
    """对候选 hits 做精排，返回重排后的 top_k。"""
    rerank_cfg = cfg["rerank"]
    mode = str(rerank_cfg.get("mode", "none")).lower()
    top_k = int(rerank_cfg.get("top_k", cfg["retrieval"].get("top_k", 5)))
    # 不精排：直接按粗筛分排序取前 top_k
    if mode == "none" or not hits:
        return sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]

    weight = float(rerank_cfg.get("weight", 0.45))
    base_scores = _normalize([hit.score for hit in hits])   # 把粗筛分归一化

    # 按模式算“精排分” rerank_scores
    if mode == "bm25":
        chunks = [Chunk(id=hit.id, text=hit.text, metadata=hit.metadata) for hit in hits]
        bm25_hits = BM25Index(chunks).search(query, top_k=len(hits))   # 只在候选内部重算 BM25
        bm25_lookup = {hit.id: hit.bm25_score or 0.0 for hit in bm25_hits}
        rerank_scores = [bm25_lookup.get(hit.id, 0.0) for hit in hits]
    elif mode in {"overlap", "keyword"}:
        rerank_scores = [_overlap_score(query, hit.text) for hit in hits]
        rerank_scores = _normalize(rerank_scores)
    elif mode in {"cross_encoder", "cross-encoder"}:
        rerank_scores = _cross_encoder_scores(query, hits, cfg)        # 最准的那种
        rerank_scores = _normalize(rerank_scores)
    else:
        raise ValueError(f"Unsupported rerank.mode: {mode}")

    # 融合：最终分 = (1-weight)×粗筛分 + weight×精排分
    updated: list[SearchHit] = []
    for hit, base_score, rerank_score in zip(hits, base_scores, rerank_scores):
        final_score = (1.0 - weight) * base_score + weight * rerank_score
        hit.score = final_score
        hit.rerank_score = rerank_score                    # 记下精排分，方便打印/调试
        hit.rank_details["rerank_mode"] = mode
        updated.append(hit)
    return sorted(updated, key=lambda hit: hit.score, reverse=True)[:top_k]


# cross-encoder 模型加载很贵；按模型名缓存，别每次查询都重新加载（降延迟）。
_CE_CACHE: dict = {}


def _get_cross_encoder(model_name: str):
    """拿到（缓存的）CrossEncoder 模型。"""
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
    """对每个 (问题, 候选文本) 用 cross-encoder 直接打一个相关性分数。

    关键：每个候选都要跑一次模型前向，所以候选越多越慢——这就是为什么
    pipeline 里要用 rerank.input_k 限制喂进来的候选数量。
    """
    model_name = str(cfg["rerank"].get("model") or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
    model = _get_cross_encoder(model_name)
    pairs = [(query, hit.text) for hit in hits]            # 把问题和每个候选配成对
    return [float(score) for score in model.predict(pairs)]
