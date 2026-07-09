"""
================================================================================
pipeline.py —— 整个项目的“心脏”
--------------------------------------------------------------------------------
RAG 只有两个核心动作，全在这个文件里：

  1) ingest_config()  建库：资料 → 切块 → 变向量 → 存进向量库（只做一次）
  2) query_config()   查询：问题 → 双路检索 → 融合 → 精排 → 返回（每次提问走一遍）

看懂这一个文件，就看懂了整个检索流程。其它文件（chunking/embeddings/
stores/bm25/rerankers）都是被这里“调用”的零件。
================================================================================
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from rag_lab.bm25 import BM25Index          # 关键词检索
from rag_lab.chunking import make_chunks     # 把文档切成 chunk
from rag_lab.config import get_path          # 从配置里取路径
from rag_lab.embeddings import get_embedder  # 文字 → 向量
from rag_lab.loaders import load_chunks, load_corpus, save_chunks
from rag_lab.models import Chunk, SearchHit  # 两个数据结构：文本块 / 检索命中
from rag_lab.rerankers import rerank_hits    # 精排
from rag_lab.stores import get_store         # 向量库（Chroma/Milvus）

# --- 性能缓存 ----------------------------------------------------------------
# 问题：每次 query 都重新读 2.7 万个 chunk + 重建 BM25 索引，非常慢（曾 26 秒/次）。
# 办法：按“chunk 缓存文件路径 + 文件修改时间(mtime)”缓存。文件没变就复用，
#       文件一变（重新 ingest 过）mtime 变化，缓存自动失效、重建。
_RETRIEVAL_CACHE: dict[str, tuple[float, list[Chunk], dict[str, Chunk], BM25Index]] = {}


def _get_retrieval_assets(chunks_path: Path) -> tuple[list[Chunk], dict[str, Chunk], BM25Index]:
    """返回 (所有chunk, id→chunk的字典, BM25索引)，带缓存。"""
    key = str(chunks_path)
    mtime = os.path.getmtime(chunks_path)          # 文件最后修改时间，当“版本号”用
    cached = _RETRIEVAL_CACHE.get(key)
    if cached is None or cached[0] != mtime:       # 没缓存 或 文件变了 → 重建
        chunks = load_chunks(chunks_path)          # 从磁盘读回切好的 chunk
        lookup = {chunk.id: chunk for chunk in chunks}   # 建 id→chunk 索引，方便按 id 取
        _RETRIEVAL_CACHE[key] = (mtime, chunks, lookup, BM25Index(chunks))  # 建一次 BM25
    _, chunks, lookup, bm25 = _RETRIEVAL_CACHE[key]
    return chunks, lookup, bm25


# ============================================================================
# 动作一：建库 (ingest)
# ============================================================================
def ingest_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """资料 → chunk → 向量 → 存库。返回一份建库统计。"""
    # 1) 读语料：把配置里指定的来源（这里是疾病 JSON）加载成统一的文档 dict
    docs, source_counts = load_corpus(cfg)
    # 2) 切块：每个文档切成若干 chunk（受 chunking.* 配置控制）
    chunks = make_chunks(docs, cfg)
    # 2.5) Contextual Retrieval：可选，给每个块加 LLM 生成的上下文前缀再 embedding
    if bool(cfg.get("chunking", {}).get("contextual", False)):
        from rag_lab.contextual import augment_chunks
        info = augment_chunks(cfg, chunks, docs,
                              max_chunks=int(cfg["chunking"].get("contextual_max_chunks", 0)))
        print(f"[contextual] {info}")
    # 3) 拿到 embedding 模型，把每个 chunk 的文本批量变成向量
    embedder = get_embedder(cfg)
    embeddings = embedder.embed([chunk.text for chunk in chunks])
    if not embeddings:
        raise RuntimeError("No chunks were generated from the corpus.")
    dimension = len(embeddings[0])                 # 向量维度（如 384），建库要用到
    # 4) 打开向量库；reset_on_ingest=true 时先清空旧集合再重建
    reset = bool(cfg["vector_store"].get("reset_on_ingest", True))
    store = get_store(cfg, dimension=dimension, reset=reset)
    # 5) 把 (chunk + 向量) 写入向量库
    store.upsert(chunks, embeddings)
    # 6) 把切好的 chunk 也存一份到磁盘（chunks_cache），query 时直接读、还用于建 BM25
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


# ============================================================================
# 动作二：查询 (query)
# ============================================================================
def query_config(cfg: dict[str, Any], query: str, history: list[str] | None = None) -> dict[str, Any]:
    """一个问题走完整条检索链路，返回最终命中 + 各阶段中间结果。

    history：多轮对话历史（可选）。开启大模型改写时用于消解“它/这个病”等指代。
    """
    # 0) 确认建库时存的 chunk 缓存在；不在说明还没 ingest
    chunks_path = get_path(cfg, "chunks_cache")
    if not Path(chunks_path).exists():
        raise FileNotFoundError(f"Chunks cache not found: {chunks_path}. Run ingest first.")
    # 1) 取出（缓存的）chunk、id索引、BM25索引
    chunks, chunk_lookup, bm25_index = _get_retrieval_assets(chunks_path)
    embedder = get_embedder(cfg)

    retrieval_cfg = cfg["retrieval"]
    candidate_k = int(retrieval_cfg.get("candidate_k", 12))   # 每路粗筛多少候选

    # 2) ★ 检索前的三层 query 改写：得到若干 (向量文本, BM25文本)。
    #    不开任何改写时就是 [(原问题, 原问题)]，行为和以前完全一致。
    from rag_lab.query_rewrite import build_retrieval_queries
    specs = build_retrieval_queries(cfg, query, history=history)

    # 3) 对每个 spec 各跑一遍向量 + BM25 粗筛（multi 模式会有多个 spec）
    store = None
    vector_lists: list[list[SearchHit]] = []
    bm25_lists: list[list[SearchHit]] = []
    for vec_text, bm25_text in specs:
        qe = embedder.embed([vec_text])[0]
        if store is None:                                     # 用第一个向量的维度打开库（只读）
            store = get_store(cfg, dimension=len(qe), reset=False)
        vector_lists.append(store.search(qe, top_k=candidate_k))
        bm25_lists.append(bm25_index.search(bm25_text, top_k=candidate_k))
    # 第一个 spec 的两路结果作为“代表”返回（评测/调试看），多 spec 时它是原问题那路
    vector_hits = vector_lists[0]
    bm25_hits = bm25_lists[0]

    # 4) 融合：开了 hybrid 就用 RRF 把所有路（可能多个 spec）合成一个排名；否则只用向量
    if bool(retrieval_cfg.get("hybrid", True)):
        candidates = combine_hits(
            vector_lists=vector_lists,
            bm25_lists=bm25_lists,
            chunk_lookup=chunk_lookup,
            vector_weight=float(retrieval_cfg.get("vector_weight", 0.7)),
            bm25_weight=float(retrieval_cfg.get("bm25_weight", 0.3)),
            rrf_k=float(retrieval_cfg.get("rrf_k", 60)),
        )
    else:
        # 仅向量：把所有向量路去重（同 id 留最高分）
        dedup: dict[str, SearchHit] = {}
        for vh in vector_lists:
            for hit in vh:
                if hit.id not in dedup or hit.score > dedup[hit.id].score:
                    dedup[hit.id] = copy.deepcopy(hit)
        candidates = list(dedup.values())

    top_k = int(retrieval_cfg.get("top_k", 5))                 # 最终返回几条
    # 6) 按融合分排序，截取候选池（最多 candidate_k 个）
    candidates = sorted(candidates, key=lambda hit: hit.score, reverse=True)[:candidate_k]
    # 7) 精排：候选池可能很大(50)，但 cross-encoder 很贵（每个候选跑一次模型），
    #    所以只把“前 input_k 个”喂给它精排。input_k=0 表示精排整个候选池。
    rerank_input_k = int(cfg["rerank"].get("input_k", 0)) or candidate_k
    final_hits = rerank_hits(query, candidates[:rerank_input_k], cfg)[:top_k]
    # 8) 返回：最终命中 + 两路原始结果（评测/调试时能看清每个阶段）
    return {
        "query": query,
        "hits": final_hits,             # 最终给用户/生成器看的 top_k
        "vector_hits": vector_hits,     # 向量这一路的原始结果（评测用）
        "bm25_hits": bm25_hits,         # BM25 这一路的原始结果（评测用）
        "candidate_count": len(candidates),
        "config": {                     # 把这次用的关键参数也带回去，方便打印/记录
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


# ============================================================================
# RRF 融合：把“向量排名”和“BM25 排名”合成一个总排名
# ----------------------------------------------------------------------------
# 核心思想：只看排名、不看原始分数（因为两路分数量纲不同，没法直接相加）。
# 某文档的融合分 = Σ  权重 / (rrf_k + 它在该路的排名)
# 两路都靠前的文档，融合分自然最高。
# ============================================================================
def combine_hits(
    vector_lists: list[list[SearchHit]],
    bm25_lists: list[list[SearchHit]],
    chunk_lookup: dict[str, Chunk],
    vector_weight: float,
    bm25_weight: float,
    rrf_k: float,
) -> list[SearchHit]:
    """RRF 融合。支持多路：multi-query 改写时会有多个向量/BM25 结果列表，
    每个列表的排名都按 RRF 公式累加到同一个 chunk 上。"""
    combined: dict[str, SearchHit] = {}   # 按 chunk id 去重合并：同一个 chunk 多路命中只算一条

    def add(hit: SearchHit, channel: str, rank: int, weight: float) -> None:
        """把某一路的一个命中累加进 combined。channel 是 'vector' 或 'bm25'。"""
        if hit.id in combined:
            target = combined[hit.id]          # 这个 chunk 之前已被另一路加过，取出来继续累加
        else:
            chunk = chunk_lookup.get(hit.id)   # 第一次见：复制一份命中对象
            target = copy.deepcopy(hit)
            if chunk is not None:              # 用缓存里的权威文本/元数据覆盖（保证一致）
                target.text = chunk.text
                target.metadata = chunk.metadata
            target.score = 0.0                 # 融合分从 0 开始累加
            combined[hit.id] = target
        # RRF 公式：排名越靠前(rank 越小) → 分母越小 → 加的分越多
        target.score += weight / (rrf_k + rank)
        # 顺便把单路的分数和排名记下来（打印/调试时能看到 vector_rank / bm25_rank）
        if channel == "vector":
            target.vector_score = hit.vector_score if hit.vector_score is not None else hit.score
            target.rank_details["vector_rank"] = rank
        if channel == "bm25":
            target.bm25_score = hit.bm25_score if hit.bm25_score is not None else hit.score
            target.rank_details["bm25_rank"] = rank

    # 遍历每一路（可能多个 spec 的多个列表），rank 从 1 开始（第 1 名 rank=1）
    for vector_hits in vector_lists:
        for rank, hit in enumerate(vector_hits, start=1):
            add(hit, "vector", rank, vector_weight)
    for bm25_hits in bm25_lists:
        for rank, hit in enumerate(bm25_hits, start=1):
            add(hit, "bm25", rank, bm25_weight)
    return list(combined.values())   # 注意：这里没排序，排序在 query_config 后面做


# ============================================================================
# 评测：给定最终命中 + 标准答案 id，算这道题答得怎么样
# ============================================================================
def evaluate_hits(hits: list[SearchHit], expected_source_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_source_ids)
    # 找出所有命中里，哪些位置(从1计)的文档属于标准答案
    ranks = [idx for idx, hit in enumerate(hits, start=1) if hit.source_id in expected]
    first_rank = ranks[0] if ranks else None     # 正确答案第一次出现的排名
    return {
        "hit": bool(ranks),                       # 是否命中（top_k 里有没有正确答案）
        "first_rank": first_rank,                 # 第一个正确答案排第几
        "mrr": 0.0 if first_rank is None else 1.0 / first_rank,   # MRR = 1/排名，越靠前越高
        "matched_source_ids": [hit.source_id for hit in hits if hit.source_id in expected],
    }
