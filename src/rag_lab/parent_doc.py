"""
================================================================================
parent_doc.py —— Parent-Document Retrieval（small-to-big，小块检索大块喂）
--------------------------------------------------------------------------------
经典矛盾：
  · 检索要"小块"——块小，向量语义聚焦，命中精准（一个症状块就命中）。
  · 生成要"大块"——只喂那一小块，LLM 缺上下文，答案残缺（只知症状不知病因治疗）。

Parent-Document 的解法：**用小块检索，命中后把它所在的整篇父文档喂给 LLM**。
检索精度和生成上下文完整性两头都要。

本项目里"父文档" = 同一个 source_id（同一疾病）的所有 chunk 按顺序拼起来。
不用改建库、不用重新 embedding——纯查询期后处理：
  检索得到 chunk 命中 → 按 source_id 去重 → 每个命中扩展成完整父文档文本 → 喂 LLM。

开关：generation.parent_document=true。评测看生成端指标（faithfulness/引用），
检索指标不变（还是同一批 chunk 命中）。
================================================================================
"""

from __future__ import annotations

from typing import Any

from rag_lab.models import Chunk, SearchHit


def _join_overlap(acc: str, nxt: str, max_ov: int = 200) -> str:
    """把 nxt 接到 acc 后面，去掉 acc 尾部与 nxt 头部重叠的那段（chunk overlap）。"""
    if not acc:
        return nxt
    lim = min(len(acc), len(nxt), max_ov)
    for k in range(lim, 10, -1):          # 从最长可能重叠往下找
        if acc[-k:] == nxt[:k]:
            return acc + nxt[k:]
    return acc + "\n" + nxt


def build_parent_context(hits: list[SearchHit], chunk_lookup: dict[str, Chunk],
                         all_chunks: list[Chunk], max_chars: int = 1500,
                         max_parents: int = 5) -> tuple[str, list[dict]]:
    """把 chunk 命中扩展成"父文档"上下文。

    - 按命中顺序取 source_id，去重（同一疾病只留第一次，保留检索排序）。
    - 每个 source_id 的所有 chunk 按 chunk_index 拼成完整父文档文本。
    - 返回 (编号资料文本, sources 清单)，接口和 build_context 一致。
    """
    # 预建 source_id -> [chunk...]（按 chunk_index 排序）索引
    by_source: dict[str, list[Chunk]] = {}
    for c in all_chunks:
        by_source.setdefault(c.metadata.get("source_id", ""), []).append(c)
    for sid in by_source:
        by_source[sid].sort(key=lambda c: c.metadata.get("chunk_index", 0))

    seen: set[str] = set()
    blocks: list[str] = []
    sources: list[dict] = []
    n = 0
    for hit in hits:
        sid = hit.source_id
        if sid in seen:
            continue
        seen.add(sid)
        n += 1
        title = hit.title or sid
        parts = by_source.get(sid, [])
        if parts:
            # 拼父文档：去每块重复的标题行；相邻块有 chunk_overlap，拼接时要去重叠，
            # 否则重叠区文本出现两遍。用"累积文本的后缀 == 下一块的前缀"来剪。
            acc = ""
            for c in parts:
                body = c.text
                if body.startswith(title + "\n"):
                    body = body[len(title) + 1:]
                acc = _join_overlap(acc, body)
            parent_text = f"{title}\n{acc}"
        else:
            parent_text = hit.text or ""
        blocks.append(f"[{n}] {title}\n{parent_text[:max_chars]}")
        sources.append({"n": n, "source_id": sid, "title": title})
        if n >= max_parents:
            break
    return "\n\n".join(blocks), sources
