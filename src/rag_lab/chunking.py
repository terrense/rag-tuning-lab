"""
================================================================================
chunking.py —— 切块：把长文档切成一个个小块 (chunk)
--------------------------------------------------------------------------------
为什么要切块？
  - embedding/检索是以“块”为单位的。块太大→一块里混了很多无关内容（噪声），
    检索分数被稀释；块太小→上下文被切断，答案不完整。
  - overlap（相邻块重叠）能减少“答案正好被切在两块边界”的概率。

本文件提供两种切法：
  - paragraph（按段落，默认）：尽量在空行处切，保持语义完整
  - fixed（定长滑窗）：不管语义，按固定长度切
还有一个关键开关 prepend_title：把标题（病名）放进每个块的开头。
================================================================================
"""

from __future__ import annotations

import re
from typing import Any

from rag_lab.models import Chunk


def _clean_text(text: str) -> str:
    """统一换行、压掉多余空格、把 3+ 个连续空行压成 1 个空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # Windows 换行 → \n
    text = re.sub(r"[ \t]+", " ", text)                     # 多个空格/制表符 → 单空格
    text = re.sub(r"\n{3,}", "\n\n", text)                  # 过多空行 → 一个空行
    return text.strip()


def _fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    """定长滑窗切：每块 size 字，相邻块重叠 overlap 字。"""
    if not text:
        return []
    size = max(20, int(size))                       # 至少 20 字，防呆
    overlap = max(0, min(int(overlap), size - 1))   # overlap 不能 >= size
    step = max(1, size - overlap)                   # 每次窗口前进的步长
    pieces: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):               # 已到末尾，结束
            break
        start += step
    return pieces


def _paragraph_chunks(text: str, size: int, overlap: int) -> list[str]:
    """按段落切：把段落往一个 buffer 里攒，攒到接近 size 就吐出一块。

    思路：
      - 先用“空行”把文本分成段落。
      - 单个段落如果就超过 size，单独用定长滑窗切碎它。
      - 否则把多个短段落拼进 buffer，直到再加下一段会超长，就把 buffer 吐出来，
        并保留尾部 overlap 个字接到下一块开头（保持上下文衔接）。
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > size:                   # 这个段落本身就太长
            if buffer:                              # 先把已攒的吐出去
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_fixed_windows(paragraph, size=size, overlap=overlap))  # 长段落切碎
            continue
        next_buffer = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(next_buffer) <= size:                # 还装得下，继续攒
            buffer = next_buffer
        else:                                       # 装不下了：吐出当前 buffer
            chunks.append(buffer.strip())
            tail = buffer[-overlap:].strip() if overlap and buffer else ""  # 取尾部做重叠
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
    if buffer:                                      # 循环结束，吐出最后一块
        chunks.append(buffer.strip())
    return chunks


def make_chunks(docs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Chunk]:
    """把一批文档 dict 切成 Chunk 列表。每个 chunk 带上 source_id/title 等 metadata。"""
    chunk_cfg = cfg["chunking"]
    strategy = str(chunk_cfg.get("strategy", "paragraph")).lower()
    size = int(chunk_cfg.get("chunk_size", 360))
    overlap = int(chunk_cfg.get("chunk_overlap", 80))
    # ★ 关键开关（L0 那一课）：开了之后，把标题（病名）重复放进“每个”块的开头，
    #   这样“只有症状/治疗”的块也带着病名，检索时不会变成无名孤块。
    prepend_title = bool(chunk_cfg.get("prepend_title", False))

    chunks: list[Chunk] = []
    for doc in docs:
        doc_id = str(doc["id"])
        title = str(doc.get("title", doc_id))
        tags = doc.get("tags", [])
        tags_text = ",".join(str(tag) for tag in tags)
        extra_metadata = dict(doc.get("metadata", {}))   # 文档的字段元数据，会透传到每个 chunk
        content = doc.get("content", "")
        # 没开 prepend_title 时，标题只拼在正文最前面一次（旧行为：标题只进第 0 块）。
        text = _clean_text(content if prepend_title else f"{title}\n\n{content}")
        # 按策略切
        if strategy == "fixed":
            pieces = _fixed_windows(text, size=size, overlap=overlap)
        else:
            pieces = _paragraph_chunks(text, size=size, overlap=overlap)
        # 把每个文本片段包成 Chunk 对象
        for idx, piece in enumerate(pieces):
            # 开了 prepend_title：在每个片段开头再补一行标题
            chunk_text = f"{title}\n{piece}" if prepend_title else piece
            chunks.append(
                Chunk(
                    id=f"{doc_id}::chunk_{idx:03d}",    # 块 id = 文档id::chunk_序号
                    text=chunk_text,
                    metadata={
                        "source_id": doc_id,            # 这个块属于哪个文档（评测按它判命中）
                        "title": title,
                        "tags": tags_text,
                        "chunk_index": idx,             # 第几块
                        **extra_metadata,               # 把疾病字段(科室/症状...)也带上
                    },
                )
            )
    return chunks
