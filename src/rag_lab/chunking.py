from __future__ import annotations

import re
from typing import Any

from rag_lab.models import Chunk


def _clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fixed_windows(text: str, size: int, overlap: int) -> list[str]:
    if not text:
        return []
    size = max(20, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    step = max(1, size - overlap)
    pieces: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(text):
            break
        start += step
    return pieces


def _paragraph_chunks(text: str, size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        if len(paragraph) > size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            chunks.extend(_fixed_windows(paragraph, size=size, overlap=overlap))
            continue
        next_buffer = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(next_buffer) <= size:
            buffer = next_buffer
        else:
            chunks.append(buffer.strip())
            tail = buffer[-overlap:].strip() if overlap and buffer else ""
            buffer = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
    if buffer:
        chunks.append(buffer.strip())
    return chunks


def make_chunks(docs: list[dict[str, Any]], cfg: dict[str, Any]) -> list[Chunk]:
    chunk_cfg = cfg["chunking"]
    strategy = str(chunk_cfg.get("strategy", "paragraph")).lower()
    size = int(chunk_cfg.get("chunk_size", 360))
    overlap = int(chunk_cfg.get("chunk_overlap", 80))

    chunks: list[Chunk] = []
    for doc in docs:
        doc_id = str(doc["id"])
        title = str(doc.get("title", doc_id))
        tags = doc.get("tags", [])
        tags_text = ",".join(str(tag) for tag in tags)
        extra_metadata = dict(doc.get("metadata", {}))
        text = _clean_text(f"{title}\n\n{doc.get('content', '')}")
        if strategy == "fixed":
            pieces = _fixed_windows(text, size=size, overlap=overlap)
        else:
            pieces = _paragraph_chunks(text, size=size, overlap=overlap)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{doc_id}::chunk_{idx:03d}",
                    text=piece,
                    metadata={
                        "source_id": doc_id,
                        "title": title,
                        "tags": tags_text,
                        "chunk_index": idx,
                        **extra_metadata,
                    },
                )
            )
    return chunks
