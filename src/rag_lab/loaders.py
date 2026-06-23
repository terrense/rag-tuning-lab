"""
================================================================================
loaders.py —— 加载各种来源的资料 + chunk 缓存的读写
--------------------------------------------------------------------------------
把不同来源统一加载成文档 dict {id, title, tags, content, metadata}：
  - 面试卡片 (jsonl)
  - data/docs/ 目录下的用户文档（pdf / txt / md）
  - 结构化数据集（交给 structured.py 处理）
另外还负责把切好的 chunk 存盘 / 读回（save_chunks / load_chunks），
以及读评测题库（load_eval_queries）。
================================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from rag_lab.models import Chunk

# 支持的用户文档类型
SUPPORTED_DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def load_documents(path: str | Path) -> list[dict[str, Any]]:
    """读 JSONL（每行一个 JSON 对象），用于内置面试卡片。"""
    docs: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                docs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return docs


def load_corpus(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """按配置 source.* 决定加载哪些来源，汇总成一个文档列表 + 统计计数。"""
    source_cfg = cfg.get("source", {})
    docs: list[dict[str, Any]] = []
    counts = {"interview_cards": 0, "files": 0, "pdf_pages": 0}

    # 1) 内置面试卡片
    if bool(source_cfg.get("include_interview_cards", True)):
        corpus_path = Path(cfg["paths"]["corpus"])
        if corpus_path.exists():
            interview_docs = load_documents(corpus_path)
            docs.extend(interview_docs)
            counts["interview_cards"] = len(interview_docs)

    # 2) data/docs/ 下的用户文档（pdf/txt/md）
    if bool(source_cfg.get("include_docs_dir", True)):
        docs_dir = Path(cfg["paths"].get("docs_dir", "data/docs"))
        multimodal = bool(source_cfg.get("pdf_multimodal", False))
        # 多模态模式下，PDF 交给 multimodal.py（文字+表格+配图），txt/md 仍走普通加载
        file_docs, file_counts = load_document_files(docs_dir, skip_pdf=multimodal)
        docs.extend(file_docs)
        counts.update(file_counts)
        if multimodal:
            from rag_lab.multimodal import load_multimodal_corpus

            mm_docs, mm_counts = load_multimodal_corpus(cfg)
            docs.extend(mm_docs)
            counts.update(mm_counts)

    # 3) 结构化数据集（疾病 JSON 等），交给 structured.py
    if source_cfg.get("structured"):
        from rag_lab.structured import load_structured_sources

        structured_docs, structured_counts = load_structured_sources(cfg)
        docs.extend(structured_docs)
        counts.update(structured_counts)

    return docs, counts


def load_document_files(docs_dir: str | Path, skip_pdf: bool = False) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """扫描 docs_dir，把支持的文件都加载成文档。PDF 按“页”拆成多个文档。

    skip_pdf=True 时跳过 PDF（多模态模式下 PDF 由 multimodal.py 处理）。
    """
    docs_dir = Path(docs_dir)
    counts = {"files": 0, "pdf_pages": 0}
    if not docs_dir.exists():
        return [], counts

    docs: list[dict[str, Any]] = []
    files = sorted(                                  # 递归找所有支持类型的文件
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
    )
    for path in files:
        if path.name.upper() == "README.MD":         # 跳过说明文件
            continue
        if path.suffix.lower() == ".pdf":
            if skip_pdf:
                continue
            pdf_docs = _load_pdf(path, docs_dir)      # PDF：一页一个文档
            docs.extend(pdf_docs)
            counts["files"] += 1
            counts["pdf_pages"] += len(pdf_docs)
        else:
            doc = _load_text_file(path, docs_dir)     # txt/md：整篇一个文档
            if doc:
                docs.append(doc)
                counts["files"] += 1
    return docs, counts


def _load_text_file(path: Path, root: Path) -> dict[str, Any] | None:
    """读一个文本文件成文档 dict；空文件返回 None。"""
    text = _read_text_with_fallbacks(path).strip()
    if not text:
        return None
    rel_path = _relative_path(path, root)
    return {
        "id": f"file_{_stable_id(rel_path)}",
        "title": path.stem,
        "tags": ["user-doc", path.suffix.lower().lstrip(".")],
        "content": text,
        "metadata": {
            "source_type": "file",
            "file_name": path.name,
            "path": rel_path,
            "extension": path.suffix.lower(),
        },
    }


def _load_pdf(path: Path, root: Path) -> list[dict[str, Any]]:
    """用 pypdf 逐页抽取文字，每页变成一个文档（带页码 metadata）。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF loading needs pypdf. Run: pip install pypdf") from exc

    reader = PdfReader(str(path))
    rel_path = _relative_path(path, root)
    docs: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:                                 # 跳过没抽出文字的页（如扫描图片页）
            continue
        docs.append(
            {
                "id": f"pdf_{_stable_id(rel_path)}_p{page_index:04d}",
                "title": f"{path.stem} p.{page_index}",
                "tags": ["user-doc", "pdf"],
                "content": text,
                "metadata": {
                    "source_type": "pdf",
                    "file_name": path.name,
                    "path": rel_path,
                    "extension": ".pdf",
                    "page": page_index,
                },
            }
        )
    return docs


def _read_text_with_fallbacks(path: Path) -> str:
    """按多种编码依次尝试读取（中文文件常见 gb18030），都失败就用替换字符兜底。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _relative_path(path: Path, root: Path) -> str:
    """尽量返回相对 root 的路径（用 / 分隔），失败就返回绝对路径。"""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_id(value: str) -> str:
    """把路径变成稳定、安全的 id 片段：小写、非字母数字/汉字换成下划线，截断 120 字。"""
    normalized = value.lower().replace("\\", "/")
    slug = re.sub(r"[^a-z0-9一-鿿]+", "_", normalized).strip("_")
    if not slug:
        slug = "doc"
    return slug[:120]


def load_eval_queries(path: str | Path) -> list[dict[str, Any]]:
    """读评测题库 YAML，返回 queries 列表（每项含 question + expected_source_ids）。"""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("queries", []))


def save_chunks(path: str | Path, chunks: list[Chunk]) -> None:
    """把切好的 chunk 存成 JSONL（建库时写，查询/评测时读回）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata},
                    ensure_ascii=False,                # 保留中文，不转义成 \uXXXX
                )
                + "\n"
            )


def load_chunks(path: str | Path) -> list[Chunk]:
    """从 JSONL 读回 chunk 列表。"""
    chunks: list[Chunk] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid chunks cache at {path}:{line_no}: {exc}") from exc
            chunks.append(Chunk(id=item["id"], text=item["text"], metadata=item["metadata"]))
    return chunks


def find_query(queries: list[dict[str, Any]], query_id: str) -> dict[str, Any]:
    """按 id 在题库里找一道题；找不到就报错并列出可用 id。"""
    for item in queries:
        if item.get("id") == query_id:
            return item
    available = ", ".join(str(item.get("id")) for item in queries)
    raise KeyError(f"Unknown query id '{query_id}'. Available: {available}")
