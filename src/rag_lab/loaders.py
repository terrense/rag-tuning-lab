from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from rag_lab.models import Chunk

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def load_documents(path: str | Path) -> list[dict[str, Any]]:
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
    source_cfg = cfg.get("source", {})
    docs: list[dict[str, Any]] = []
    counts = {"interview_cards": 0, "files": 0, "pdf_pages": 0}

    if bool(source_cfg.get("include_interview_cards", True)):
        corpus_path = Path(cfg["paths"]["corpus"])
        if corpus_path.exists():
            interview_docs = load_documents(corpus_path)
            docs.extend(interview_docs)
            counts["interview_cards"] = len(interview_docs)

    if bool(source_cfg.get("include_docs_dir", True)):
        docs_dir = Path(cfg["paths"].get("docs_dir", "data/docs"))
        file_docs, file_counts = load_document_files(docs_dir)
        docs.extend(file_docs)
        counts.update(file_counts)

    return docs, counts


def load_document_files(docs_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    docs_dir = Path(docs_dir)
    counts = {"files": 0, "pdf_pages": 0}
    if not docs_dir.exists():
        return [], counts

    docs: list[dict[str, Any]] = []
    files = sorted(
        path
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
    )
    for path in files:
        if path.name.upper() == "README.MD":
            continue
        if path.suffix.lower() == ".pdf":
            pdf_docs = _load_pdf(path, docs_dir)
            docs.extend(pdf_docs)
            counts["files"] += 1
            counts["pdf_pages"] += len(pdf_docs)
        else:
            doc = _load_text_file(path, docs_dir)
            if doc:
                docs.append(doc)
                counts["files"] += 1
    return docs, counts


def _load_text_file(path: Path, root: Path) -> dict[str, Any] | None:
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
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF loading needs pypdf. Run: pip install pypdf") from exc

    reader = PdfReader(str(path))
    rel_path = _relative_path(path, root)
    docs: list[dict[str, Any]] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
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
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stable_id(value: str) -> str:
    normalized = value.lower().replace("\\", "/")
    slug = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "_", normalized).strip("_")
    if not slug:
        slug = "doc"
    return slug[:120]


def load_eval_queries(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("queries", []))


def save_chunks(path: str | Path, chunks: list[Chunk]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(
                json.dumps(
                    {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_chunks(path: str | Path) -> list[Chunk]:
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
    for item in queries:
        if item.get("id") == query_id:
            return item
    available = ", ".join(str(item.get("id")) for item in queries)
    raise KeyError(f"Unknown query id '{query_id}'. Available: {available}")
