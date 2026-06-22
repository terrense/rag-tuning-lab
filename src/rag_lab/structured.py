"""Structured-record ingestion (L0 of the roadmap).

Turns row/record style data (JSON now, xlsx later) into the same
``{id, title, tags, content, metadata}`` doc dicts the rest of the pipeline
expects. Two ideas drive the design:

1. **Verbalize for recall.** A record is rewritten into labelled natural-language
   sections so both the embedding model and BM25 have real text to match.
2. **Keep fields as metadata for filtering.** The original columns are preserved
   as scalar metadata (lists are comma-joined, because Chroma only stores
   scalars), so later we can filter/route by department, symptom, etc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# Split a multi-value cell on Chinese/ASCII commas and pause marks.
_LIST_SEP = re.compile(r"[,，、;；]+")


def _split_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = [str(v).strip() for v in value]
    else:
        items = [part.strip() for part in _LIST_SEP.split(str(value))]
    return [item for item in items if item]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def verbalize_disease(record: dict[str, Any], index: int) -> dict[str, Any] | None:
    """Rewrite one disease record into a retrieval-friendly doc dict.

    Returns ``None`` for records with no name (skips junk rows).
    """
    name = _clean(record.get("疾病名称"))
    if not name:
        return None

    description = _clean(record.get("描述"))
    cause = _clean(record.get("病因"))
    category = _clean(record.get("分类"))
    prevalence = _clean(record.get("患病概率"))
    departments = _split_list(record.get("就诊科室"))
    symptoms = _split_list(record.get("症状"))
    treatments = _split_list(record.get("治疗方式"))
    exams = _split_list(record.get("检查项目"))

    # Verbalized content: a header line plus labelled sections, separated by
    # blank lines so the paragraph chunker can split on natural boundaries.
    sections: list[str] = []
    header = f"疾病名称：{name}"
    if departments:
        header += f"\n就诊科室：{'、'.join(departments)}"
    if prevalence:
        header += f"\n患病概率：{prevalence}"
    sections.append(header)

    if description:
        sections.append(f"【描述】\n{description}")
    if cause:
        sections.append(f"【病因】\n{cause}")
    if symptoms:
        sections.append(f"【症状】\n{'、'.join(symptoms)}")
    if treatments:
        sections.append(f"【治疗方式】\n{'、'.join(treatments)}")
    if exams:
        sections.append(f"【检查项目】\n{'、'.join(exams)}")

    content = "\n\n".join(sections)

    return {
        "id": f"disease_{index:05d}",
        "title": name,
        "tags": ["disease"] + departments,
        "content": content,
        "metadata": {
            "source_type": "disease",
            "disease_name": name,
            "department": "、".join(departments),
            "category": category,
            "symptoms": "、".join(symptoms),
            "treatments": "、".join(treatments),
            "exams": "、".join(exams),
            "prevalence": prevalence,
        },
    }


def load_disease_json(
    path: str | Path, max_records: int | None = None
) -> list[dict[str, Any]]:
    """Load a list-of-records disease JSON file into verbalized doc dicts."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of records at {path}, got {type(data).__name__}")

    docs: list[dict[str, Any]] = []
    for index, record in enumerate(data):
        if not isinstance(record, dict):
            continue
        doc = verbalize_disease(record, index)
        if doc is not None:
            docs.append(doc)
        if max_records is not None and len(docs) >= max_records:
            break
    return docs


# Dispatch table: structured-source "format" -> loader function.
_LOADERS = {
    "disease_json": load_disease_json,
}


def load_structured_sources(
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load every entry in ``source.structured`` (a list of dataset specs)."""
    source_cfg = cfg.get("source", {})
    specs = source_cfg.get("structured", []) or []
    # Global override usable via --set source.structured_max_records=N.
    # Not set -> use each spec's own max_records. <=0 -> no limit (all records).
    global_max = source_cfg.get("structured_max_records")
    docs: list[dict[str, Any]] = []
    counts = {"structured_files": 0, "structured_records": 0}

    for spec in specs:
        fmt = spec.get("format")
        path = spec.get("path")
        if not fmt or not path:
            raise ValueError(f"structured source needs 'format' and 'path': {spec}")
        loader = _LOADERS.get(fmt)
        if loader is None:
            raise ValueError(
                f"Unknown structured format '{fmt}'. Known: {', '.join(_LOADERS)}"
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"structured source not found: {path}")
        if global_max is not None:
            max_records = None if int(global_max) <= 0 else int(global_max)
        else:
            max_records = spec.get("max_records")
        loaded = loader(path, max_records=max_records)
        docs.extend(loaded)
        counts["structured_files"] += 1
        counts["structured_records"] += len(loaded)

    return docs, counts
