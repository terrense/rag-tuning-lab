from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "rag_tuning_lab"},
    "paths": {
        "corpus": "data/interview_cards.jsonl",
        "docs_dir": "data/docs",
        "eval_queries": "data/eval_queries.yaml",
        "chunks_cache": "storage/chunks.jsonl",
        "chroma_dir": "storage/chroma",
    },
    "source": {
        "include_interview_cards": True,
        "include_docs_dir": True,
        # L0 structured datasets: list of {format, path, max_records?} specs.
        "structured": [],
    },
    "vector_store": {
        "type": "chroma",
        "collection": "rag_lab",
        "reset_on_ingest": True,
        "metric": "cosine",
    },
    "embedding": {
        "provider": "hashing",
        "dimension": 768,
        "analyzer": "char_wb",
        "ngram_min": 2,
        "ngram_max": 4,
        "normalize": True,
    },
    "chunking": {
        "strategy": "paragraph",
        "chunk_size": 360,
        "chunk_overlap": 80,
    },
    "retrieval": {
        "candidate_k": 12,
        "top_k": 5,
        "hybrid": True,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "rrf_k": 60,
    },
    "rerank": {
        "mode": "bm25",
        "top_k": 5,
        "weight": 0.45,
        "model": "",
    },
    "display": {"snippet_chars": 220},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT_CONFIG, loaded)
    cfg["_config_path"] = str(path)
    return cfg


def parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def get_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])
