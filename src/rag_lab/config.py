"""
================================================================================
config.py —— 配置系统
--------------------------------------------------------------------------------
做三件事：
  1) 提供一份“默认配置” DEFAULT_CONFIG（所有可调项的完整列表 + 默认值）。
  2) load_config()：读 YAML 文件，与默认配置“深度合并”——YAML 里写了的覆盖默认，
     没写的用默认。这样每个 configs/*.yaml 只需写关心的部分。
  3) 支持命令行 --set a.b.c=value 临时覆盖任意一项（parse_value + set_dotted）。
================================================================================
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


# 所有可调参数的“总表”。每个 configs/*.yaml 都是在它基础上做覆盖。
DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "rag_tuning_lab"},
    "paths": {                                       # 各类文件路径
        "corpus": "data/interview_cards.jsonl",
        "docs_dir": "data/docs",
        "eval_queries": "data/eval_queries.yaml",
        "chunks_cache": "storage/chunks.jsonl",
        "chroma_dir": "storage/chroma",
    },
    "source": {                                      # 用哪些资料建库
        "include_interview_cards": True,
        "include_docs_dir": True,
        # L0 结构化数据集：列表，每项是 {format, path, max_records?}
        "structured": [],
        # 开启后 docs_dir 里的 PDF 走多模态导入（文字+表格+配图），而非纯文本
        "pdf_multimodal": False,
    },
    "multimodal": {                                  # 多模态 PDF 导入参数（L2~L5）
        "tables": True,           # 抽表格
        "figures": True,          # 抽配图（渲染整页）
        "caption": True,          # 用 M3 视觉给图配描述
        "max_figures_per_doc": 8, # 每篇最多 caption 几页图
        "max_figures_total": 60,  # 全局 caption 上限（控成本，可 --set 调高）
        "render_zoom": 2.0,       # 渲染清晰度
        "max_image_px": 1400,     # 渲染后长边上限（控 base64 体积）
        "min_table_rows": 2,      # 少于这么多行的“表”忽略
    },
    "vector_store": {                                # 向量库
        "type": "chroma",
        "collection": "rag_lab",
        "reset_on_ingest": True,
        "metric": "cosine",
    },
    "embedding": {                                   # 文字→向量。默认用零依赖的 hashing
        "provider": "hashing",
        "dimension": 768,
        "analyzer": "char_wb",
        "ngram_min": 2,
        "ngram_max": 4,
        "normalize": True,
    },
    "query": {                                       # 检索前的三层 query 改写（默认全关）
        "rules": False,   # 第1层 规则：口语→术语 同义词补全
        "nlp": False,     # 第2层 传统NLP：分词去停用词（整理 BM25 查询）
        "llm": "none",    # 第3层 大模型：none | rewrite | hyde | multi
        "num_variants": 3,        # multi 模式生成几个变体
        "hyde_max_tokens": 256,   # hyde 假设答案的最大长度
    },
    "chunking": {                                    # 切块
        "strategy": "paragraph",
        "chunk_size": 360,
        "chunk_overlap": 80,
    },
    "retrieval": {                                   # 检索
        "candidate_k": 12,
        "top_k": 5,
        "hybrid": True,
        "vector_weight": 0.7,
        "bm25_weight": 0.3,
        "rrf_k": 60,
    },
    "rerank": {                                      # 精排
        "mode": "bm25",
        "top_k": 5,
        "weight": 0.45,
        "model": "",
        # 只精排候选池前 input_k 个（0=全部）。独立于 candidate_k 控制 cross-encoder 开销。
        "input_k": 0,
    },
    "display": {"snippet_chars": 220},               # 打印摘要长度
    "generation": {                                  # L1 生成（MiniMax M3）
        "model": "",  # 留空则用环境变量 MINIMAX_MODEL（MiniMax-M3）
        "temperature": 0.2,
        "max_tokens": 2048,
        "context_chars": 600,
        "timeout": 60,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并两个 dict：override 覆盖 base，但嵌套 dict 递归合并而不是整块替换。"""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)   # 两边都是 dict → 递归合并
        else:
            result[key] = value                            # 否则直接覆盖
    return result


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读 YAML 配置，与默认配置合并，返回最终 cfg。还记下配置文件路径备用。"""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    cfg = deep_merge(DEFAULT_CONFIG, loaded)
    cfg["_config_path"] = str(path)                  # 实验记录里要用到
    return cfg


def parse_value(raw: str) -> Any:
    """把命令行 --set 里的字符串值，尽量转成合适的类型（bool/None/int/float/str）。"""
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
        return raw                                   # 都不是就当字符串


def set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    """按点号路径写入嵌套配置，例如 set_dotted(cfg, 'retrieval.top_k', 8)。"""
    current = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:                           # 一路往下钻，缺的层就建空 dict
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def get_path(cfg: dict[str, Any], key: str) -> Path:
    """从 cfg['paths'] 取某个路径并包成 Path。"""
    return Path(cfg["paths"][key])
