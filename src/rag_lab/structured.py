"""
================================================================================
structured.py —— 结构化数据导入（路线图 L0）
--------------------------------------------------------------------------------
作用：把“一行一条记录”的结构化数据（现在是疾病 JSON，以后还能加 xlsx）
变成项目通用的文档格式 dict：{id, title, tags, content, metadata}。

两条设计思想（很重要）：
  1) 为检索而“语言化”：把一条记录改写成带小标题的自然语言段落，
     这样 embedding 和 BM25 才有真正的“文本”可以匹配（而不是一堆字段名）。
  2) 字段保留进 metadata 供过滤：原始字段都存进 metadata。
     注意 Chroma 只能存标量，所以列表（如症状列表）要用顿号拼成字符串。
================================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# 一个单元格里可能塞了多个值，用中英文逗号/顿号/分号切开。
_LIST_SEP = re.compile(r"[,，、;；]+")


def _split_list(value: Any) -> list[str]:
    """把 '恶心, 抽搐、感觉障碍' 这种多值字符串切成 ['恶心','抽搐','感觉障碍']。"""
    if value is None:
        return []
    if isinstance(value, list):                       # 本来就是列表，逐项清洗
        items = [str(v).strip() for v in value]
    else:                                             # 是字符串，按分隔符切
        items = [part.strip() for part in _LIST_SEP.split(str(value))]
    return [item for item in items if item]           # 去掉空串


def _clean(value: Any) -> str:
    """取字段的字符串值，None 当空串，去首尾空白。"""
    return "" if value is None else str(value).strip()


def verbalize_disease(record: dict[str, Any], index: int) -> dict[str, Any] | None:
    """把一条疾病记录改写成“可检索的文档 dict”。没有病名的脏记录返回 None（跳过）。

    index 是这条记录在 JSON 列表里的位置，用来生成稳定的 id（disease_00000）。
    """
    name = _clean(record.get("疾病名称"))
    if not name:                          # 没病名的记录直接丢弃
        return None

    # 逐个取出字段
    description = _clean(record.get("描述"))
    cause = _clean(record.get("病因"))
    category = _clean(record.get("分类"))
    prevalence = _clean(record.get("患病概率"))
    departments = _split_list(record.get("就诊科室"))
    symptoms = _split_list(record.get("症状"))
    treatments = _split_list(record.get("治疗方式"))
    exams = _split_list(record.get("检查项目"))

    # 组装“语言化”正文：一个头部 + 若干带【小标题】的段落。
    # 段落之间用空行隔开，这样段落切块器(chunking.py)能在自然边界切。
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

    content = "\n\n".join(sections)   # 段落间空行

    return {
        "id": f"disease_{index:05d}",            # 稳定 id，如 disease_00042
        "title": name,                            # 标题=病名（chunking 会把它放进每个块）
        "tags": ["disease"] + departments,        # 标签
        "content": content,                       # 语言化后的正文（用来切块+检索）
        "metadata": {                             # 结构化字段，留作过滤/展示（必须是标量）
            "source_type": "disease",
            "disease_name": name,
            "department": "、".join(departments),  # 列表 → 顿号拼接的字符串
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
    """读“列表型”疾病 JSON 文件，逐条语言化成文档 dict。max_records 可限量（练手用）。"""
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
        if max_records is not None and len(docs) >= max_records:  # 达到上限就停
            break
    return docs


# 格式分发表：配置里写的 format 字段 → 对应的加载函数。以后加 xlsx 就在这里登记。
_LOADERS = {
    "disease_json": load_disease_json,
}


def load_structured_sources(
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """加载配置 source.structured 里的每一个数据集（是个列表，可配多个）。"""
    source_cfg = cfg.get("source", {})
    specs = source_cfg.get("structured", []) or []
    # 全局上限开关：命令行 --set source.structured_max_records=N 可覆盖各数据集自己的 max_records。
    # 不设 → 用每个 spec 自己的 max_records；设成 <=0 → 不限量（全量）。
    global_max = source_cfg.get("structured_max_records")
    docs: list[dict[str, Any]] = []
    counts = {"structured_files": 0, "structured_records": 0}

    for spec in specs:
        fmt = spec.get("format")
        path = spec.get("path")
        if not fmt or not path:
            raise ValueError(f"structured source needs 'format' and 'path': {spec}")
        loader = _LOADERS.get(fmt)                 # 按 format 找加载器
        if loader is None:
            raise ValueError(
                f"Unknown structured format '{fmt}'. Known: {', '.join(_LOADERS)}"
            )
        if not Path(path).exists():
            raise FileNotFoundError(f"structured source not found: {path}")
        # 决定这个数据集取多少条
        if global_max is not None:
            max_records = None if int(global_max) <= 0 else int(global_max)
        else:
            max_records = spec.get("max_records")
        loaded = loader(path, max_records=max_records)
        docs.extend(loaded)
        counts["structured_files"] += 1
        counts["structured_records"] += len(loaded)

    return docs, counts
