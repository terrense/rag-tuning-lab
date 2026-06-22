"""
================================================================================
sources.py —— 命令行入口：建库前先看看语料里有什么
--------------------------------------------------------------------------------
用法：python -m rag_lab.sources --config configs/diseases.yaml
不建库、不检索，只加载语料并打印统计（多少卡片/文件/PDF页/结构化记录、
支持哪些文件类型、docs 目录下检测到哪些文件）。用来确认“系统看见了我放的资料”。
================================================================================
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag_lab.config import load_config
from rag_lab.loaders import SUPPORTED_DOC_EXTENSIONS, load_corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect corpus files before ingest.")
    parser.add_argument("--config", default="configs/play.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    docs, counts = load_corpus(cfg)                 # 走一遍加载，但不切块/不建库
    docs_dir = Path(cfg["paths"].get("docs_dir", "data/docs"))
    files = []
    if docs_dir.exists():                           # 列出 docs 目录下被支持的文件
        files = sorted(
            path
            for path in docs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
        )

    # 打印各来源的计数
    print("Corpus sources")
    print(f"  config      : {args.config}")
    print(f"  docs dir    : {docs_dir}")
    print(f"  cards       : {counts.get('interview_cards', 0)}")      # 面试卡片
    print(f"  files       : {counts.get('files', 0)}")               # 用户文件数
    print(f"  pdf pages   : {counts.get('pdf_pages', 0)}")           # PDF 总页数
    print(f"  structured  : {counts.get('structured_records', 0)} records "
          f"from {counts.get('structured_files', 0)} dataset(s)")    # 结构化记录
    print(f"  total docs  : {len(docs)}")
    print("Supported file types")
    print("  " + ", ".join(sorted(SUPPORTED_DOC_EXTENSIONS)))
    # 列出检测到的具体文件
    display_files = [path for path in files if path.name.upper() != "README.MD"]
    if display_files:
        print("Detected files")
        for path in display_files:
            print(f"  - {path.relative_to(docs_dir)}")
    else:
        print("Detected files")
        print("  - none yet")


if __name__ == "__main__":
    main()
