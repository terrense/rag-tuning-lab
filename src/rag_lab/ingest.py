"""
================================================================================
ingest.py —— 命令行入口：建库
--------------------------------------------------------------------------------
用法：python -m rag_lab.ingest --config configs/diseases.yaml [--set a.b=c ...]
本身很薄：解析参数 → 载入配置（含 --set 覆盖）→ 调 pipeline.ingest_config 干活
→ 打印统计。真正的建库逻辑在 pipeline.py。
================================================================================
"""

from __future__ import annotations

import argparse

from rag_lab.config import load_config, parse_value, set_dotted
from rag_lab.formatting import print_ingest_summary
from rag_lab.pipeline import ingest_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG lab index.")
    parser.add_argument("--config", default="configs/chroma.yaml")
    parser.add_argument(
        "--set",
        action="append",          # 可多次：--set a=1 --set b=2
        default=[],
        help="Override a config value, for example --set chunking.chunk_size=260",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:                          # 逐个应用命令行覆盖
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))
    summary = ingest_config(cfg)                   # ★ 真正建库
    print_ingest_summary(summary)


if __name__ == "__main__":
    main()
