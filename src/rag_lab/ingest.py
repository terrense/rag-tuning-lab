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
        action="append",
        default=[],
        help="Override a config value, for example --set chunking.chunk_size=260",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for item in args.set:
        key, raw_value = item.split("=", 1)
        set_dotted(cfg, key, parse_value(raw_value))
    summary = ingest_config(cfg)
    print_ingest_summary(summary)


if __name__ == "__main__":
    main()
