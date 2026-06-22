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
    docs, counts = load_corpus(cfg)
    docs_dir = Path(cfg["paths"].get("docs_dir", "data/docs"))
    files = []
    if docs_dir.exists():
        files = sorted(
            path
            for path in docs_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
        )

    print("Corpus sources")
    print(f"  config      : {args.config}")
    print(f"  docs dir    : {docs_dir}")
    print(f"  cards       : {counts.get('interview_cards', 0)}")
    print(f"  files       : {counts.get('files', 0)}")
    print(f"  pdf pages   : {counts.get('pdf_pages', 0)}")
    print(f"  structured  : {counts.get('structured_records', 0)} records "
          f"from {counts.get('structured_files', 0)} dataset(s)")
    print(f"  total docs  : {len(docs)}")
    print("Supported file types")
    print("  " + ", ".join(sorted(SUPPORTED_DOC_EXTENSIONS)))
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
