#!/usr/bin/env python3
"""Build tokenized corpus from data/chunks/**/*.md for BM25 retrieval.

Usage (from project root):
    python run_prep_corpus.py
    python run_prep_corpus.py --root /path/to/project

Output: data/prep/tokenized_corpus.jsonl
Each line: {"source": "data/chunks/doc_id/chunk_N.md", "tokens": [...]}
The raw chunk text is never stored—only stemmed tokens for BM25 indexing.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.retrieval.bm25_retrieval import tokenize

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main(root: Path, output_path: Path) -> None:
    chunks_dir = root / "data" / "chunks"
    if not chunks_dir.exists():
        logger.warning("Chunks dir %s does not exist. Creating empty corpus.", chunks_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for md_path in sorted(chunks_dir.rglob("*.md")):
            rel = md_path.relative_to(root)
            source_str = str(rel).replace("\\", "/")
            text = md_path.read_text(encoding="utf-8")
            tokens = tokenize(text)
            f.write(json.dumps({"source": source_str, "tokens": tokens}, ensure_ascii=False) + "\n")
            count += 1
    logger.info("Wrote %d chunks to %s", count, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build tokenized corpus for BM25 from data/chunks."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: directory containing run_prep_corpus.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/prep/tokenized_corpus.jsonl)",
    )
    args = parser.parse_args()

    root = args.root or Path(__file__).resolve().parent
    output = args.output or root / "data" / "prep" / "tokenized_corpus.jsonl"

    try:
        main(root, output)
    except Exception as e:
        logger.exception("Prep corpus failed: %s", e)
        sys.exit(1)
