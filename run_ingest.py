#!/usr/bin/env python3
"""Run ingestion: load PDF/markdown -> chunk -> embed -> upsert to Qdrant.

Usage (from project root):
    python run_ingest.py --path data/my.pdf --doc-id my_doc
    python run_ingest.py --path data/my.pdf --doc-id my_doc --root /path/to/project

Requires: Qdrant running (e.g. docker compose up -d), embedding model available.
"""

import argparse
import logging
import sys
from pathlib import Path

# Project root on PYTHONPATH so src.rag resolves
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.rag.ingestion.pipeline import run_ingestion
from src.rag.observability import init_telemetry

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDF into vector store (chunk, embed, upsert to Qdrant)."
    )
    parser.add_argument("--path", required=True, help="Path to PDF file (e.g. data/file.pdf)")
    parser.add_argument("--doc-id", required=True, help="Document ID for this ingestion")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Project root (default: directory containing run_ingest.py)",
    )
    args = parser.parse_args()

    root = args.root or Path(__file__).resolve().parent
    init_telemetry()

    try:
        result = run_ingestion(
            doc_id=args.doc_id,
            pdf_filename=args.path,
            root=root,
        )
        logger.info(
            "Ingested doc_id=%s: %d chunks, %d embedded",
            result.doc_id,
            result.num_chunks,
            result.num_embedded,
        )
    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
