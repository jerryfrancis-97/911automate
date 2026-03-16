"""CLI entry point for 911automate RAG agent.

Usage:
    python -m src.rag.cli              # Interactive chat
    python -m src.rag.cli ingest --path data/file.pdf --doc-id my_doc  # Ingest PDF

Environment variables:
    QDRANT_URL   Qdrant server URL (default: http://localhost:6333)
    OLLAMA_URL   Ollama server URL (default: http://localhost:11434)
    API_BASE_URL OpenAI-compatible API base URL (if set, overrides Ollama)
    API_KEY      API key for the OpenAI-compatible endpoint
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _cmd_ingest(path: str, doc_id: str) -> None:
    """Run ingestion pipeline for a PDF."""
    from src.rag.ingestion.pipeline import run_ingestion
    from src.rag.observability import init_telemetry

    init_telemetry()
    result = run_ingestion(doc_id=doc_id, pdf_filename=path)
    logger.info("Ingested %d chunks for doc_id=%s", result.num_embedded, result.doc_id)


def _cmd_chat() -> None:
    from src.rag.retrieval.embedder import Embedder
    from src.rag.observability import init_telemetry
    from src.rag.retrieval.vectordb_qdrant import VectorDBQdrant
    from src.rag.retrieval.retriever import Retriever
    from src.rag.agent.agent import Agent

    init_telemetry()

    from src.rag.core.config import Config

    config = Config.from_env()

    try:
        embedder = Embedder(config)
        vectordb = VectorDBQdrant(config)
        retriever = Retriever(embedder, vectordb, config)
    except Exception as exc:
        print(f"[cli] Failed to initialise retriever: {exc}", file=sys.stderr)
        print(
            "[cli] Make sure Qdrant is reachable and the embedding model is available.",
            file=sys.stderr,
        )
        sys.exit(1)

    agent = Agent(retriever=retriever, config=config)
    agent.run_agent_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="911automate RAG agent")
    subparsers = parser.add_subparsers(dest="cmd", help="Command", required=False)

    # Chat (default)
    subparsers.add_parser("chat", help="Interactive chat (default)")

    # Ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF into vector store")
    ingest_parser.add_argument("--path", required=True, help="Path to PDF file")
    ingest_parser.add_argument("--doc-id", required=True, help="Document ID")

    args = parser.parse_args()
    if args.cmd is None or args.cmd == "chat":
        _cmd_chat()
    elif args.cmd == "ingest":
        _cmd_ingest(args.path, args.doc_id)


if __name__ == "__main__":
    main()
