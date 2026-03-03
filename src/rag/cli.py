"""Minimal CLI entry point for 911automate RAG agent.

Usage:
    python -m src.rag.cli

Environment variables:
    QDRANT_URL   Qdrant server URL (default: http://localhost:6333)
    OLLAMA_URL   Ollama server URL (default: http://localhost:11434)
    API_BASE_URL OpenAI-compatible API base URL (if set, overrides Ollama)
    API_KEY      API key for the OpenAI-compatible endpoint
"""

import os
import sys


def main() -> None:
    from src.rag.config import Config
    from src.rag.embedder import Embedder
    from src.rag.observability import init_telemetry
    from src.rag.vectordb_qdrant import VectorDBQdrant
    from src.rag.retriever import Retriever
    from src.rag.agent import Agent

    init_telemetry()

    config = Config(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        api_base_url=os.environ.get("API_BASE_URL") or None,
        api_key=os.environ.get("API_KEY") or None,
    )

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


if __name__ == "__main__":
    main()
