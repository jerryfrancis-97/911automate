"""Re-export from ingestion.chunking for backward compatibility."""

from src.rag.ingestion.chunking import create_chunks, deterministic_chunk_id

__all__ = ["create_chunks", "deterministic_chunk_id"]
