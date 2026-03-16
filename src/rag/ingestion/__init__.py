"""Document loading, chunking, and ingestion pipeline."""

from src.rag.ingestion.loader import (
    DoclingParserAdapter,
    DocLoader,
    DocumentLoader,
    PyMuPDFDocumentLoader,
    URLDocumentLoader,
    save_markdown_files,
)
from src.rag.ingestion.chunking import create_chunks, deterministic_chunk_id
from src.rag.ingestion.pipeline import IngestionPipeline, IngestionResult, run_ingestion

__all__ = [
    "create_chunks",
    "deterministic_chunk_id",
    "DoclingParserAdapter",
    "DocLoader",
    "DocumentLoader",
    "IngestionPipeline",
    "IngestionResult",
    "PyMuPDFDocumentLoader",
    "run_ingestion",
    "save_markdown_files",
    "URLDocumentLoader",
]
