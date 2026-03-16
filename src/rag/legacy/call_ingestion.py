"""Re-export from ingestion.pipeline for backward compatibility."""

from src.rag.ingestion.pipeline import IngestionPipeline, IngestionResult, run_ingestion

from src.rag.core.config import Config
from src.rag.retrieval.embedder import Embedder
from src.rag.retrieval.vectordb_qdrant import VectorDBQdrant

__all__ = [
    "Config",
    "Embedder",
    "IngestionPipeline",
    "IngestionResult",
    "VectorDBQdrant",
    "run_ingestion",
]
