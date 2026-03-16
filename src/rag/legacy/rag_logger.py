"""Re-export from observability package for backward compatibility."""

from src.rag.observability.rag_logger import log_rag_request

__all__ = ["log_rag_request"]
