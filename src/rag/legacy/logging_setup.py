"""Re-export from observability package for backward compatibility."""

from src.rag.observability.logging_setup import JSONFormatter, LOGS_DIR, configure_logging

__all__ = ["JSONFormatter", "LOGS_DIR", "configure_logging"]
