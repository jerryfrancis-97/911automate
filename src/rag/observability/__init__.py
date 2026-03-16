"""Observability: telemetry and structured logging."""

import logging

from src.rag.observability.logging_setup import configure_logging

_INITIALISED = False


def init_telemetry(service_name: str = "911automate") -> None:
    """Initialize telemetry. Currently a stub — all telemetry disabled."""
    global _INITIALISED
    if _INITIALISED:
        return
    _INITIALISED = True
    configure_logging()
    logging.getLogger(__name__).info("Telemetry disabled (stub)")
