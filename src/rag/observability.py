"""Observability stub. All telemetry disabled."""

import logging
from src.rag.logging_setup import configure_logging

_INITIALISED = False


def init_telemetry(service_name: str = "911automate") -> None:
    global _INITIALISED
    if _INITIALISED:
        return
    _INITIALISED = True
    configure_logging()
    logging.getLogger(__name__).info("Telemetry disabled (stub)")
