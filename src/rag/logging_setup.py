"""Structured logging configuration for 911automate."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = True,
) -> None:
    """Configure logging with structured JSON output and default level.

    Args:
        level: Logging level (default INFO).
        json_format: If True, use JSON formatter; otherwise use standard format.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root.addHandler(handler)
