"""Structured logging configuration for 911automate."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGS_DIR = Path("logs")


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
    log_dir: Path | None = None,
) -> None:
    """Configure logging to a timestamped file in logs/ (no stdout).

    Args:
        level: Logging level (default INFO).
        json_format: If True, use JSON formatter; otherwise use standard format.
        log_dir: Directory for log files. Defaults to logs/.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if root.handlers:
        return

    dir_path = log_dir if log_dir is not None else LOGS_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = dir_path / f"log_{timestamp}.txt"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    root.addHandler(handler)
