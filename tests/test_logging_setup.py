"""Unit tests for src.rag.logging_setup."""

import logging
import tempfile
from pathlib import Path

import pytest

from src.rag.logging_setup import JSONFormatter, configure_logging


def test_configure_logging_importable() -> None:
    """configure_logging is importable from src.rag.logging_setup."""
    from src.rag.logging_setup import configure_logging as fn

    assert fn is not None


def test_configure_logging_runs_without_error() -> None:
    """configure_logging() runs without raising errors."""
    with tempfile.TemporaryDirectory() as tmp:
        configure_logging(log_dir=Path(tmp))


def test_configure_logging_sets_handlers() -> None:
    """configure_logging sets up root logger with a handler."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)

    with tempfile.TemporaryDirectory() as tmp:
        configure_logging(level=logging.DEBUG, log_dir=Path(tmp))
        assert len(root.handlers) >= 1
        assert root.level == logging.DEBUG
        for h in root.handlers[:]:
            h.close()
            root.removeHandler(h)


def test_configure_logging_idempotent() -> None:
    """Calling configure_logging twice does not duplicate handlers."""
    root = logging.getLogger()
    root.handlers.clear()

    with tempfile.TemporaryDirectory() as tmp:
        log_dir = Path(tmp)
        configure_logging(log_dir=log_dir)
        first_count = len(root.handlers)

        configure_logging(log_dir=log_dir)
        second_count = len(root.handlers)

        assert first_count == second_count, "Handlers should not be duplicated"
        for h in root.handlers[:]:
            h.close()
            root.removeHandler(h)


def test_configure_logging_uses_json_formatter_when_enabled() -> None:
    """configure_logging with json_format=True uses JSONFormatter."""
    root = logging.getLogger()
    root.handlers.clear()

    with tempfile.TemporaryDirectory() as tmp:
        configure_logging(json_format=True, log_dir=Path(tmp))
        assert len(root.handlers) >= 1
        assert isinstance(root.handlers[0].formatter, JSONFormatter)
        for h in root.handlers[:]:
            h.close()
            root.removeHandler(h)
