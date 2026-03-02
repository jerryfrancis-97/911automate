"""Unit tests for src.rag.logging_setup."""

import logging

import pytest

from src.rag.logging_setup import configure_logging


def test_configure_logging_importable() -> None:
    """configure_logging is importable from src.rag.logging_setup."""
    from src.rag.logging_setup import configure_logging as fn

    assert fn is not None


def test_configure_logging_runs_without_error() -> None:
    """configure_logging() runs without raising errors."""
    configure_logging()


def test_configure_logging_sets_handlers() -> None:
    """configure_logging sets up root logger with a handler."""
    root = logging.getLogger()
    # Clear handlers to test fresh setup
    root.handlers.clear()
    root.setLevel(logging.NOTSET)

    configure_logging(level=logging.DEBUG)
    assert len(root.handlers) >= 1
    assert root.level == logging.DEBUG


def test_configure_logging_idempotent() -> None:
    """Calling configure_logging twice does not duplicate handlers."""
    root = logging.getLogger()
    root.handlers.clear()

    configure_logging()
    first_count = len(root.handlers)

    configure_logging()
    second_count = len(root.handlers)

    assert first_count == second_count, "Handlers should not be duplicated"


def test_configure_logging_uses_json_formatter_when_enabled() -> None:
    """configure_logging with json_format=True uses JSONFormatter."""
    from src.rag.logging_setup import JSONFormatter

    root = logging.getLogger()
    root.handlers.clear()

    configure_logging(json_format=True)
    assert len(root.handlers) >= 1
    assert isinstance(root.handlers[0].formatter, JSONFormatter)
