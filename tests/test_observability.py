"""Unit tests for src/rag/observability stub."""

import pytest


def test_init_telemetry_does_not_raise():
    """init_telemetry() completes without raising."""
    from src.rag import observability

    observability._INITIALISED = False
    observability.init_telemetry()


def test_init_telemetry_is_idempotent():
    """Calling init_telemetry() twice does not raise."""
    from src.rag import observability

    observability._INITIALISED = False
    observability.init_telemetry()
    observability.init_telemetry()
