"""Unit tests for src/rag/observability.

All tests use in-process console/no-op exporters -- no OTel collector needed.
"""

import pytest

from opentelemetry import metrics, trace
from opentelemetry.sdk.trace import TracerProvider


# ---------------------------------------------------------------------------
# init_telemetry
# ---------------------------------------------------------------------------


def test_init_telemetry_sets_tracer_provider():
    """After init_telemetry(), the global tracer provider is a real TracerProvider."""
    from src.rag import observability
    observability._INITIALISED = False  # reset for test isolation
    observability.init_telemetry()
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)


def test_init_telemetry_is_idempotent():
    """Calling init_telemetry() twice does not raise."""
    from src.rag import observability
    observability._INITIALISED = False
    observability.init_telemetry()
    observability.init_telemetry()  # second call is a no-op


# ---------------------------------------------------------------------------
# Tracer / spans
# ---------------------------------------------------------------------------


def test_tracer_creates_span():
    """Starting and ending a span completes without error."""
    from src.rag.observability import tracer

    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("key", "value")


# ---------------------------------------------------------------------------
# Metrics instruments
# ---------------------------------------------------------------------------


def test_retrieval_latency_histogram_records():
    """Recording a value on the histogram does not raise."""
    from src.rag.observability import retrieval_latency
    retrieval_latency.record(42.0)


def test_llm_latency_histogram_records():
    """Recording a value on the LLM latency histogram does not raise."""
    from src.rag.observability import llm_latency
    llm_latency.record(150.0)


def test_escalation_counter_increments():
    """Incrementing the escalation counter does not raise."""
    from src.rag.observability import escalation_counter
    escalation_counter.add(1)


def test_retrieval_hit_counter_increments():
    """Incrementing the retrieval hit counter does not raise."""
    from src.rag.observability import retrieval_hit_counter
    retrieval_hit_counter.add(5)
