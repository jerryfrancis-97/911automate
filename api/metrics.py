"""Prometheus metrics for RAG pipeline observability.

Instrumentation: counters for requests/errors, histograms for latency.

"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

# Histogram buckets: fine-grained for fast ops, coarse for slow (LLM can take 60+ s)
_LATENCY_BUCKETS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

rag_requests_total = Counter(
    "rag_requests_total",
    "Total number of RAG chat requests",
)

rag_errors_total = Counter(
    "rag_errors_total",
    "Total number of failed RAG chat requests",
)

rag_request_latency_seconds = Histogram(
    "rag_request_latency_seconds",
    "End-to-end RAG request latency in seconds",
    buckets=_LATENCY_BUCKETS,
)

rag_retrieval_latency_seconds = Histogram(
    "rag_retrieval_latency_seconds",
    "Retrieval (embed + vector search) latency in seconds",
    buckets=_LATENCY_BUCKETS,
)

rag_llm_latency_seconds = Histogram(
    "rag_llm_latency_seconds",
    "LLM generate latency in seconds",
    buckets=_LATENCY_BUCKETS,
)


# ---------------------------------------------------------------------------
# Helper functions (keep instrumentation logic isolated)
# ---------------------------------------------------------------------------


def increment_rag_requests() -> None:
    """Increment total RAG requests counter."""
    rag_requests_total.inc()


def increment_rag_errors() -> None:
    """Increment RAG errors counter."""
    rag_errors_total.inc()


def record_rag_request_latency(seconds: float) -> None:
    """Record end-to-end request latency."""
    rag_request_latency_seconds.observe(seconds)


def record_rag_retrieval_latency(seconds: float) -> None:
    """Record retrieval (retrieve()) latency."""
    rag_retrieval_latency_seconds.observe(seconds)


def record_rag_llm_latency(seconds: float) -> None:
    """Record LLM generate (_invoke_llm) latency."""
    rag_llm_latency_seconds.observe(seconds)


def get_metrics_content() -> tuple[bytes, str]:
    """Return (body, content_type) for /metrics endpoint.

    Prometheus expects text/plain with exposition format.
    """
    return generate_latest(), CONTENT_TYPE_LATEST
