"""OpenTelemetry-based observability for the RAG pipeline.

Call ``init_telemetry()`` once at startup (e.g. in cli.py) to wire up
tracing and metrics.  Individual modules import ``tracer`` / the metric
instruments directly and use them in hot paths.

Defaults to console exporters; swap to OTLP by setting
``OTEL_EXPORTER_OTLP_ENDPOINT`` before calling ``init_telemetry()``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from src.rag.logging_setup import configure_logging

logger = logging.getLogger(__name__)

_INITIALISED = False

# Module-level convenience handles (safe even before init_telemetry; the SDK
# returns no-op implementations until a real provider is registered).
tracer = trace.get_tracer("rag")
meter = metrics.get_meter("rag")

retrieval_latency = meter.create_histogram(
    name="rag.retrieval.latency_ms",
    description="Retrieval latency in milliseconds",
    unit="ms",
)

llm_latency = meter.create_histogram(
    name="rag.llm.latency_ms",
    description="LLM call latency in milliseconds",
    unit="ms",
)

escalation_counter = meter.create_counter(
    name="rag.agent.escalations",
    description="Number of escalations to human operator",
)

retrieval_hit_counter = meter.create_counter(
    name="rag.retrieval.hits",
    description="Total retrieval results returned",
)


def init_telemetry(service_name: str = "911automate") -> None:
    """Initialise OTel tracing + metrics providers (idempotent).

    Also calls ``configure_logging()`` so structured JSON logs remain active
    alongside OTel spans.
    """
    global _INITIALISED
    if _INITIALISED:
        return
    _INITIALISED = True

    configure_logging()

    _setup_tracing(service_name)
    _setup_metrics(service_name)

    logger.info("Telemetry initialised for service=%s", service_name)


def _setup_tracing(service_name: str) -> None:
    provider = TracerProvider()
    exporter = _get_span_exporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Re-bind module-level tracer so callers that already imported it get
    # the real provider.
    global tracer
    tracer = trace.get_tracer("rag")


def _setup_metrics(service_name: str) -> None:
    exporter = _get_metric_exporter()
    reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10_000)
    provider = MeterProvider(metric_readers=[reader])
    metrics.set_meter_provider(provider)

    global meter, retrieval_latency, llm_latency, escalation_counter, retrieval_hit_counter
    meter = metrics.get_meter("rag")
    retrieval_latency = meter.create_histogram(
        name="rag.retrieval.latency_ms",
        description="Retrieval latency in milliseconds",
        unit="ms",
    )
    llm_latency = meter.create_histogram(
        name="rag.llm.latency_ms",
        description="LLM call latency in milliseconds",
        unit="ms",
    )
    escalation_counter = meter.create_counter(
        name="rag.agent.escalations",
        description="Number of escalations to human operator",
    )
    retrieval_hit_counter = meter.create_counter(
        name="rag.retrieval.hits",
        description="Total retrieval results returned",
    )


def _get_span_exporter():
    """Return OTLP exporter if endpoint is configured, else console."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter(endpoint=endpoint)
        except ImportError:
            logger.warning("OTLP trace exporter not installed; falling back to console")
    return ConsoleSpanExporter()


def _get_metric_exporter():
    """Return OTLP exporter if endpoint is configured, else console."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
            return OTLPMetricExporter(endpoint=endpoint)
        except ImportError:
            logger.warning("OTLP metric exporter not installed; falling back to console")
    return ConsoleMetricExporter()
