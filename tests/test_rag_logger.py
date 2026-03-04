"""Tests for src.rag.rag_logger structured JSONL logging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.rag.rag_logger import log_rag_request


def test_log_rag_request_writes_valid_jsonl(tmp_path: Path) -> None:
    """log_rag_request writes one valid JSON line with expected keys."""
    with patch("src.rag.rag_logger._RAG_LOGS_DIR", tmp_path):
        # Reset module state so it uses our patched dir
        import src.rag.rag_logger as m
        m._log_file = None

        log_rag_request(
            question="What is CPR?",
            retrieved_chunks=[{"chunk_id": "c1", "doc_id": "d1", "page": 1, "snippet": "abc", "score": 0.9}],
            retrieval_scores=[0.9],
            final_answer="CPR is...",
            used_facts=[{"chunk_id": "c1", "doc_id": "d1", "page": 1, "snippet": "abc", "score": 0.9}],
            confidence=0.85,
            escalated=False,
            latency_ms=150,
            request_id="test-req-1",
        )

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    content = files[0].read_text(encoding="utf-8").strip()
    record = json.loads(content)
    assert record["request_id"] == "test-req-1"
    assert record["question"] == "What is CPR?"
    assert record["final_answer"] == "CPR is..."
    assert record["confidence"] == 0.85
    assert record["escalated"] is False
    assert record["latency_ms"] == 150
    assert "retrieved_chunks" in record
    assert "retrieval_scores" in record
    assert "used_facts" in record
    assert "timestamp" in record


def test_log_rag_request_never_raises() -> None:
    """log_rag_request never raises even on invalid input."""
    log_rag_request(
        question="x",
        retrieved_chunks=[],
        retrieval_scores=[],
        final_answer="y",
        used_facts=[],
        confidence=0.0,
        escalated=False,
        latency_ms=0,
    )
    # No exception
