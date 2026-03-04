"""Structured JSONL logging for RAG evaluation.

Writes one JSON object per line to rag_logs/rag_logs_{timestamp}.jsonl.
Thread-safe, non-blocking. Logging failures are silently ignored.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Log directory and file (created on first write)
_RAG_LOGS_DIR = Path("rag_logs")
_lock = threading.Lock()
_log_file: Path | None = None


def _get_log_path() -> Path:
    """Return log file path. New file per day (YYYYMMDD) for manageable size."""
    global _log_file
    if _log_file is not None:
        return _log_file
    _RAG_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    _log_file = _RAG_LOGS_DIR / f"rag_logs_{ts}.jsonl"
    return _log_file


def log_rag_request(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    retrieval_scores: list[float],
    final_answer: str,
    used_facts: list[dict[str, Any]],
    confidence: float,
    escalated: bool,
    latency_ms: int,
    *,
    request_id: str | None = None,
) -> None:
    """Append one JSONL record for a RAG request. Never raises."""
    try:
        record = {
            "request_id": request_id or str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_scores": retrieval_scores,
            "final_answer": final_answer,
            "used_facts": used_facts,
            "confidence": confidence,
            "escalated": escalated,
            "latency_ms": latency_ms,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        path = _get_log_path()
        with _lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass  # Silently ignore logging errors; never fail the request
