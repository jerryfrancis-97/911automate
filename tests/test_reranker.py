"""Tests for Reranker: deduplication, top_n, and Ollama scoring (mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from src.rag.core.types import RetrievedChunk
from src.rag.retrieval.reranker import Reranker, _dedup_key


def test_dedup_key_uses_chunk_id() -> None:
    """_dedup_key prefers chunk_id from metadata."""
    chunk: RetrievedChunk = {
        "text": "content",
        "metadata": {"chunk_id": "chunk_5.md", "source_path": "/path/to/chunk_5.md"},
        "score": 0.9,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    assert _dedup_key(chunk) == "chunk_5.md"


def test_dedup_key_fallback_to_source_path() -> None:
    """_dedup_key uses source_path when chunk_id is missing."""
    chunk: RetrievedChunk = {
        "text": "content",
        "metadata": {"source_path": "/data/chunks/doc/chunk_3.md"},
        "score": 0.9,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    assert _dedup_key(chunk) == "/data/chunks/doc/chunk_3.md"


def test_dedup_key_fallback_to_source_filename() -> None:
    """_dedup_key uses source filename when chunk_id and source_path missing."""
    chunk: RetrievedChunk = {
        "text": "content",
        "metadata": {"source": "/data/chunks/doc/chunk_2.md"},
        "score": 0.9,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    assert _dedup_key(chunk) == "chunk_2.md"


def test_dedup_key_fallback_to_text_preview() -> None:
    """_dedup_key uses text prefix when no metadata identifiers."""
    chunk: RetrievedChunk = {
        "text": "Unique passage about ALS bleeding criteria.",
        "metadata": {},
        "score": 0.9,
        "provenance": {"doc_id": "", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    assert _dedup_key(chunk) == "Unique passage about ALS bleeding criteria."[:200]


def test_rerank_empty_chunks() -> None:
    """rerank returns [] for empty input."""
    reranker = Reranker(model="test-model", top_n=3)
    assert reranker.rerank("query", []) == []


def test_rerank_deduplicates_by_chunk_id() -> None:
    """Duplicate chunks (same chunk_id) are deduplicated."""
    chunk_a: RetrievedChunk = {
        "text": "ALS bleeding criteria.",
        "metadata": {"chunk_id": "chunk_0.md"},
        "score": 0.9,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    chunk_b: RetrievedChunk = {
        "text": "ALS bleeding criteria.",
        "metadata": {"chunk_id": "chunk_0.md"},
        "score": 0.8,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.8},
    }
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.message.content = "8"
        mock_client.chat.return_value = mock_resp

        reranker = Reranker(model="test", top_n=5)
        results = reranker.rerank("bleeding", [chunk_a, chunk_b])

    assert len(results) == 1
    assert results[0]["text"] == "ALS bleeding criteria."
    mock_client.chat.assert_called_once()


def test_rerank_returns_top_n() -> None:
    """rerank returns at most top_n chunks."""
    chunks: list[RetrievedChunk] = [
        {
            "text": f"Passage {i}.",
            "metadata": {"chunk_id": f"chunk_{i}.md"},
            "score": float(i),
            "provenance": {"doc_id": "x", "page": 0, "chunk_index": i, "source_path": "", "score": float(i)},
        }
        for i in range(5)
    ]
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.side_effect = [
            MagicMock(message=MagicMock(content=f"{9 - i}"))
            for i in range(5)
        ]

        reranker = Reranker(model="test", top_n=2)
        results = reranker.rerank("query", chunks)

    assert len(results) == 2
    assert mock_client.chat.call_count == 5


def test_rerank_preserves_chunk_text_unchanged() -> None:
    """Chunk text is preserved exactly—no stemming or transformation."""
    raw_text = "ALS bleeding criteria: unconscious or not breathing normally."
    chunk: RetrievedChunk = {
        "text": raw_text,
        "metadata": {"chunk_id": "chunk_0.md"},
        "score": 0.9,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.message.content = "9"
        mock_client.chat.return_value = mock_resp

        reranker = Reranker(model="test", top_n=3)
        results = reranker.rerank("bleeding ALS", [chunk])

        assert len(results) == 1
        assert results[0]["text"] == raw_text
    assert "ALS" in results[0]["text"]


def test_rerank_sorts_by_score_descending() -> None:
    """Chunks are returned ordered by relevance score (highest first)."""
    chunks: list[RetrievedChunk] = [
        {"text": "Low relevance.", "metadata": {"chunk_id": "chunk_0.md"}, "score": 0.1, "provenance": {"doc_id": "", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.1}},
        {"text": "High relevance.", "metadata": {"chunk_id": "chunk_1.md"}, "score": 0.9, "provenance": {"doc_id": "", "page": 0, "chunk_index": 1, "source_path": "", "score": 0.9}},
        {"text": "Medium relevance.", "metadata": {"chunk_id": "chunk_2.md"}, "score": 0.5, "provenance": {"doc_id": "", "page": 0, "chunk_index": 2, "source_path": "", "score": 0.5}},
    ]
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.side_effect = [
            MagicMock(message=MagicMock(content="2")),
            MagicMock(message=MagicMock(content="9")),
            MagicMock(message=MagicMock(content="5")),
        ]

        reranker = Reranker(model="test", top_n=3)
        results = reranker.rerank("query", chunks)

    assert [r["text"] for r in results] == ["High relevance.", "Medium relevance.", "Low relevance."]


def test_rerank_ollama_failure_returns_chunks_as_is() -> None:
    """When Ollama client fails, return unique chunks up to top_n."""
    chunk: RetrievedChunk = {
        "text": "Fallback content.",
        "metadata": {"chunk_id": "chunk_0.md"},
        "score": 0.9,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.9},
    }
    with patch("ollama.Client", side_effect=ConnectionError("Ollama unavailable")):
        reranker = Reranker(model="test", top_n=3)
        results = reranker.rerank("query", [chunk])

    assert len(results) == 1
    assert results[0]["text"] == "Fallback content."


def test_rerank_empty_text_chunk_gets_zero_score() -> None:
    """Chunks with empty text get score 0 and are still included."""
    chunk: RetrievedChunk = {
        "text": "",
        "metadata": {"chunk_id": "chunk_empty.md"},
        "score": 0.0,
        "provenance": {"doc_id": "x", "page": 0, "chunk_index": 0, "source_path": "", "score": 0.0},
    }
    with patch("ollama.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        reranker = Reranker(model="test", top_n=5)
        results = reranker.rerank("query", [chunk])

    assert len(results) == 1
    mock_client.chat.assert_not_called()
