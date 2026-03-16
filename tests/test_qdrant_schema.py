"""Unit tests for QdrantPayload schema and validate_payload."""

import pytest

from src.rag.core.types import QdrantPayload
from src.rag.retrieval.vectordb_qdrant import _TEXT_PREVIEW_MAX, validate_payload


# ---------------------------------------------------------------------------
# validate_payload: defaults
# ---------------------------------------------------------------------------


def test_validate_payload_fills_defaults():
    """An empty dict gets all required keys with zero-value defaults."""
    result = validate_payload({})
    assert result["chunk_id"] == ""
    assert result["doc_id"] == ""
    assert result["page"] == 0
    assert result["chunk_index"] == 0
    assert result["text_preview"] == ""
    assert result["source_path"] == ""
    assert result["embedding_version"] == "v1"


def test_validate_payload_preserves_values():
    """Provided values are kept as-is."""
    meta = {
        "chunk_id": "abc",
        "doc_id": "doc1",
        "page": 3,
        "chunk_index": 7,
        "text": "Hello world",
        "source_path": "/data/doc1.pdf",
        "embedding_version": "v2",
    }
    result = validate_payload(meta)
    assert result["chunk_id"] == "abc"
    assert result["doc_id"] == "doc1"
    assert result["page"] == 3
    assert result["chunk_index"] == 7
    assert result["text_preview"] == "Hello world"
    assert result["source_path"] == "/data/doc1.pdf"
    assert result["embedding_version"] == "v2"


# ---------------------------------------------------------------------------
# text_preview truncation
# ---------------------------------------------------------------------------


def test_validate_payload_truncates_text_preview():
    """text_preview is capped at _TEXT_PREVIEW_MAX characters."""
    long_text = "x" * (_TEXT_PREVIEW_MAX + 100)
    result = validate_payload({"text": long_text})
    assert len(result["text_preview"]) == _TEXT_PREVIEW_MAX


def test_validate_payload_uses_page_content_fallback():
    """Falls back to page_content when text is missing."""
    result = validate_payload({"page_content": "fallback content"})
    assert result["text_preview"] == "fallback content"


# ---------------------------------------------------------------------------
# source_path resolution
# ---------------------------------------------------------------------------


def test_validate_payload_source_path_fallback():
    """Falls back to 'source' when 'source_path' is absent."""
    result = validate_payload({"source": "/alt/path.pdf"})
    assert result["source_path"] == "/alt/path.pdf"


# ---------------------------------------------------------------------------
# embedding_version override
# ---------------------------------------------------------------------------


def test_validate_payload_custom_embedding_version():
    """The embedding_version parameter overrides the default."""
    result = validate_payload({}, embedding_version="v3")
    assert result["embedding_version"] == "v3"


def test_validate_payload_meta_embedding_version_takes_precedence():
    """Value in meta takes precedence over the function parameter."""
    result = validate_payload({"embedding_version": "v2"}, embedding_version="v3")
    assert result["embedding_version"] == "v2"


# ---------------------------------------------------------------------------
# Extra fields are ignored (no crash)
# ---------------------------------------------------------------------------


def test_validate_payload_ignores_extra_keys():
    """Extra keys in meta don't cause errors; only schema keys in output."""
    result = validate_payload({"extra_field": "surprise", "doc_id": "d1"})
    assert result["doc_id"] == "d1"
    assert "extra_field" not in result
