"""Integration tests for VectorDBQdrant: upsert chunks and search by vector."""

import math
import uuid

import pytest

from src.rag.core.config import Config
from src.rag.core.types import ChunkItem
from src.rag.retrieval.vectordb_qdrant import VectorDBQdrant, _id


def _normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _make_vector(dim: int, seed: int) -> list[float]:
    """Create a deterministic normalized vector of given dimension."""
    vec = [float((seed + i) % 100) / 100.0 for i in range(dim)]
    return _normalize(vec)


@pytest.fixture
def vectordb() -> VectorDBQdrant:
    """VectorDBQdrant with test collection. Skips if Qdrant unavailable."""
    try:
        config = Config(
            qdrant_url="http://localhost:6333",
            collection_name=f"911automate_test_{uuid.uuid4().hex[:8]}",
            embedding_dim=384,
            top_k=5,
        )
        vdb = VectorDBQdrant(config=config)
        vdb._ensure_collection()
        return vdb
    except Exception as e:
        pytest.skip(f"Qdrant unavailable: {e}")


def test_upsert_and_search_returns_payloads_and_scores(vectordb: VectorDBQdrant) -> None:
    """Upsert sample chunks and retrieve them by searching with their vectors."""
    dim = vectordb._config.embedding_dim
    chunks: list[ChunkItem] = [
        {
            "id": 1,
            "vector": _make_vector(dim, 1),
            "metadata": {"doc_id": "doc1", "page": 1, "text": "First chunk"},
        },
        {
            "id": 2,
            "vector": _make_vector(dim, 2),
            "metadata": {"doc_id": "doc1", "page": 2, "text": "Second chunk"},
        },
        {
            "id": 3,
            "vector": _make_vector(dim, 3),
            "metadata": {"doc_id": "doc2", "page": 1, "text": "Third chunk"},
        },
    ]
    vectordb.upsert_chunks(chunks)

    query_vec = chunks[0]["vector"]
    results = vectordb.search_vector(query_vec, top_k=2)

    assert len(results) == 2
    payloads = [r[0] for r in results]
    scores = [r[1] for r in results]
    assert scores[0] >= scores[1], "Results should be ordered by score desc"
    assert scores[0] >= 0.99, "Self-match should have score near 1.0 (cosine)"
    assert payloads[0]["doc_id"] == "doc1"
    assert payloads[0]["text"] == "First chunk"


def test_search_vector_with_vectors_returns_vectors(vectordb: VectorDBQdrant) -> None:
    """search_vector_with_vectors returns (payload, score, vector) tuples."""
    dim = vectordb._config.embedding_dim
    chunks: list[ChunkItem] = [
        {
            "id": 0,
            "vector": _make_vector(dim, 1),
            "metadata": {"doc_id": "d1", "text": "Alpha"},
        },
    ]
    vectordb.upsert_chunks(chunks)
    results = vectordb.search_vector_with_vectors(chunks[0]["vector"], top_k=1)
    assert len(results) == 1
    payload, score, vec = results[0]
    assert payload["text"] == "Alpha"
    assert len(vec) == dim
    assert all(isinstance(x, (int, float)) for x in vec)


def test_id_converts_hex_string_to_int():
    """16-char hex strings (from deterministic_chunk_id) are converted to int for Qdrant."""
    hex_id = "10aa6a4ec73fd5a6"
    converted = _id({"id": hex_id})
    assert converted == int(hex_id, 16)
    assert isinstance(converted, int)
    assert _id({"id": 42}) == 42
    assert converted != hex_id


def test_upsert_hex_string_id_accepts_qdrant(vectordb: VectorDBQdrant) -> None:
    """Chunks with hex string ids (like deterministic_chunk_id) upsert successfully."""
    dim = vectordb._config.embedding_dim
    chunks: list[ChunkItem] = [
        {
            "id": "10aa6a4ec73fd5a6",  # Qdrant rejects this as-is; _id converts to int
            "vector": _make_vector(dim, 99),
            "metadata": {"doc_id": "emergency_childbirth", "page": 1, "text": "Test chunk"},
        },
    ]
    vectordb.upsert_chunks(chunks)
    results = vectordb.search_vector(chunks[0]["vector"], top_k=1)
    assert len(results) == 1
    assert results[0][0]["doc_id"] == "emergency_childbirth"


def test_search_empty_collection_returns_empty(vectordb: VectorDBQdrant) -> None:
    """Search on empty collection returns empty list."""
    dim = vectordb._config.embedding_dim
    results = vectordb.search_vector(_make_vector(dim, 99), top_k=5)
    assert results == []
