"""Integration tests for VectorDBQdrant: upsert chunks and search by vector."""

import math
import uuid

import pytest

from src.rag.config import Config
from src.rag.vectordb_qdrant import ChunkItem, VectorDBQdrant


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


def test_search_empty_collection_returns_empty(vectordb: VectorDBQdrant) -> None:
    """Search on empty collection returns empty list."""
    dim = vectordb._config.embedding_dim
    results = vectordb.search_vector(_make_vector(dim, 99), top_k=5)
    assert results == []
