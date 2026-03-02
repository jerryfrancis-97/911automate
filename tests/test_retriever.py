"""Unit tests for Retriever with MMR re-ranking."""

import math
import uuid

import pytest

from src.rag.config import Config
from src.rag.embedder import Embedder
from src.rag.retriever import CalculateMMR, Retriever, RetrievedChunk
from src.rag.vectordb_qdrant import ChunkItem, VectorDBQdrant


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def _make_vector(dim: int, seed: int) -> list[float]:
    vec = [float((seed + i) % 100) / 100.0 for i in range(dim)]
    return _normalize(vec)


@pytest.fixture
def embedder() -> Embedder:
    """Embedder with default Config. Skips if sentence-transformers unavailable."""
    try:
        return Embedder(config=Config(), normalize=True)
    except Exception as e:
        pytest.skip(f"sentence-transformers not available: {e}")


@pytest.fixture
def vectordb() -> VectorDBQdrant:
    """VectorDBQdrant with test collection. Skips if Qdrant unavailable."""
    try:
        config = Config(
            qdrant_url="http://localhost:6333",
            collection_name=f"911automate_retriever_test_{uuid.uuid4().hex[:8]}",
            embedding_dim=384,
            top_k=5,
        )
        vdb = VectorDBQdrant(config=config)
        vdb._ensure_collection()
        return vdb
    except Exception as e:
        pytest.skip(f"Qdrant unavailable: {e}")


@pytest.fixture
def seeded_vectordb(vectordb: VectorDBQdrant) -> VectorDBQdrant:
    """Vectordb pre-seeded with chunks that have text in payload."""
    dim = vectordb._config.embedding_dim
    chunks: list[ChunkItem] = [
        {
            "id": 0,
            "vector": _make_vector(dim, 1),
            "metadata": {
                "doc_id": "doc1",
                "page": 1,
                "text": "Emergency childbirth protocol first steps",
            },
        },
        {
            "id": 1,
            "vector": _make_vector(dim, 2),
            "metadata": {"doc_id": "doc1", "page": 2, "text": "Second chunk content"},
        },
        {
            "id": 2,
            "vector": _make_vector(dim, 3),
            "metadata": {"doc_id": "doc2", "page": 1, "text": "Third chunk content"},
        },
    ]
    vectordb.upsert_chunks(chunks)
    return vectordb


@pytest.fixture
def retriever(embedder: Embedder, seeded_vectordb: VectorDBQdrant) -> Retriever:
    """Retriever with embedder and pre-seeded vectordb."""
    return Retriever(
        embedder=embedder,
        vectordb=seeded_vectordb,
        use_mmr=True,
        mmr_lambda=0.5,
        candidate_multiplier=2,
    )


def test_retriever_importable() -> None:
    """Retriever importable from src.rag.retriever."""
    assert Retriever is not None
    assert RetrievedChunk is not None
    assert CalculateMMR is not None


def test_calculate_mmr_cosine_sim() -> None:
    """CalculateMMR._cosine_sim: same normalized vector gives 1.0, orthogonal gives 0."""
    vec = _normalize([1.0, 0.0, 0.0])
    assert abs(CalculateMMR._cosine_sim(vec, vec) - 1.0) < 1e-6
    orth = _normalize([0.0, 1.0, 0.0])
    assert abs(CalculateMMR._cosine_sim(vec, orth)) < 1e-6


def test_calculate_mmr_calc_mmr_score_no_selected() -> None:
    """When selected_vectors is empty, _calc_mmr_score = (1 - lambda) * relevance."""
    vec = _make_vector(4, 1)
    score = CalculateMMR._calc_mmr_score(vec, 0.9, [], mmr_lambda=0.5)
    assert abs(score - 0.45) < 1e-6


def test_calculate_mmr_calc_mmr_score_with_selected() -> None:
    """When selected vectors exist, MMR penalizes similarity to selected."""
    vec_a = _make_vector(4, 1)
    vec_b = _make_vector(4, 2)
    score_same = CalculateMMR._calc_mmr_score(vec_a, 0.9, [vec_a], mmr_lambda=0.5)
    score_diff = CalculateMMR._calc_mmr_score(vec_b, 0.9, [vec_a], mmr_lambda=0.5)
    assert score_diff > score_same


def test_retrieve_returns_top_k(retriever: Retriever) -> None:
    """For a query, len(retrieve(query, top_k=3)) <= 3."""
    results = retriever.retrieve("emergency childbirth", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0


def test_mmr_output_from_qdrant_results(retriever: Retriever) -> None:
    """MMR output items are a subset of initial Qdrant results (no new items)."""
    results = retriever.retrieve("childbirth protocol", top_k=2, use_mmr=True)
    valid_texts = {
        "Emergency childbirth protocol first steps",
        "Second chunk content",
        "Third chunk content",
    }
    for r in results:
        assert r["text"] in valid_texts
        assert "metadata" in r
        assert "score" in r


def test_mmr_does_not_crash(retriever: Retriever) -> None:
    """retrieve(query, top_k=2, use_mmr=True) completes without error."""
    results = retriever.retrieve("emergency", top_k=2, use_mmr=True)
    assert isinstance(results, list)


def test_retrieve_consistency(retriever: Retriever) -> None:
    """Same query twice returns same number of results."""
    r1 = retriever.retrieve("protocol", top_k=3)
    r2 = retriever.retrieve("protocol", top_k=3)
    assert len(r1) == len(r2)


def test_no_mmr_returns_first_k(retriever: Retriever) -> None:
    """With use_mmr=False, order matches Qdrant order (relevance only)."""
    results_mmr = retriever.retrieve("childbirth", top_k=2, use_mmr=True)
    results_no_mmr = retriever.retrieve("childbirth", top_k=2, use_mmr=False)
    assert len(results_no_mmr) <= 2
    assert len(results_mmr) <= 2
    if len(results_no_mmr) >= 2 and len(results_mmr) >= 2:
        assert results_no_mmr[0]["score"] >= results_no_mmr[1]["score"]


def test_retrieve_empty_query_returns_empty(retriever: Retriever) -> None:
    """Empty query may return empty or embedder-dependent; should not crash."""
    results = retriever.retrieve("", top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
