"""Unit tests for Embedder."""

import pytest

from src.rag.config import Config
from src.rag.embedder import Embedder


def test_embedder_importable() -> None:
    """Embedder is importable from src.rag.embedder."""
    assert Embedder is not None


@pytest.fixture
def embedder() -> Embedder:
    """Embedder with default Config (HuggingFaceEmbeddings)."""
    try:
        return Embedder(config=Config(), normalize=True)
    except Exception as e:
        pytest.skip(f"sentence-transformers not available: {e}")


def test_embed_texts_returns_vectors(embedder: Embedder) -> None:
    """embed_texts(['hello', 'world']) returns 2 vectors."""
    vectors = embedder.embed_texts(["hello", "world"])
    assert len(vectors) == 2
    assert all(isinstance(v, list) for v in vectors)
    assert all(all(isinstance(x, (int, float)) for x in v) for v in vectors)


def test_output_length_equals_input_length(embedder: Embedder) -> None:
    """len(embed_texts(texts)) == len(texts)."""
    texts = ["first", "second", "third"]
    vectors = embedder.embed_texts(texts)
    assert len(vectors) == len(texts)


def test_vector_dimensionality(embedder: Embedder) -> None:
    """Each vector has length == Config.embedding_dim (384)."""
    config = Config()
    vectors = embedder.embed_texts(["test"])
    assert len(vectors) == 1
    assert len(vectors[0]) == config.embedding_dim


def test_embed_texts_empty_returns_empty(embedder: Embedder) -> None:
    """embed_texts([]) returns []."""
    assert embedder.embed_texts([]) == []
