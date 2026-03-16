"""Unit tests for src.rag.config."""

import pytest

from src.rag.core.config import Config


def test_config_importable() -> None:
    """Config is importable from src.rag.config."""
    assert Config is not None


def test_config_instantiable_with_defaults() -> None:
    """Config can be instantiated with all defaults."""
    config = Config()
    assert config is not None


def test_config_default_values() -> None:
    """Config has expected default values (from YAML or built-in defaults)."""
    config = Config()
    assert config.qdrant_url == "http://localhost:6333"
    assert config.qdrant_api_key is None
    assert config.collection_name == "911automate"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.ollama_embedding_model == "all-minilm"
    assert config.embedding_dim == 384
    assert config.top_k == 5
    assert config.confidence_threshold == 0.7
    assert config.max_clarify_rounds == 2
    assert config.eval_mode is False


def test_config_override_values() -> None:
    """Config accepts overridden values via kwargs."""
    config = Config(
        qdrant_url="http://custom:6334",
        qdrant_api_key="secret",
        collection_name="custom_collection",
        top_k=10,
    )
    assert config.qdrant_url == "http://custom:6334"
    assert config.qdrant_api_key == "secret"
    assert config.collection_name == "custom_collection"
    assert config.top_k == 10
    assert config.embedding_dim == 384
