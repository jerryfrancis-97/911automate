"""Unit tests for src/rag/cli.py.

All external dependencies (Embedder, VectorDBQdrant, Retriever, Agent) are
mocked so no Qdrant server, embedding model download, or LLM is required.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.rag.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockRetriever:
    """Always returns an empty chunk list."""

    def retrieve(self, query: str) -> list:
        return []


def _make_mock_agent(action: str = "answer", response: str = "ok") -> MagicMock:
    """Return a mock Agent whose handle() returns a minimal result dict."""
    agent = MagicMock()
    agent.handle.return_value = {
        "action": action,
        "response": response,
        "confidence": 0.9,
        "session": {},
    }
    return agent


# ---------------------------------------------------------------------------
# Config resolution from environment variables
# ---------------------------------------------------------------------------


def test_cli_config_defaults(monkeypatch):
    """main() builds Config with default URLs when env vars are absent."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)

    captured_config: list[Config] = []

    def fake_embedder(config):
        captured_config.append(config)
        return MagicMock()

    mock_agent = _make_mock_agent()
    mock_agent.run_agent_loop.return_value = None

    with patch("src.rag.cli.os") as mock_os, \
         patch("src.rag.embedder.Embedder", side_effect=fake_embedder), \
         patch("src.rag.vectordb_qdrant.VectorDBQdrant", return_value=MagicMock()), \
         patch("src.rag.retriever.Retriever", return_value=MockRetriever()), \
         patch("src.rag.agent.Agent", return_value=mock_agent):

        # Simulate env vars not set → os.environ.get returns the default
        mock_os.environ.get.side_effect = lambda key, default=None: default

        from src.rag import cli
        import importlib
        importlib.reload(cli)  # ensure fresh import state

        cli.main()

    config = captured_config[0]
    assert config.qdrant_url == "http://localhost:6333"
    assert config.ollama_base_url == "http://localhost:11434"
    assert config.api_base_url is None
    assert config.api_key is None


def test_cli_config_from_env(monkeypatch):
    """main() picks up QDRANT_URL, OLLAMA_URL, API_BASE_URL, API_KEY from env."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant-host:6333")
    monkeypatch.setenv("OLLAMA_URL", "http://ollama-host:11434")
    monkeypatch.setenv("API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("API_KEY", "sk-test-key")

    captured: list[Config] = []

    def fake_embedder(config):
        captured.append(config)
        return MagicMock()

    mock_agent = _make_mock_agent()
    mock_agent.run_agent_loop.return_value = None

    with patch("src.rag.embedder.Embedder", side_effect=fake_embedder), \
         patch("src.rag.vectordb_qdrant.VectorDBQdrant", return_value=MagicMock()), \
         patch("src.rag.retriever.Retriever", return_value=MockRetriever()), \
         patch("src.rag.agent.Agent", return_value=mock_agent):

        from src.rag import cli
        cli.main()

    cfg = captured[0]
    assert cfg.qdrant_url == "http://qdrant-host:6333"
    assert cfg.ollama_base_url == "http://ollama-host:11434"
    assert cfg.api_base_url == "https://api.openai.com/v1"
    assert cfg.api_key == "sk-test-key"


def test_cli_api_base_url_empty_string_becomes_none(monkeypatch):
    """An empty API_BASE_URL env var is normalised to None (not an empty string)."""
    monkeypatch.setenv("API_BASE_URL", "")
    monkeypatch.setenv("API_KEY", "")

    captured: list[Config] = []

    def fake_embedder(config):
        captured.append(config)
        return MagicMock()

    mock_agent = _make_mock_agent()
    mock_agent.run_agent_loop.return_value = None

    with patch("src.rag.embedder.Embedder", side_effect=fake_embedder), \
         patch("src.rag.vectordb_qdrant.VectorDBQdrant", return_value=MagicMock()), \
         patch("src.rag.retriever.Retriever", return_value=MockRetriever()), \
         patch("src.rag.agent.Agent", return_value=mock_agent):

        from src.rag import cli
        cli.main()

    cfg = captured[0]
    assert cfg.api_base_url is None
    assert cfg.api_key is None


# ---------------------------------------------------------------------------
# Config field sanity checks (no external services needed)
# ---------------------------------------------------------------------------


def test_config_required_fields_present():
    """Config exposes every field the CLI depends on."""
    cfg = Config()
    assert hasattr(cfg, "qdrant_url")
    assert hasattr(cfg, "ollama_base_url")
    assert hasattr(cfg, "api_base_url")
    assert hasattr(cfg, "api_key")
    assert hasattr(cfg, "confidence_threshold")
    assert hasattr(cfg, "max_clarify_rounds")
    assert hasattr(cfg, "prompts_dir")
    assert hasattr(cfg, "embedding_model")
    assert hasattr(cfg, "collection_name")


def test_config_default_values_are_sane():
    """Default Config values are non-empty and within reasonable ranges."""
    cfg = Config()
    assert cfg.qdrant_url.startswith("http")
    assert cfg.ollama_base_url.startswith("http")
    assert cfg.collection_name != ""
    assert cfg.embedding_model != ""
    assert 0.0 < cfg.confidence_threshold < 1.0
    assert cfg.max_clarify_rounds >= 1
    assert cfg.top_k >= 1
    assert cfg.embedding_dim > 0


def test_config_custom_values_are_stored():
    """Config correctly stores custom values passed at construction."""
    cfg = Config(
        qdrant_url="http://custom:6333",
        confidence_threshold=0.5,
        max_clarify_rounds=2,
        collection_name="my_collection",
    )
    assert cfg.qdrant_url == "http://custom:6333"
    assert cfg.confidence_threshold == 0.5
    assert cfg.max_clarify_rounds == 2
    assert cfg.collection_name == "my_collection"


def test_config_is_immutable():
    """Config is a frozen dataclass — mutation raises an error."""
    cfg = Config()
    with pytest.raises((AttributeError, TypeError)):
        cfg.qdrant_url = "http://other:6333"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Agent wiring: CLI correctly connects components
# ---------------------------------------------------------------------------


def test_cli_agent_is_constructed_and_loop_is_called(monkeypatch):
    """main() constructs an Agent and calls run_agent_loop() exactly once."""
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)

    mock_agent = _make_mock_agent()
    mock_agent.run_agent_loop.return_value = None

    with patch("src.rag.embedder.Embedder", return_value=MagicMock()), \
         patch("src.rag.vectordb_qdrant.VectorDBQdrant", return_value=MagicMock()), \
         patch("src.rag.retriever.Retriever", return_value=MockRetriever()), \
         patch("src.rag.agent.Agent", return_value=mock_agent):

        from src.rag import cli
        cli.main()

    mock_agent.run_agent_loop.assert_called_once()


def test_cli_passes_config_to_agent(monkeypatch):
    """Agent is constructed with the same Config built from env vars."""
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:9999")
    monkeypatch.delenv("API_BASE_URL", raising=False)

    agent_configs: list[Config] = []

    def fake_agent(retriever, config):
        agent_configs.append(config)
        m = _make_mock_agent()
        m.run_agent_loop.return_value = None
        return m

    with patch("src.rag.embedder.Embedder", return_value=MagicMock()), \
         patch("src.rag.vectordb_qdrant.VectorDBQdrant", return_value=MagicMock()), \
         patch("src.rag.retriever.Retriever", return_value=MockRetriever()), \
         patch("src.rag.agent.Agent", side_effect=fake_agent):

        from src.rag import cli
        cli.main()

    assert agent_configs[0].qdrant_url == "http://qdrant:9999"


# ---------------------------------------------------------------------------
# Error handling: retriever init failure → sys.exit(1)
# ---------------------------------------------------------------------------


def test_cli_exits_with_code_1_when_retriever_fails(monkeypatch, capsys):
    """When Embedder/VectorDB/Retriever raise, main() prints to stderr and exits 1."""
    monkeypatch.delenv("API_BASE_URL", raising=False)

    with patch("src.rag.embedder.Embedder", side_effect=RuntimeError("no model")):
        with pytest.raises(SystemExit) as exc_info:
            from src.rag import cli
            cli.main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to initialise retriever" in captured.err


def test_cli_stderr_message_mentions_qdrant(monkeypatch, capsys):
    """Error message hints the user to check Qdrant connectivity."""
    monkeypatch.delenv("API_BASE_URL", raising=False)

    with patch("src.rag.embedder.Embedder", side_effect=ConnectionError("refused")):
        with pytest.raises(SystemExit):
            from src.rag import cli
            cli.main()

    captured = capsys.readouterr()
    assert "Qdrant" in captured.err
