"""Configuration for 911automate RAG. Loads from config.yml with env overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


# From core/config.py: parent=core, parent.parent=rag, parent.parent.parent=src, parent.parent.parent.parent=project_root
_DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file. Raises FileNotFoundError if missing."""
    import yaml

    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw) or {}


def _find_config_path(path: str | Path | None) -> Path:
    """Resolve config path: explicit path, CONFIG_PATH env, or default."""
    if path is not None:
        p = Path(path)
        if p.is_absolute():
            return p
        return Path.cwd() / p
    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _DEFAULT_PATH


class Config:
    """RAG configuration loaded from YAML. Typed attributes for IDE support and type checking."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "911automate"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ollama_embedding_model: str = "all-minilm"
    embedding_version: str = "v1"
    embedding_dim: int = 384
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5
    confidence_threshold: float = 0.7
    max_clarify_rounds: int = 2
    max_history_turns: int = 4
    eval_mode: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    api_base_url: str | None = None
    api_model: str = "gpt-4o-mini"
    api_key: str | None = None
    use_ollama_by_default: bool = True
    prompts_dir: str = "prompts"
    prompts_file: str = "prompts/system_prompts.yaml"
    agent_warm_on_start: bool = True

    def __init__(
        self,
        path: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        """Load config from YAML. Overrides can be passed as keyword args."""
        config_path = _find_config_path(path)
        data = _load_yaml(config_path) if config_path.exists() else {}
        data.update(overrides)

        # Set typed attributes from data
        self.qdrant_url = str(data.get("qdrant_url", "http://localhost:6333"))
        self.qdrant_api_key = data.get("qdrant_api_key")
        self.collection_name = str(data.get("collection_name", "911automate"))
        self.embedding_model = str(
            data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.ollama_embedding_model = str(data.get("ollama_embedding_model", "all-minilm"))
        self.embedding_version = str(data.get("embedding_version", "v1"))
        self.embedding_dim = int(data.get("embedding_dim", 384))
        self.chunk_size = int(data.get("chunk_size", 400))
        self.chunk_overlap = int(data.get("chunk_overlap", 50))
        self.top_k = int(data.get("top_k", 5))
        self.confidence_threshold = float(data.get("confidence_threshold", 0.7))
        self.max_clarify_rounds = int(data.get("max_clarify_rounds", 2))
        self.max_history_turns = int(data.get("max_history_turns", 4))
        self.eval_mode = bool(data.get("eval_mode", False))
        self.ollama_base_url = str(data.get("ollama_base_url", "http://localhost:11434"))
        self.ollama_model = str(data.get("ollama_model", "llama3.2"))
        self.api_base_url = data.get("api_base_url")
        self.api_model = str(data.get("api_model", "gpt-4o-mini"))
        self.api_key = data.get("api_key")
        self.use_ollama_by_default = bool(data.get("use_ollama_by_default", True))
        self.prompts_dir = str(data.get("prompts_dir", "prompts"))
        self.prompts_file = str(data.get("prompts_file", "prompts/system_prompts.yaml"))
        self.agent_warm_on_start = bool(data.get("agent_warm_on_start", True))

    @classmethod
    def from_env(cls, path: str | Path | None = None) -> Config:
        """Build Config from YAML with environment variable overrides.

        Env vars: QDRANT_URL, OLLAMA_URL, API_BASE_URL, API_KEY, AGENT_WARM_ON_START.
        Replaces _build_config() from api/deps.
        """
        overrides: dict[str, Any] = {}
        if "QDRANT_URL" in os.environ:
            overrides["qdrant_url"] = os.environ["QDRANT_URL"]
        if "OLLAMA_URL" in os.environ:
            overrides["ollama_base_url"] = os.environ["OLLAMA_URL"]
        if "API_BASE_URL" in os.environ:
            overrides["api_base_url"] = os.environ["API_BASE_URL"] or None
        if "API_KEY" in os.environ:
            overrides["api_key"] = os.environ["API_KEY"] or None
        if "AGENT_WARM_ON_START" in os.environ:
            overrides["agent_warm_on_start"] = (
                os.environ["AGENT_WARM_ON_START"].lower() in ("true", "1", "yes")
            )
        return cls(path=path, **overrides)
