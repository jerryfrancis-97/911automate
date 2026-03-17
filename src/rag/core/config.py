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

    qdrant_url: str
    qdrant_api_key: str | None
    collection_name: str
    embedding_model: str
    ollama_embedding_model: str
    embedding_version: str
    embedding_dim: int
    chunk_size: int
    chunk_overlap: int
    top_k: int
    confidence_threshold: float
    max_clarify_rounds: int
    max_history_turns: int
    eval_mode: bool
    ollama_base_url: str
    ollama_model: str
    api_base_url: str | None
    api_model: str
    api_key: str | None
    use_ollama_by_default: bool
    prompts_dir: str
    prompts_file: str
    agent_warm_on_start: bool

    def __init__(
        self,
        path: str | Path | None = None,
        **overrides: Any,
    ) -> None:
        """Load config from YAML (single source of truth).

        KISS: we rely on config.yml keys being present. If a required key is
        missing, we raise a clear error instead of silently falling back.
        """
        config_path = _find_config_path(path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Set CONFIG_PATH or pass path=... to Config()."
            )
        data = _load_yaml(config_path)
        data.update(overrides)

        required = (
            "qdrant_url",
            "collection_name",
            "embedding_model",
            "ollama_embedding_model",
            "embedding_version",
            "embedding_dim",
            "chunk_size",
            "chunk_overlap",
            "top_k",
            "confidence_threshold",
            "max_clarify_rounds",
            "eval_mode",
            "ollama_base_url",
            "ollama_model",
            "api_model",
            "use_ollama_by_default",
            "prompts_dir",
            "prompts_file",
            "agent_warm_on_start",
        )
        missing = [k for k in required if k not in data]
        if missing:
            raise KeyError(f"Missing required config keys in {config_path}: {', '.join(missing)}")

        self.qdrant_url = str(data["qdrant_url"])
        self.qdrant_api_key = data.get("qdrant_api_key")
        self.collection_name = str(data["collection_name"])
        self.embedding_model = str(data["embedding_model"])
        self.ollama_embedding_model = str(data["ollama_embedding_model"])
        self.embedding_version = str(data["embedding_version"])
        self.embedding_dim = int(data["embedding_dim"])
        self.chunk_size = int(data["chunk_size"])
        self.chunk_overlap = int(data["chunk_overlap"])

        self.top_k = int(data["top_k"])
        self.confidence_threshold = float(data["confidence_threshold"])
        self.max_clarify_rounds = int(data["max_clarify_rounds"])
        self.max_history_turns = int(data.get("max_history_turns", 4))
        self.eval_mode = bool(data["eval_mode"])

        self.ollama_base_url = str(data["ollama_base_url"])
        self.ollama_model = str(data["ollama_model"])
        self.api_base_url = data.get("api_base_url")
        self.api_model = str(data["api_model"])
        self.api_key = data.get("api_key")
        self.use_ollama_by_default = bool(data["use_ollama_by_default"])

        self.prompts_dir = str(data["prompts_dir"])
        self.prompts_file = str(data["prompts_file"])
        self.agent_warm_on_start = bool(data["agent_warm_on_start"])

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
