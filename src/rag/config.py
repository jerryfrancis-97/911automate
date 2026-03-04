"""Configuration for 911automate RAG."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """RAG configuration with sensible defaults."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "911automate_test_live"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # legacy; use ollama_embedding_model
    ollama_embedding_model: str = "all-minilm"
    embedding_version: str = "v1"
    embedding_dim: int = 384
    # Chunking for document splitting
    chunk_size: int = 400
    chunk_overlap: int = 50
    top_k: int = 5
    confidence_threshold: float = 0.7
    max_clarify_rounds: int = 3
    # LLM: Ollama (default) vs API
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    api_base_url: Optional[str] = None
    api_model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    use_ollama_by_default: bool = True
    prompts_dir: str = "prompts"
    # Agent warm on startup (background task, no blocking)
    agent_warm_on_start: bool = True
