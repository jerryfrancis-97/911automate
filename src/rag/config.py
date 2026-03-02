"""Configuration for 911automate RAG."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Config:
    """RAG configuration with sensible defaults."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    collection_name: str = "911automate"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
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
