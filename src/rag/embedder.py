"""Embedder wrapping LangChain embeddings for embed_texts(list[str]) -> list[list[float]]."""

import math

from src.rag.config import Config


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalize a vector. Returns unit vector or zeros if norm is 0."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class Embedder:
    """Wraps LangChain embeddings for embed_texts(list[str]) -> list[list[float]]."""

    def __init__(self, config: Config | None = None, normalize: bool = True) -> None:
        """Use config.embedding_model; normalize vectors for cosine similarity if normalize=True."""
        self._config = config or Config()
        self._normalize = normalize
        self._embeddings = self._build_embeddings()

    def _build_embeddings(self):
        """Build HuggingFaceEmbeddings from config.embedding_model."""
        from langchain_huggingface import HuggingFaceEmbeddings

        model = self._config.embedding_model
        encode_kwargs = {"normalize_embeddings": True} if self._normalize else {}
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs=encode_kwargs,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts; returns list of vectors. Empty input -> empty output."""
        if not texts:
            return []
        vectors = self._embeddings.embed_documents(texts)
        # Fallback: normalize in Python if backend didn't
        if self._normalize and vectors:
            norm = math.sqrt(sum(x * x for x in vectors[0]))
            if norm > 0 and abs(norm - 1.0) > 0.01:
                vectors = [_l2_normalize(v) for v in vectors]
        return vectors
