"""Test fakes for VectorDB and Embedder. Use with Retriever when Qdrant/Ollama unavailable."""

from __future__ import annotations

from collections.abc import Iterable

from src.rag.core.types import ChunkItem


class InMemoryVectorDB:
    """In-memory vector store for tests. Implements VectorDB protocol."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._points: list[tuple[str | int, list[float], dict]] = []

    def upsert_chunks(self, chunks: Iterable[ChunkItem]) -> None:
        for c in chunks:
            pid = c["id"]
            vec = c["vector"]
            meta = dict(c["metadata"])
            meta["text"] = meta.get("text_preview") or meta.get("text", "")
            self._points.append((pid, vec, meta))

    def search_vector_with_vectors(
        self,
        q_vector: list[float],
        top_k: int | None = None,
    ) -> list[tuple[dict, float, list[float]]]:
        k = top_k or 5
        if not self._points or k <= 0:
            return []
        # Simple cosine-like scoring (dot product for normalized vectors)
        scored: list[tuple[dict, float, list[float]]] = []
        for pid, vec, meta in self._points:
            if len(vec) != len(q_vector):
                continue
            score = sum(a * b for a, b in zip(q_vector, vec))
            text = meta.get("text") or meta.get("text_preview") or ""
            payload = {**meta, "text": text}
            scored.append((payload, float(score), vec))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]


class FakeEmbedder:
    """Returns fixed-dimension zero vectors. Implements EmbedderProtocol for tests."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]
