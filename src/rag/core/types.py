"""Shared data types and protocols for the 911automate RAG system."""

from __future__ import annotations

from typing import Any, Iterable, Protocol, TypedDict, runtime_checkable


class ProvenanceInfo(TypedDict):
    """Provenance metadata surfaced to the operator / UI."""

    doc_id: str
    page: int
    chunk_index: int
    source_path: str
    score: float


class RetrievedChunk(TypedDict):
    """Retrieved chunk with text, metadata, provenance, and score."""

    text: str
    metadata: dict[str, Any]
    score: float
    provenance: ProvenanceInfo


class QdrantPayload(TypedDict, total=False):
    """Typed schema for Qdrant point payloads.

    All fields are optional at the TypedDict level so that legacy data
    without every key can still be read back, but ``validate_payload``
    ensures every key is present before upsert.
    """

    chunk_id: str
    doc_id: str
    page: int
    chunk_index: int
    text_preview: str
    source_path: str
    embedding_version: str


class ChunkItem(TypedDict):
    """Chunk item for upsert: id, vector, and metadata."""

    id: str | int
    vector: list[float]
    metadata: dict[str, Any]


@runtime_checkable
class MetricsRecorder(Protocol):
    """Protocol for recording agent metrics (latency, counts, etc.)."""

    def record_retrieval_latency(self, seconds: float) -> None: ...

    def record_llm_latency(self, seconds: float) -> None: ...


@runtime_checkable
class VectorDB(Protocol):
    """Protocol for vector database operations."""

    def upsert_chunks(self, chunks: Iterable[ChunkItem]) -> None: ...

    def search_vector_with_vectors(
        self,
        q_vector: list[float],
        top_k: int | None = None,
    ) -> list[tuple[dict[str, Any], float, list[float]]]: ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for text embedding."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
