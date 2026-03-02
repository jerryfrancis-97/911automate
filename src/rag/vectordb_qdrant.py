"""Qdrant vector database wrapper for upserting chunks and searching by vector."""

from typing import Any, Iterable, TypedDict

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.rag.config import Config


class ChunkItem(TypedDict):
    """Chunk item for upsert: id, vector, and metadata."""

    id: str | int
    vector: list[float]
    metadata: dict[str, Any]


class VectorDBQdrant:
    """Qdrant wrapper for upserting chunks and searching by vector."""

    def __init__(self, config: Config | None = None) -> None:
        """Initialize with Config. Uses qdrant_url, qdrant_api_key, collection_name, embedding_dim."""
        self._config = config or Config()
        kwargs: dict[str, Any] = {"url": self._config.qdrant_url}
        if self._config.qdrant_api_key:
            kwargs["api_key"] = self._config.qdrant_api_key
        self._client = QdrantClient(**kwargs)

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if self._client.collection_exists(self._config.collection_name):
            return
        self._client.create_collection(
            collection_name=self._config.collection_name,
            vectors_config=VectorParams(
                size=self._config.embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def upsert_chunks(self, chunks: Iterable[ChunkItem]) -> None:
        """Upsert chunks to Qdrant. Each chunk has id, vector, and metadata."""
        chunk_list = list(chunks)
        if not chunk_list:
            return
        self._ensure_collection()
        points = [
            PointStruct(
                id=_id(c),
                vector=_vector(c),
                payload=_payload(c),
            )
            for c in chunk_list
        ]
        self._client.upsert(
            collection_name=self._config.collection_name,
            points=points,
            wait=True,
        )

    def search_vector(
        self,
        q_vector: list[float],
        top_k: int | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search by vector. Returns list of (payload, score) tuples."""
        k = top_k if top_k is not None else self._config.top_k
        self._ensure_collection()
        results = self._client.query_points(
            collection_name=self._config.collection_name,
            query=q_vector,
            limit=k,
            with_payload=True,
        )
        return [
            (dict(hit.payload or {}), float(hit.score or 0.0))
            for hit in results.points
        ]

    def search_vector_with_vectors(
        self,
        q_vector: list[float],
        top_k: int | None = None,
    ) -> list[tuple[dict[str, Any], float, list[float]]]:
        """Search by vector. Returns list of (payload, score, vector) tuples."""
        k = top_k if top_k is not None else self._config.top_k
        self._ensure_collection()
        results = self._client.query_points(
            collection_name=self._config.collection_name,
            query=q_vector,
            limit=k,
            with_payload=True,
            with_vectors=True,
        )
        out: list[tuple[dict[str, Any], float, list[float]]] = []
        for hit in results.points:
            payload = dict(hit.payload or {})
            score = float(hit.score or 0.0)
            raw = getattr(hit, "vector", None)
            if isinstance(raw, list):
                vec = raw
            elif hasattr(raw, "__iter__") and not isinstance(raw, (dict, str)):
                vec = list(raw)
            elif isinstance(raw, dict) and raw:
                vec = next(iter(raw.values())) if raw else []
            else:
                vec = []
            out.append((payload, score, vec))
        return out


def _id(chunk: ChunkItem | dict | Any) -> str | int:
    """Extract id from chunk (dict or object with id attr)."""
    return chunk["id"] if isinstance(chunk, dict) else getattr(chunk, "id")


def _vector(chunk: ChunkItem | dict | Any) -> list[float]:
    """Extract vector from chunk."""
    return chunk["vector"] if isinstance(chunk, dict) else getattr(chunk, "vector")


def _payload(chunk: ChunkItem | dict | Any) -> dict[str, Any]:
    """Extract metadata as payload from chunk. Ensures JSON-serializable values."""
    meta = chunk["metadata"] if isinstance(chunk, dict) else getattr(chunk, "metadata")
    if not isinstance(meta, dict):
        return {}
    return {k: _to_serializable(v) for k, v in meta.items()}


def _to_serializable(val: Any) -> Any:
    """Coerce value for Qdrant payload (JSON-serializable)."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [_to_serializable(x) for x in val]
    if isinstance(val, dict):
        return {str(k): _to_serializable(v) for k, v in val.items()}
    return str(val)
