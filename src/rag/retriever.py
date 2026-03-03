"""Retriever that queries Qdrant and optionally re-ranks with MMR."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

from src.rag.config import Config
from src.rag.embedder import Embedder
from src.rag.vectordb_qdrant import VectorDBQdrant

logger = logging.getLogger(__name__)


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


def _build_provenance(payload: dict[str, Any], score: float) -> ProvenanceInfo:
    """Extract provenance fields from a Qdrant payload with safe defaults."""
    return ProvenanceInfo(
        doc_id=str(payload.get("doc_id", "")),
        page=int(payload.get("page", 0)),
        chunk_index=int(payload.get("chunk_index", 0)),
        source_path=str(payload.get("source_path") or payload.get("source", "")),
        score=score,
    )


class CalculateMMR:
    """Computes MMR score and cosine similarity for re-ranking."""

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity (dot product for normalized vectors)."""
        if not a or not b or len(a) != len(b):
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    @classmethod
    def _calc_mmr_score(
        cls,
        cand_vec: list[float],
        cand_relevance: float,
        selected_vectors: list[list[float]],
        mmr_lambda: float,
    ) -> float:
        """Compute MMR score for a candidate.

        mmr_score = (1 - lambda) * relevance - lambda * max_sim_to_selected
        """
        if not cand_vec:
            return (1 - mmr_lambda) * cand_relevance
        max_sim = (
            max(cls._cosine_sim(cand_vec, s) for s in selected_vectors)
            if selected_vectors
            else 0.0
        )
        return (1 - mmr_lambda) * cand_relevance - mmr_lambda * max_sim


def _resolve_text(payload: dict[str, Any]) -> str:
    """Resolve chunk text from payload (text, page_content, or path)."""
    text = payload.get("text") or payload.get("page_content")
    if isinstance(text, str) and text:
        return text
    path_str = payload.get("path")
    if path_str:
        try:
            return Path(path_str).read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("Could not read chunk path %s: %s", path_str, e)
    return ""


def _mmr_select(
    query_vec: list[float],
    candidates: list[tuple[dict[str, Any], float, list[float]]],
    top_k: int,
    mmr_lambda: float,
) -> list[tuple[dict[str, Any], float, list[float]]]:
    """Select top_k items using Maximal Marginal Relevance."""
    if top_k >= len(candidates):
        return candidates
    if top_k <= 0:
        return []
    selected: list[tuple[dict[str, Any], float, list[float]]] = []
    pool = list(candidates)
    while len(selected) < top_k and pool:
        best_idx = -1
        best_mmr = float("-inf")
        selected_vecs = [s[2] for s in selected]
        for i, (payload, rel, vec) in enumerate(pool):
            mmr_score = CalculateMMR._calc_mmr_score(
                vec, rel, selected_vecs, mmr_lambda
            )
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(pool.pop(best_idx))
    return selected


class Retriever:
    """Retriever that embeds queries, searches Qdrant, and optionally applies MMR re-ranking."""

    def __init__(
        self,
        embedder: Embedder,
        vectordb: VectorDBQdrant,
        config: Config | None = None,
        use_mmr: bool = True,
        mmr_lambda: float = 0.5,
        candidate_multiplier: int = 2,
    ) -> None:
        self._embedder = embedder
        self._vectordb = vectordb
        self._config = config or Config()
        self._use_mmr = use_mmr
        self._mmr_lambda = mmr_lambda
        self._candidate_multiplier = candidate_multiplier

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        use_mmr: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks for *query*. Returns list of RetrievedChunk dicts."""
        k = top_k if top_k is not None else self._config.top_k
        do_mmr = use_mmr if use_mmr is not None else self._use_mmr
        if k <= 0:
            return []

        query_vec = self._embedder.embed_texts([query])
        if not query_vec:
            return []
        query_vec = query_vec[0]
        candidate_k = k * self._candidate_multiplier
        results = self._vectordb.search_vector_with_vectors(
            query_vec, top_k=candidate_k
        )
        if not results:
            return []
        if do_mmr and len(results) > k:
            results = _mmr_select(
                query_vec, results, top_k=k, mmr_lambda=self._mmr_lambda
            )
        else:
            results = results[:k]

        out: list[RetrievedChunk] = []
        for payload, score, _ in results:
            text = _resolve_text(payload)
            out.append(
                RetrievedChunk(
                    text=text,
                    metadata=payload,
                    score=score,
                    provenance=_build_provenance(payload, score),
                )
            )
        return out
