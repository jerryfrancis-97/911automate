"""Reranker using Ollama to score query-passage pairs. Uses raw chunk text from disk."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from src.rag.core.types import RetrievedChunk

logger = logging.getLogger(__name__)

_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _dedup_key(chunk: RetrievedChunk) -> str:
    """Use chunk_id or source_path for deduplication."""
    meta = chunk.get("metadata") or {}
    key = meta.get("chunk_id") or meta.get("source_path") or ""
    if key:
        return key
    path = meta.get("source", "")
    if path:
        return Path(path).name
    return chunk.get("text", "")[:200]


class Reranker:
    """Reranks chunks via Ollama chat (query-passage relevance scoring)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        top_n: int = 3,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._top_n = top_n

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Deduplicate by chunk_id/source_path, score via Ollama, return top_n.
        chunk['text'] is raw disk content—never stemmed or transformed."""
        if not chunks:
            return []
        seen: set[str] = set()
        unique: list[RetrievedChunk] = []
        for c in chunks:
            key = _dedup_key(c)
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        if not unique:
            return []

        scored: list[tuple[float, RetrievedChunk]] = []
        try:
            from ollama import Client

            client = Client(host=self._base_url)
        except Exception as e:
            logger.warning("Ollama client init failed, returning chunks as-is: %s", e)
            return unique[: self._top_n]

        for chunk in unique:
            text = chunk.get("text") or ""
            if not text.strip():
                scored.append((0.0, chunk))
                continue
            prompt = (
                f"Rate relevance of the passage to the query, 0-10. Output only a number.\n"
                f"Query: {query}\n"
                f"Passage: {text[:2000]}"
            )
            try:
                resp = client.chat(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = (resp.message.content or "").strip()
                match = _SCORE_RE.search(content)
                score = float(match.group(1)) if match else 0.0
            except Exception as e:
                logger.warning("Rerank score failed for chunk: %s", e)
                score = 0.0
            scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: self._top_n]]
