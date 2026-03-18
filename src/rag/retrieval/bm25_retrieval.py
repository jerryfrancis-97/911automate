"""BM25 retrieval using a pre-tokenized corpus. Chunk text is always read from disk."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.rag.core.config import Config
from src.rag.core.types import ProvenanceInfo, RetrievedChunk

logger = logging.getLogger(__name__)

_STEMMER: object | None = None


def _get_stemmer():
    """Lazy-load NLTK PorterStemmer."""
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer

        _STEMMER = PorterStemmer()
    return _STEMMER


def tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, apply Porter stemming. For BM25 indexing only."""
    if not text or not isinstance(text, str):
        return []
    stemmer = _get_stemmer()
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens if t]


class BM25Retriever:
    """BM25 retriever using tokenized corpus JSONL. Returns raw chunk text from disk."""

    def __init__(
        self,
        config: Config,
        corpus_path: str | Path | None = None,
    ) -> None:
        path = Path(corpus_path or config.tokenized_corpus_path)
        if not path.is_absolute():
            root = Path(__file__).resolve().parent.parent.parent.parent
            path = root / path

        self._config = config
        self._sources: list[str] = []
        self._tokenized_corpus: list[list[str]] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                source = obj.get("source", "")
                tokens = obj.get("tokens", [])
                self._sources.append(source)
                self._tokenized_corpus.append(tokens)

        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks. Chunk text is read directly from disk (raw, unstemmed)."""
        k = top_k if top_k is not None else getattr(
            self._config, "bm25_top_k", 5
        )
        if k <= 0 or not self._tokenized_corpus:
            return []

        query_tokens = tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        out: list[RetrievedChunk] = []
        root = Path(__file__).resolve().parent.parent.parent.parent
        for idx in top_indices:
            source = self._sources[idx]
            score = float(scores[idx])
            chunk_path = root / source if not Path(source).is_absolute() else Path(source)
            try:
                text = chunk_path.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Could not read chunk %s: %s", source, e)
                text = ""
            chunk_name = chunk_path.name
            meta = {
                "chunk_id": chunk_name,
                "source_path": str(chunk_path.resolve()),
                "doc_id": chunk_path.parent.name,
            }
            prov = ProvenanceInfo(
                doc_id=chunk_path.parent.name,
                page=0,
                chunk_index=idx,
                source_path=str(chunk_path.resolve()),
                score=score,
            )
            out.append(
                RetrievedChunk(
                    text=text,
                    metadata=meta,
                    score=score,
                    provenance=prov,
                )
            )
        return out
