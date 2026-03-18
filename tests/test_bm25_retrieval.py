"""Tests for BM25 retrieval: tokenize() and BM25Retriever."""

import json
from pathlib import Path

import pytest

from src.rag.retrieval.bm25_retrieval import tokenize, BM25Retriever
from src.rag.core.config import Config


def test_tokenize_lowercase() -> None:
    """tokenize lowercases input."""
    assert "hello" in tokenize("Hello World")
    assert "world" in tokenize("Hello World")


def test_tokenize_removes_non_alphanumeric() -> None:
    """tokenize removes non-alphanumeric tokens."""
    result = tokenize("Hello, world! 123")
    assert "hello" in result
    assert "world" in result
    assert "123" in result
    assert "," not in result and "!" not in result


def test_tokenize_stems() -> None:
    """tokenize applies Porter stemming."""
    # running -> run, bleeding -> bleed
    assert tokenize("running") == ["run"]
    assert tokenize("bleeding") == ["bleed"]


def test_tokenize_empty_input() -> None:
    """tokenize returns [] for empty or invalid input."""
    assert tokenize("") == []
    assert tokenize("   ") == []


@pytest.fixture
def bm25_corpus(tmp_path: Path) -> Path:
    """Create a minimal tokenized corpus JSONL and chunk files."""
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    chunk_a = chunk_dir / "chunk_0.md"
    chunk_a.write_text("ALS bleeding criteria: unconscious or not breathing.", encoding="utf-8")
    chunk_b = chunk_dir / "chunk_1.md"
    chunk_b.write_text("Pizza recipes and cooking tips.", encoding="utf-8")

    corpus_path = tmp_path / "tokenized_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"source": str(chunk_a.resolve()), "tokens": tokenize(chunk_a.read_text())},
                ensure_ascii=False,
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {"source": str(chunk_b.resolve()), "tokens": tokenize(chunk_b.read_text())},
                ensure_ascii=False,
            )
            + "\n"
        )
    return corpus_path


def test_bm25_retriever_retrieve(bm25_corpus: Path) -> None:
    """BM25Retriever returns chunks with raw text from disk, ranked by relevance."""
    config = Config(tokenized_corpus_path=str(bm25_corpus), bm25_top_k=2)
    retriever = BM25Retriever(config=config, corpus_path=bm25_corpus)
    results = retriever.retrieve("bleeding ALS", top_k=2)
    assert len(results) == 2
    # Top result should be the bleeding/ALS chunk
    assert "bleeding" in results[0]["text"].lower() or "als" in results[0]["text"].lower()
    assert results[0]["text"] == "ALS bleeding criteria: unconscious or not breathing."
    assert "chunk_0" in results[0]["metadata"].get("chunk_id", "")
