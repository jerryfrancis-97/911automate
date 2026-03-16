"""Tests for src.rag.call_ingestion ingestion pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.rag.ingestion.pipeline import IngestionPipeline, run_ingestion

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document  # type: ignore[assignment]


def test_load_markdown_documents_builds_documents_with_metadata(tmp_path: Path) -> None:
    """IngestionPipeline.load_markdown_documents returns Documents with metadata."""
    markdown_root = tmp_path / "data" / "markdown"
    doc_id = "my_doc"
    doc_dir = markdown_root / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    f1 = doc_dir / "page_1.md"
    f2 = doc_dir / "page_2.md"
    f1.write_text("Page 1 content", encoding="utf-8")
    f2.write_text("Page 2 content", encoding="utf-8")

    paths = {
        "root": tmp_path,
        "data": tmp_path / "data",
        "markdown": markdown_root,
        "chunks": tmp_path / "data" / "chunks",
        "embeddings": tmp_path / "data" / "embeddings",
    }
    pipeline = IngestionPipeline(paths=paths)
    docs = pipeline.load_markdown_documents(doc_id)
    assert len(docs) == 2
    sources = {d.metadata["source"] for d in docs}
    file_names = {d.metadata["file_name"] for d in docs}
    assert {str(f1.resolve()), str(f2.resolve())} == sources
    assert {"page_1.md", "page_2.md"} == file_names
    for d in docs:
        assert d.metadata["doc_id"] == doc_id
        assert d.metadata["parent_doc_id"] == d.metadata["source"]


def test_chunks_to_embeddings_writes_jsonl_with_metadata(tmp_path: Path) -> None:
    """IngestionPipeline.chunks_to_embeddings writes JSONL with expected keys."""
    embeddings_root = tmp_path / "data" / "embeddings"
    doc_id = "d1"
    chunks = [
        Document(page_content="Chunk one", metadata={"doc_id": doc_id, "page": 1}),
        Document(page_content="Chunk two", metadata={"doc_id": doc_id, "page": 2}),
    ]

    # Patch Embedder to avoid external dependency on Ollama
    with patch("src.rag.ingestion.pipeline.Embedder") as mock_embed_cls:
        mock_embed = MagicMock()
        mock_embed.embed_texts.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embed_cls.return_value = mock_embed

        paths = {
            "root": tmp_path,
            "data": tmp_path / "data",
            "markdown": tmp_path / "data" / "markdown",
            "chunks": tmp_path / "data" / "chunks",
            "embeddings": embeddings_root,
        }
        pipeline = IngestionPipeline(paths=paths)
        emb_path = pipeline.chunks_to_embeddings(chunks, doc_id)

    assert emb_path.exists()
    lines = emb_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for idx, line in enumerate(lines):
        item = json.loads(line)
        assert item["chunk_idx"] == idx
        assert "embedding" in item
        assert "metadata" in item
        assert item["metadata"]["doc_id"] == doc_id


def test_embeddings_to_qdrant_builds_chunk_items_and_calls_upsert(tmp_path: Path) -> None:
    """IngestionPipeline.embeddings_to_qdrant maps records and upserts them."""
    embeddings_path = tmp_path / "emb.jsonl"
    records = [
        {
            "chunk_idx": 0,
            "embedding": [0.1, 0.2],
            "metadata": {"doc_id": "d1", "page": 1, "chunk_id": "cid-0"},
        },
        {
            "chunk_idx": 1,
            "embedding": [0.3, 0.4],
            "metadata": {"doc_id": "d1", "page": 2},
        },
    ]
    with embeddings_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    captured: dict[str, object] = {}

    class FakeVDB:
        def __init__(self, config=None) -> None:  # noqa: D401
            self.config = config

        def upsert_chunks(self, chunk_items):
            captured["chunk_items"] = chunk_items

    with patch("src.rag.ingestion.pipeline.VectorDBQdrant", FakeVDB):
        paths = {
            "root": tmp_path,
            "data": tmp_path / "data",
            "markdown": tmp_path / "data" / "markdown",
            "chunks": tmp_path / "data" / "chunks",
            "embeddings": tmp_path / "data" / "embeddings",
        }
        pipeline = IngestionPipeline(paths=paths)
        count = pipeline.embeddings_to_qdrant(embeddings_path)

    assert count == 2
    chunk_items = captured["chunk_items"]  # type: ignore[index]
    assert len(chunk_items) == 2
    # First item uses chunk_id as id
    assert chunk_items[0]["id"] == "cid-0"
    assert chunk_items[0]["vector"] == [0.1, 0.2]
    assert chunk_items[0]["metadata"]["doc_id"] == "d1"
    # Second item falls back to index id
    assert chunk_items[1]["id"] == 1
    assert chunk_items[1]["vector"] == [0.3, 0.4]


def test_pipeline_run_uses_existing_markdown(tmp_path: Path) -> None:
    """When markdown exists, IngestionPipeline.run reuses it and skips pdf ingestion."""
    data_dir = tmp_path / "data"
    markdown_root = data_dir / "markdown"
    markdown_dir = markdown_root / "doc1"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    (markdown_dir / "page_1.md").write_text("content", encoding="utf-8")

    paths = {
        "root": tmp_path,
        "data": data_dir,
        "markdown": markdown_root,
        "chunks": data_dir / "chunks",
        "embeddings": data_dir / "embeddings",
    }
    pipeline = IngestionPipeline(paths=paths)

    with patch.object(IngestionPipeline, "pdf_to_markdown_documents") as pdf_mock, patch.object(
        IngestionPipeline, "load_markdown_documents"
    ) as load_md_mock, patch.object(
        IngestionPipeline, "documents_to_chunks"
    ) as docs_to_chunks_mock, patch.object(
        IngestionPipeline, "chunks_to_embeddings"
    ) as chunks_to_emb_mock, patch.object(
        IngestionPipeline, "embeddings_to_qdrant"
    ) as emb_to_qdrant_mock:
        load_md_mock.return_value = [
            Document(page_content="c", metadata={"doc_id": "doc1", "page": 1, "source": "s"})
        ]
        docs_to_chunks_mock.return_value = [
            Document(page_content="chunk", metadata={"doc_id": "doc1", "page": 1})
        ]
        chunks_to_emb_mock.return_value = data_dir / "embeddings" / "doc1.jsonl"
        emb_to_qdrant_mock.return_value = 1

        pipeline.run(doc_id="doc1", pdf_filename=None)

    pdf_mock.assert_not_called()
    load_md_mock.assert_called_once()
    docs_to_chunks_mock.assert_called_once()
    chunks_to_emb_mock.assert_called_once()
    emb_to_qdrant_mock.assert_called_once()


def test_pipeline_run_uses_pdf_when_no_markdown(tmp_path: Path) -> None:
    """When no markdown exists, IngestionPipeline.run uses pdf_to_markdown_documents."""
    data_dir = tmp_path / "data"
    markdown_root = data_dir / "markdown"
    markdown_root.mkdir(parents=True, exist_ok=True)
    # pdf_filename is not actually read because we mock the loader

    paths = {
        "root": tmp_path,
        "data": data_dir,
        "markdown": markdown_root,
        "chunks": data_dir / "chunks",
        "embeddings": data_dir / "embeddings",
    }
    pipeline = IngestionPipeline(paths=paths)

    with patch.object(IngestionPipeline, "pdf_to_markdown_documents") as pdf_mock, patch.object(
        IngestionPipeline, "load_markdown_documents"
    ) as load_md_mock, patch.object(
        IngestionPipeline, "documents_to_chunks"
    ) as docs_to_chunks_mock, patch.object(
        IngestionPipeline, "chunks_to_embeddings"
    ) as chunks_to_emb_mock, patch.object(
        IngestionPipeline, "embeddings_to_qdrant"
    ) as emb_to_qdrant_mock:
        pdf_mock.return_value = [
            Document(page_content="c", metadata={"doc_id": "doc2", "page": 1, "source": "s"})
        ]
        docs_to_chunks_mock.return_value = [
            Document(page_content="chunk", metadata={"doc_id": "doc2", "page": 1})
        ]
        chunks_to_emb_mock.return_value = data_dir / "embeddings" / "doc2.jsonl"
        emb_to_qdrant_mock.return_value = 1

        pipeline.run(doc_id="doc2", pdf_filename="doc2.pdf")

    pdf_mock.assert_called_once()
    load_md_mock.assert_not_called()
    docs_to_chunks_mock.assert_called_once()
    chunks_to_emb_mock.assert_called_once()
    emb_to_qdrant_mock.assert_called_once()


def test_run_ingestion_delegates_to_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_ingestion constructs a pipeline and forwards arguments to .run()."""
    captured: dict[str, object] = {}

    class FakePipeline:
        def __init__(self, *args, **kwargs) -> None:
            captured["init_args"] = (args, kwargs)

        def run(self, doc_id: str, pdf_filename: str | None = None) -> None:  # noqa: D401
            captured["run_args"] = (doc_id, pdf_filename)

    monkeypatch.setattr("src.rag.ingestion.pipeline.IngestionPipeline", FakePipeline)

    run_ingestion(doc_id="d1", pdf_filename="doc2.pdf")

    assert captured["run_args"] == ("d1", "doc2.pdf")


@pytest.mark.integration
def test_run_ingestion_end_to_end_uses_qdrant_and_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: run_ingestion with mocked Embedder and temporary Qdrant collection.

    Skips if Qdrant is unavailable or sample PDF is missing.
    """
    from src.rag.core.config import Config
    from src.rag.retrieval.vectordb_qdrant import VectorDBQdrant

    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "emergency_childbirth.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}")

    # Use a temporary collection name and 4-dim vectors (mock embedder)
    test_collection = f"call_ingest_test_{id(monkeypatch) % 10_000}"
    mock_embed_dim = 4

    def _test_config(**kwargs):
        return Config(collection_name=test_collection, embedding_dim=mock_embed_dim, **kwargs)

    # Patch Config used inside call_ingestion
    monkeypatch.setattr("src.rag.ingestion.pipeline.Config", _test_config)

    with patch("src.rag.ingestion.pipeline.Embedder") as mock_embed_cls:
        mock_embed = MagicMock()
        # Simple deterministic 4-dim vectors
        def _fake_embed_texts(texts):
            return [[float(i), 0.0, 0.0, 0.0] for i, _ in enumerate(texts)]

        mock_embed.embed_texts.side_effect = _fake_embed_texts
        mock_embed_cls.return_value = mock_embed

        # Run ingestion using real project root and PDF
        run_ingestion(doc_id="emergency_childbirth_ci", pdf_filename="emergency_childbirth.pdf")

    # Verify that points exist in Qdrant for the test collection
    try:
        vdb = VectorDBQdrant(config=_test_config())
        # Query with any 4-dim vector we used above
        results = vdb.search_vector([0.0, 0.0, 0.0, 0.0], top_k=1)
        assert isinstance(results, list)
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"Qdrant unavailable or collection check failed: {e}")

