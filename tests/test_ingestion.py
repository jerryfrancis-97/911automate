"""Unit tests for src.rag.ingestion."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.rag.ingestion import (
    DocumentLoader,
    DoclingParserAdapter,
    save_markdown_files,
)

SAMPLE_PDF = Path(__file__).resolve().parent.parent / "data" / "emergency_childbirth.pdf"


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF (skip if not present)."""
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found: {SAMPLE_PDF}")
    return SAMPLE_PDF


def test_parse_to_markdown_returns_non_empty(sample_pdf_path: Path) -> None:
    """Adapter produces >0 page dicts for sample PDF."""
    adapter = DoclingParserAdapter()
    page_dicts = adapter.parse_to_markdown(sample_pdf_path)
    assert len(page_dicts) > 0


def test_page_dicts_have_page_and_content(sample_pdf_path: Path) -> None:
    """Each page dict has 'page' (int) and 'content' (str)."""
    adapter = DoclingParserAdapter()
    page_dicts = adapter.parse_to_markdown(sample_pdf_path)
    for d in page_dicts:
        assert "page" in d
        assert "content" in d
        assert isinstance(d["page"], int)
        assert isinstance(d["content"], str)


def test_load_documents_returns_documents(sample_pdf_path: Path) -> None:
    """DocumentLoader returns >0 LangChain Documents."""
    loader = DocumentLoader()
    docs = loader.load_documents(sample_pdf_path, doc_id="emergency_childbirth")
    assert len(docs) > 0
    assert all(hasattr(d, "page_content") and hasattr(d, "metadata") for d in docs)


def test_document_metadata_keys(sample_pdf_path: Path) -> None:
    """Documents include metadata keys: doc_id, source, page."""
    loader = DocumentLoader()
    docs = loader.load_documents(sample_pdf_path, doc_id="test_doc")
    for doc in docs:
        assert "doc_id" in doc.metadata
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert doc.metadata["doc_id"] == "test_doc"
        assert doc.metadata["source"] == str(sample_pdf_path.resolve())
        assert isinstance(doc.metadata["page"], int)


def test_save_markdown_files_creates_files(sample_pdf_path: Path) -> None:
    """save_markdown_files writes files under output dir."""
    adapter = DoclingParserAdapter()
    page_dicts = adapter.parse_to_markdown(sample_pdf_path)
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = save_markdown_files(
            page_dicts, output_dir=tmp, doc_id="test_save"
        )
        assert out_dir.exists()
        assert out_dir.name == "test_save"
        md_files = list(out_dir.glob("page_*.md"))
        assert len(md_files) == len(page_dicts)
        for md_path in md_files:
            content = md_path.read_text(encoding="utf-8")
            assert isinstance(content, str)


@patch("src.rag.ingestion.loader._docling_available", return_value=False)
def test_fallback_pypdf_when_no_docling(mock_available: object, sample_pdf_path: Path) -> None:
    """When docling unavailable, PyPDF fallback produces valid page dicts."""
    adapter = DoclingParserAdapter()
    page_dicts = adapter.parse_to_markdown(sample_pdf_path)
    assert len(page_dicts) > 0
    for d in page_dicts:
        assert "page" in d
        assert "content" in d
        assert isinstance(d["page"], int)
        assert isinstance(d["content"], str)

    mock_available.assert_called()
