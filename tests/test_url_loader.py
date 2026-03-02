"""Unit tests for URLDocumentLoader."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.ingestion import URLDocumentLoader

# Stable public PDF URL for integration-style tests
SAMPLE_PDF_URL = "https://arxiv.org/pdf/2408.09869"


def test_url_document_loader_importable() -> None:
    """Class importable from src.rag.ingestion."""
    assert URLDocumentLoader is not None


def test_load_documents_pdf_url_returns_documents_live() -> None:
    """Load from stable public PDF URL, assert len(docs) > 0 (live network test)."""
    loader = URLDocumentLoader()
    docs = loader.load_documents(SAMPLE_PDF_URL, doc_id="arxiv_test")
    assert len(docs) > 0


def test_load_documents_pdf_url_page_dicts_saved() -> None:
    """Verify markdown files created under data/markdown/{doc_id}/."""
    loader = URLDocumentLoader()
    docs = loader.load_documents(SAMPLE_PDF_URL, doc_id="arxiv_page_saved")
    assert len(docs) > 0
    project_root = Path(__file__).resolve().parent.parent
    markdown_dir = project_root / "data" / "markdown" / "arxiv_page_saved"
    assert markdown_dir.exists()
    md_files = list(markdown_dir.glob("page_*.md"))
    assert len(md_files) == len(docs)
    assert md_files[0].read_text()


def test_load_documents_pdf_url_metadata_keys() -> None:
    """Documents have doc_id, source, page in metadata."""
    loader = URLDocumentLoader()
    docs = loader.load_documents(SAMPLE_PDF_URL, doc_id="arxiv_meta_test")
    assert len(docs) > 0
    for doc in docs:
        assert "doc_id" in doc.metadata
        assert "source" in doc.metadata
        assert "page" in doc.metadata
        assert doc.metadata["doc_id"] == "arxiv_meta_test"
        assert doc.metadata["source"] == SAMPLE_PDF_URL
        assert isinstance(doc.metadata["page"], int)


def test_load_documents_empty_handling() -> None:
    """Mock loader returning empty list; verify no error, returns []."""
    loader = URLDocumentLoader()
    with patch("src.rag.ingestion.OnlinePDFLoader") as mock_loader_cls:
        mock_loader = MagicMock()
        mock_loader.load.return_value = []
        mock_loader_cls.return_value = mock_loader

        docs = loader.load_documents(SAMPLE_PDF_URL, doc_id="empty")
        assert docs == []


@pytest.mark.skip(reason="HTML loader not implemented")
def test_load_documents_html_url() -> None:
    """Placeholder for future WebBaseLoader integration when HTML support is added."""
    pass
