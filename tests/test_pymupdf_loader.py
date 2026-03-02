"""Unit tests for PyMuPDFDocumentLoader."""

from pathlib import Path

import pytest

from src.rag.ingestion import PyMuPDFDocumentLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PDF = PROJECT_ROOT / "data" / "emergency_childbirth.pdf"
TEST_OUTPUT_DIR = PROJECT_ROOT / "test_data" / "markdown"


@pytest.fixture
def sample_pdf_path() -> Path:
    """Path to sample PDF (skip if not present)."""
    if not SAMPLE_PDF.exists():
        pytest.skip(f"Sample PDF not found: {SAMPLE_PDF}")
    return SAMPLE_PDF


def test_pymupdf_document_loader_importable() -> None:
    """PyMuPDFDocumentLoader importable from src.rag.ingestion."""
    assert PyMuPDFDocumentLoader is not None


def test_pymupdf_load_documents_returns_non_empty(sample_pdf_path: Path) -> None:
    """pymupdf_loader.load_documents returns len(pymupdf_docs) > 0."""
    pymupdf_loader = PyMuPDFDocumentLoader()
    pymupdf_docs = pymupdf_loader.load_documents(
        sample_pdf_path, doc_id="emergency_childbirth", output_dir=TEST_OUTPUT_DIR
    )
    assert len(pymupdf_docs) > 0


def test_pymupdf_page_dicts_have_page_and_content(sample_pdf_path: Path) -> None:
    """Each pymupdf_doc has metadata page and non-empty page_content."""
    pymupdf_loader = PyMuPDFDocumentLoader()
    pymupdf_docs = pymupdf_loader.load_documents(
        sample_pdf_path, doc_id="test_pages", output_dir=TEST_OUTPUT_DIR
    )
    for pymupdf_doc in pymupdf_docs:
        assert "page" in pymupdf_doc.metadata
        assert isinstance(pymupdf_doc.metadata["page"], int)
        assert len(pymupdf_doc.page_content) >= 0


def test_pymupdf_document_metadata_keys(sample_pdf_path: Path) -> None:
    """pymupdf_docs include doc_id, source, page in metadata."""
    pymupdf_loader = PyMuPDFDocumentLoader()
    pymupdf_docs = pymupdf_loader.load_documents(
        sample_pdf_path, doc_id="test_meta", output_dir=TEST_OUTPUT_DIR
    )
    for pymupdf_doc in pymupdf_docs:
        assert "doc_id" in pymupdf_doc.metadata
        assert "source" in pymupdf_doc.metadata
        assert "page" in pymupdf_doc.metadata
        assert pymupdf_doc.metadata["doc_id"] == "test_meta"
        assert str(sample_pdf_path.resolve()) in pymupdf_doc.metadata["source"]


def test_pymupdf_save_markdown_files_called(sample_pdf_path: Path) -> None:
    """save_markdown_files invoked; verify markdown dir exists."""
    pymupdf_loader = PyMuPDFDocumentLoader()
    pymupdf_docs = pymupdf_loader.load_documents(
        sample_pdf_path, doc_id="pymupdf_save_test", output_dir=TEST_OUTPUT_DIR
    )
    assert len(pymupdf_docs) > 0
    markdown_dir = TEST_OUTPUT_DIR / "pymupdf_save_test"
    assert markdown_dir.exists()
    md_files = list(markdown_dir.glob("page_*.md"))
    assert len(md_files) == len(pymupdf_docs)


def test_pymupdf_file_not_found_raises() -> None:
    """load_documents with missing path raises FileNotFoundError."""
    pymupdf_loader = PyMuPDFDocumentLoader()
    with pytest.raises(FileNotFoundError):
        pymupdf_loader.load_documents("/nonexistent/path/doc.pdf", doc_id="fail")
