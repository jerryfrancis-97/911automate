"""PDF ingestion loader (PyMuPDF-based)."""

from pathlib import Path
from typing import Any

import pymupdf4llm

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    from langchain.schema import Document

def save_markdown_files(
    page_dicts: list[dict[str, Any]],
    output_dir: str | Path = "data/markdown",
    doc_id: str = "document",
) -> Path:
    """Writes page dicts to output_dir/doc_id/page_N.md. Returns output dir path."""
    output_dir = Path(output_dir)
    doc_dir = output_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    for d in page_dicts:
        page = d.get("page", 0)
        content = d.get("content", "")
        out_path = doc_dir / f"page_{page}.md"
        out_path.write_text(content, encoding="utf-8")
    return doc_dir


class PyMuPDFDocumentLoader:
    """Loads PDFs using pymupdf4llm and writes markdown pages to disk."""

    def __init__(self):
        """
        PyMuPDFDocumentLoader does not require initialization arguments.
        This constructor is defined for interface consistency.
        """
        print("Initiliazed PymuPDF loader")

    def load_documents(
        self,
        path: str | Path,
        doc_id: str,
        output_dir: str | Path = "data/markdown",
    ) -> list[Document]:
        """
        Parses PDF via pymupdf4llm, saves markdown via save_markdown_files,
        returns List[Document] with metadata doc_id, source, page.
        """

        pdf_path = Path(path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")

        pymupdf_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
        pymupdf_page_dicts = [
            {
                "page": c.get("metadata", {}).get("page_number", i + 1),
                "content": c.get("text", ""),
            }
            for i, c in enumerate(pymupdf_chunks)
        ]
        save_markdown_files(pymupdf_page_dicts, output_dir=output_dir, doc_id=doc_id)
        return [
            Document(
                page_content=d.get("content", ""),
                metadata={
                    "doc_id": doc_id,
                    "source": str(pdf_path.resolve()),
                    "page": d.get("page", 0),
                },
            )
            for d in pymupdf_page_dicts
        ]
