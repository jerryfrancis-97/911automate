"""PDF ingestion with Docling or PyPDF fallback."""

from pathlib import Path
from typing import Any, Protocol

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
from pypdf import PdfReader
from langchain_community.document_loaders import OnlinePDFLoader
import pymupdf4llm


class DocLoader(Protocol):
    """Protocol for document loaders. Align all loaders to this interface."""

    def load_documents(
        self, path: str | Path, doc_id: str, **kwargs: Any
    ) -> list[Document]: ...


_DOCLING_AVAILABLE: bool | None = None


def _docling_available() -> bool:
    """Check if docling is importable and usable."""
    global _DOCLING_AVAILABLE
    if _DOCLING_AVAILABLE is not None:
        return _DOCLING_AVAILABLE
    try:
        import docling.document_converter  # noqa: F401

        _DOCLING_AVAILABLE = True
        return True
    except ImportError:
        _DOCLING_AVAILABLE = False
        return False


class DoclingParserAdapter:
    """Parses PDF to markdown; uses docling if available, else PyPDF."""

    def parse_to_markdown(self, path: str | Path) -> list[dict[str, Any]]:
        """
        Returns list of {"page": int, "content": str} dicts.
        Uses docling if importable, else PyPDF for text extraction.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        reader = PdfReader(str(path))
        num_pages = len(reader.pages)

        if _docling_available():
            print("Using docling")
            return self._parse_with_docling(path, num_pages)

        print("Using pypdf")
        return self._parse_with_pypdf(reader, num_pages)

    def _parse_with_docling(self, path: Path, num_pages: int) -> list[dict[str, Any]]:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result: list[dict[str, Any]] = []
        for i in range(1, num_pages + 1):
            conv_result = converter.convert(path, page_range=(i, i))
            markdown = conv_result.document.export_to_markdown()
            result.append({"page": i, "content": markdown or ""})
        return result

    def _parse_with_pypdf(
        self, reader: PdfReader, num_pages: int
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for i in range(num_pages):
            text = reader.pages[i].extract_text() or ""
            result.append({"page": i + 1, "content": text})
        return result


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


class DocumentLoader:
    """Loads PDFs and returns LangChain Documents."""

    def __init__(self, adapter: DoclingParserAdapter | None = None) -> None:
        self.adapter = adapter or DoclingParserAdapter()

    def load_documents(self, path: str | Path, doc_id: str) -> list[Document]:
        """
        Parses PDF at path and returns List[Document] with metadata doc_id, source, page.
        Parsed markdown is saved to data/markdown/{doc_id}/ for reuse.
        """
        path = Path(path)
        page_dicts = self.adapter.parse_to_markdown(path)
        save_markdown_files(page_dicts, doc_id=doc_id)
        docs: list[Document] = []
        for d in page_dicts:
            docs.append(
                Document(
                    page_content=d.get("content", ""),
                    metadata={
                        "doc_id": doc_id,
                        "source": str(path.resolve()),
                        "page": d.get("page", 0),
                    },
                )
            )
        return docs


class URLDocumentLoader:
    """Loads PDFs from URLs via OnlinePDFLoader and saves to markdown via save_markdown_files."""

    def load_documents(self, url: str, doc_id: str) -> list[Document]:
        """
        Fetches PDF from URL, parses, saves markdown via save_markdown_files,
        returns List[Document] with metadata doc_id, source, page.
        Caller must provide a valid PDF URL.
        """

        loader = OnlinePDFLoader(url)
        raw_docs = loader.load()
        if not raw_docs:
            return []

        page_dicts = [
            {"page": d.metadata.get("page", i + 1), "content": d.page_content}
            for i, d in enumerate(raw_docs)
        ]
        save_markdown_files(page_dicts, doc_id=doc_id)
        return [
            Document(
                page_content=d.get("content", ""),
                metadata={"doc_id": doc_id, "source": url, "page": d.get("page", 0)},
            )
            for d in page_dicts
        ]


class PyMuPDFDocumentLoader:
    """Loads PDFs using pymupdf4llm (distinct from DocumentLoader which uses Docling/PyPDF)."""

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
