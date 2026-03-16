"""Ingestion pipeline: load documents, chunk, embed, upsert to vector store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.rag.core.config import Config
from src.rag.core.types import ChunkItem
from src.rag.ingestion.chunking import create_chunks
from src.rag.ingestion.loader import PyMuPDFDocumentLoader
from src.rag.retrieval.embedder import Embedder
from src.rag.retrieval.vectordb_qdrant import VectorDBQdrant

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of running the ingestion pipeline."""

    num_chunks: int
    num_embedded: int
    doc_id: str


def _default_paths(root: Path | None = None) -> dict[str, Path]:
    """Build default paths dict from project root."""
    root = root or Path(__file__).resolve().parent.parent.parent.parent
    data = root / "data"
    return {
        "root": root,
        "data": data,
        "markdown": data / "markdown",
        "chunks": data / "chunks",
        "embeddings": data / "embeddings",
    }


class IngestionPipeline:
    """Pipeline: load PDF/markdown -> chunk -> embed -> upsert to Qdrant."""

    def __init__(
        self,
        paths: dict[str, Path] | None = None,
        config: Config | None = None,
        loader=None,
        embedder=None,
        vectordb=None,
    ) -> None:
        self._paths = paths or _default_paths()
        self._config = config or Config()
        self._loader = loader or PyMuPDFDocumentLoader()
        self._embedder = embedder or Embedder(self._config)
        self._vectordb = vectordb or VectorDBQdrant(self._config)

    def load_markdown_documents(self, doc_id: str) -> list[Document]:
        """Load documents from markdown files in paths['markdown']/doc_id/."""
        markdown_root = self._paths["markdown"]
        doc_dir = markdown_root / doc_id
        docs: list[Document] = []
        if not doc_dir.exists():
            return docs
        for md_path in sorted(doc_dir.glob("*.md")):
            content = md_path.read_text(encoding="utf-8")
            meta = {
                "doc_id": doc_id,
                "source": str(md_path.resolve()),
                "parent_doc_id": str(md_path.resolve()),
                "file_name": md_path.name,
            }
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def pdf_to_markdown_documents(
        self,
        pdf_filename: str | None,
        doc_id: str,
    ) -> list[Document]:
        """Load PDF via loader and return Documents. Saves markdown to paths['markdown']."""
        if not pdf_filename:
            return []
        root = self._paths["root"]
        pdf_path = root / "data" / pdf_filename
        if not pdf_path.exists():
            pdf_path = root / pdf_filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        markdown_dir = self._paths["markdown"]
        return self._loader.load_documents(
            path=pdf_path,
            doc_id=doc_id,
            output_dir=markdown_dir,
        )

    def documents_to_chunks(self, documents: list[Document]) -> list[Document]:
        """Chunk documents and save to paths['chunks']."""
        chunks_dir = self._paths["chunks"]
        return create_chunks(
            documents,
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
            output_dir=chunks_dir,
        )

    def chunks_to_embeddings(self, chunks: list[Document], doc_id: str) -> Path:
        """Embed chunks, write JSONL to paths['embeddings']/{doc_id}.jsonl, return path."""
        if not chunks:
            emb_dir = self._paths["embeddings"]
            emb_dir.mkdir(parents=True, exist_ok=True)
            p = emb_dir / f"{doc_id}.jsonl"
            p.write_text("", encoding="utf-8")
            return p

        texts = [c.page_content for c in chunks]
        vectors = self._embedder.embed_texts(texts)
        emb_dir = self._paths["embeddings"]
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_path = emb_dir / f"{doc_id}.jsonl"
        with emb_path.open("w", encoding="utf-8") as f:
            for idx, (chunk, vec) in enumerate(zip(chunks, vectors)):
                item = {
                    "chunk_idx": idx,
                    "embedding": vec,
                    "metadata": dict(chunk.metadata),
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return emb_path

    def embeddings_to_qdrant(self, embeddings_path: Path) -> int:
        """Read JSONL, build ChunkItems, upsert to vectordb. Returns count upserted."""
        chunk_items: list[ChunkItem] = []
        with embeddings_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                idx = item.get("chunk_idx", len(chunk_items))
                meta = dict(item.get("metadata", {}))
                chunk_id = meta.get("chunk_id")
                point_id: str | int = chunk_id if chunk_id else idx
                chunk_items.append(
                    {
                        "id": point_id,
                        "vector": item["embedding"],
                        "metadata": {
                            **meta,
                            "text": meta.get("text") or meta.get("page_content", ""),
                            "source_path": meta.get("source_path") or meta.get("source", ""),
                        },
                    }
                )
        if chunk_items:
            self._vectordb.upsert_chunks(chunk_items)
        return len(chunk_items)

    def run(
        self,
        doc_id: str,
        pdf_filename: str | None = None,
    ) -> IngestionResult:
        """Run full pipeline. Reuses markdown if present, else loads from PDF."""
        markdown_dir = self._paths["markdown"] / doc_id
        if markdown_dir.exists() and any(markdown_dir.glob("*.md")):
            docs = self.load_markdown_documents(doc_id)
        else:
            docs = self.pdf_to_markdown_documents(pdf_filename, doc_id)
        if not docs:
            logger.warning("No documents loaded for doc_id=%s", doc_id)
            return IngestionResult(num_chunks=0, num_embedded=0, doc_id=doc_id)

        chunks = self.documents_to_chunks(docs)
        emb_path = self.chunks_to_embeddings(chunks, doc_id)
        count = self.embeddings_to_qdrant(emb_path)
        return IngestionResult(
            num_chunks=len(chunks),
            num_embedded=count,
            doc_id=doc_id,
        )


def run_ingestion(
    doc_id: str,
    pdf_filename: str | None = None,
    root: Path | None = None,
) -> IngestionResult:
    """Run the ingestion pipeline with default paths. Entry point for CLI and tests."""
    paths = _default_paths(root)
    pipeline = IngestionPipeline(paths=paths)
    return pipeline.run(doc_id=doc_id, pdf_filename=pdf_filename)
