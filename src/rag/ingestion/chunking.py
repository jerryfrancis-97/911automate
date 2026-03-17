"""Chunk documents using nomic-embed-text tokenizer (token-based splitting)."""

import hashlib
from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag.core.tokenizer_utils import count_tokens, get_nomic_tokenizer


def deterministic_chunk_id(doc_id: str, page: int, chunk_index: int) -> str:
    """Return a deterministic, collision-resistant ID for a chunk.

    Uses SHA-256 of ``doc_id:page:chunk_index`` truncated to 16 hex chars.
    The same inputs always produce the same ID, making re-ingestion
    idempotent when used as the Qdrant point ID.
    """
    raw = f"{doc_id}:{page}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


MIN_CHUNK_TOKENS = 5


def create_chunks(
    documents: list[Document],
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    output_dir: str | Path = "data/chunks",
) -> list[Document]:
    """Split documents into token-sized chunks using nomic-embed-text tokenizer.

    chunk_size and chunk_overlap are in tokens. Saves to output_dir/{doc_id}/chunk_{i}.md.
    Chunks with fewer than MIN_CHUNK_TOKENS are skipped.
    """
    output_dir = Path(output_dir)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        get_nomic_tokenizer(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    all_chunks: list[Document] = []
    chunk_index = 0

    for doc in documents:
        if not doc.page_content.strip():
            continue

        doc_id = doc.metadata.get("doc_id", "document")
        page = doc.metadata.get("page", 0)
        source = doc.metadata.get("source", "")

        split_docs = splitter.split_documents([doc])
        doc_dir = output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        for sub_chunk in split_docs:
            if count_tokens(sub_chunk.page_content) < MIN_CHUNK_TOKENS:
                continue
            meta = {
                "doc_id": doc_id,
                "page": page,
                "chunk_index": chunk_index,
                "chunk_id": deterministic_chunk_id(doc_id, page, chunk_index),
                "source": source,
                "parent_doc_id": source,
            }
            meta.update(sub_chunk.metadata)
            chunk_doc = Document(page_content=sub_chunk.page_content, metadata=meta)
            all_chunks.append(chunk_doc)

            out_path = doc_dir / f"chunk_{chunk_index}.md"
            out_path.write_text(sub_chunk.page_content, encoding="utf-8")
            chunk_index += 1

    # Validate no chunk exceeds limit (safety check)
    tolerance = 10
    over_limit = [
        (i, count_tokens(c.page_content))
        for i, c in enumerate(all_chunks)
        if count_tokens(c.page_content) > chunk_size + tolerance
    ]
    if over_limit:
        details = ", ".join(f"chunk_{i}={n} tok" for i, n in over_limit[:5])
        if len(over_limit) > 5:
            details += f" ... and {len(over_limit) - 5} more"
        raise ValueError(
            f"Chunks exceed chunk_size={chunk_size} tokens: {details}. "
            "Reduce chunk_size in config."
        )
    return all_chunks
