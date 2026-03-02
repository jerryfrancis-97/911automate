"""Chunk documents using MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter."""

from pathlib import Path

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter


def create_chunks(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    output_dir: str | Path = "data/chunks",
) -> list[Document]:
    """Split documents into chunks using MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter.

    Saves chunks to output_dir/{doc_id}/chunk_{chunk_index}.md. Returns chunk Documents.
    """
    output_dir = Path(output_dir)
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    recursive_splitter = RecursiveCharacterTextSplitter(
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

        header_chunks = header_splitter.split_text(doc.page_content)
        size_chunks = recursive_splitter.split_documents(header_chunks)

        doc_dir = output_dir / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)

        for sub_chunk in size_chunks:
            meta = {
                "doc_id": doc_id,
                "page": page,
                "chunk_index": chunk_index,
                "source": source,
                "parent_doc_id": source,  # path to parent file (or URL)
            }
            meta.update(sub_chunk.metadata)
            chunk_doc = Document(page_content=sub_chunk.page_content, metadata=meta)
            all_chunks.append(chunk_doc)

            out_path = doc_dir / f"chunk_{chunk_index}.md"
            out_path.write_text(sub_chunk.page_content, encoding="utf-8")

            chunk_index += 1

    return all_chunks
