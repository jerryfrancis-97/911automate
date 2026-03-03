import sys
from pathlib import Path

# data/ is at project root (sibling of src/), so go up 2 levels from src/rag/
_project_root = Path(__file__).resolve().parent.parent.parent
print(_project_root)
sys.path.insert(0, str(_project_root))

from chunking import create_chunks
from ingestion import PyMuPDFDocumentLoader

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

import glob

def load_markdown_documents(markdown_dir: Path, doc_id: str) -> list:
    docs = []
    md_files = sorted((markdown_dir / doc_id).glob("*.md"))
    # print(md_files)
    # raise RuntimeError("Stop here")
    for md_path in md_files:
        page_content = md_path.read_text(encoding="utf-8")
        metadata = {
            "doc_id": doc_id,
            "source": str(md_path.resolve()),
            "parent_doc_id": str(md_path.resolve()),
            "file_name": md_path.name,
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

# Settings
# doc_id = "alemd_guidebook"
doc_id = "emergency_childbirth"
markdown_dir = _project_root / "data" / "markdown"
chunk_output_dir = _project_root / "data" / "chunks"

# Logic: If Markdown already exists, use it, else call ingestion
# if any(markdown_dir.glob("*.md")):
#     print(f"Found markdown files in {markdown_dir}, loading them as documents.")
#     docs = load_markdown_documents(markdown_dir, doc_id=doc_id)
# else:
#     print(f"No markdown files found in {markdown_dir}, calling PDF ingestion.")
#     # raise RuntimeError("Srop")
#     path = _project_root / "data" / "emergency_childbirth.pdf"
#     doc_loader = PyMuPDFDocumentLoader()
#     docs = doc_loader.load_documents(
#         path=path,
#         doc_id=doc_id,
#         output_dir=markdown_dir
#     )

# chunks = create_chunks(
#     docs,
#     chunk_size=1000,
#     chunk_overlap=200,
#     output_dir=chunk_output_dir
# )

# print(f"Chunked {len(docs)} documents into {len(chunks)} chunks.")

# from embedder import Embedder
# from pathlib import Path
# import json

# # Prepare output directory for embeddings
# embeddings_dir = _project_root / "data" / "embeddings"
# embeddings_dir.mkdir(parents=True, exist_ok=True)

# embedder = Embedder()

# # Extract texts from chunks for embedding; assume each chunk has a "page_content" key
# chunk_texts = [chunk.page_content for chunk in chunks]
# embeddings = embedder.embed_texts(chunk_texts)

# # Save embeddings as a JSONL file; each line: {"chunk_idx": int, "embedding": [...], "metadata": {...}}
# embeddings_path = embeddings_dir / f"{doc_id}_embeddings.jsonl"
# with open(embeddings_path, "w", encoding="utf-8") as f:
#     for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#         item = {
#             "chunk_idx": idx,
#             "embedding": embedding,
#             "metadata": chunk.metadata,
#         }
#         f.write(json.dumps(item) + "\n")

# print(f"Saved embeddings for {len(embeddings)} chunks to {embeddings_path}")


# Push the embeddings into the Qdrant vector store

from config import Config
from vectordb_qdrant import VectorDBQdrant, ChunkItem
import json
from pathlib import Path

# # Load chunks from markdown files in data/chunks/emergency_childbirth/
# chunk_dir = Path(__file__).parent.parent.parent / "data" / "chunks" / "emergency_childbirth"
# chunks = []
# for chunk_file in sorted(chunk_dir.glob("chunk_*.md")):
#     with open(chunk_file, "r", encoding="utf-8") as f:
#         content = f.read()
#     # Each chunk can be a simple object with minimal metadata for the example
#     chunks.append(type("Chunk", (), {
#         "page_content": content,
#         "metadata": {"id": chunk_file.stem, "path": str(chunk_file)}
#     })())

# # Load embeddings from JSONL
# embeddings_path = Path(__file__).parent.parent.parent / "data" / "embeddings" / "emergency_childbirth_embeddings.jsonl"
# embeddings = []
# with open(embeddings_path, "r", encoding="utf-8") as f:
#     for line in f:
#         item = json.loads(line)
#         embeddings.append(item["embedding"])

# doc_id = "emergency_childbirth"

# # Prepare chunk items for upsert
# # Qdrant requires point IDs to be integers or valid UUIDs; use integer idx
# chunk_items: list[ChunkItem] = []
# for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
#     meta = dict(chunk.metadata)
#     meta["chunk_id"] = meta.get("id", f"{doc_id}_chunk_{idx}")  # keep human-readable id in payload
#     chunk_items.append({
#         "id": idx,
#         "vector": embedding,
#         "metadata": meta,
#     })

# # Initialize Qdrant vector DB with default or custom config
# vectordb = VectorDBQdrant(config=Config())

# # Upsert (push) all chunks
# vectordb.upsert_chunks(chunk_items)
# print(f"Pushed {len(chunk_items)} chunks into Qdrant vector store.")

# # Do an example search using the first embedding
# if embeddings:
#     query_vector = embeddings[0]
#     results = vectordb.search_vector(query_vector, top_k=3)
#     print("Example search results (top 3):")
#     for idx, (payload, score) in enumerate(results):
#         print(f"{idx+1}. Score: {score:.4f}, Payload: {payload}")
# else:
#     print("No embeddings to search with.")

# # Example retrieval using Retriever
# from embedder import Embedder
# from retriever import Retriever

# # Initialize embedder and retriever
# embedder = Embedder(config=Config(), normalize=True)
# retriever = Retriever(embedder=embedder, vectordb=vectordb)

# # Example query
# query = "How do you recognize signs of imminent childbirth?"
# retrieved_chunks = retriever.retrieve(query, top_k=3)
# print("--- Retrieval using Retriever ---")
# for idx, chunk in enumerate(retrieved_chunks):
#     print(f"{idx+1}. Score: {chunk['score']:.4f}")
#     print(f"Text: {chunk['text'][:100]} ...")
#     print(f"Metadata: {chunk['metadata']}")
#     print("---")


#     # INSERT_YOUR_CODE
# from src.rag.agent import Agent

# # Create an Agent instance with our Retriever
# agent = Agent(retriever=retriever)

# # Try asking a question to the agent
# result = agent.handle(query)
# print("--- Agent result ---")

# def pretty_preview(obj, maxlen=180):
#     """Show a preview of each key/value for a (possibly nested) dict result, truncating if needed."""
#     if isinstance(obj, dict):
#         preview = {}
#         for k, v in obj.items():
#             if isinstance(v, (str, bytes)):
#                 preview[k] = (v[:maxlen] + "..." if len(v) > maxlen else v)
#             elif isinstance(v, list):
#                 # show preview for first item
#                 if v:
#                     if isinstance(v[0], dict):
#                         preview[k] = [pretty_preview(v[0], maxlen)]
#                     else:
#                         preview[k] = [str(v[0])[:maxlen] + ("..." if len(str(v[0])) > maxlen else "")]
#                 else:
#                     preview[k] = []
#             elif isinstance(v, dict):
#                 preview[k] = pretty_preview(v, maxlen)
#             else:
#                 preview[k] = str(v)
#         return preview
#     else:
#         return str(obj)[:maxlen] + ("..." if len(str(obj)) > maxlen else "")

# import pprint
# pprint.pprint(pretty_preview(result), width=110)

if __name__ == "__main__":
    agent.run_agent_loop()