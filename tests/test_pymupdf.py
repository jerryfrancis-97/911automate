"""Standalone script to test PyMuPDF markdown extraction.

Run from project root: python tests/test_pymupdf.py
"""
import sys
from pathlib import Path

# Project root for imports and data paths
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pymupdf4llm

from src.rag.ingestion import save_markdown_files

# Input: read from data/
_pdf_path = _project_root / "data" / "ALEMD_Guidebook.pdf"
# Output: save to test_data/ (not data/)
_test_output_dir = _project_root / "test_data" / "markdown"
_doc_id = "ALEMD_Guidebook"

pymupdf_chunks = pymupdf4llm.to_markdown(str(_pdf_path), page_chunks=True)
pymupdf_page_dicts = [
    {"page": c.get("metadata", {}).get("page_number", i + 1), "content": c.get("text", "")}
    for i, c in enumerate(pymupdf_chunks)
]
save_markdown_files(pymupdf_page_dicts, output_dir=_test_output_dir, doc_id=_doc_id)
print(f"Saved {len(pymupdf_page_dicts)} pages to {_test_output_dir / _doc_id}")
