"""Unit tests for deterministic_chunk_id and idempotent ingestion."""

import pytest

from src.rag.ingestion.chunking import deterministic_chunk_id


# ---------------------------------------------------------------------------
# deterministic_chunk_id
# ---------------------------------------------------------------------------


def test_same_inputs_produce_same_id():
    """Identical (doc_id, page, chunk_index) always yields the same hash."""
    id1 = deterministic_chunk_id("doc1", 1, 0)
    id2 = deterministic_chunk_id("doc1", 1, 0)
    assert id1 == id2


def test_different_doc_id_produces_different_id():
    a = deterministic_chunk_id("doc1", 1, 0)
    b = deterministic_chunk_id("doc2", 1, 0)
    assert a != b


def test_different_page_produces_different_id():
    a = deterministic_chunk_id("doc1", 1, 0)
    b = deterministic_chunk_id("doc1", 2, 0)
    assert a != b


def test_different_chunk_index_produces_different_id():
    a = deterministic_chunk_id("doc1", 1, 0)
    b = deterministic_chunk_id("doc1", 1, 1)
    assert a != b


def test_id_is_hex_string_of_length_16():
    """Output is a 16-char hex string."""
    cid = deterministic_chunk_id("doc1", 3, 7)
    assert len(cid) == 16
    assert all(c in "0123456789abcdef" for c in cid)


def test_many_ids_are_collision_free():
    """1000 distinct inputs produce 1000 distinct IDs."""
    ids = set()
    for doc_idx in range(10):
        for page in range(10):
            for chunk in range(10):
                ids.add(deterministic_chunk_id(f"doc{doc_idx}", page, chunk))
    assert len(ids) == 1000
