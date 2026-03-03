"""Unit tests for src/rag/session_state."""

import pytest

from src.rag.session_state import (
    InMemorySessionStore,
    SessionState,
    SessionStore,
    get_or_create_session,
)


# ---------------------------------------------------------------------------
# SessionState defaults
# ---------------------------------------------------------------------------


def test_session_state_defaults():
    """A fresh SessionState has sensible zero-value defaults."""
    s = SessionState()
    assert isinstance(s.session_id, str) and len(s.session_id) > 0
    assert s.history == []
    assert s.clarify_rounds == 0
    assert s.last_confidence == 0.0
    assert s.last_retrieval_ids == []
    assert s.metadata == {}


def test_session_state_custom_id():
    """session_id can be set explicitly."""
    s = SessionState(session_id="abc-123")
    assert s.session_id == "abc-123"


def test_session_state_is_mutable():
    """SessionState fields can be updated in-place."""
    s = SessionState()
    s.clarify_rounds = 3
    s.last_confidence = 0.85
    s.last_retrieval_ids = ["id_1", "id_2"]
    s.metadata["key"] = "value"
    assert s.clarify_rounds == 3
    assert s.last_confidence == 0.85
    assert s.last_retrieval_ids == ["id_1", "id_2"]
    assert s.metadata == {"key": "value"}


def test_two_sessions_have_unique_ids():
    """Two default SessionState instances get distinct session_ids."""
    a = SessionState()
    b = SessionState()
    assert a.session_id != b.session_id


# ---------------------------------------------------------------------------
# InMemorySessionStore
# ---------------------------------------------------------------------------


def test_store_put_and_get():
    """put() stores a session that get() can retrieve by ID."""
    store = InMemorySessionStore()
    s = SessionState(session_id="s1")
    store.put(s)
    assert store.get("s1") is s


def test_store_get_missing_returns_none():
    """get() returns None for unknown session_id."""
    store = InMemorySessionStore()
    assert store.get("nonexistent") is None


def test_store_overwrites_on_put():
    """put() with same session_id replaces the previous entry."""
    store = InMemorySessionStore()
    s1 = SessionState(session_id="s1", clarify_rounds=1)
    store.put(s1)
    s1_updated = SessionState(session_id="s1", clarify_rounds=5)
    store.put(s1_updated)
    assert store.get("s1").clarify_rounds == 5


def test_store_implements_protocol():
    """InMemorySessionStore satisfies the SessionStore protocol."""
    assert isinstance(InMemorySessionStore(), SessionStore)


# ---------------------------------------------------------------------------
# get_or_create_session
# ---------------------------------------------------------------------------


def test_get_or_create_new_session():
    """When session_id is None a brand-new session is created and stored."""
    store = InMemorySessionStore()
    s = get_or_create_session(store)
    assert store.get(s.session_id) is s


def test_get_or_create_returns_existing():
    """When session_id matches an existing session, return it."""
    store = InMemorySessionStore()
    original = SessionState(session_id="existing", clarify_rounds=2)
    store.put(original)
    fetched = get_or_create_session(store, session_id="existing")
    assert fetched is original
    assert fetched.clarify_rounds == 2


def test_get_or_create_with_unknown_id_creates_new():
    """When session_id is provided but not in store, create with that ID."""
    store = InMemorySessionStore()
    s = get_or_create_session(store, session_id="brand-new")
    assert s.session_id == "brand-new"
    assert store.get("brand-new") is s
