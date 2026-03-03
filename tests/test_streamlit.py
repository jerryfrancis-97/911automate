"""Tests for streamlit timeout logic.

The /chat endpoint can take 60-180s (embed + Qdrant + LLM). We verify that
get_post_timeout returns CHAT_TIMEOUT for /chat and DEFAULT_TIMEOUT for others.
"""

from __future__ import annotations

from streamlit_helpers import (
    CHAT_TIMEOUT,
    DEFAULT_TIMEOUT,
    get_post_timeout,
)


def test_get_post_timeout_chat_path():
    """POST /chat uses CHAT_TIMEOUT (180s)."""
    assert get_post_timeout("/chat") == CHAT_TIMEOUT
    assert CHAT_TIMEOUT == 180


def test_get_post_timeout_other_paths():
    """POST to non-/chat paths uses DEFAULT_TIMEOUT (30s)."""
    assert get_post_timeout("/session") == DEFAULT_TIMEOUT
    assert get_post_timeout("/escalate") == DEFAULT_TIMEOUT
    assert DEFAULT_TIMEOUT == 30


def test_get_post_timeout_explicit_overrides_path():
    """Explicit timeout parameter overrides path-based default."""
    assert get_post_timeout("/chat", explicit=60) == 60
    assert get_post_timeout("/session", explicit=5) == 5
