"""Explicit session state management for the RAG agent.

Separates session lifecycle from agent logic. InMemorySessionStore is the
default; swap in a Redis-backed implementation of the same SessionStore
protocol for horizontal scaling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class SessionState:
    """Mutable session state carried across agent turns."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    history: list[dict[str, str]] = field(default_factory=list)
    clarify_rounds: int = 0
    last_confidence: float = 0.0
    last_retrieval_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SessionStore(Protocol):
    """Minimal store interface -- implement with Redis, SQL, etc."""

    def get(self, session_id: str) -> SessionState | None: ...

    def put(self, session: SessionState) -> None: ...

    def list_ids(self) -> list[str]: ...


class InMemorySessionStore:
    """Dict-backed session store suitable for single-process deployments."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def put(self, session: SessionState) -> None:
        self._sessions[session.session_id] = session

    def list_ids(self) -> list[str]:
        return list(self._sessions.keys())


def get_or_create_session(
    store: SessionStore,
    session_id: str | None = None,
) -> SessionState:
    """Return an existing session or create and persist a new one."""
    if session_id:
        existing = store.get(session_id)
        if existing is not None:
            return existing
    session = SessionState(session_id=session_id or uuid.uuid4().hex)
    store.put(session)
    return session
