"""Pydantic request/response models for the 911automate RAG API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    question: str


class UsedFact(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    snippet: str
    score: float


class ChatResponse(BaseModel):
    action: Literal["answer", "clarify", "escalate", "blocked"]
    answer: str
    confidence: float
    used_facts: list[UsedFact]
    escalation_reason: str | None = None


class EscalateRequest(BaseModel):
    session_id: str
    reason: str


class SessionResponse(BaseModel):
    session_id: str
    history: list[dict[str, Any]]
    clarify_rounds: int
    last_confidence: float
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    qdrant: bool
    llm: bool


class ReadyResponse(BaseModel):
    agent_ready: bool
    agent_initializing: bool
    embedder_ready: bool = False
    retriever_ready: bool = False
    error: str | None = None


class SessionsListResponse(BaseModel):
    session_ids: list[str]
