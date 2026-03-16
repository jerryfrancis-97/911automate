"""FastAPI application for the 911automate RAG service.

Launch:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
    QDRANT_URL   Qdrant server URL      (default: http://localhost:6333)
    OLLAMA_URL   Ollama server URL      (default: http://localhost:11434)
    API_BASE_URL OpenAI-compatible URL  (if set, used instead of Ollama)
    API_KEY      API key for above endpoint
"""

from __future__ import annotations

import time
import uuid

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from src.rag.agent.agent import Agent
from src.rag.core.config import Config
from src.rag.observability.rag_logger import log_rag_request
from src.rag.core.session_state import SessionStore, get_or_create_session

from api.async_utils import run_in_thread
from api.deps import get_agent, get_session_store, lifespan
from api.metrics import (
    get_metrics_content,
    increment_rag_errors,
    increment_rag_requests,
    record_rag_request_latency,
)
from api.models import (
    ChatRequest,
    ChatResponse,
    EscalateRequest,
    HealthResponse,
    ReadyResponse,
    SessionResponse,
    SessionsListResponse,
    UsedFact,
)

app = FastAPI(title="911automate RAG API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health probes
# ---------------------------------------------------------------------------


def _probe_qdrant(client: QdrantClient) -> bool:
    """Return True if Qdrant responds to a list-collections call."""
    try:
        client.get_collections()
        return True
    except Exception:
        return False


async def _probe_llm(config: Config) -> bool:
    """Return True if the configured LLM endpoint responds within 2 s."""
    if config.api_base_url:
        url = config.api_base_url.rstrip("/") + "/models"
    else:
        url = config.ollama_base_url.rstrip("/") + "/api/tags"

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            return resp.status_code < 400
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_to_response(sess) -> SessionResponse:
    return SessionResponse(
        session_id=sess.session_id,
        history=sess.history,
        clarify_rounds=sess.clarify_rounds,
        last_confidence=sess.last_confidence,
        metadata=sess.metadata,
    )


def _sources_to_facts(sources: list[dict]) -> list[UsedFact]:
    return [
        UsedFact(
            chunk_id=f"{s.get('doc_id', '')}:{s.get('chunk_index', 0)}",
            doc_id=s.get("doc_id", ""),
            page=s.get("page", 0),
            snippet=s.get("source_path", ""),
            score=s.get("score", 0.0),
        )
        for s in sources
    ]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/sessions", response_model=SessionsListResponse)
def list_sessions(
    store: SessionStore = Depends(get_session_store),
) -> SessionsListResponse:
    """List all active session IDs."""
    ids = store.list_ids()
    return SessionsListResponse(session_ids=ids)


@app.post("/session")
def create_session(
    store: SessionStore = Depends(get_session_store),
) -> dict[str, str]:
    sess = get_or_create_session(store)
    return {"session_id": sess.session_id}


@app.get("/session/{session_id}", response_model=SessionResponse)
def get_session(
    session_id: str,
    store: SessionStore = Depends(get_session_store),
) -> SessionResponse:
    sess = store.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_response(sess)


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus scrape endpoint. Returns metrics in exposition format."""
    body, content_type = get_metrics_content()
    return Response(content=body, media_type=content_type)


def _sources_to_log_dicts(sources: list[dict]) -> list[dict]:
    """Convert sources to dicts for JSONL logging."""
    return [
        {
            "chunk_id": f"{s.get('doc_id', '')}:{s.get('chunk_index', 0)}",
            "doc_id": s.get("doc_id", ""),
            "page": s.get("page", 0),
            "snippet": s.get("source_path", ""),
            "score": s.get("score", 0.0),
        }
        for s in sources
    ]


@app.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    agent: Agent = Depends(get_agent),
) -> ChatResponse:
    increment_rag_requests()
    request_id = str(uuid.uuid4())
    start = time.perf_counter()
    try:
        result = await run_in_thread(
            agent.handle,
            body.question,
            session_id=body.session_id,
        )
    except Exception:
        increment_rag_errors()
        raise
    finally:
        record_rag_request_latency(time.perf_counter() - start)

    escalation_reason: str | None = None
    if result["action"] == "escalate":
        escalation_reason = result["response"]

    # Structured logging for RAG evaluation (non-blocking, never fails request)
    sources = result.get("sources", [])
    log_rag_request(
        question=body.question,
        retrieved_chunks=_sources_to_log_dicts(sources),
        retrieval_scores=[s.get("score", 0.0) for s in sources],
        final_answer=result["response"],
        used_facts=_sources_to_log_dicts(sources),
        confidence=result["confidence"],
        escalated=result["action"] == "escalate",
        latency_ms=int((time.perf_counter() - start) * 1000),
        request_id=request_id,
    )

    return ChatResponse(
        action=result["action"],
        answer=result["response"],
        confidence=result["confidence"],
        used_facts=_sources_to_facts(sources),
        escalation_reason=escalation_reason,
    )


@app.post("/escalate", response_model=SessionResponse)
def escalate(
    body: EscalateRequest,
    request: Request,
    store: SessionStore = Depends(get_session_store),
) -> SessionResponse:
    config = getattr(request.app.state, "config", None)
    if config and getattr(config, "eval_mode", False):
        raise HTTPException(
            status_code=403,
            detail="Manual escalation disabled in eval mode",
        )
    sess = store.get(body.session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="Session not found")

    sess.metadata["escalated"] = True
    sess.metadata["escalation_reason"] = body.reason
    store.put(sess)
    return _session_to_response(sess)


@app.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    """Agent warm status. Use for readiness probes. Embedder and retriever are ready when agent is ready."""
    agent_ready = getattr(request.app.state, "agent_ready", False)
    config = getattr(request.app.state, "config", None)
    eval_mode = getattr(config, "eval_mode", False) if config else False
    return ReadyResponse(
        agent_ready=agent_ready,
        agent_initializing=getattr(request.app.state, "agent_initializing", False),
        embedder_ready=agent_ready,
        retriever_ready=agent_ready,
        error=getattr(request.app.state, "agent_error", None),
        eval_mode=eval_mode,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness + readiness probe.  Always returns HTTP 200."""
    qdrant_ok = await run_in_thread(
        _probe_qdrant,
        app.state.qdrant_client,
    )
    llm_ok = await _probe_llm(app.state.config)
    return HealthResponse(
        status="ok" if qdrant_ok and llm_ok else "degraded",
        qdrant=qdrant_ok,
        llm=llm_ok,
    )
