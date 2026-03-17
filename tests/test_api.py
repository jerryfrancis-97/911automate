"""Unit tests for api/main.py.

All external calls (Qdrant, LLM, Embedder) are mocked so tests run
without any live infrastructure.

Convention:
- Endpoint tests keep ``TestClient(app)`` INSIDE the patch context so
  the ASGI lifespan fires while all mocks are still active.
- The Agent is lazy-loaded, so chat tests inject a mock via app.state.agent.
- Probe helpers are tested in isolation.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ---------------------------------------------------------------------------
# Shared mock factories
# ---------------------------------------------------------------------------


def _mock_agent_handle(
    action: str = "answer",
    response: str = "mock answer",
    confidence: float = 0.9,
    sources: list | None = None,
    session_id: str = "sess-1",
):
    """Return a dict shaped exactly like Agent.handle()."""
    return {
        "action": action,
        "response": response,
        "confidence": confidence,
        "sources": sources or [],
        "session": {
            "session_id": session_id,
            "history": [],
            "clarify_rounds": 0,
            "last_confidence": confidence,
            "last_retrieval_ids": [],
            "metadata": {},
        },
    }


def _lifespan_patches():
    """Patch only the lightweight lifespan deps (no Embedder/Agent)."""
    mock_agent = MagicMock()
    mock_agent.handle.return_value = _mock_agent_handle()
    return (
        patch("api.deps.init_telemetry"),
        patch("api.deps.QdrantClient", return_value=MagicMock()),
        patch("api.deps._build_agent", return_value=mock_agent),
    )


def _make_mock_agent(**handle_kwargs):
    mock_agent = MagicMock()
    mock_agent.handle.return_value = _mock_agent_handle(**handle_kwargs)
    return mock_agent


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


def test_metrics_endpoint():
    """GET /metrics returns 200 and Prometheus exposition format with RAG metrics."""
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "rag_requests_total" in body
    assert "rag_errors_total" in body
    assert "rag_request_latency_seconds" in body
    assert "rag_retrieval_latency_seconds" in body
    assert "rag_llm_latency_seconds" in body


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------


def test_list_sessions():
    """GET /sessions returns list of active session IDs."""
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            resp = client.get("/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert "session_ids" in body
    assert isinstance(body["session_ids"], list)

    # Create a session and list again
    with p1, p2, p3:
        with TestClient(app) as client:
            create = client.post("/session").json()
            sid = create["session_id"]
            resp = client.get("/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert sid in body["session_ids"]


# ---------------------------------------------------------------------------
# POST /session
# ---------------------------------------------------------------------------


def test_create_session():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            resp = client.post("/session")

    assert resp.status_code == 200
    body = resp.json()
    assert "session_id" in body
    assert isinstance(body["session_id"], str)
    assert len(body["session_id"]) > 0


# ---------------------------------------------------------------------------
# GET /session/{session_id}
# ---------------------------------------------------------------------------


def test_get_session_found():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            create = client.post("/session").json()
            sid = create["session_id"]
            resp = client.get(f"/session/{sid}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["session_id"] == sid
    assert body["history"] == []
    assert body["clarify_rounds"] == 0


def test_get_session_not_found():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            resp = client.get("/session/nonexistent")

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# POST /chat  — happy path (answer)
# ---------------------------------------------------------------------------


def test_chat_answer():
    mock_agent = _make_mock_agent(
        action="answer",
        response="Test answer",
        confidence=0.95,
        sources=[{
            "doc_id": "d1",
            "page": 2,
            "chunk_index": 3,
            "source_path": "/docs/test.pdf",
            "score": 0.88,
        }],
    )
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, patch("api.deps._build_agent", return_value=mock_agent):
        with TestClient(app) as client:
            create = client.post("/session").json()
            resp = client.post("/chat", json={
                "session_id": create["session_id"],
                "question": "What is CPR?",
            })

    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "answer"
    assert body["answer"] == "Test answer"
    assert body["confidence"] == 0.95
    assert body["escalation_reason"] is None
    assert len(body["used_facts"]) == 1
    fact = body["used_facts"][0]
    assert fact["doc_id"] == "d1"
    assert fact["page"] == 2
    assert fact["score"] == 0.88


# ---------------------------------------------------------------------------
# POST /chat  — clarify
# ---------------------------------------------------------------------------


def test_chat_clarify():
    mock_agent = _make_mock_agent(
        action="clarify",
        response="Can you describe the symptoms?",
        confidence=0.4,
    )
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, patch("api.deps._build_agent", return_value=mock_agent):
        with TestClient(app) as client:
            create = client.post("/session").json()
            resp = client.post("/chat", json={
                "session_id": create["session_id"],
                "question": "Help me",
            })

    body = resp.json()
    assert body["action"] == "clarify"
    assert body["escalation_reason"] is None


# ---------------------------------------------------------------------------
# POST /chat  — escalate
# ---------------------------------------------------------------------------


def test_chat_escalate():
    mock_agent = _make_mock_agent(
        action="escalate",
        response="Escalating to human operator.",
        confidence=0.1,
    )
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, patch("api.deps._build_agent", return_value=mock_agent):
        with TestClient(app) as client:
            create = client.post("/session").json()
            resp = client.post("/chat", json={
                "session_id": create["session_id"],
                "question": "I need help",
            })
    assert resp.status_code == 200
    body = resp.json()
    assert body["action"] == "escalate"
    assert body["escalation_reason"] == "Escalating to human operator."


# ---------------------------------------------------------------------------
# POST /chat  — blocked
# ---------------------------------------------------------------------------


def test_chat_blocked():
    mock_agent = _make_mock_agent(
        action="blocked",
        response="Flagged for inappropriate content.",
        confidence=0.0,
    )
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, patch("api.deps._build_agent", return_value=mock_agent):
        with TestClient(app) as client:
            create = client.post("/session").json()
            resp = client.post("/chat", json={
                "session_id": create["session_id"],
                "question": "bad input",
            })

    body = resp.json()
    assert body["action"] == "blocked"
    assert body["confidence"] == 0.0


# ---------------------------------------------------------------------------
# POST /chat  — validation (missing fields)
# ---------------------------------------------------------------------------


def test_chat_missing_question():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            app.state.agent = _make_mock_agent()
            app.state.agent_ready = True
            app.state.agent_initializing = False
            resp = client.post("/chat", json={"session_id": "x"})

    assert resp.status_code == 422


def test_chat_missing_session_id():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            app.state.agent = _make_mock_agent()
            app.state.agent_ready = True
            app.state.agent_initializing = False
            resp = client.post("/chat", json={"question": "hi"})

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /escalate
# ---------------------------------------------------------------------------


def test_escalate_marks_session():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            # /escalate is blocked when eval_mode=True (see api.main.escalate)
            if getattr(app.state, "config", None) is not None:
                app.state.config.eval_mode = False
            create = client.post("/session").json()
            sid = create["session_id"]
            resp = client.post("/escalate", json={
                "session_id": sid,
                "reason": "caller unresponsive",
            })

    assert resp.status_code == 200
    body = resp.json()
    assert body["metadata"]["escalated"] is True
    assert body["metadata"]["escalation_reason"] == "caller unresponsive"


def test_escalate_unknown_session():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            # Ensure we test the unknown-session path, not eval_mode lockout.
            if getattr(app.state, "config", None) is not None:
                app.state.config.eval_mode = False
            resp = client.post("/escalate", json={
                "session_id": "nope",
                "reason": "test",
            })

    assert resp.status_code == 404


def test_escalate_forbidden_in_eval_mode():
    """When eval_mode is enabled, manual escalation is disabled (HTTP 403)."""
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            if getattr(app.state, "config", None) is not None:
                app.state.config.eval_mode = True
            create = client.post("/session").json()
            sid = create["session_id"]
            resp = client.post("/escalate", json={"session_id": sid, "reason": "test"})
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def test_health_both_ok():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3, \
         patch("api.main._probe_qdrant", return_value=True), \
         patch("api.main._probe_llm", new=AsyncMock(return_value=True)):
        with TestClient(app) as client:
            resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["qdrant"] is True
    assert body["llm"] is True


def test_health_qdrant_down():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3, \
         patch("api.main._probe_qdrant", return_value=False), \
         patch("api.main._probe_llm", new=AsyncMock(return_value=True)):
        with TestClient(app) as client:
            resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] is False


def test_health_llm_down():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3, \
         patch("api.main._probe_qdrant", return_value=True), \
         patch("api.main._probe_llm", new=AsyncMock(return_value=False)):
        with TestClient(app) as client:
            resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["llm"] is False


def test_health_both_down():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3, \
         patch("api.main._probe_qdrant", return_value=False), \
         patch("api.main._probe_llm", new=AsyncMock(return_value=False)):
        with TestClient(app) as client:
            resp = client.get("/health")

    body = resp.json()
    assert body["status"] == "degraded"
    assert body["qdrant"] is False
    assert body["llm"] is False


# ---------------------------------------------------------------------------
# GET /ready — agent warm lifecycle
# ---------------------------------------------------------------------------


def test_ready_response_shape():
    """GET /ready returns JSON with agent_ready, agent_initializing, embedder_ready, retriever_ready, error keys."""
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            resp = client.get("/ready")

    assert resp.status_code == 200
    body = resp.json()
    assert "agent_ready" in body
    assert "agent_initializing" in body
    assert "embedder_ready" in body
    assert "retriever_ready" in body
    assert "error" in body
    assert isinstance(body["agent_ready"], bool)
    assert isinstance(body["agent_initializing"], bool)


def test_ready_initializing_then_ready():
    """With warming on, /ready eventually shows agent_ready=True."""
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3:
        with TestClient(app) as client:
            for _ in range(20):
                resp = client.get("/ready")
                body = resp.json()
                if body["agent_ready"]:
                    break
                assert body["agent_initializing"] or not body["agent_ready"]
                time.sleep(0.05)
            else:
                pytest.fail("agent_ready did not become True within 1s")
    assert body["agent_ready"] is True
    assert body["agent_initializing"] is False
    assert body["error"] is None


def test_ready_when_warming_disabled():
    """When AGENT_WARM_ON_START=false, /ready shows not initializing; /chat triggers lazy build."""
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3, patch.dict("os.environ", {"AGENT_WARM_ON_START": "false"}, clear=False):
        with TestClient(app) as client:
            resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_initializing"] is False
    assert body["agent_ready"] is False

    with p1, p2, p3, patch.dict("os.environ", {"AGENT_WARM_ON_START": "false"}, clear=False):
        with TestClient(app) as client:
            create = client.post("/session").json()
            chat_resp = client.post("/chat", json={
                "session_id": create["session_id"],
                "question": "test",
            })
            ready_resp = client.get("/ready")
    assert chat_resp.status_code == 200
    assert ready_resp.json()["agent_ready"] is True


def test_chat_503_while_warming():
    """POST /chat returns 503 while agent is warming (slow _build_agent)."""
    def slow_build(*args, **kwargs):
        time.sleep(0.3)
        return _make_mock_agent()

    p1, p2, p3 = _lifespan_patches()
    with p1, p2, patch("api.deps._build_agent", side_effect=slow_build):
            with TestClient(app) as client:
                create = client.post("/session").json()
                resp = client.post("/chat", json={
                    "session_id": create["session_id"],
                    "question": "test",
                })

    assert resp.status_code == 503
    detail = resp.json().get("detail", "").lower()
    assert "warming" in detail or "agent" in detail


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------


def test_cors_headers():
    p1, p2, p3 = _lifespan_patches()
    with p1, p2, p3, \
         patch("api.main._probe_qdrant", return_value=True), \
         patch("api.main._probe_llm", new=AsyncMock(return_value=True)):
        with TestClient(app) as client:
            resp = client.get("/health", headers={"Origin": "http://example.com"})

    assert resp.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# Unit tests: probe helpers in isolation
# ---------------------------------------------------------------------------


def test_probe_qdrant_success():
    from api.main import _probe_qdrant
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock()
    assert _probe_qdrant(mock_client) is True


def test_probe_qdrant_failure():
    from api.main import _probe_qdrant
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = Exception("timeout")
    assert _probe_qdrant(mock_client) is False


@pytest.mark.asyncio
async def test_probe_llm_ollama_ok():
    from api.main import _probe_llm
    from src.rag.core.config import Config

    config = Config(ollama_base_url="http://localhost:11434")
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("api.main.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await _probe_llm(config)

    assert result is True
    mock_http.get.assert_called_once_with("http://localhost:11434/api/tags")


@pytest.mark.asyncio
async def test_probe_llm_api_url_ok():
    from api.main import _probe_llm
    from src.rag.core.config import Config

    config = Config(api_base_url="https://api.example.com/v1")
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("api.main.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.get.return_value = mock_resp
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await _probe_llm(config)

    assert result is True
    mock_http.get.assert_called_once_with("https://api.example.com/v1/models")


@pytest.mark.asyncio
async def test_probe_llm_connection_error():
    from api.main import _probe_llm
    from src.rag.core.config import Config

    config = Config()

    with patch("api.main.httpx.AsyncClient") as mock_cls:
        mock_http = AsyncMock()
        mock_http.get.side_effect = Exception("connection refused")
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_http)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await _probe_llm(config)

    assert result is False
