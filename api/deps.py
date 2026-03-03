"""Shared application state built once at startup.

Lightweight state (Config, SessionStore, QdrantClient) is created during
the lifespan so the server starts immediately.  The heavy Agent (Embedder,
VectorDB, Retriever) is built lazily on the first /chat request to avoid
blocking startup with large model loads.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from qdrant_client import QdrantClient

from api.async_utils import run_in_thread
from src.rag.config import Config
from src.rag.observability import init_telemetry
from src.rag.session_state import SessionStore

logger = logging.getLogger(__name__)
_agent_lock = threading.Lock()


def _build_config() -> Config:
    return Config(
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_base_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        api_base_url=os.environ.get("API_BASE_URL") or None,
        api_key=os.environ.get("API_KEY") or None,
        agent_warm_on_start=os.environ.get("AGENT_WARM_ON_START", "true").lower()
        in ("true", "1", "yes"),
    )


def _build_agent(config: Config, session_store: SessionStore):
    """Heavy initialisation: imports torch, loads embedding model, connects Qdrant."""
    from src.rag.agent import Agent
    from src.rag.embedder import Embedder
    from src.rag.retriever import Retriever
    from src.rag.vectordb_qdrant import VectorDBQdrant

    embedder = Embedder(config)
    vectordb = VectorDBQdrant(config)
    retriever = Retriever(embedder, vectordb, config)
    return Agent(retriever=retriever, config=config, session_store=session_store)


async def _init_agent_background(app: FastAPI) -> None:
    """Build Agent in threadpool; update app.state on completion."""
    try:
        agent = await run_in_thread(
            _build_agent,
            app.state.config,
            app.state.session_store,
        )
        app.state.agent = agent
        app.state.agent_ready = True
    except Exception as e:
        logger.exception("Agent warm failed")
        app.state.agent_error = str(e)
        app.state.agent_ready = False
    finally:
        app.state.agent_initializing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_telemetry()

    config = _build_config()
    from src.rag.session_state import InMemorySessionStore

    session_store = InMemorySessionStore()
    app.state.config = config
    app.state.qdrant_client = QdrantClient(url=config.qdrant_url)
    app.state.session_store = session_store
    app.state.agent = None
    app.state.agent_ready = False
    app.state.agent_initializing = False
    app.state.agent_error = None

    if config.agent_warm_on_start:
        app.state.agent_initializing = True
        asyncio.create_task(_init_agent_background(app))

    yield


def get_agent(request: Request):
    """Return the Agent, building it lazily on first call when warming disabled (thread-safe)."""
    app = request.app
    agent_ready = getattr(app.state, "agent_ready", False)
    agent_initializing = getattr(app.state, "agent_initializing", False)
    agent_error = getattr(app.state, "agent_error", None)

    if agent_ready and app.state.agent is not None:
        return app.state.agent
    if agent_initializing:
        raise HTTPException(status_code=503, detail="Agent warming")
    if agent_error:
        raise HTTPException(status_code=503, detail=f"Agent not ready: {agent_error}")

    # Warming disabled: fall back to lazy build
    with _agent_lock:
        if app.state.agent is not None:
            return app.state.agent
        try:
            logger.info("Initialising Agent (first request — loading models)...")
            app.state.agent = _build_agent(
                app.state.config,
                app.state.session_store,
            )
            app.state.agent_ready = True
            logger.info("Agent ready.")
        except Exception as exc:
            logger.error("Agent initialisation failed: %s", exc)
            raise HTTPException(status_code=503, detail=f"Agent not ready: {exc}")
    return app.state.agent


def get_session_store(request: Request) -> SessionStore:
    return request.app.state.session_store
