"""RAG Agent with answer/clarify/escalate logic."""

import json
import re
import time
from typing import Any

from src.rag.core.config import Config
from src.rag.agent.guardrails import GuardrailCheck, default_guardrail, should_escalate
from src.rag.agent.llm_adapters import LLM, get_llm
from src.rag.agent.prompt_handler import PromptHandler
from src.rag.retrieval.retriever import Retriever
from src.rag.retrieval.bm25_retrieval import BM25Retriever
from src.rag.retrieval.reranker import Reranker
from src.rag.core.types import MetricsRecorder, ProvenanceInfo, RetrievedChunk
from src.rag.core.session_state import (
    InMemorySessionStore,
    SessionState,
    SessionStore,
    get_or_create_session,
)


def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
    """Compute confidence from retrieval scores (mean of top chunk scores)."""
    if not chunks:
        return 0.0
    scores = [c["score"] for c in chunks]
    return sum(scores) / len(scores)


def _extract_llm_confidence(llm_response: str) -> float | None:
    """Parse ``{"answer": str, "confidence": float}`` from LLM output if present."""
    json_match = re.search(
        r'\{[^{}]*"confidence"\s*:\s*[\d.]+[^{}]*\}', llm_response
    )
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "confidence" in data:
                return float(data["confidence"])
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return None


class Agent:
    """RAG Agent that retrieves, builds prompts, calls LLM, and decides answer/clarify/escalate."""

    _QUIT_COMMANDS: frozenset[str] = frozenset({"quit", "exit", "q"})
    QUIT_RESPONSE: str = "Thank you. Goodbye!"

    def __init__(
        self,
        retriever: Retriever,
        config: Config,
        llm: LLM | None = None,
        prompt_handler: PromptHandler | None = None,
        session_store: SessionStore | None = None,
        metrics: MetricsRecorder | None = None,
        guardrail: GuardrailCheck | None = None,
        bm25_retriever: BM25Retriever | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._bm25_retriever = bm25_retriever
        self._reranker = reranker
        self._config = config
        self._llm = llm or get_llm(self._config)
        self._prompt_handler = prompt_handler or PromptHandler(
            prompts_path=self._config.prompts_file
        )
        self._session_store: SessionStore = (
            session_store or InMemorySessionStore()
        )
        self._metrics = metrics
        self._guardrail = guardrail or default_guardrail()

    def handle(
        self,
        question: str,
        session_id: str | None = None,
        *,
        session: dict | None = None,
    ) -> dict[str, Any]:
        """Process a question and return a result dict.

        Accepts *either* ``session_id`` (preferred, backed by the store) or a
        raw ``session`` dict for backward-compatibility with existing callers
        and tests.
        """
        sess = (
            self._session_from_dict(session)
            if session is not None
            else get_or_create_session(self._session_store, session_id)
        )

        # blocked = self._check_guardrails(question, sess)
        # if blocked is not None:
        #     return blocked

        chunks, sources = self._retrieve(question, sess)
        context_block = self._build_context(chunks)
        return self._decide_and_respond(
            question, sess, chunks, sources, context_block
        )

    def _check_guardrails(
        self, question: str, sess: SessionState
    ) -> dict[str, Any] | None:
        """If banned content, return result dict; else None."""
        result = self._guardrail.check(question)
        if not result.flagged:
            return None
        sess.history.append({"role": "user", "content": question})
        refusal = (
            "Your message was flagged for inappropriate content. "
            "Please rephrase your question."
        )
        sess.history.append({"role": "assistant", "content": refusal})
        sess.last_confidence = 0.0
        self._session_store.put(sess)
        return self._result("blocked", refusal, 0.0, sess, sources=[])

    def _retrieve(
        self, question: str, sess: SessionState
    ) -> tuple[list[RetrievedChunk], list[ProvenanceInfo]]:
        """Retrieve chunks, update sess, return (chunks, sources)."""
        start = time.perf_counter()
        chunks = self._retriever.retrieve(question)
        if self._bm25_retriever:
            chunks = chunks + self._bm25_retriever.retrieve(question)
        if self._reranker:
            chunks = self._reranker.rerank(question, chunks)
        if self._metrics is not None:
            self._metrics.record_retrieval_latency(time.perf_counter() - start)
        sources = [c["provenance"] for c in chunks if "provenance" in c]
        sess.last_retrieval_ids = [
            c["metadata"].get("chunk_id", "") for c in chunks
        ]
        return chunks, sources

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context block from chunks. Includes doc_id and page from provenance when available."""
        parts: list[str] = []
        for c in chunks:
            text = c.get("text") or ""
            if not text.strip():
                continue
            prov = c.get("provenance") or {}
            doc_id = prov.get("doc_id")
            page = prov.get("page")
            if doc_id is not None or page is not None:
                header = f"[doc_id: {doc_id or ''}, page: {page if page is not None else ''}]"
                parts.append(f"{header}\n\n{text}")
            else:
                parts.append(text)
        return "\n\n---\n\n".join(parts) if parts else "(No relevant context retrieved.)"

    def _build_messages_with_history(
        self,
        system_content: str,
        user_content: str,
        sess: SessionState,
    ) -> list[dict[str, str]]:
        """Build messages with optional history window between system and user."""
        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
        max_turns = getattr(self._config, "max_history_turns", 4)
        if sess.history and max_turns > 0:
            window = sess.history[-max_turns:]
            for turn in window:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _decide_and_respond(
        self,
        question: str,
        sess: SessionState,
        chunks: list[RetrievedChunk],
        sources: list[ProvenanceInfo],
        context_block: str,
    ) -> dict[str, Any]:
        """Route to answer/clarify/escalate and return result."""
        confidence = _compute_confidence(chunks)
        threshold = self._config.confidence_threshold
        eval_mode = self._config.eval_mode
        would_escalate = should_escalate(
            confidence, sess.clarify_rounds, self._config
        )

        if would_escalate and not eval_mode:
            action = "escalate"
            response_text = (
                "I don't have enough information to answer confidently. "
                "I'm escalating to a human operator for assistance."
            )
        # Clarify prompt handling (commented out for now)
        # elif confidence < threshold and not would_escalate and not (
        #     eval_mode and sess.clarify_rounds >= self._config.max_clarify_rounds
        # ):
        #     clarify_prompt = self._prompt_handler.get_prompt("clarify")
        #     user_content = (
        #         f"Context:\n{context_block}\n\n"
        #         f"User question: {question}\n\n"
        #         "Generate 1–2 clarifying questions."
        #     )
        #     messages = self._build_messages_with_history(
        #         clarify_prompt, user_content, sess
        #     )
        #     response_text = self._invoke_llm(messages, action="clarify")
        #     action = "clarify"
        #     sess.clarify_rounds += 1
        else:
            sys_prompt = self._prompt_handler.get_prompt("agent_system")
            user_content = (
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the context above:"
            )
            messages = self._build_messages_with_history(
                sys_prompt, user_content, sess
            )
            response_text = self._invoke_llm(messages, action="answer")
            llm_conf = _extract_llm_confidence(response_text)
            if llm_conf is not None:
                confidence = llm_conf
            action = "answer"

        sess.history.append({"role": "user", "content": question})
        sess.history.append({"role": "assistant", "content": response_text})
        sess.last_confidence = confidence
        self._session_store.put(sess)
        return self._result(action, response_text, confidence, sess, sources=sources)

    def run_agent_loop(self) -> None:
        """Interactive CLI loop. Input 'quit' to exit without calling LLM."""
        session_id: str | None = None
        while True:
            query = input("You: ").strip()
            if query.lower() in self._QUIT_COMMANDS:
                print(self.QUIT_RESPONSE)
                break
            if not query:
                continue
            result = self.handle(query, session_id=session_id)
            print(f"Agent: {result['response']}")
            session_id = result["session"]["session_id"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _invoke_llm(self, messages: list, *, action: str = "") -> str:
        """Call LLM."""
        start = time.perf_counter()
        result = self._llm.invoke(messages)
        if self._metrics is not None:
            self._metrics.record_llm_latency(time.perf_counter() - start)
        return result

    @staticmethod
    def _session_from_dict(raw: dict) -> SessionState:
        """Build a SessionState from a legacy raw dict."""
        return SessionState(
            session_id=raw.get("session_id", SessionState().session_id),
            history=raw.get("history") or [],
            clarify_rounds=raw.get("clarify_rounds") or 0,
            last_confidence=raw.get("last_confidence") or 0.0,
            last_retrieval_ids=raw.get("last_retrieval_ids") or [],
            metadata=raw.get("metadata") or {},
        )

    @staticmethod
    def _result(
        action: str,
        response: str,
        confidence: float,
        sess: SessionState,
        *,
        sources: list[ProvenanceInfo] | None = None,
    ) -> dict[str, Any]:
        """Build the standard result dict returned by handle()."""
        return {
            "action": action,
            "response": response,
            "confidence": confidence,
            "sources": sources or [],
            "session": {
                "session_id": sess.session_id,
                "history": sess.history,
                "clarify_rounds": sess.clarify_rounds,
                "last_confidence": sess.last_confidence,
                "last_retrieval_ids": sess.last_retrieval_ids,
                "metadata": sess.metadata,
            },
        }
