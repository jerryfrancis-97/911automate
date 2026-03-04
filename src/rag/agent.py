"""RAG Agent with answer/clarify/escalate logic."""

import json
import re
import time
from typing import Any

from api.metrics import record_rag_llm_latency, record_rag_retrieval_latency
from src.rag.config import Config
from src.rag.guardrails import check_for_banned_content, should_escalate
from src.rag.llm_adapters import OllamaLLM, APILLM, get_llm
from src.rag.prompt_handler import PromptHandler
from src.rag.retriever import ProvenanceInfo, Retriever, RetrievedChunk
from src.rag.session_state import (
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
        config: Config | None = None,
        llm: OllamaLLM | APILLM | None = None,
        prompt_handler: PromptHandler | None = None,
        session_store: SessionStore | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config or Config()
        self._llm = llm or get_llm(self._config)
        self._prompt_handler = prompt_handler or PromptHandler(
            self._config.prompts_dir
        )
        self._session_store: SessionStore = (
            session_store or InMemorySessionStore()
        )

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
        if session is not None:
            sess = self._session_from_dict(session)
        else:
            sess = get_or_create_session(self._session_store, session_id)

        banned = check_for_banned_content(question)
        if banned["flagged"]:
            sess.history.append({"role": "user", "content": question})
            refusal = (
                "Your message was flagged for inappropriate content. "
                "Please rephrase your question."
            )
            sess.history.append({"role": "assistant", "content": refusal})
            sess.last_confidence = 0.0
            self._session_store.put(sess)
            return self._result("blocked", refusal, 0.0, sess, sources=[])

        start = time.perf_counter()
        chunks = self._retriever.retrieve(question)
        record_rag_retrieval_latency(time.perf_counter() - start)

        sources: list[ProvenanceInfo] = [
            c["provenance"] for c in chunks if "provenance" in c
        ]

        sess.last_retrieval_ids = [
            c["metadata"].get("chunk_id", "") for c in chunks
        ]

        confidence = _compute_confidence(chunks)

        context_block = "\n\n---\n\n".join(
            c["text"] for c in chunks if c.get("text")
        ) or "(No relevant context retrieved.)"

        threshold = self._config.confidence_threshold

        if confidence < threshold and not should_escalate(
            confidence, sess.clarify_rounds, self._config
        ):
            clarify_prompt = self._prompt_handler.get_prompt("clarify")
            messages = [
                {"role": "system", "content": clarify_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"User question: {question}\n\n"
                        "Generate 1\u20132 clarifying questions."
                    ),
                },
            ]
            response_text = self._invoke_llm(messages, action="clarify")
            action = "clarify"
            sess.clarify_rounds += 1
        elif should_escalate(
            confidence, sess.clarify_rounds, self._config
        ):
            action = "escalate"
            response_text = (
                "I don't have enough information to answer confidently. "
                "I'm escalating to a human operator for assistance."
            )
        else:
            sys_prompt = self._prompt_handler.get_prompt("agent_system")
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question: {question}\n\n"
                        "Answer based on the context above:"
                    ),
                },
            ]
            response_text = self._invoke_llm(messages, action="answer")
            llm_conf = _extract_llm_confidence(response_text)
            if llm_conf is not None:
                confidence = llm_conf
            action = "answer"

        sess.history.append({"role": "user", "content": question})
        sess.history.append({"role": "assistant", "content": response_text})
        sess.last_confidence = confidence
        self._session_store.put(sess)

        return self._result(
            action, response_text, confidence, sess, sources=sources
        )

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
        record_rag_llm_latency(time.perf_counter() - start)
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
