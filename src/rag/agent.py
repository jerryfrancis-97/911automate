"""RAG Agent with answer/clarify/escalate logic."""

import json
import re
from typing import Any, TypedDict

from src.rag.config import Config
from src.rag.llm_adapters import OllamaLLM, APILLM, get_llm
from src.rag.prompt_handler import PromptHandler
from src.rag.retriever import Retriever, RetrievedChunk


class AgentSession(TypedDict, total=False):
    """Session state for the agent."""

    history: list[dict[str, str]]
    clarify_rounds: int
    last_confidence: float


def _compute_confidence(chunks: list[RetrievedChunk]) -> float:
    """Compute confidence from retrieval scores (mean of top chunk scores)."""
    if not chunks:
        return 0.0
    scores = [c["score"] for c in chunks]
    return sum(scores) / len(scores)


def _extract_llm_confidence(llm_response: str) -> float | None:
    """
    Parse JSON from LLM response if present: {"answer": str, "confidence": float}.
    Returns confidence when present, else None.
    """
    # Try to find JSON block in response
    json_match = re.search(r'\{[^{}]*"confidence"\s*:\s*[\d.]+[^{}]*\}', llm_response)
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

    def __init__(
        self,
        retriever: Retriever,
        config: Config | None = None,
        llm: OllamaLLM | APILLM | None = None,
        prompt_handler: PromptHandler | None = None,
    ) -> None:
        self._retriever = retriever
        self._config = config or Config()
        self._llm = llm or get_llm(self._config)
        self._prompt_handler = prompt_handler or PromptHandler(
            self._config.prompts_dir
        )

    def handle(
        self, question: str, session: dict | None = None
    ) -> dict[str, Any]:
        """
        Process a question. Returns {
            "action": "answer" | "clarify" | "escalate",
            "response": str,
            "confidence": float,
            "session": dict (updated),
        }
        """
        sess: AgentSession = dict(session) if session else {}
        history: list[dict[str, str]] = sess.get("history") or []
        clarify_rounds: int = sess.get("clarify_rounds") or 0

        # 1. Retrieve chunks
        chunks = self._retriever.retrieve(question)

        # 2. Compute confidence (retrieval-based; LLM override when structured output used)
        retrieval_confidence = _compute_confidence(chunks)
        confidence = retrieval_confidence

        context_block = "\n\n---\n\n".join(
            c["text"] for c in chunks if c.get("text")
        ) or "(No relevant context retrieved.)"

        threshold = self._config.confidence_threshold
        max_clarify = self._config.max_clarify_rounds

        # 3–6. Decide action
        if confidence < threshold and clarify_rounds < max_clarify:
            # Clarify: use clarify prompt to generate clarifying question(s)
            clarify_prompt = self._prompt_handler.get_prompt("clarify")
            messages = [
                {"role": "system", "content": clarify_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nUser question: {question}\n\nGenerate 1–2 clarifying questions.",
                },
            ]
            response_text = self._llm.invoke(messages)
            action = "clarify"
            clarify_rounds += 1
        elif confidence < threshold and clarify_rounds >= max_clarify:
            # Escalate
            action = "escalate"
            response_text = (
                "I don't have enough information to answer confidently. "
                "I'm escalating to a human operator for assistance."
            )
        else:
            # Answer: use agent_system prompt
            system_prompt = self._prompt_handler.get_prompt("agent_system")
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer based on the context above:",
                },
            ]
            response_text = self._llm.invoke(messages)
            llm_conf = _extract_llm_confidence(response_text)
            if llm_conf is not None:
                confidence = llm_conf
            action = "answer"

        # 7. Update session
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": response_text})
        updated_session: AgentSession = {
            "history": history,
            "clarify_rounds": clarify_rounds,
            "last_confidence": confidence,
        }

        return {
            "action": action,
            "response": response_text,
            "confidence": confidence,
            "session": updated_session,
        }
