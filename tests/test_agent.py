"""Unit tests for RAG Agent with mocked Retriever and LLM."""

from pathlib import Path

import pytest

from src.rag.agent import Agent, AgentSession, _extract_llm_confidence
from src.rag.config import Config
from src.rag.llm_adapters import APILLM, OllamaLLM, get_llm
from src.rag.prompt_handler import PromptHandler
from src.rag.retriever import RetrievedChunk


# --- Mocks ---


class MockRetriever:
    """Retriever that returns configurable chunks."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self._chunks


class MockLLM:
    """LLM that returns a fixed string."""

    def __init__(self, response: str = "Mock response") -> None:
        self._response = response

    def invoke(self, messages: list) -> str:
        return self._response


# --- Tests ---


def test_agent_importable() -> None:
    """Agent, PromptHandler, LLM adapters are importable."""
    from src.rag.agent import Agent, AgentSession
    from src.rag.llm_adapters import APILLM, OllamaLLM
    from src.rag.prompt_handler import PromptHandler

    assert Agent is not None
    assert AgentSession is not None
    assert PromptHandler is not None
    assert OllamaLLM is not None
    assert APILLM is not None


def test_handle_returns_action_keys() -> None:
    """handle() returns dict with action, response, confidence, session."""
    retriever = MockRetriever(
        chunks=[{"text": "Some context.", "metadata": {}, "score": 0.9}]
    )
    agent = Agent(retriever=retriever, llm=MockLLM("Answer here"))
    result = agent.handle("What is X?")
    assert "action" in result
    assert "response" in result
    assert "confidence" in result
    assert "session" in result
    assert result["action"] in ("answer", "clarify", "escalate")
    assert isinstance(result["response"], str)
    assert isinstance(result["confidence"], (int, float))
    assert isinstance(result["session"], dict)


def test_low_confidence_clarify() -> None:
    """Mock retriever returning low scores -> action='clarify', clarify_rounds incremented."""
    config = Config(confidence_threshold=0.7, max_clarify_rounds=3)
    retriever = MockRetriever(
        chunks=[
            {"text": "Vague context", "metadata": {}, "score": 0.3},
            {"text": "More vague", "metadata": {}, "score": 0.2},
        ]
    )
    agent = Agent(
        retriever=retriever,
        config=config,
        llm=MockLLM("Could you clarify: what exactly do you need?"),
    )
    result = agent.handle("Complex question", session={})
    assert result["action"] == "clarify"
    assert result["confidence"] < 0.7
    assert result["session"]["clarify_rounds"] == 1
    assert "clarify" in result["response"].lower() or "?" in result["response"]


def test_clarify_then_escalate() -> None:
    """After max_clarify_rounds with low confidence -> action='escalate'."""
    config = Config(confidence_threshold=0.8, max_clarify_rounds=2)
    retriever = MockRetriever(
        chunks=[{"text": "Weak", "metadata": {}, "score": 0.2}]
    )
    agent = Agent(
        retriever=retriever,
        config=config,
        llm=MockLLM("Clarifying question?"),
    )
    session: AgentSession = {"history": [], "clarify_rounds": 0}
    # First call: clarify
    r1 = agent.handle("Q1", session=session)
    assert r1["action"] == "clarify"
    assert r1["session"]["clarify_rounds"] == 1
    # Second call: clarify again
    r2 = agent.handle("Q2", session=r1["session"])
    assert r2["action"] == "clarify"
    assert r2["session"]["clarify_rounds"] == 2
    # Third call: escalate (max reached)
    r3 = agent.handle("Q3", session=r2["session"])
    assert r3["action"] == "escalate"
    assert "escalat" in r3["response"].lower()


def test_high_confidence_answer() -> None:
    """Mock retriever high scores -> action='answer'."""
    retriever = MockRetriever(
        chunks=[
            {"text": "Relevant answer content here.", "metadata": {}, "score": 0.92},
            {"text": "Additional context.", "metadata": {}, "score": 0.88},
        ]
    )
    agent = Agent(
        retriever=retriever,
        llm=MockLLM("Based on the context: here is the answer."),
    )
    result = agent.handle("What is the protocol?")
    assert result["action"] == "answer"
    assert result["confidence"] >= 0.7
    assert "answer" in result["response"].lower() or len(result["response"]) > 0


def test_prompt_handler_loads() -> None:
    """PromptHandler.get_prompt('agent_system') returns non-empty string."""
    project_root = Path(__file__).resolve().parent.parent
    prompts_dir = project_root / "prompts"
    handler = PromptHandler(prompts_dir)
    text = handler.get_prompt("agent_system")
    assert isinstance(text, str)
    assert len(text) > 0


def test_ollama_llm_mock() -> None:
    """Use MockLLM in tests to avoid real Ollama/API calls."""
    retriever = MockRetriever(
        chunks=[{"text": "Test", "metadata": {}, "score": 0.95}]
    )
    agent = Agent(retriever=retriever, llm=MockLLM("Test reply"))
    result = agent.handle("Test question")
    assert result["action"] == "answer"
    assert result["response"] == "Test reply"


def test_get_llm_returns_ollama_when_no_api_base() -> None:
    """get_llm returns OllamaLLM when api_base_url is not set."""
    config = Config(api_base_url=None)
    llm = get_llm(config)
    assert isinstance(llm, OllamaLLM)


def test_get_llm_returns_api_when_api_base_set() -> None:
    """get_llm returns APILLM when api_base_url is set."""
    config = Config(api_base_url="https://api.openai.com/v1")
    llm = get_llm(config)
    assert isinstance(llm, APILLM)


def test_extract_llm_confidence() -> None:
    """_extract_llm_confidence parses JSON when present."""
    assert _extract_llm_confidence('{"answer": "x", "confidence": 0.85}') == 0.85
    assert _extract_llm_confidence('Some text {"confidence": 0.5} more') == 0.5
    assert _extract_llm_confidence("Plain text answer") is None
    assert _extract_llm_confidence("") is None
