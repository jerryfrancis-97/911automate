"""Unit tests for RAG Agent with mocked Retriever and LLM."""

from pathlib import Path

import pytest

from src.rag.agent.agent import Agent, _extract_llm_confidence
from src.rag.core.config import Config
from src.rag.agent.llm_adapters import APILLM, OllamaLLM, get_llm
from src.rag.agent.prompt_handler import PromptHandler
from src.rag.core.types import ProvenanceInfo, RetrievedChunk


# --- Mocks ---


class MockRetriever:
    """Retriever that returns configurable chunks."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        return self._chunks


class MockLLM:
    """LLM that returns a fixed string and tracks invocation count."""

    def __init__(self, response: str = "Mock response") -> None:
        self._response = response
        self.call_count = 0

    def invoke(self, messages: list) -> str:
        self.call_count += 1
        return self._response


# --- Tests ---


def test_agent_importable() -> None:
    """Agent, PromptHandler, LLM adapters are importable."""
    from src.rag.agent.agent import Agent
    from src.rag.agent.llm_adapters import APILLM, OllamaLLM
    from src.rag.agent.prompt_handler import PromptHandler
    from src.rag.core.session_state import SessionState

    assert Agent is not None
    assert SessionState is not None
    assert PromptHandler is not None
    assert OllamaLLM is not None
    assert APILLM is not None


def test_handle_returns_action_keys() -> None:
    """handle() returns dict with action, response, confidence, session."""
    retriever = MockRetriever(
        chunks=[{"text": "Some context.", "metadata": {}, "score": 0.9}]
    )
    agent = Agent(retriever=retriever, config=Config(), llm=MockLLM("Answer here"))
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
    """With clarify disabled: low scores -> action='answer' (LLM still invoked)."""
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
    # Clarify path is commented out; low confidence goes to answer
    assert result["action"] == "answer"
    assert result["response"] == "Could you clarify: what exactly do you need?"


@pytest.mark.skip(reason="Clarify and clarify-then-escalate flow are commented out")
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
    session = {"history": [], "clarify_rounds": 0}
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
        config=Config(),
        llm=MockLLM("Based on the context: here is the answer."),
    )
    result = agent.handle("What is the protocol?")
    assert result["action"] == "answer"
    assert result["confidence"] >= 0.7
    assert "answer" in result["response"].lower() or len(result["response"]) > 0


def test_prompt_handler_loads() -> None:
    """PromptHandler.get_prompt('agent_system') returns non-empty string."""
    project_root = Path(__file__).resolve().parent.parent
    prompts_path = project_root / "prompts" / "system_prompts.yaml"
    handler = PromptHandler(prompts_path=prompts_path)
    text = handler.get_prompt("agent_system")
    assert isinstance(text, str)
    assert len(text) > 0


def test_ollama_llm_mock() -> None:
    """Use MockLLM in tests to avoid real Ollama/API calls."""
    retriever = MockRetriever(
        chunks=[{"text": "Test", "metadata": {}, "score": 0.95}]
    )
    agent = Agent(retriever=retriever, config=Config(), llm=MockLLM("Test reply"))
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


# --- Agent loop tests ---


def _make_loop_agent(llm: MockLLM) -> Agent:
    """Create an Agent with high-confidence chunks so handle() always answers."""
    retriever = MockRetriever(
        chunks=[{"text": "Relevant context.", "metadata": {}, "score": 0.95}]
    )
    return Agent(retriever=retriever, config=Config(), llm=llm)


def test_loop_quit_no_llm_call(monkeypatch, capsys) -> None:
    """Typing 'quit' exits the loop, prints QUIT_RESPONSE, and never calls LLM."""
    llm = MockLLM("Should not be returned")
    agent = _make_loop_agent(llm)
    monkeypatch.setattr("builtins.input", lambda _: "quit")

    agent.run_agent_loop()

    assert llm.call_count == 0
    captured = capsys.readouterr()
    assert Agent.QUIT_RESPONSE in captured.out


def test_loop_responds_then_quits(monkeypatch, capsys) -> None:
    """Agent responds to a real question, then exits cleanly on 'quit'."""
    llm = MockLLM("Here is my answer.")
    agent = _make_loop_agent(llm)
    inputs = iter(["What is the protocol?", "quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    agent.run_agent_loop()

    assert llm.call_count == 1
    captured = capsys.readouterr()
    assert "Here is my answer." in captured.out
    assert Agent.QUIT_RESPONSE in captured.out


# --- Provenance tests ---


def _chunk_with_provenance(
    text: str, score: float, doc_id: str = "doc1", page: int = 1,
    chunk_index: int = 0, source_path: str = "/data/doc1.pdf",
) -> RetrievedChunk:
    """Build a RetrievedChunk with full provenance metadata."""
    meta = {
        "doc_id": doc_id, "page": page, "chunk_index": chunk_index,
        "source_path": source_path,
    }
    return RetrievedChunk(
        text=text, metadata=meta, score=score,
        provenance=ProvenanceInfo(
            doc_id=doc_id, page=page, chunk_index=chunk_index,
            source_path=source_path, score=score,
        ),
    )


def test_handle_returns_sources_key() -> None:
    """handle() result contains a 'sources' list."""
    retriever = MockRetriever(chunks=[
        _chunk_with_provenance("Context A", 0.9, doc_id="d1", page=1, chunk_index=0),
    ])
    agent = Agent(retriever=retriever, config=Config(), llm=MockLLM("Answer"))
    result = agent.handle("question")
    assert "sources" in result
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) == 1


def test_provenance_fields_present_in_sources() -> None:
    """Each source dict has doc_id, page, chunk_index, source_path, score."""
    retriever = MockRetriever(chunks=[
        _chunk_with_provenance("A", 0.92, "doc1", 2, 5, "/data/doc1.pdf"),
        _chunk_with_provenance("B", 0.88, "doc2", 1, 0, "/data/doc2.pdf"),
    ])
    agent = Agent(retriever=retriever, config=Config(), llm=MockLLM("Answer"))
    result = agent.handle("question")
    for src in result["sources"]:
        assert "doc_id" in src
        assert "page" in src
        assert "chunk_index" in src
        assert "source_path" in src
        assert "score" in src
    assert result["sources"][0]["doc_id"] == "doc1"
    assert result["sources"][0]["page"] == 2
    assert result["sources"][1]["source_path"] == "/data/doc2.pdf"


@pytest.mark.skip(reason="Guardrails check is commented out in handle()")
def test_blocked_message_has_empty_sources() -> None:
    """Banned content returns sources=[]."""
    retriever = MockRetriever(chunks=[])
    agent = Agent(retriever=retriever, config=Config(), llm=MockLLM("x"))
    result = agent.handle("fuck this", session={})
    assert result["action"] == "blocked"
    assert result["sources"] == []
