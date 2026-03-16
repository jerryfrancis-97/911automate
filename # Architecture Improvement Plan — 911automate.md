# Architecture Improvement Plan — 911automate

## Context
The codebase is a RAG system for 911/EMS dispatch running llama3.2-3B via local Ollama,
with 1-2 documents in Qdrant, experiencing 10-30s latency (dominated by LLM generation)
and accuracy concerns. This plan addresses structural issues that directly impact latency,
correctness, and maintainability — without over-engineering.

---

## ARCH-1: Fix LLM client re-creation on every call (Latency — HIGH PRIORITY)
(This is done, but just verify the code correctness and modify if there are any errors)
### Problem
`OllamaLLM.invoke()` and `APILLM.invoke()` both instantiate a new `ChatOllama` / `ChatOpenAI`
inside every `.invoke()` call. This creates a new HTTP client, connection pool, and potentially
triggers model negotiation on each request.

### Files
- `src/rag/llm_adapters.py`

### Changes
- Move `ChatOllama(...)` / `ChatOpenAI(...)` construction from `.invoke()` into `__init__()`.
- Store as `self._llm` instance attribute.
- `.invoke()` should only call `self._llm.invoke(messages)`.

### Acceptance Criteria
- [ ] `ChatOllama` / `ChatOpenAI` instantiated once per adapter lifetime
- [ ] No import statements inside `.invoke()` — move to top of file
- [ ] Existing tests still pass
- [ ] Measure latency delta on repeated calls (expect 0.5-2s improvement per call)

---

## ARCH-2: Remove inverted dependency — Agent must not import from API layer (Architecture — HIGH PRIORITY)

### Problem
`src/rag/agent.py` imports `from api.metrics import record_rag_llm_latency, record_rag_retrieval_latency`.
Core domain depends on infrastructure layer. This breaks testability and violates dependency direction
(domain should never know about transport/infra).

`src/rag/cli.py` imports `from api.deps import _build_config` — a private function from the API layer.

### Files
- `src/rag/agent.py`
- `src/rag/cli.py`
- `api/deps.py`

### Changes
1. Define a `MetricsCallback` protocol or simple callable type in `src/rag/` (e.g., in a new
   `src/rag/types.py` or in `config.py`):
   ```python
   from typing import Protocol

   class MetricsRecorder(Protocol):
       def record_retrieval_latency(self, seconds: float) -> None: ...
       def record_llm_latency(self, seconds: float) -> None: ...