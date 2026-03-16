"""RAG module for 911automate.

Imports are guarded so that missing optional dependencies (e.g. qdrant-client)
don't prevent the package from being partially usable.
"""

__all__: list[str] = []


def _safe(fn) -> None:
    try:
        fn()
    except (ImportError, ModuleNotFoundError):
        pass


def _core():
    from src.rag.core.config import Config
    from src.rag.core.types import ChunkItem, ProvenanceInfo, QdrantPayload, RetrievedChunk

    for name in ("Config", "ChunkItem", "ProvenanceInfo", "QdrantPayload", "RetrievedChunk"):
        globals()[name] = locals()[name]
        __all__.append(name)


def _guardrails():
    from src.rag.agent.guardrails import check_for_banned_content, should_escalate, system_prompt

    for name in ("check_for_banned_content", "should_escalate", "system_prompt"):
        globals()[name] = locals()[name]
        __all__.append(name)


def _prompt_handler():
    from src.rag.agent.prompt_handler import PromptHandler

    globals()["PromptHandler"] = PromptHandler
    __all__.append("PromptHandler")


def _llm_adapters():
    from src.rag.agent.llm_adapters import APILLM, LLM, OllamaLLM, get_llm

    for name in ("APILLM", "LLM", "OllamaLLM", "get_llm"):
        globals()[name] = locals()[name]
        __all__.append(name)


# Chunking is NOT eagerly imported — it pulls in langchain_text_splitters
# -> sentence_transformers -> transformers (30+ s).
# Use "from src.rag.ingestion.chunking import create_chunks" when needed.


def _embedder():
    from src.rag.retrieval.embedder import Embedder

    globals()["Embedder"] = Embedder
    __all__.append("Embedder")


def _retriever():
    from src.rag.retrieval.retriever import Retriever
    from src.rag.core.types import RetrievedChunk

    globals()["Retriever"] = Retriever
    globals()["RetrievedChunk"] = RetrievedChunk
    __all__.extend(["Retriever", "RetrievedChunk"])


def _vectordb():
    from src.rag.retrieval.vectordb_qdrant import VectorDBQdrant
    from src.rag.core.types import ChunkItem

    globals()["VectorDBQdrant"] = VectorDBQdrant
    globals()["ChunkItem"] = ChunkItem
    __all__.extend(["VectorDBQdrant", "ChunkItem"])


def _agent():
    from src.rag.agent.agent import Agent

    globals()["Agent"] = Agent
    __all__.append("Agent")


for _fn in (_core, _prompt_handler, _guardrails, _llm_adapters, _embedder, _retriever, _vectordb, _agent):
    _safe(_fn)
