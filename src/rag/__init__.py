"""RAG module for 911automate."""

# Imports are guarded so that missing optional dependencies (e.g. qdrant-client)
# don't prevent the package from being partially usable.

__all__: list[str] = []


def _safe_import(import_fn: callable, names: list[str]) -> None:
    try:
        import_fn()
    except (ImportError, ModuleNotFoundError):
        pass


def _import_config():
    from src.rag.config import Config  # noqa: F401
    globals()["Config"] = Config
    __all__.append("Config")

def _import_guardrails():
    from src.rag.guardrails import check_for_banned_content, should_escalate, system_prompt  # noqa: F401
    globals()["check_for_banned_content"] = check_for_banned_content
    globals()["should_escalate"] = should_escalate
    globals()["system_prompt"] = system_prompt
    __all__.extend(["check_for_banned_content", "should_escalate", "system_prompt"])

def _import_prompt_handler():
    from src.rag.prompt_handler import PromptHandler  # noqa: F401
    globals()["PromptHandler"] = PromptHandler
    __all__.append("PromptHandler")

def _import_llm_adapters():
    from src.rag.llm_adapters import APILLM, OllamaLLM, get_llm  # noqa: F401
    globals()["APILLM"] = APILLM
    globals()["OllamaLLM"] = OllamaLLM
    globals()["get_llm"] = get_llm
    __all__.extend(["APILLM", "OllamaLLM", "get_llm"])

def _import_chunking():
    from src.rag.chunking import create_chunks  # noqa: F401
    globals()["create_chunks"] = create_chunks
    __all__.append("create_chunks")

def _import_embedder():
    from src.rag.embedder import Embedder  # noqa: F401
    globals()["Embedder"] = Embedder
    __all__.append("Embedder")

def _import_retriever():
    from src.rag.retriever import CalculateMMR, RetrievedChunk, Retriever  # noqa: F401
    globals()["CalculateMMR"] = CalculateMMR
    globals()["RetrievedChunk"] = RetrievedChunk
    globals()["Retriever"] = Retriever
    __all__.extend(["CalculateMMR", "RetrievedChunk", "Retriever"])

def _import_vectordb():
    from src.rag.vectordb_qdrant import ChunkItem, VectorDBQdrant  # noqa: F401
    globals()["ChunkItem"] = ChunkItem
    globals()["VectorDBQdrant"] = VectorDBQdrant
    __all__.extend(["ChunkItem", "VectorDBQdrant"])

def _import_agent():
    from src.rag.agent import Agent, AgentSession  # noqa: F401
    globals()["Agent"] = Agent
    globals()["AgentSession"] = AgentSession
    __all__.extend(["Agent", "AgentSession"])


for _fn in (
    _import_config,
    _import_prompt_handler,
    _import_guardrails,
    _import_llm_adapters,
    _import_chunking,
    _import_embedder,
    _import_retriever,
    _import_vectordb,
    _import_agent,
):
    _safe_import(_fn, [])
