"""LLM adapters for Ollama and generic OpenAI-compatible API."""

import os
from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.rag.core.config import Config


class LLM(Protocol):
    """Protocol for LLM adapters. Use structural typing — no inheritance required."""

    def invoke(self, messages: list[dict[str, str]]) -> str: ...


def _to_langchain_messages(messages: list) -> list[BaseMessage]:
    """Convert list of dicts {role, content} to LangChain message objects."""
    out: list[BaseMessage] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content") or ""
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


class OllamaLLM:
    """Wrapper around ChatOllama. Uses config.ollama_base_url, ollama_model."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._llm = ChatOllama(
            base_url=self._config.ollama_base_url,
            model=self._config.ollama_model,
        )

    def invoke(self, messages: list) -> str:
        """Invoke the model with messages (list of {role, content}). Returns content string."""
        lc_messages = _to_langchain_messages(messages)
        response = self._llm.invoke(lc_messages)
        return response.content if hasattr(response, "content") else str(response)


class APILLM:
    """Generic OpenAI-compatible API. Uses config.api_base_url, api_model, api_key."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._api_key = (
            self._config.api_key or os.environ.get("OPENAI_API_KEY") or ""
        ).strip()

        kwargs: dict = {
            "model": self._config.api_model,
            "api_key": self._api_key or "placeholder",
        }
        if self._config.api_base_url:
            kwargs["base_url"] = self._config.api_base_url

        self._llm = ChatOpenAI(**kwargs)

    def invoke(self, messages: list) -> str:
        """Invoke the API with messages. Returns content string."""
        lc_messages = _to_langchain_messages(messages)
        response = self._llm.invoke(lc_messages)
        return response.content if hasattr(response, "content") else str(response)


def get_llm(config: Config) -> LLM:
    """Return APILLM if api_base_url set, else OllamaLLM."""
    if config.api_base_url:
        return APILLM(config)
    return OllamaLLM(config)
