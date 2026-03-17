"""Shared tokenizer utilities for nomic-embed-text. Used by chunking and embedder."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# nomic-embed-text context limit (used for embedding truncation)
NOMIC_MAX_TOKENS = 8192

# HuggingFace model ID for nomic-embed-text tokenizer
NOMIC_TOKENIZER_MODEL = "nomic-ai/nomic-embed-text-v1.5"


@lru_cache(maxsize=1)
def _get_tokenizer():
    """Lazy-load nomic-embed-text tokenizer. Cached for reuse."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        NOMIC_TOKENIZER_MODEL,
        trust_remote_code=True,
    )


def get_nomic_tokenizer():
    """Return nomic-embed-text tokenizer for RecursiveCharacterTextSplitter.from_huggingface_tokenizer."""
    return _get_tokenizer()


def count_tokens(text: str) -> int:
    """Return token count for text using nomic-embed-text tokenizer."""
    if not text or not text.strip():
        return 0
    tokenizer = _get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_to_tokens(text: str, max_tokens: int = 512) -> str:
    """Truncate text to max_tokens. Prevents embedding failures from oversized input."""
    if not text or not text.strip():
        return text
    tokenizer = _get_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(ids) <= max_tokens:
        return text
    truncated = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    logger.debug("Truncated from %d to %d tokens", len(ids), max_tokens)
    return truncated
