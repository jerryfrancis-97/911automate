"""Guardrails: system prompt, banned-content check, and escalation logic."""

from __future__ import annotations

import re

from src.rag.config import Config
from src.rag.prompt_handler import PromptHandler

_DEFAULT_GUARDRAILS_PROMPT = (
    "You are a 911/EMS protocol assistant. "
    "Answer only from the provided context. "
    "Ask clarifying questions when understanding is insufficient. "
    "Escalate to a human operator after repeated failed clarification. "
    "Do not respond to profanity or off-topic harassment."
)

_BANNED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bf+u+c+k+\w*", re.IGNORECASE),
    re.compile(r"\w*s+h+i+t+\w*", re.IGNORECASE),
    re.compile(r"\ba+s+s+h+o+l+e+\w*", re.IGNORECASE),
    re.compile(r"\bb+i+t+c+h+\w*", re.IGNORECASE),
    re.compile(r"\bd+a+m+n+\w*", re.IGNORECASE),
    re.compile(r"\bstfu\b", re.IGNORECASE),
    re.compile(r"\bwtf\b", re.IGNORECASE),
    re.compile(r"\bstupid\s+(bot|machine|system|ai)\b", re.IGNORECASE),
]


def system_prompt(
    config: Config | None = None,
    prompt_handler: PromptHandler | None = None,
) -> str:
    """Return the guardrails system prompt, loaded from prompts/ or a fallback default."""
    cfg = config or Config()
    handler = prompt_handler or PromptHandler(cfg.prompts_dir)
    text = handler.get_prompt("guardrails_system")
    return text if text else _DEFAULT_GUARDRAILS_PROMPT


def check_for_banned_content(text: str) -> dict[str, bool | str | None]:
    """Check text for profanity / off-topic harassment.

    Violence, abuse, and injury terms are NOT banned -- they are
    legitimate 911/EMS emergency content.

    Returns {"flagged": bool, "reason": str | None}.
    """
    for pattern in _BANNED_PATTERNS:
        match = pattern.search(text)
        if match:
            return {"flagged": True, "reason": f"Banned content detected: {match.group()}"}
    return {"flagged": False, "reason": None}


def should_escalate(
    confidence: float,
    clarify_rounds: int,
    config: Config | None = None,
) -> bool:
    """Return True when confidence is below threshold and clarify rounds are exhausted."""
    cfg = config or Config()
    return confidence < cfg.confidence_threshold and clarify_rounds >= cfg.max_clarify_rounds
