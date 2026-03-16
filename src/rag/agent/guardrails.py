"""Guardrails: system prompt, banned-content check, and escalation logic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from src.rag.core.config import Config
from src.rag.agent.prompt_handler import PromptHandler


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    flagged: bool
    reason: str | None = None


class GuardrailCheck(Protocol):
    """Protocol for content checks. Compose multiple checks via CompositeGuardrail."""

    def check(self, text: str) -> GuardrailResult: ...


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


class RegexBannedContentCheck:
    """Check text for profanity / off-topic harassment via regex patterns.

    Violence, abuse, and injury terms are NOT banned — legitimate 911/EMS content.
    """

    def __init__(self, patterns: list[re.Pattern[str]] | None = None) -> None:
        self._patterns = patterns or _BANNED_PATTERNS

    def check(self, text: str) -> GuardrailResult:
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                return GuardrailResult(flagged=True, reason=f"Banned content detected: {match.group()}")
        return GuardrailResult(flagged=False, reason=None)


class CompositeGuardrail:
    """Runs multiple GuardrailCheck implementations. Flags if any check flags."""

    def __init__(self, checks: list[GuardrailCheck]) -> None:
        self._checks = checks

    def check(self, text: str) -> GuardrailResult:
        for c in self._checks:
            result = c.check(text)
            if result.flagged:
                return result
        return GuardrailResult(flagged=False, reason=None)


def default_guardrail() -> GuardrailCheck:
    """Return the default guardrail (RegexBannedContentCheck). Used by Agent when none injected."""
    return RegexBannedContentCheck()


_DEFAULT_GUARDRAILS_PROMPT = (
    "You are a 911/EMS protocol assistant. "
    "Answer only from the provided context. "
    "Ask clarifying questions when understanding is insufficient. "
    "Escalate to a human operator after repeated failed clarification. "
    "Do not respond to profanity or off-topic harassment."
)


def system_prompt(
    config: Config | None = None,
    prompt_handler: PromptHandler | None = None,
) -> str:
    """Return the guardrails system prompt, loaded from prompts/ or a fallback default."""
    cfg = config or Config()
    prompts_path = getattr(cfg, "prompts_file", "prompts/system_prompts.yaml")
    handler = prompt_handler or PromptHandler(prompts_path=prompts_path)
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
    """Return True when confidence is below threshold and clarify rounds are exhausted.
    Always returns False when eval_mode is True (RAG keeps suggesting responses).
    """
    cfg = config or Config()
    if getattr(cfg, "eval_mode", False):
        return False
    return confidence < cfg.confidence_threshold and clarify_rounds >= cfg.max_clarify_rounds
