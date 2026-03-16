"""Compatibility: re-export guardrails from agent subpackage."""

from src.rag.agent.guardrails import (
    CompositeGuardrail,
    GuardrailCheck,
    GuardrailResult,
    RegexBannedContentCheck,
    check_for_banned_content,
    default_guardrail,
    should_escalate,
    system_prompt,
)

__all__ = [
    "CompositeGuardrail",
    "GuardrailCheck",
    "GuardrailResult",
    "RegexBannedContentCheck",
    "check_for_banned_content",
    "default_guardrail",
    "should_escalate",
    "system_prompt",
]
