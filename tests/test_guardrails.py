"""Unit tests for src/rag/guardrails."""

import pytest

from src.rag.config import Config
from src.rag.guardrails import (
    check_for_banned_content,
    should_escalate,
    system_prompt,
)
from src.rag.prompt_handler import PromptHandler


# ---------------------------------------------------------------------------
# system_prompt tests
# ---------------------------------------------------------------------------


def test_system_prompt_returns_string():
    """system_prompt() returns a non-empty string with key policy phrases."""
    result = system_prompt()
    assert isinstance(result, str)
    assert len(result) > 0
    assert "escalate" in result.lower()
    assert "profanity" in result.lower() or "harassment" in result.lower()


def test_system_prompt_fallback():
    """When the prompt file is missing, a sensible default is returned."""
    handler = PromptHandler(prompts_dir="nonexistent_dir_for_testing")
    result = system_prompt(prompt_handler=handler)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "escalate" in result.lower()


# ---------------------------------------------------------------------------
# check_for_banned_content tests
# ---------------------------------------------------------------------------


def test_check_for_banned_content_clean():
    """Clean emergency-related input is not flagged."""
    result = check_for_banned_content("What is the protocol for cardiac arrest?")
    assert result["flagged"] is False
    assert result["reason"] is None


def test_check_for_banned_content_flagged():
    """Input with profanity is flagged with a reason."""
    result = check_for_banned_content("This is bullshit, give me a real answer")
    assert result["flagged"] is True
    assert result["reason"] is not None
    assert "Banned content" in result["reason"]


def test_check_for_banned_content_allows_emergency_language():
    """Violence/abuse/injury descriptions are NOT flagged -- legitimate 911 content."""
    emergency_inputs = [
        "Someone was stabbed in the chest",
        "There is a domestic abuse situation at 123 Main St",
        "The victim has severe bleeding from a gunshot wound",
        "A child was hit by a car and is unconscious",
        "There is an active shooter at the school",
    ]
    for text in emergency_inputs:
        result = check_for_banned_content(text)
        assert result["flagged"] is False, f"Incorrectly flagged legitimate emergency text: {text!r}"


# ---------------------------------------------------------------------------
# should_escalate tests
# ---------------------------------------------------------------------------

_CFG = Config(confidence_threshold=0.7, max_clarify_rounds=3)


def test_should_escalate_true():
    """Low confidence + clarify rounds exhausted -> escalate."""
    assert should_escalate(confidence=0.3, clarify_rounds=3, config=_CFG) is True
    assert should_escalate(confidence=0.5, clarify_rounds=5, config=_CFG) is True


def test_should_escalate_false_high_confidence():
    """High confidence -> no escalation regardless of rounds."""
    assert should_escalate(confidence=0.9, clarify_rounds=10, config=_CFG) is False
    assert should_escalate(confidence=0.7, clarify_rounds=3, config=_CFG) is False


def test_should_escalate_false_rounds_remaining():
    """Low confidence but rounds still remaining -> no escalation yet."""
    assert should_escalate(confidence=0.3, clarify_rounds=0, config=_CFG) is False
    assert should_escalate(confidence=0.5, clarify_rounds=2, config=_CFG) is False
