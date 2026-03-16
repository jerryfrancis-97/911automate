"""Compatibility: re-export PromptHandler from agent subpackage."""

from src.rag.agent.prompt_handler import PromptHandler, load_prompts

__all__ = ["PromptHandler", "load_prompts"]
