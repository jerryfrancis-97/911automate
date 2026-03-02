"""Loads system prompts from prompts/ folder."""

from pathlib import Path


class PromptHandler:
    """Loads system prompts from prompts/ folder."""

    def __init__(self, prompts_dir: str | Path = "prompts") -> None:
        self._prompts_dir = Path(prompts_dir)

    def get_prompt(self, task: str) -> str:
        """Load prompts/{task}.txt or prompts/{task}.md. Returns content or default."""
        for ext in (".txt", ".md"):
            path = self._prompts_dir / f"{task}{ext}"
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
        return self._default_for_task(task)

    def _default_for_task(self, task: str) -> str:
        """Fallback minimal string when file is missing."""
        if task == "agent_system":
            return "You are a helpful assistant. Answer based on the context provided."
        if task == "clarify":
            return "Generate 1–2 clarifying questions based on the context and user question."
        return ""
