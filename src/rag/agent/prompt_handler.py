"""Loads system prompts from a single YAML file."""

from pathlib import Path


def load_prompts(path: str | Path) -> dict[str, str]:
    """Load prompts from YAML file. Returns {task: content}."""
    path = Path(path)
    if not path.exists():
        return {}

    import yaml

    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) or {}
    out: dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, str):
            out[str(k)] = v.strip()
    return out


class PromptHandler:
    """Loads system prompts from a single YAML file."""

    def __init__(
        self,
        prompts_path: str | Path = "prompts/system_prompts.yaml",
        prompts_dir: str | Path | None = None,
    ) -> None:
        """Load from prompts_path (project-relative or absolute). prompts_dir is deprecated."""
        self._prompts_path = Path(prompts_path)
        if not self._prompts_path.is_absolute():
            # agent/prompt_handler.py -> agent/ -> rag/ -> src/ -> project_root
            root = Path(__file__).resolve().parent.parent.parent.parent
            self._prompts_path = root / self._prompts_path
        self._prompts = load_prompts(self._prompts_path)

    def get_prompt(self, task: str) -> str:
        """Return prompt for task, or default fallback if missing."""
        if task in self._prompts:
            return self._prompts[task]
        return self._default_for_task(task)

    def _default_for_task(self, task: str) -> str:
        """Fallback minimal string when task is missing from YAML."""
        if task == "agent_system":
            return "You are a helpful assistant. Answer based on the context provided."
        if task == "clarify":
            return "Generate 1–2 clarifying questions based on the context and user question."
        return ""
