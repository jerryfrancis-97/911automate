"""Loads system prompts from task-based YAML files (prompts/{task}.yaml) or legacy YAML."""

from pathlib import Path


def load_prompts(path: str | Path) -> dict[str, str]:
    """Load prompts from YAML file. Returns {task: content}. Used as fallback."""
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


def _load_task_yaml(path: Path) -> str | None:
    """Load prompt content from YAML with structure {key: {role: str, content: str}}."""
    if not path.exists():
        return None
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not data:
        return None
    # First (and typically only) top-level key holds role + content
    block = next(iter(data.values()))
    if isinstance(block, dict) and "content" in block:
        return str(block["content"]).strip()
    if isinstance(block, str):
        return block.strip()
    return None


def _project_root() -> Path:
    """Project root: agent/ -> rag/ -> src/ -> project_root."""
    return Path(__file__).resolve().parent.parent.parent.parent


class PromptHandler:
    """Loads prompts from task-based YAML (prompts/{task}.yaml) or legacy YAML."""

    def __init__(
        self,
        prompts_path: str | Path = "prompts/system_prompts.yaml",
        prompts_dir: str | Path | None = None,
    ) -> None:
        """Load from prompts_path (YAML fallback). Task files at prompts_dir/{task}.yaml."""
        self._prompts_path = Path(prompts_path)
        if not self._prompts_path.is_absolute():
            self._prompts_path = _project_root() / self._prompts_path
        self._prompts_dir = Path(prompts_dir) if prompts_dir else self._prompts_path.parent
        if not self._prompts_dir.is_absolute():
            self._prompts_dir = _project_root() / self._prompts_dir
        self._prompts = load_prompts(self._prompts_path)

    def get_prompt(self, task: str) -> str:
        """Return prompt for task. Prefers prompts_dir/{task}.yaml with role+content structure."""
        task_file = self._prompts_dir / f"{task}.yaml"
        content = _load_task_yaml(task_file)
        if content:
            return content
        if task in self._prompts:
            return self._prompts[task]
        return self._default_for_task(task)

    def _default_for_task(self, task: str) -> str:
        """Fallback minimal string when task is missing."""
        if task == "agent_system":
            return "You are a helpful assistant. Answer based on the context provided."
        if task == "clarify":
            return "Generate 1–2 clarifying questions based on the context and user question."
        return ""
