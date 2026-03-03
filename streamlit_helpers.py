"""Helpers for streamlit_app (testable without Streamlit)."""

CHAT_TIMEOUT = 180
DEFAULT_TIMEOUT = 30


def get_post_timeout(path: str, explicit: int | None = None) -> int:
    """Return timeout in seconds for a POST request.

    /chat can take 60-180s (embed + Qdrant + LLM). Other endpoints are fast.
    """
    if explicit is not None:
        return explicit
    return CHAT_TIMEOUT if path == "/chat" else DEFAULT_TIMEOUT
