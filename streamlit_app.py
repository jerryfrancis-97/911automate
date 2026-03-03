"""Streamlit frontend for the 911automate RAG agent.

Launch:
    streamlit run streamlit_app.py

Requires the FastAPI backend running at API_URL (default http://localhost:8000).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="911automate",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("911automate RAG Agent")

# ---------------------------------------------------------------------------
# Session state defaults (must run before any conditional exit)
# ---------------------------------------------------------------------------

def _init_session_state():
    defaults = {
        "session_id": None,
        "messages": [],
        "last_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


from streamlit_helpers import get_post_timeout


def _post(path: str, payload: dict | None = None, timeout: int | None = None) -> dict | None:
    t = get_post_timeout(path, timeout)
    try:
        resp = requests.post(f"{API_URL}{path}", json=payload, timeout=t)
        resp.raise_for_status()
        return resp.json()
    except requests.Timeout:
        st.error(f"Request timed out after {t}s. The chat endpoint may need more time (LLM inference).")
        return None
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


def _get(path: str) -> dict | None:
    try:
        resp = requests.get(f"{API_URL}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        st.error(f"API error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Section 1: Session Control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Session Control")
    if st.button("Start New Session", use_container_width=True):
        data = _post("/session")
        if data:
            st.session_state.session_id = data["session_id"]
            st.session_state.messages = []
            st.session_state.last_result = None
            st.rerun()

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:12]}...`")
    else:
        st.info("Click **Start New Session** to begin.")

    sessions_data = _get("/sessions")
    if sessions_data and sessions_data.get("session_ids"):
        st.caption(f"Active sessions: {len(sessions_data['session_ids'])}")

    # ----- Backend status -----
    st.divider()
    st.header("Backend Status")
    ready_data = _get("/ready")
    if ready_data:
        st.write("Agent:", "ready" if ready_data.get("agent_ready") else "not ready")
        st.write("Embedder:", "ready" if ready_data.get("embedder_ready") else "not ready")
        st.write("Retriever:", "ready" if ready_data.get("retriever_ready") else "not ready")
        if ready_data.get("error"):
            st.error(ready_data["error"])
    else:
        st.warning("Cannot reach backend. Is it running?")

    # ----- Manual Escalation -----
    st.divider()
    st.header("Manual Escalation")
    escalation_reason = st.text_input("Reason", key="esc_reason")
    if st.button("Escalate to Human", use_container_width=True):
        if not st.session_state.session_id:
            st.warning("Start a session first.")
        elif not escalation_reason.strip():
            st.warning("Enter a reason.")
        else:
            data = _post("/escalate", {
                "session_id": st.session_state.session_id,
                "reason": escalation_reason.strip(),
            })
            if data:
                st.success("Session escalated to a human operator.")

# ---------------------------------------------------------------------------
# Section 2: Chat Interface
# ---------------------------------------------------------------------------

if not st.session_state.session_id:
    st.info("Start a new session from the sidebar to chat.")
    st.stop()

st.subheader("Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not st.session_state.messages:
    st.caption("Type your question below and press Enter or click Send.")

# Chat input: use form + text_input for reliable visibility (st.chat_input can be below fold)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question",
        placeholder="Ask a question about 911 procedures...",
        key="chat_input",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input and user_input.strip():
    question = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    data = _post("/chat", {
        "session_id": st.session_state.session_id,
        "question": question,
    })

    if data:
        st.session_state.last_result = data
        assistant_text = data.get("answer", "")
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Sorry, the agent is not ready or there was an error. Check the sidebar for backend status.",
        })
    st.rerun()

# ---------------------------------------------------------------------------
# Section 3: Agent Result Panel
# ---------------------------------------------------------------------------

result = st.session_state.last_result
if result:
    st.divider()

    col1, col2 = st.columns([1, 3])
    with col1:
        action = result.get("action", "")
        color_map = {
            "answer": "green",
            "clarify": "orange",
            "escalate": "red",
            "blocked": "red",
        }
        color = color_map.get(action, "gray")
        st.markdown(f"**Action:** :{color}[{action.upper()}]")

    with col2:
        confidence = result.get("confidence", 0.0)
        st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.2f}")

    if action == "clarify":
        st.info(f"**Clarification needed:** {result.get('answer', '')}")

    if action == "escalate":
        reason = result.get("escalation_reason") or result.get("answer", "")
        st.error(f"**Escalation:** {reason}")

    # -------------------------------------------------------------------
    # Section 4: Retrieved Evidence
    # -------------------------------------------------------------------
    facts = result.get("used_facts", [])
    if facts:
        with st.expander(f"Retrieved Evidence ({len(facts)} chunks)", expanded=False):
            for i, fact in enumerate(facts, 1):
                st.markdown(
                    f"**{i}.** `{fact.get('doc_id', '')}` "
                    f"p.{fact.get('page', '?')} — "
                    f"score {fact.get('score', 0):.3f}"
                )
                snippet = fact.get("snippet", "")
                if snippet:
                    st.caption(snippet)
