"""OPenly – Child Development Assessment Chatbot (Streamlit Web UI)

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push repo to GitHub
    2. Connect at share.streamlit.io
    3. Add ANTHROPIC_API_KEY in Streamlit Cloud secrets
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be first st call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="OPenly – Child Development Chatbot",
    page_icon="🧒",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Resolve API key: st.secrets → env → .env fallback
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Resolve API key from Streamlit secrets, env var, or .env file."""
    # 1. Streamlit secrets (production / cloud)
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key and key != "your-key-here":
            return key
    except Exception:
        pass

    # 2. Environment variable
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key and key != "your-key-here":
        return key

    # 3. .env file fallback (local dev)
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if key and key != "your-key-here":
            return key

    return ""


# ---------------------------------------------------------------------------
# Build / cache the conversation graph (runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading clinical domain data…")
def _build_graph(api_key: str):
    """Build the OPenly conversation graph (cached across reruns)."""
    from src.openly.graph import build_graph

    project_root = Path(__file__).resolve().parent
    domain_tree = str(project_root / "Domain_tree_UPDATED.xlsx")
    cross_domain = str(project_root / "Cross_Domain_Logic_UPDATED.xlsx")

    return build_graph(
        domain_tree_path=domain_tree,
        cross_domain_path=cross_domain,
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_session(graph):
    """Start a new conversation session and save to session_state."""
    session_id = f"session_{uuid.uuid4().hex[:8]}"
    state = graph.start_session(session_id)
    st.session_state["conv_state"] = state
    st.session_state["messages"] = []

    # Add opening bot message
    bot_msg = state.get("bot_message", "")
    if bot_msg:
        st.session_state["messages"].append({"role": "assistant", "content": bot_msg})


def _severity_color(level: str) -> str:
    return {
        "low": "🟢",
        "mild_concern": "🟡",
        "moderate": "🟠",
        "high": "🔴",
    }.get(level, "⚪")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    # --- API key check ---
    api_key = _get_api_key()
    if not api_key:
        st.error(
            "**ANTHROPIC_API_KEY not found.**\n\n"
            "Please set it in one of:\n"
            "- `.streamlit/secrets.toml` (local dev)\n"
            "- Streamlit Cloud secrets (production)\n"
            "- `.env` file in the project root\n"
            "- `ANTHROPIC_API_KEY` environment variable"
        )
        st.stop()

    # --- Build graph ---
    try:
        graph = _build_graph(api_key)
    except Exception as e:
        st.error(f"Failed to initialize: {e}")
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.title("🧒 OPenly")
        st.caption("Child Development Assessment Chatbot (MVP)")
        st.divider()

        # New session button
        if st.button("🔄 New Session", use_container_width=True):
            _init_session(graph)
            st.rerun()

        # Show trace toggle
        show_trace = st.toggle("Show reasoning trace", value=False)

        st.divider()

        # Session stats
        if "conv_state" in st.session_state:
            state = st.session_state["conv_state"]
            severity = state.get("severity_level", "low")

            st.subheader("Session Info")
            st.markdown(f"**Severity:** {_severity_color(severity)} {severity.replace('_', ' ').title()}")
            st.markdown(f"**Phase:** {state.get('phase', '–').title()}")

            explored = state.get("explored_domains", [])
            if explored:
                st.markdown(f"**Domains explored:** {len(explored)}")
                for d in explored:
                    st.markdown(f"  - {d.replace('_', ' ').title()}")

            tags = state.get("discovered_tags", set())
            if tags:
                st.markdown(f"**Tags discovered:** {len(tags)}")

            queue = state.get("domain_queue", [])
            if queue:
                st.markdown(f"**Queued domains:** {len(queue)}")

            intake = state.get("intake_fields", {})
            if intake:
                st.markdown("**Intake captured:**")
                for k, v in intake.items():
                    st.markdown(f"  - *{k}*: {v[:50]}{'…' if len(v) > 50 else ''}")

            if state.get("safety_escalated"):
                st.error("⚠️ SAFETY ESCALATION TRIGGERED")

        st.divider()
        st.caption("Built with Claude + LangGraph")

    # --- Initialize session on first load ---
    if "conv_state" not in st.session_state:
        _init_session(graph)

    # --- Chat header ---
    st.title("OPenly")
    st.caption("A warm, supportive space to discuss your child's development")

    # --- Render conversation history ---
    for msg in st.session_state.get("messages", []):
        with st.chat_message(msg["role"], avatar="🧒" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # --- Session ended notice ---
    state = st.session_state.get("conv_state", {})
    if state.get("should_end"):
        st.info("Session complete. Click **New Session** in the sidebar to start over.")

        # Show final trace if enabled
        if show_trace:
            with st.expander("📋 Final Session Trace", expanded=False):
                trace_json = graph.get_trace_json(state)
                st.json(json.loads(trace_json))
        return

    # --- Chat input ---
    if prompt := st.chat_input("Share your concern…"):
        # Add user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Process through graph
        with st.spinner("Thinking…"):
            state = graph.process_message(state, prompt)
            st.session_state["conv_state"] = state

        # Display bot response
        bot_msg = state.get("bot_message", "")
        if bot_msg:
            st.session_state["messages"].append({"role": "assistant", "content": bot_msg})
            with st.chat_message("assistant", avatar="🧒"):
                st.markdown(bot_msg)

        # Safety escalation alert
        if state.get("safety_escalated"):
            st.error(
                "⚠️ **Important:** Based on what you've shared, we strongly recommend "
                "reaching out to a professional immediately. If there is an immediate "
                "safety concern, please contact your local emergency services."
            )

        # Show trace if enabled
        if show_trace:
            with st.expander("🔍 Reasoning Trace", expanded=False):
                trace_json = graph.get_trace_json(state)
                st.json(json.loads(trace_json))

        # Rerun to update sidebar stats
        if state.get("should_end"):
            st.rerun()


if __name__ == "__main__":
    main()
