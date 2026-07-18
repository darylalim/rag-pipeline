"""Streamlit chat UI over the RAG pipeline.

Run with:  uv run streamlit run app.py

Uses the same Settings and RAGPipeline as the CLI. The pipeline (embedding model
+ persisted index + Claude client) is cached with st.cache_resource, so it loads
once per session rather than on every rerun.
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline, unique_sources

st.set_page_config(page_title="RAG Pipeline", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_pipeline() -> RAGPipeline:
    """Build the pipeline once and cache it across reruns."""
    return RAGPipeline(Settings.from_env())


st.title("🔎 RAG Pipeline")
st.caption(
    "Ask questions about the indexed documents. Answers are grounded in "
    "retrieved context and cite their source files."
)

# Build the pipeline, turning setup errors (no index yet, missing API key) into
# a clear on-screen message instead of a stack trace.
try:
    pipeline = load_pipeline()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(str(exc))
    st.info("Then reload this page.")
    st.stop()

settings = pipeline.settings
with st.sidebar:
    st.header("Configuration")
    st.markdown(
        f"""
- **Chat model:** `{settings.chat_model}`
- **Embeddings:** `{settings.embedding_model}`
- **Retrieved chunks (k):** `{settings.retrieval_k}`
- **Data dir:** `{settings.data_dir.name}/`
"""
    )
    st.caption("Edit documents in `data/`, then re-run `rag ingest` to update.")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []


def _render_sources(sources: list[str]) -> None:
    if sources:
        with st.expander("Sources"):
            for src in sources:
                st.markdown(f"- `{src}`")


# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

# Handle a new question.
if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating an answer..."):
                result = pipeline.answer(question)
        except Exception as exc:
            # Any failure (bad/expired key, rate limit, network) — show it in
            # the chat instead of a raw traceback, and record it so the
            # question isn't left unanswered in the replayed history.
            error_msg = f"⚠️ Generation failed: {exc}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg, "sources": []}
            )
            st.stop()
        st.markdown(result.text)
        sources = unique_sources(result.sources)
        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "sources": sources}
    )
