"""Streamlit chat UI over the RAG pipeline.

Run with:  uv run streamlit run app.py

Uses the same Settings and RAGPipeline as the CLI. The pipeline (embedding model
+ persisted index + Claude client) is cached with st.cache_resource, so it loads
once per session rather than on every rerun.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from chromadb.api.shared_system_client import SharedSystemClient

from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline, unique_sources

st.set_page_config(page_title="RAG Pipeline", page_icon="🔎", layout="centered")


def _index_version(persist_dir: Path) -> float:
    """Newest mtime under the index directory.

    Changes whenever `rag ingest` rebuilds the store, so it can key the cache
    below and make the app pick up a re-ingest without a manual restart.
    """
    if not persist_dir.exists():
        return 0.0
    return max((p.stat().st_mtime for p in persist_dir.rglob("*")), default=0.0)


@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_pipeline(index_version: float) -> RAGPipeline:
    """Build the pipeline, cached until the on-disk index changes.

    `index_version` busts this cache when `rag ingest` rebuilds the store.
    chromadb caches its client per directory within a process, so we also clear
    that cache here — otherwise the rebuilt pipeline would reuse a stale cached
    client instead of reading the fresh index from disk.
    """
    del index_version  # used only as the cache key; not needed in the body
    SharedSystemClient.clear_system_cache()
    return RAGPipeline(Settings.from_env())


st.title("🔎 RAG Pipeline")
st.caption(
    "Ask questions about the indexed documents. Answers are grounded in "
    "retrieved context and cite their source files."
)

# Build the pipeline, turning setup errors (no index yet, missing API key) into
# a clear on-screen message instead of a stack trace.
try:
    cfg = Settings.from_env()
    pipeline = load_pipeline(_index_version(cfg.persist_dir))
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    # FileNotFoundError: no/empty index. RuntimeError: missing API key.
    # ValueError: a malformed numeric env var (e.g. CHUNK_SIZE=abc).
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
    st.caption(
        "Edit documents in `data/` and re-run `rag ingest` — the app reloads "
        "the new index automatically."
    )
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
