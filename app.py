"""Streamlit chat UI over the RAG pipeline.

Run with:  uv run streamlit run app.py

Uses the same Settings and RAGPipeline as the CLI. The pipeline (embedding model
+ persisted index + Claude client) is cached with st.cache_resource, so it loads
once per session rather than on every rerun.
"""

from __future__ import annotations

import streamlit as st

from rag_pipeline.config import Settings
from rag_pipeline.ingest import index_version, reset_store_cache
from rag_pipeline.pipeline import RAGPipeline, unique_sources

st.set_page_config(
    page_title="RAG Pipeline", page_icon=":material/search:", layout="centered"
)

# Initialized here rather than beside the replay loop: the sidebar runs first and
# already writes this key, so a reader added there would otherwise KeyError on
# the first load.
st.session_state.setdefault("messages", [])


# max_entries=1 because `version` is a deliberately changing key: every rebuild
# mints a new entry, and cache_resource never evicts on its own. Unbounded, each
# `rag ingest` against a running server would strand another embedding model and
# Chroma client in memory for the life of the process. Only the newest index is
# ever wanted, so one entry is enough.
@st.cache_resource(max_entries=1, show_spinner="Loading index and embedding model...")
def load_pipeline(_settings: Settings, version: float) -> RAGPipeline:
    """Build the pipeline, cached until the on-disk index changes.

    `version` (the index's fingerprint) is the cache key: it changes when
    `rag ingest` rebuilds the store, busting this cache. `_settings` is passed
    in — the leading underscore tells Streamlit not to hash it — so we don't
    re-read the environment here. On a rebuild we clear chromadb's client cache
    first, or the fresh pipeline would reuse a stale cached client instead of
    reading the new index from disk.
    """
    del version  # used only as the cache key; not needed in the body
    reset_store_cache()
    return RAGPipeline(_settings)


st.title(":material/search: RAG Pipeline")
st.caption(
    "Ask questions about the indexed documents. Answers are grounded in "
    "retrieved context and cite their source files."
)

# Build the pipeline, turning setup errors (no index yet, missing API key) into
# a clear on-screen message instead of a stack trace.
try:
    cfg = Settings.from_env()
    pipeline = load_pipeline(cfg, index_version(cfg))
except (FileNotFoundError, RuntimeError, ValueError) as exc:
    # FileNotFoundError: no/empty index. RuntimeError: missing API key.
    # ValueError: a malformed numeric env var (e.g. CHUNK_SIZE=abc).
    # One callout, not an error stacked on an info: "Then reload this page" is a
    # continuation of the error, meaningless on its own.
    st.error(f"{exc}\n\nThen reload this page.", icon=":material/error:")
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
    # No st.rerun() needed: the click already triggered this run, and the sidebar
    # executes above the replay loop, so falling through renders the cleared UI.
    if st.button("Clear conversation", icon=":material/delete:"):
        st.session_state.messages = []


def _render_sources(sources: list[str]) -> None:
    if sources:
        with st.expander("Sources", icon=":material/description:"):
            # One markdown body, not one call per item: separate calls render as
            # separate single-item lists, each with its own block spacing.
            st.markdown("\n".join(f"- `{src}`" for src in sources))


# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            # Unparsed, so a question about `snake_case` or one starting with `#`
            # echoes back as typed rather than as italics or a heading.
            st.text(message["content"])
        elif message.get("error"):
            # Same renderer as the live failure, so a failed turn keeps its error
            # styling instead of decaying into an ordinary-looking answer.
            st.error(message["content"], icon=":material/error:")
        else:
            st.markdown(message["content"])
            _render_sources(message.get("sources", []))

# Handle a new question.
# submit_mode="disable" is load-bearing, not cosmetic: answering blocks for
# seconds, and a second submit during it reruns the script, killing this one
# between the user append below and the assistant append at the end. That leaves
# a question in the history with no answer under it, permanently — the same state
# the error branch below exists to prevent. Disabling the input makes it
# unreachable.
if question := st.chat_input(
    "Ask a question about the documents...", submit_mode="disable"
):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.text(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving context and generating an answer..."):
                result = pipeline.answer(question)
        except Exception as exc:
            # Any failure (bad/expired key, rate limit, network) — show it in
            # the chat instead of a raw traceback, and record it so the
            # question isn't left unanswered in the replayed history.
            # The `error` flag keeps the stored text free of presentation, so the
            # replay above can route it back through st.error rather than
            # rendering a failure as if it were an answer.
            error_msg = f"Generation failed: {exc}"
            st.error(error_msg, icon=":material/error:")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "sources": [],
                    "error": True,
                }
            )
            st.stop()
        st.markdown(result.text)
        sources = unique_sources(result.sources)
        _render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.text, "sources": sources}
    )
