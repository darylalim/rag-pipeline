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


# Bounded because `version` is a deliberately changing key: every rebuild mints
# a new entry, and cache_resource never evicts on its own, so unbounded each
# `rag ingest` against a running server strands another embedding model and
# Chroma client for the life of the process.
#
# Two entries rather than one. The cache is process-wide, not per-session, and
# during an ingest two browser sessions can read `chroma.sqlite3` at different
# moments and derive different mtimes. At size one those keys evict each other
# on every rerun, reloading the embedding model each time; the second slot lets
# the outgoing and incoming versions coexist until the writes settle.
@st.cache_resource(max_entries=2, show_spinner="Loading index and embedding model...")
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


def _error_reply(text: str) -> dict:
    """A stored assistant turn that failed rather than answered.

    Cites nothing, because an unanswered turn has no grounding to claim, and
    carries the flag the replay loop routes on.
    """
    return {"role": "assistant", "content": text, "sources": [], "error": True}


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
# submit_mode="disable" removes the common way to interrupt an in-flight answer
# (submitting again), but it is not sufficient on its own: the toolbar Stop
# button raises a BaseException that no `except Exception` sees, and streaming
# leaves it reachable for the whole generation. The pairing below, not this
# parameter, is what actually guarantees a consistent history.
if question := st.chat_input(
    "Ask a question about the documents...", submit_mode="disable"
):
    # Rendered now but deliberately not stored yet — the turn is committed as a
    # pair in the `finally` below.
    with st.chat_message("user"):
        st.text(question)

    # Stands unless a branch below replaces it, so an interruption still records
    # *something* under the question rather than leaving it dangling.
    reply = _error_reply("Interrupted before an answer was generated.")
    with st.chat_message("assistant"):
        try:
            # stream_answer() has finished retrieving when it returns but has
            # not started generating, so the spinner covers exactly the step
            # with nothing to show.
            with st.spinner("Retrieving context..."):
                docs, chunks = pipeline.stream_answer(question)
            answer_text = st.write_stream(chunks)
            # write_stream already rendered this; it is re-read only to store
            # it, and joined because the declared return is list[Any] | str —
            # a str is itself an iterable of str, so one join covers both. Kept
            # inside the try so a non-str list raises into the handler below
            # rather than as a crash page.
            text = "".join(answer_text)
            sources = unique_sources(docs)
            _render_sources(sources)
            reply = {"role": "assistant", "content": text, "sources": sources}
        except Exception as exc:
            # Any failure (bad/expired key, rate limit, empty response) — show
            # it in the chat instead of a raw traceback. The `error` flag keeps
            # the stored text free of presentation, so the replay above can
            # route it back through st.error rather than rendering a failure as
            # if it were an answer.
            error_msg = f"Generation failed: {exc}"
            st.error(error_msg, icon=":material/error:")
            reply = _error_reply(error_msg)
        finally:
            # Both halves land together, so no exit path — success, handled
            # failure, or an interruption that unwinds past the handler — can
            # leave a question in the history with no answer under it.
            st.session_state.messages.extend(
                [{"role": "user", "content": question}, reply]
            )
