"""Streamlit chat UI over the RAG pipeline.

Run with:  uv run streamlit run app.py

Uses the same Settings and RAGPipeline as the CLI. The pipeline (embedding model
+ persisted index + Claude client) is cached with st.cache_resource, so it loads
once per session rather than on every rerun.
"""

from __future__ import annotations

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from rag_pipeline.config import Settings
from rag_pipeline.ingest import (
    SUPPORTED_SUFFIXES,
    index_version,
    ingest,
    save_upload,
)
from rag_pipeline.pipeline import Excerpt, RAGPipeline, source_excerpts

st.set_page_config(
    page_title="RAG Pipeline", page_icon=":material/search:", layout="centered"
)

# Initialized here rather than beside the replay loop: the sidebar runs first and
# already writes this key, so a reader added there would otherwise KeyError on
# the first load.
st.session_state.setdefault("messages", [])


# Bounded because `version` is a deliberately changing key: every rebuild mints
# a new entry, and cache_resource never evicts on its own, so unbounded each
# `rag ingest` against a running server strands another embedding model for the
# life of the process. (The Mongo client is not multiplied — it is a single
# process-wide pool every pipeline shares.)
#
# Two entries rather than one. The digest is a single server-authoritative
# value, so the two-sessions-derive-different-keys race the mtime had is gone;
# the second slot now just cushions reload churn, letting the outgoing and
# incoming versions coexist for a rerun rather than evicting each other.
@st.cache_resource(max_entries=2, show_spinner="Loading index and embedding model...")
def load_pipeline(_settings: Settings, version: str) -> RAGPipeline:
    """Build the pipeline, cached until the indexed corpus changes.

    `version` (the corpus fingerprint digest) is the cache key: it changes when
    `rag ingest` re-embeds the store, busting this cache. `_settings` is passed
    in — the leading underscore tells Streamlit not to hash it — so we don't
    re-read the environment here. The rebuilt pipeline reuses the process-wide
    Mongo client rather than resetting it: a live pool is always current, so
    there is nothing stale to drop, and closing it would strand the still-cached
    outgoing pipeline (kept by `max_entries`) on a dead connection.
    """
    del version  # used only as the cache key; not needed in the body
    return RAGPipeline(_settings)


def _add_documents(settings: Settings, uploads: list[UploadedFile]) -> None:
    """Save uploaded files into ``data_dir``, then rebuild the index from it.

    Each file is saved on its own so one rejected name does not discard the
    rest, and the rebuild covers whatever landed rather than only a clean batch:
    a file already written *is* a document in ``data_dir``, so omitting it here
    would hide it until some later, unrelated ingest happened to pick it up.

    Deliberately no ``st.rerun()``. The submit already triggered this run and
    this executes above the pipeline load, so falling through re-reads
    ``index_version()`` — which the rebuild just changed — and the cached
    pipeline is rebuilt against the new index on this same run.
    """
    saved, rejected = [], []
    for upload in uploads:
        try:
            saved.append(save_upload(settings.data_dir, upload.name, upload.getvalue()))
        except (OSError, ValueError) as exc:
            # Per file rather than per batch because OSError genuinely varies
            # within one: a name past the filesystem's length limit fails while
            # its neighbours write fine. ValueError (an unsupported suffix) is
            # belt-and-braces — Streamlit re-checks the extension server-side, so
            # `type=` above already blocks it — but save_upload is a public
            # function and this frontend should not be the thing assuming it.
            rejected.append(str(exc))

    if saved:
        try:
            with st.spinner(f"Indexing {len(saved)} file(s)..."):
                n_chunks = ingest(settings)
        except (OSError, RuntimeError, ValueError) as exc:
            # The files are on disk regardless, so this reports a failed *index*
            # rather than a failed upload — `rag ingest` retries it. RuntimeError
            # covers a missing VOYAGE_API_KEY and any Voyage/index error now that
            # embedding is an API call; without it those escape and crash the
            # sidebar mid-upload.
            st.error(f"Indexing failed: {exc}", icon=":material/error:")
        else:
            st.success(
                f"Indexed {n_chunks} chunks. Added: {', '.join(saved)}",
                icon=":material/check:",
            )
    for message in rejected:
        st.warning(message, icon=":material/error:")


st.title(":material/search: RAG Pipeline")
st.caption(
    "Ask questions about the indexed documents. Answers are grounded in "
    "retrieved context, and each one shows the passages it was generated from."
)

# Settings are resolved on their own, ahead of the index, because the sidebar
# below needs them and has to render even when the index does not load — an app
# that cannot answer anything is exactly when a user reaches for the uploader
# that fixes it. A malformed numeric env var (CHUNK_SIZE=abc) is the one setup
# failure nothing can proceed past, so it alone stops the script here.
try:
    cfg = Settings.from_env()
except ValueError as exc:
    st.error(f"{exc}\n\nFix it, then reload this page.", icon=":material/error:")
    st.stop()

with st.sidebar:
    st.header("Configuration")
    st.markdown(
        f"""
- **Chat model:** `{cfg.chat_model}`
- **Embeddings:** `{cfg.embedding_model}`
- **Retrieved chunks (k):** `{cfg.retrieval_k}`
- **Data dir:** `{cfg.data_dir.name}/`
"""
    )

    # `st.file_uploader` re-reports its files on every rerun, so re-indexing on
    # sight would rebuild the whole corpus once per chat message. What confines
    # the work to one run is `submitted` below, not the form: a submit button is
    # a *trigger*, reset to False after the run it was clicked on, whereas the
    # uploader keeps its batch attached (no clear_on_submit) and stays truthy
    # afterwards. Measured, not assumed — a form leaves `uploads` truthy on
    # every subsequent rerun, so a bare `st.button` would gate this identically.
    #
    # The form earns its place on selection, not on gating: it batches a
    # multi-file drop into a single submit instead of a rerun per file.
    #
    # `type` is derived from SUPPORTED_SUFFIXES rather than restated, so a loader
    # added for a new suffix is offered here without a second edit. It only
    # filters the browser's file picker, which Streamlit documents as
    # best-effort — though it also re-checks the extension server-side, so
    # save_upload's own suffix check is the third layer rather than the second.
    with st.form("add-documents", border=False):
        uploads = st.file_uploader(
            "Add documents",
            type=sorted(SUPPORTED_SUFFIXES),
            accept_multiple_files=True,
            key="uploads",
            help=(
                "Files are saved into the data directory and the index is "
                "rebuilt. A file with an existing name replaces it."
            ),
        )
        submitted = st.form_submit_button(
            "Add to index", icon=":material/upload:", key="add-documents-submit"
        )
    if submitted and uploads:
        _add_documents(cfg, uploads)
    elif submitted:
        st.warning("Choose at least one file first.", icon=":material/error:")

    st.caption(
        "Or edit `data/` directly and run `rag ingest` — the app reloads the "
        "new index either way."
    )
    # No st.rerun() needed: the click already triggered this run, and the sidebar
    # executes above the replay loop, so falling through renders the cleared UI.
    # Keyed because a second sidebar button now exists and index order is not a
    # stable way to name either of them.
    if st.button("Clear conversation", icon=":material/delete:", key="clear-chat"):
        st.session_state.messages = []

# Build the pipeline, turning setup errors (no index yet, missing API key) into
# a clear on-screen message instead of a stack trace. Below the sidebar so that
# an upload made on this run is already indexed: index_version() is read here,
# after the rebuild bumped it, so the cached pipeline misses and reloads.
try:
    pipeline = load_pipeline(cfg, index_version(cfg))
except (FileNotFoundError, RuntimeError) as exc:
    # FileNotFoundError: no/empty index. RuntimeError: missing API key.
    # One callout, not an error stacked on an info: "Then reload this page" is a
    # continuation of the error, meaningless on its own. Left generic because it
    # covers both cases and each exception already names its own remedy; the
    # sidebar has rendered above this guard, so the uploader that resolves the
    # missing-index case is on screen to speak for itself.
    st.error(f"{exc}\n\nThen reload this page.", icon=":material/error:")
    st.stop()


def _error_reply(text: str) -> dict:
    """A stored assistant turn that failed rather than answered.

    Cites nothing, because an unanswered turn has no grounding to claim, and
    carries the flag the replay loop routes on.
    """
    return {"role": "assistant", "content": text, "sources": [], "error": True}


def _render_sources(excerpts: list[Excerpt]) -> None:
    """Show the passages, not only the names of the files they came from.

    A filename is not checkable evidence: it tells a reader which document to go
    and search, which is the work the citation was supposed to save them. The
    passage text is what lets them see whether a sentence came from the
    retrieved context or was invented around it -- the one thing a RAG interface
    exists to make checkable -- and `stream_answer` already handed it over.

    Unparsed, for the reason the user's question is: a chunk is a *fragment*,
    and the splitter does not pair code fences, so rendering one as Markdown
    lets a half-open fence swallow the rest of the very panel that exists to
    audit the answer. `st.code` would wrap nothing by default, which turns a
    PDF page -- extracted as one long line -- into a horizontal scrollbar.
    """
    if excerpts:
        with st.expander(
            f"Sources ({len(excerpts)} retrieved passages)",
            icon=":material/description:",
        ):
            for rank, excerpt in enumerate(excerpts, start=1):
                st.caption(f"{rank}. `{excerpt['source']}`")
                st.text(excerpt["text"])


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
        # Names the step in flight, so a failure is labelled for the phase it hit
        # rather than always blamed on generation: embedding is an API call now,
        # so retrieval can raise a provider error before generation ever starts.
        phase = "Retrieval"
        try:
            # stream_answer() has finished retrieving when it returns but has
            # not started generating, so the spinner covers exactly the step
            # with nothing to show.
            with st.spinner("Retrieving context..."):
                docs, chunks = pipeline.stream_answer(question)
            phase = "Generation"
            answer_text = st.write_stream(chunks)
            # write_stream already rendered this; it is re-read only to store
            # it, and joined because the declared return is list[Any] | str —
            # a str is itself an iterable of str, so one join covers both. Kept
            # inside the try so a non-str list raises into the handler below
            # rather than as a crash page.
            text = "".join(answer_text)
            sources = source_excerpts(docs)
            _render_sources(sources)
            reply = {"role": "assistant", "content": text, "sources": sources}
        except Exception as exc:
            # Any failure (bad/expired key, rate limit, empty response) — show
            # it in the chat instead of a raw traceback. The `error` flag keeps
            # the stored text free of presentation, so the replay above can
            # route it back through st.error rather than rendering a failure as
            # if it were an answer.
            error_msg = f"{phase} failed: {exc}"
            st.error(error_msg, icon=":material/error:")
            reply = _error_reply(error_msg)
        finally:
            # Both halves land together, so no exit path — success, handled
            # failure, or an interruption that unwinds past the handler — can
            # leave a question in the history with no answer under it.
            st.session_state.messages.extend(
                [{"role": "user", "content": question}, reply]
            )
