"""Tests for the Streamlit frontend, run headless via ``streamlit.testing``.

`app.py` cannot take injected fakes the way `ingest()` and `RAGPipeline` do — it
is a script, not a function. But it reaches the embedding model, the reranker
and the chat model only through `build_embeddings()`, `build_reranker()` and
`build_chat_model()`, which the architecture already requires of every caller,
and all three are resolved as module globals at call time. Patching them there
is the same dependency-injection seam entering by a different door, and it is
what keeps these tests inside the suite's guarantees: no model loaded, no MLX,
no socket.

What earns these tests their runtime is the turn-pairing invariant. Every other
guarantee in this repo is about a function's return value, which an ordinary
test can assert directly. This one is about what `st.session_state` looks like
*after* the script has been torn down mid-run, and nothing below the frontend
can observe that.
"""

from __future__ import annotations

import gc
import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest
import streamlit as st
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.language_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from streamlit.testing.v1 import AppTest

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline import tracing as tracing_mod
from rag_pipeline.mlx_models import MLXChatModel

APP = Path(__file__).resolve().parent.parent / "app.py"

_ZEBRA = "Zebras are striped equids from Africa."


@pytest.fixture
def app(wired_env, fake_embeddings) -> AppTest:
    """An `AppTest` over `app.py`, wired to fakes and a freshly ingested index.

    `wired_env` supplies everything both frontends need; what is left here is
    what only a Streamlit script does.
    """
    ingest_mod.ingest(wired_env, embeddings=fake_embeddings)

    # load_pipeline is cached across script runs and its key ignores _settings,
    # so a previous test's pipeline would otherwise answer this one.
    st.cache_resource.clear()
    return AppTest.from_file(str(APP), default_timeout=60)


def _roles(at: AppTest) -> list[str]:
    return [m["role"] for m in at.session_state["messages"]]


def test_app_answers_a_question_with_sources(app, canned_answer):
    at = app.run()
    assert not at.exception, [e.value for e in at.exception]

    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    user, assistant = at.session_state["messages"]
    assert user == {"role": "user", "content": "Why do chunks overlap?"}
    # A str, not write_stream's list form: replay feeds this to st.markdown.
    # Whole, too: the fake streams a piece per character, and the first one is
    # pulled early (under the spinner that covers the wait for it), so an answer
    # missing its opening character is that piece dropped on the way to the page.
    assert assistant["content"] == canned_answer
    assert assistant["sources"], "answer stored without the docs that grounded it"
    assert "error" not in assistant


def test_the_stored_sources_carry_the_passage_text_and_survive_replay(app):
    """A citation the reader cannot check is the failure this panel exists for.

    Asserted against the retrieved chunk rather than a literal, so it stays true
    if the fixture corpus changes -- and re-run afterwards, because the passage
    lives in session_state and a replayed turn that dropped it would still look
    correct on the run that produced it.
    """
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()

    _, assistant = at.session_state["messages"]
    excerpt = assistant["sources"][0]
    assert excerpt["source"].endswith((".md", ".txt"))
    assert excerpt["text"].strip(), "a passage stored with no text cites nothing"

    at.run()  # replay from session_state, the path a rerun takes
    assert at.session_state["messages"][1]["sources"] == assistant["sources"]
    assert excerpt["text"] in [t.value for t in at.text]


def test_a_retrieved_passage_is_shown_unparsed(app):
    """The corpus is Markdown, and a chunk is a fragment of it.

    Parsed, a passage's own `#` heading would render as a heading inside the
    chat and an unpaired code fence could swallow the rest of the panel -- so
    the audit surface would be formatted by the documents it exists to audit.
    """
    at = app.run()
    at.chat_input[0].set_value("Tell me about apples.").run()
    at.run()

    passages = [t.value for t in at.text if "Alpha" in t.value]
    assert passages, "the retrieved passage was not rendered as text"
    assert "# Alpha" in passages[0], "the heading was parsed instead of shown"


def test_the_passages_are_labelled_as_retrieved_not_as_sources(app, monkeypatch):
    """The panel says what the passages are, not that the answer used them.

    Asserted under a refusal, the case the label exists for: the passages are
    still shown -- they are how a reader checks that the documents really lack
    the answer -- but a panel titled "Sources" would credit them with grounding
    an answer that declined to use them.
    """
    monkeypatch.setattr(
        pipeline_mod,
        "build_chat_model",
        lambda _s: FakeListChatModel(
            responses=["I don't know based on the provided documents."]
        ),
    )
    at = app.run()
    at.chat_input[0].set_value("What is the capital of Australia?").run()
    assert not at.exception, [e.value for e in at.exception]

    _, assistant = at.session_state["messages"]
    labels = [panel.label for panel in at.get("status")]
    assert labels == [f"Retrieved passages ({len(assistant['sources'])})"]


def test_user_turn_is_echoed_unparsed(app):
    """A question containing Markdown must come back as typed, not rendered."""
    at = app.run()
    at.chat_input[0].set_value("what about `snake_case` and # headings?").run()
    at.run()  # replay from session_state, which is the path that used to differ

    assert "what about `snake_case` and # headings?" in [t.value for t in at.text]


def test_failed_generation_is_recorded_as_an_error_turn(app, fail_mid_stream):
    """A failure must replay as an error, not as an ordinary answer."""
    fail_mid_stream(
        RuntimeError("Generation with the chat model failed: out of memory")
    )
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    assert _roles(at) == ["user", "assistant"]
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert "out of memory" in reply["content"]

    at.run()  # replayed, it must still render through st.error
    assert any("out of memory" in e.value for e in at.error)
    assert not any("out of memory" in m.value for m in at.markdown)


def test_an_interrupted_run_still_pairs_the_turn(app, fail_mid_stream):
    """The regression test for a question left permanently unanswered.

    Streamlit's toolbar Stop raises `StopException`, which derives from
    `BaseException` — so `except Exception` in the frontend never sees it and the
    script is torn down mid-answer. The user half of the turn is appended before
    generation starts, so without the `finally` that commits both halves
    together, session_state keeps a question with nothing under it, on every
    later rerun, with no code path that ever fills it in.
    """
    fail_mid_stream(StopException("user pressed stop"))
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()

    assert _roles(at) == ["user", "assistant"], (
        "the run was torn down mid-answer and left an unpaired turn"
    )
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert "Interrupted" in reply["content"]


def _stop_the_run() -> None:
    """Press the toolbar Stop, from inside the running script.

    The call AppSession makes for the button (and for a widget clicked while an
    answer streams): the runner is asked to stop, and raises StopException at
    the script's next Streamlit call -- then stays stopped, so every later one
    raises again.
    """
    ctx = get_script_run_ctx()
    assert ctx is not None
    assert ctx.script_requests is not None
    ctx.script_requests.request_stop()


def test_a_real_stop_releases_the_model_and_keeps_the_turn(
    app, fake_mlx, model_dir, monkeypatch
):
    """The Stop button as Streamlit delivers it, against the real local model.

    Unlike `fail_mid_stream`, which raises from inside generation, the button
    asks the runner to stop; the exception arrives inside write_stream, with the
    answer stream suspended mid-generation. Two things then rest on app.py:

    - The stream is *closed*, so the model stops and releases the process-wide
      generation lock there and then. Merely dropped, it lives on as a global of
      the script module, which Streamlit keeps after the run, and the next
      question -- from any session -- waits on the lock for good. The garbage
      collector is off for the stopped run, so nothing else can release it.
    - The turn is still stored as a pair. The runner stays stopped, so reading
      st.session_state in the `finally` would raise again and drop both halves.
    """
    fake_mlx.pieces = [f"word{i} " for i in range(20)]
    fake_mlx.on_piece = lambda i: _stop_the_run() if i == 3 else None
    chat = MLXChatModel(model_id=model_dir, max_tokens=50)
    monkeypatch.setattr(pipeline_mod, "build_chat_model", lambda _s: chat)
    at = app.run()

    gc.disable()
    try:
        at.chat_input[0].set_value("Why do chunks overlap?").run()
        held = mlx_models._GENERATION_LOCK.locked()
    finally:
        gc.enable()
        if mlx_models._GENERATION_LOCK.locked():
            # So a failure here fails this test, not every later one that waits.
            mlx_models._GENERATION_LOCK.release()

    assert not held, "the stopped answer kept the generation lock"
    assert fake_mlx.pieces_generated < len(fake_mlx.pieces), (
        "the model generated on after the Stop"
    )
    assert _roles(at) == ["user", "assistant"], "the stopped turn was not stored"
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert "Interrupted" in reply["content"]

    fake_mlx.on_piece = None
    at.chat_input[0].set_value("Tell me about apples.").run()
    assert not at.exception, [e.value for e in at.exception]
    assert _roles(at) == ["user", "assistant"] * 2
    assert "error" not in at.session_state["messages"][3]


class _StopDuringRerank(BaseDocumentCompressor):
    """A reranker during whose work the user presses Stop."""

    def compress_documents(self, documents, query, callbacks=None):
        _stop_the_run()
        return documents[:2]


def test_a_stop_during_retrieval_still_sends_the_questions_trace(
    app, spans, monkeypatch
):
    """The one Stop that lands with the answer stream returned but unread.

    The retrieval spinner's exit is itself a Streamlit call, so a Stop pressed
    while retrieval runs is raised there -- after stream_answer has handed the
    stream back, before anything reads it. No lock is held yet, but the stream
    holds the question's root span, which ends when the stream is closed:
    dropped, the question never reaches Phoenix, and its search and rerank
    arrive with a parent that never does. The garbage collector is off, as it
    is free to be, so nothing but an explicit close can end it.
    """
    monkeypatch.setattr(pipeline_mod, "build_reranker", lambda _s: _StopDuringRerank())
    at = app.run()

    gc.disable()
    try:
        at.chat_input[0].set_value("Why do chunks overlap?").run()
    finally:
        gc.enable()

    assert _roles(at) == ["user", "assistant"], "the stopped turn was not stored"
    assert "Interrupted" in at.session_state["messages"][1]["content"]
    roots = [span for span in spans.get_finished_spans() if span.name == "RAGPipeline"]
    assert len(roots) == 1, "the stopped question's trace was never ended"
    assert [event.name for event in roots[0].events] == ["stopped"]


def test_the_app_sets_up_tracing_from_its_own_settings(app, monkeypatch):
    """The app is one of the two places a process decides to trace, and it
    must decide from the Settings it built, like everything else it does."""
    seen: list[str] = []
    monkeypatch.setattr(
        tracing_mod,
        "setup_tracing",
        lambda settings: seen.append(settings.phoenix_collector_endpoint),
    )
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.test:6006")

    at = app.run()

    assert not at.exception, [e.value for e in at.exception]
    assert seen == ["http://phoenix.test:6006"]


def test_every_turn_stays_paired_across_mixed_outcomes(app, fail_mid_stream):
    """Success, failure and interruption in one history, still strictly paired.

    Pins the invariant itself rather than one path: whatever happens to a turn,
    a user message is always immediately followed by an assistant message.
    """
    at = app.run()
    at.chat_input[0].set_value("first").run()

    fail_mid_stream(RuntimeError("Generation with the chat model failed: boom"))
    at.chat_input[0].set_value("second").run()

    fail_mid_stream(StopException("stop"))
    at.chat_input[0].set_value("third").run()

    assert _roles(at) == ["user", "assistant"] * 3
    contents = [m["content"] for m in at.session_state["messages"]]
    assert contents[0::2] == ["first", "second", "third"]


def _fail_in_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    def retrieve(_self, _question):
        raise RuntimeError("Vector store request failed: boom")

    monkeypatch.setattr(pipeline_mod.RAGPipeline, "retrieve", retrieve)


def _fail_before_the_first_piece(monkeypatch: pytest.MonkeyPatch) -> None:
    # The model itself fails, as it would reading a prompt too large for memory,
    # so the real chain's laziness decides when the error surfaces.
    def refuse(_prompt):
        raise RuntimeError("Generation with the chat model failed: out of memory")

    monkeypatch.setattr(
        pipeline_mod, "build_chat_model", lambda _s: RunnableLambda(refuse)
    )


@pytest.mark.parametrize(
    ("arrange", "label"),
    [
        pytest.param(_fail_in_retrieval, "Retrieval failed", id="retrieval"),
        pytest.param(
            _fail_before_the_first_piece,
            "Generation failed",
            id="generation-before-the-first-piece",
        ),
    ],
)
def test_a_failure_is_labelled_with_the_phase_that_raised(
    app,
    monkeypatch,
    arrange: Callable[[pytest.MonkeyPatch], None],
    label: str,
):
    """An error turn names the step that failed, so it points at the right fix.

    The case that is easy to get wrong is generation failing before its first
    piece. The local model reads the whole prompt before it produces anything,
    so the app waits for that piece under a spinner of its own -- and a failure
    during the wait has streamed nothing yet. It is still generation's: labelled
    "Retrieval failed", it would send the user to check an index that worked.
    """
    arrange(monkeypatch)
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    assert _roles(at) == ["user", "assistant"]
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert reply["content"].startswith(f"{label}: "), reply["content"]


@pytest.mark.parametrize(
    ("exc", "logged"),
    [
        pytest.param(
            RuntimeError("Generation with the chat model failed: out of memory"),
            False,
            id="a-failure-mode",
        ),
        pytest.param(
            AttributeError("'NoneType' object has no attribute 'text'"),
            True,
            id="a-bug",
        ),
    ],
)
def test_only_a_failure_outside_the_union_logs_its_traceback(
    app, fail_mid_stream, caplog, exc: Exception, logged: bool
):
    """Either way the failure is shown in the chat and stored as an error turn.

    The union is what every failure mode is translated into, and its message
    names the fix, so a traceback would only bury it -- the CLI prints one line.
    Anything else is a bug whose message alone rarely locates it, and catching
    it to keep the chat usable must not also lose the traceback that Streamlit
    would have logged for it uncaught.
    """
    fail_mid_stream(exc)
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    assert _roles(at) == ["user", "assistant"]
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert reply["content"] == f"Generation failed: {exc}"
    logged_with_traceback = [r.exc_info[1] for r in caplog.records if r.exc_info]
    assert logged_with_traceback == ([exc] if logged else [])


def test_an_empty_answer_is_not_stored_as_a_grounded_turn(app, monkeypatch):
    """Whitespace-only generation must not look like a cited answer.

    Stored as-is it would render a blank assistant bubble above a populated
    panel of passages — the strongest possible claim of grounding attached to no
    content at all. The model is faked rather than `_generate`, so the guard
    under test is the real one in the pipeline.
    """
    monkeypatch.setattr(
        pipeline_mod,
        "build_chat_model",
        lambda _s: FakeListChatModel(responses=["   "]),
    )

    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    assert _roles(at) == ["user", "assistant"]
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert reply["sources"] == [], "an empty answer must cite nothing"
    assert "empty answer" in reply["content"]


def test_clear_conversation_empties_the_history(app):
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert at.session_state["messages"]

    # By key, not by index: the sidebar holds the upload form's submit button
    # too, so position is no longer a stable way to name either of them.
    at.sidebar.button(key="clear-chat").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["messages"] == []


# --- adding documents from the sidebar ---------------------------------------


def _upload(at: AppTest, *files: tuple[str, bytes, str]) -> AppTest:
    """Attach `files` to the uploader and submit the form, as a user would."""
    at.sidebar.file_uploader(key="uploads").set_value(list(files))
    return at.sidebar.button(key="add-documents-submit").click().run()


def test_an_uploaded_document_is_indexed_and_answerable(app, wired_env):
    """The whole point: a file added in the browser reaches retrieval.

    Asserted through a question rather than by checking that the file exists,
    because writing it into `data/` is only half the job — the index is what
    answers, and it is rebuilt on this same run rather than after a reload.
    """
    at = app.run()

    at = _upload(at, ("zebra.md", _ZEBRA.encode(), "text/x-md"))
    assert not at.exception, [e.value for e in at.exception]
    assert (wired_env.data_dir / "zebra.md").is_file()

    at.chat_input[0].set_value(_ZEBRA).run()
    _, assistant = at.session_state["messages"]
    assert "zebra.md" in [e["source"] for e in assistant["sources"]], (
        "the uploaded document was saved but never reached retrieval"
    )


# The two states with nothing to query yet: a fresh checkout, with no persist
# dir at all, and a persist dir holding no collection of the configured name --
# never ingested into, or a COLLECTION_NAME that differs from the one that was.
NO_INDEX = [
    pytest.param(
        lambda s: ("PERSIST_DIR", str(s.persist_dir / "no-such-index")),
        "No index found at",
        id="no-persist-dir",
    ),
    pytest.param(
        lambda s: ("COLLECTION_NAME", "never_ingested"),
        "is empty",
        id="collection-never-ingested",
    ),
]


@pytest.mark.parametrize(("point_at", "message"), NO_INDEX)
def test_documents_can_be_added_before_any_index_exists(
    app, settings, monkeypatch, point_at, message
):
    """The bootstrap case, and the reason the sidebar renders above the guard.

    With no index the app stops before the chat input, which is exactly when a
    user needs the uploader most — so it must be on screen in the failed state,
    and using it must lift the app out of that state without a reload.
    """
    monkeypatch.setenv(*point_at(settings))
    st.cache_resource.clear()

    at = app.run()
    assert any(message in e.value for e in at.error), "not the no-index state"
    assert not at.chat_input, "the app must stop before offering an input"
    assert at.sidebar.file_uploader, "no way to fix an empty index from the app"

    at = _upload(at, ("first.md", b"The first document in a fresh index.", "text/x-md"))
    assert not at.exception, [e.value for e in at.exception]
    assert at.chat_input, "indexing the first document did not bring the app up"


def test_an_ordinary_rerun_does_not_reindex(app, monkeypatch):
    """Re-indexing must happen on the submit run and on no other.

    `st.file_uploader` re-reports its files on every rerun, and the enclosing
    form does not change that — without `clear_on_submit` the batch stays
    attached, so `uploads` is still truthy afterwards. What confines the work to
    one run is that a submit button is a trigger, reset to False after the run
    it was clicked on: this fails on `if uploads:` and survives the form being
    replaced by a bare button, which is the honest statement of what it pins.

    Without that guard every chat message would rebuild the whole index —
    correct output at absurd cost, and invisible in any assertion about what the
    app displays. Counted instead.
    """
    calls = []
    real_ingest = ingest_mod.ingest

    def counting_ingest(settings, embeddings=None):
        calls.append(settings.data_dir)
        return real_ingest(settings, embeddings=embeddings)

    monkeypatch.setattr(ingest_mod, "ingest", counting_ingest)

    at = app.run()
    at = _upload(at, ("extra.md", b"An extra uploaded document.", "text/x-md"))
    assert len(calls) == 1

    at.chat_input[0].set_value("Why do chunks overlap?").run()
    at.run()  # a plain rerun, with the file still sitting in the widget
    assert len(calls) == 1, "the uploaded file was re-indexed on a later rerun"


def _count_store_resets(monkeypatch: pytest.MonkeyPatch) -> list[None]:
    """Record every `reset_store_cache()` the app makes, still performing it."""
    calls: list[None] = []
    real_reset = ingest_mod.reset_store_cache

    def counting_reset() -> None:
        calls.append(None)
        real_reset()

    monkeypatch.setattr(ingest_mod, "reset_store_cache", counting_reset)
    return calls


def test_every_pipeline_build_starts_from_a_fresh_store_client(app, monkeypatch):
    """Each rebuild drops chromadb's client cache first, and a cache hit does not.

    chromadb shares one System per persist directory per process, and its
    vector view does not see writes another process makes afterwards. After a
    `rag ingest` in a terminal, `index_version()` (read fresh) already names the
    new corpus, so the pipeline is rebuilt -- but on the cached System it would
    search the old index. The stale view does not reproduce within one process
    (a System here sees writes made through any other), which is all this test
    has, so the reset is counted instead: one per build, none on a rerun the
    cache serves -- plus the one every ingest makes before it writes, since a
    writer must not work from a stale view either.
    """
    resets = _count_store_resets(monkeypatch)

    at = app.run()
    assert len(resets) == 1, "the first pipeline was built on a cached client"

    at.chat_input[0].set_value("Why do chunks overlap?").run()
    at.run()
    assert len(resets) == 1, "a rerun rebuilt the pipeline the cache already held"

    _upload(at, ("extra.md", b"An extra uploaded document.", "text/x-md"))
    # The upload's ingest, then the rebuild after it.
    assert len(resets) == 3, "the rebuild after an upload reused the cached client"


def test_a_corpus_that_changes_back_is_rebuilt_not_served_stale(
    app, wired_env, fake_embeddings, monkeypatch
):
    """One pipeline is cached, not two, so an old key is rebuilt, not revived.

    `index_version()` digests the corpus, so a corpus that changes and changes
    back -- a file uploaded, then deleted by hand and re-ingested from a
    terminal -- mints the first key again. A second cache slot would answer it
    with the pipeline built for that key before the last reset, on a chromadb
    System whose vector view misses the terminal's writes: it would search for
    the deleted file's chunks and fail on ids that no longer exist. With one
    slot the key misses and the pipeline is rebuilt on a fresh client.

    As with the reset itself, the stale view needs a second process to show,
    so the rebuild is what is counted. The answer is checked as well, for what
    the user must see either way: the removed file gone from retrieval.
    """
    resets = _count_store_resets(monkeypatch)
    at = app.run()
    at = _upload(at, ("zebra.md", _ZEBRA.encode(), "text/x-md"))

    # What a `rag ingest` from a terminal does after the file is deleted.
    (wired_env.data_dir / "zebra.md").unlink()
    ingest_mod.ingest(wired_env, embeddings=fake_embeddings)

    at.chat_input[0].set_value(_ZEBRA).run()
    assert not at.exception, [e.value for e in at.exception]

    _, assistant = at.session_state["messages"]
    assert "error" not in assistant, assistant["content"]
    assert "zebra.md" not in [e["source"] for e in assistant["sources"]], (
        "a removed document was still retrieved"
    )
    # A build, the upload's ingest and the rebuild after it, the terminal's
    # ingest, and the rebuild for the key the corpus came back to: one fewer
    # would be that last pipeline served from cache.
    assert len(resets) == 5, "the corpus's earlier pipeline was served from cache"


def test_submitting_with_no_file_is_reported_not_indexed(app, monkeypatch):
    """An empty submit must say so rather than silently rebuilding the index."""
    calls = []
    monkeypatch.setattr(ingest_mod, "ingest", lambda *_a, **_k: calls.append(1))

    at = app.run()
    at = at.sidebar.button(key="add-documents-submit").click().run()

    assert not at.exception, [e.value for e in at.exception]
    assert not calls, "an empty submit rebuilt the index"
    assert any("at least one file" in w.value for w in at.sidebar.warning)


def test_one_unsaveable_file_does_not_discard_the_rest_of_the_batch(app, wired_env):
    """A failed save must cost its own file and no other.

    The reachable trigger is OSError from the write — a name past the
    filesystem's length limit, a full disk — which genuinely varies within one
    batch, so aborting on the first would silently drop files the user watched
    upload. Injected at `save_upload` because the realistic causes are not
    arrangeable from a test, and what is under test is the loop around it.
    """
    real_save = ingest_mod.save_upload

    def failing_save(data_dir, filename, data):
        if filename == "broken.md":
            raise OSError("File name too long")
        return real_save(data_dir, filename, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ingest_mod, "save_upload", failing_save)
        at = app.run()
        at = _upload(
            at,
            ("broken.md", b"Never lands.", "text/x-md"),
            ("fine.md", b"A document that must survive its neighbour.", "text/x-md"),
        )

    assert not at.exception, [e.value for e in at.exception]
    assert (wired_env.data_dir / "fine.md").is_file(), "a good file was discarded"
    assert not (wired_env.data_dir / "broken.md").exists()
    assert any("too long" in w.value for w in at.sidebar.warning)
    assert any("fine.md" in s.value for s in at.sidebar.success)


def test_an_upload_with_no_text_is_not_reported_as_added(app, wired_env):
    """ "Added" must mean answerable.

    Ingest skips a file with no text -- a blank one, or a scanned PDF, which has
    no text layer -- without failing the run, so reporting every *saved* file as
    added promises answers the index can never give. The upload beside it shows
    the check is per file.
    """
    at = app.run()

    at = _upload(
        at,
        ("blank.md", b"   \n", "text/x-md"),
        ("fine.md", b"A document with real text in it.", "text/x-md"),
    )

    assert not at.exception, [e.value for e in at.exception]
    successes = [s.value for s in at.sidebar.success]
    assert any("fine.md" in s for s in successes)
    assert not any("blank.md" in s for s in successes), "a textless file was 'Added'"
    assert any("blank.md" in w.value for w in at.sidebar.warning)


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            ValueError("No readable documents found in data"), id="value-error"
        ),
        pytest.param(
            RuntimeError(
                "Cannot load 'm': the local models run on MLX, which needs Apple "
                "Silicon macOS."
            ),
            id="runtime-error",
        ),
        pytest.param(
            FileNotFoundError(
                "Model 'm' is not in the Hugging Face cache (or incomplete)."
            ),
            id="model-not-downloaded",
        ),
    ],
)
def test_a_failed_rebuild_is_reported_not_raised(app, monkeypatch, exc):
    """A failed index rebuild is reported, not raised: the files are on disk
    either way, so this is a failed *index*, not a failed upload — the user
    should re-run `rag ingest`, not re-upload.

    Parametrized over the exception because _add_documents must catch the whole
    frontend union. ValueError is the empty-corpus case; RuntimeError is MLX
    being unavailable, an embedding model that fails to load or run, or a
    translated store error; FileNotFoundError is an embedding model not yet in
    the Hugging Face cache. The historical (OSError, ValueError) catch would
    have let the RuntimeError escape straight through the sidebar.
    """

    def failing_ingest(_settings, embeddings=None):
        raise exc

    monkeypatch.setattr(ingest_mod, "ingest", failing_ingest)

    at = app.run()
    at = _upload(at, ("doc.md", b"Saved but not indexed.", "text/x-md"))

    assert not at.exception, [e.value for e in at.exception]
    assert any("Indexing failed" in e.value for e in at.sidebar.error)
    assert not at.sidebar.success


@pytest.mark.parametrize(
    ("var", "value"),
    [
        pytest.param("CHUNK_SIZE", "not-a-number", id="number"),
        # pathlib reports a `~user` with no home directory as a RuntimeError,
        # which this guard would let through as a crash page.
        pytest.param("PERSIST_DIR", "~no_such_user_xyz/chroma", id="path"),
    ],
)
def test_a_malformed_setting_stops_before_the_sidebar(app, monkeypatch, var, value):
    """The one setup failure nothing can proceed past, including the uploader.

    Every other guard renders the sidebar first so the uploader stays reachable.
    This one cannot: `Settings` is what tells the uploader where `data_dir` is,
    so there is nothing to offer and the script stops above it.
    """
    monkeypatch.setenv(var, value)
    st.cache_resource.clear()

    at = app.run()

    assert not at.exception, [e.value for e in at.exception]
    assert any(value in e.value for e in at.error), (
        "a malformed setting must render as a message quoting it, not a traceback"
    )
    assert not at.sidebar.file_uploader, "the app must stop before the sidebar"
    assert not at.chat_input


# Two loads of app.py, reported as JSON: what the first page load shows, and
# what a reload after it does.
_TWO_LOADS = """
import json, sys
from streamlit.testing.v1 import AppTest
loads = []
for _ in range(2):
    at = AppTest.from_file(sys.argv[1], default_timeout=60).run()
    loads.append({
        "exception": [e.value for e in at.exception],
        "error": [e.value for e in at.error],
        "uploader": len(at.sidebar.file_uploader),
        "chat_input": len(at.chat_input),
    })
print(json.dumps(loads))
"""


@pytest.mark.parametrize(
    ("var", "value", "message"),
    [
        # Refused by opentelemetry.sdk.trace as it is imported, which chromadb
        # does -- so with tracing off too.
        pytest.param(
            "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT",
            "abc",
            "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT",
            id="opentelemetry",
        ),
        # One of chromadb's own settings, which it validates as it is imported,
        # and names by its field.
        pytest.param(
            "CHROMA_SERVER_HTTP_PORT", "abc", "chroma_server_http_port", id="chromadb"
        ),
        # numpy's message is int()'s own, naming only the value -- and a failed
        # numpy import cannot be repeated, which is what the reload checks.
        pytest.param("NUMPY_MADVISE_HUGEPAGE", "abc", "'abc'", id="numpy"),
    ],
)
def test_a_variable_refused_at_import_stops_before_the_sidebar(
    fresh_interpreter, var, value, message
):
    """The same stop, for a malformed variable a library reads for itself.

    These are read once, as the pipeline's imports first load chromadb, and
    refused there with a ValueError. Imported above the settings guard, that
    was a traceback in place of the whole page on every load. Reloaded, it must
    stay the same message: the import is not tried again. The traceback is kept
    in the server's log, where the page no longer shows it. In a fresh
    interpreter, because this one imported everything long ago -- which is also
    why no test run in-process could see it.
    """
    result = fresh_interpreter(_TWO_LOADS, str(APP), **{var: value})
    assert result.returncode == 0, result.stderr
    loads = json.loads(result.stdout.splitlines()[-1])

    for load, shown in zip(("first load", "reload"), loads, strict=True):
        assert shown["exception"] == [], (load, shown["exception"])
        assert any(message in e for e in shown["error"]), (
            f"the {load} must show the refusal ({message}): {shown['error']}"
        )
        assert shown["uploader"] == 0, f"the {load} must stop before the sidebar"
        assert shown["chat_input"] == 0, load
    assert "Traceback" in result.stderr, "the server's log must keep the traceback"


def test_an_uploaded_name_cannot_escape_the_data_dir(app, wired_env, tmp_path):
    """The traversal defense, exercised through the widget that delivers it.

    Streamlit validates an uploaded filename's *extension* server-side
    (`enforce_filename_restriction`) and nothing else, so `../../escape.md`
    passes the widget's own checks and arrives at the app intact. That makes
    this the reachable half of `save_upload`'s guard rather than a theoretical
    one, and the reason it is tested here and not only as a unit.
    """
    at = app.run()

    at = _upload(at, ("../../escape.md", b"Escaped document.", "text/x-md"))

    assert not at.exception, [e.value for e in at.exception]
    assert (wired_env.data_dir / "escape.md").is_file()
    assert not (tmp_path / "escape.md").exists(), "wrote outside data_dir"
    assert not (tmp_path.parent / "escape.md").exists(), "wrote outside data_dir"


@pytest.mark.parametrize(("point_at", "message"), NO_INDEX)
def test_missing_index_is_reported_not_raised(
    app, settings, monkeypatch, point_at, message
):
    """A setup failure must land in the caught union and render as a message."""
    monkeypatch.setenv(*point_at(settings))
    st.cache_resource.clear()

    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(message in e.value for e in at.error)
    assert not at.chat_input, "the app must stop before offering an input"


def test_a_tracing_setup_failure_is_reported_below_the_sidebar(
    app, monkeypatch, undo_tracing
):
    """Tracing is set up on the pipeline-load path, whose handler catches
    FileNotFoundError and RuntimeError only. The tracing SDK refuses some
    malformed OTEL_* variables with a builtins ValueError -- which must arrive
    translated, as the message the handler shows, with the sidebar above it."""
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:9")
    # A batch larger than its queue, refused as setup_tracing builds the
    # processor. (An unknown compression, the trigger before OpenTelemetry
    # 1.45, has since only been logged.)
    monkeypatch.setenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "4096")

    at = app.run()

    assert not at.exception, [e.value for e in at.exception]
    assert any("OTEL_" in e.value for e in at.error)
    assert at.sidebar.button, "the sidebar must render above the error"
    assert not at.chat_input


def test_looking_for_a_missing_index_creates_nothing(app, settings, monkeypatch):
    """Reporting that there is no index must not leave an empty one behind.

    Opening a Chroma client creates its directory, and the app looks for the
    index on every rerun. A directory conjured that way would turn the next
    check's "No index found ... run `rag ingest`" into a misleading "is empty",
    and leave a stray store in whatever PERSIST_DIR a typo pointed at.
    """
    missing = settings.persist_dir / "no-such-index"
    monkeypatch.setenv("PERSIST_DIR", str(missing))
    st.cache_resource.clear()

    at = app.run()
    at.run()

    assert any("No index found at" in e.value for e in at.error)
    assert not missing.exists(), "looking for the index created its directory"


def test_a_broken_store_is_reported_on_every_rerun(app, monkeypatch, tmp_path):
    """A store that fails to open says so every time, never as a crash page.

    chromadb used to keep a store that failed to start cached, half-built, so
    the next rerun died on a builtins AttributeError instead of the store
    error -- the message and the traceback page taking turns, rerun by rerun.
    """
    broken = tmp_path / "broken-store"
    broken.mkdir()
    (broken / "chroma.sqlite3").write_bytes(b"not a database")
    monkeypatch.setenv("PERSIST_DIR", str(broken))
    st.cache_resource.clear()

    at = app
    for _ in range(3):
        at = at.run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Vector store request failed" in e.value for e in at.error)


def test_the_app_config_keeps_streamlit_local_and_quiet():
    """`.streamlit/config.toml` carries three settings the app depends on.

    Usage statistics off: Streamlit's front end otherwise reports to Streamlit's
    servers, and nothing about this app is meant to leave the machine. The file
    watcher off: it walks every loaded module on every run, and transformers'
    lazy modules (loaded with the tokenizers) make it log a traceback for each
    -- over a hundred per chat turn. The server on loopback: Streamlit otherwise
    listens on every interface, which puts the app -- no login, and an uploader
    that writes into data/ -- on the local network; and started headless with
    no address, it asks checkip.amazonaws.com for the machine's external IP
    address, to print it. None of the three is visible to a test run, which
    starts no server, so the file itself is what is checked.
    """
    config = tomllib.loads((APP.parent / ".streamlit" / "config.toml").read_text())

    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["fileWatcherType"] == "none"
    assert config["server"]["address"] == "127.0.0.1"
