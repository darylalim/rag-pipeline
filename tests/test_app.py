"""Tests for the Streamlit frontend, run headless via ``streamlit.testing``.

`app.py` cannot take injected fakes the way `ingest()` and `RAGPipeline` do — it
is a script, not a function. But it reaches the embedding model and the chat
model only through `build_embeddings()` and `build_chat_model()`, which the
architecture already requires of every caller, and both are resolved as module
globals at call time. Patching them there is the same dependency-injection seam
entering by a different door, and it is what keeps these tests inside the
suite's offline guarantee: no model download, no API key, no socket.

What earns these tests their runtime is the turn-pairing invariant. Every other
guarantee in this repo is about a function's return value, which an ordinary
test can assert directly. This one is about what `st.session_state` looks like
*after* the script has been torn down mid-run, and nothing below the frontend
can observe that.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from langchain_core.language_models import FakeListChatModel
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from streamlit.testing.v1 import AppTest

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod

APP = Path(__file__).resolve().parent.parent / "app.py"


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


def test_user_turn_is_echoed_unparsed(app):
    """A question containing Markdown must come back as typed, not rendered."""
    at = app.run()
    at.chat_input[0].set_value("what about `snake_case` and # headings?").run()
    at.run()  # replay from session_state, which is the path that used to differ

    assert "what about `snake_case` and # headings?" in [t.value for t in at.text]


def test_failed_generation_is_recorded_as_an_error_turn(app, fail_mid_stream):
    """A failure must replay as an error, not as an ordinary answer."""
    fail_mid_stream(RuntimeError("Claude API request failed: rate limited"))
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    assert _roles(at) == ["user", "assistant"]
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert "rate limited" in reply["content"]

    at.run()  # replayed, it must still render through st.error
    assert any("rate limited" in e.value for e in at.error)
    assert not any("rate limited" in m.value for m in at.markdown)


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


def test_every_turn_stays_paired_across_mixed_outcomes(app, fail_mid_stream):
    """Success, failure and interruption in one history, still strictly paired.

    Pins the invariant itself rather than one path: whatever happens to a turn,
    a user message is always immediately followed by an assistant message.
    """
    at = app.run()
    at.chat_input[0].set_value("first").run()

    fail_mid_stream(RuntimeError("Claude API request failed: boom"))
    at.chat_input[0].set_value("second").run()

    fail_mid_stream(StopException("stop"))
    at.chat_input[0].set_value("third").run()

    assert _roles(at) == ["user", "assistant"] * 3
    contents = [m["content"] for m in at.session_state["messages"]]
    assert contents[0::2] == ["first", "second", "third"]


def test_an_empty_answer_is_not_stored_as_a_grounded_turn(app, monkeypatch):
    """Whitespace-only generation must not look like a cited answer.

    Stored as-is it would render a blank assistant bubble above a populated
    Sources expander — the strongest possible claim of grounding attached to no
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

    at = _upload(
        at, ("zebra.md", b"Zebras are striped equids from Africa.", "text/x-md")
    )
    assert not at.exception, [e.value for e in at.exception]
    assert (wired_env.data_dir / "zebra.md").is_file()

    at.chat_input[0].set_value("Zebras are striped equids from Africa.").run()
    _, assistant = at.session_state["messages"]
    assert "zebra.md" in [e["source"] for e in assistant["sources"]], (
        "the uploaded document was saved but never reached retrieval"
    )


def test_documents_can_be_added_before_any_index_exists(app, monkeypatch):
    """The bootstrap case, and the reason the sidebar renders above the guard.

    With no index the app stops before the chat input, which is exactly when a
    user needs the uploader most — so it must be on screen in the failed state,
    and using it must lift the app out of that state without a reload.
    """
    # A collection never ingested into (within the fixture's own database), so it
    # has no vector index — the "no index yet" state, without a persist dir.
    monkeypatch.setenv("COLLECTION_NAME", "fresh_uningested_collection")
    st.cache_resource.clear()

    at = app.run()
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


@pytest.mark.parametrize(
    "exc",
    [
        pytest.param(
            ValueError("No readable documents found in data"), id="value-error"
        ),
        pytest.param(RuntimeError("VOYAGE_API_KEY is not set."), id="runtime-error"),
    ],
)
def test_a_failed_rebuild_is_reported_not_raised(app, monkeypatch, exc):
    """A failed index rebuild is reported, not raised: the files are on disk
    either way, so this is a failed *index*, not a failed upload — the user
    should re-run `rag ingest`, not re-upload.

    Parametrized over the exception because _add_documents must catch the whole
    frontend union. ValueError is the empty-corpus case; RuntimeError is the one
    embedding-as-an-API-call newly introduced (missing VOYAGE_API_KEY, or a
    translated Voyage/index error), which the historical (OSError, ValueError)
    catch would have let escape straight through the sidebar.
    """

    def failing_ingest(_settings, embeddings=None):
        raise exc

    monkeypatch.setattr(ingest_mod, "ingest", failing_ingest)

    at = app.run()
    at = _upload(at, ("doc.md", b"Saved but not indexed.", "text/x-md"))

    assert not at.exception, [e.value for e in at.exception]
    assert any("Indexing failed" in e.value for e in at.sidebar.error)
    assert not at.sidebar.success


def test_a_malformed_numeric_setting_stops_before_the_sidebar(app, monkeypatch):
    """The one setup failure nothing can proceed past, including the uploader.

    Every other guard renders the sidebar first so the uploader stays reachable.
    This one cannot: `Settings` is what tells the uploader where `data_dir` is,
    so there is nothing to offer and the script stops above it.
    """
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")
    st.cache_resource.clear()

    at = app.run()

    assert not at.exception, [e.value for e in at.exception]
    assert at.error, "a malformed setting must render as a message, not a traceback"
    assert not at.sidebar.file_uploader, "the app must stop before the sidebar"
    assert not at.chat_input


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


def test_missing_index_is_reported_not_raised(app, monkeypatch):
    """A setup failure must land in the caught union and render as a message."""
    monkeypatch.setenv("COLLECTION_NAME", "never_ingested_collection")
    st.cache_resource.clear()

    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No queryable vector index" in e.value for e in at.error)
    assert not at.chat_input, "the app must stop before offering an input"
