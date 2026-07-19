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

CANNED = "Chunks overlap to preserve context across boundaries. (rag_concepts.md)"


@pytest.fixture
def app(settings, fake_embeddings, monkeypatch) -> AppTest:
    """An `AppTest` over `app.py`, wired to fakes and a freshly ingested index."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    # app.py builds its own Settings.from_env(), so the environment is how the
    # fixture's temp index gets through to it.
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir))
    monkeypatch.setenv("PERSIST_DIR", str(settings.persist_dir))
    monkeypatch.setenv("COLLECTION_NAME", settings.collection_name)
    monkeypatch.setenv("CHUNK_SIZE", str(settings.chunk_size))
    monkeypatch.setenv("CHUNK_OVERLAP", str(settings.chunk_overlap))
    monkeypatch.setenv("RETRIEVAL_K", str(settings.retrieval_k))
    # RAGPipeline fast-fails on a missing key whenever it builds the model
    # itself, which it does here. The factory is patched below, so nothing ever
    # authenticates with this value.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")

    monkeypatch.setattr(ingest_mod, "build_embeddings", lambda _s: fake_embeddings)
    monkeypatch.setattr(
        pipeline_mod,
        "build_chat_model",
        lambda _s: FakeListChatModel(responses=[CANNED]),
    )

    # load_pipeline is cached across script runs and its key ignores _settings,
    # so a previous test's pipeline would otherwise answer this one.
    st.cache_resource.clear()
    return AppTest.from_file(str(APP), default_timeout=60)


def _fail_mid_stream(exc: BaseException):
    """A `stream_answer` that emits, then fails — as a real one would.

    Failing partway rather than at the call is the honest shape: `.stream()` is
    lazy, so a provider error lands while the frontend is already rendering.
    """

    def stream(_self, _question, _docs):
        yield "a partial ans"
        raise exc

    return stream


def _roles(at: AppTest) -> list[str]:
    return [m["role"] for m in at.session_state["messages"]]


def test_app_answers_a_question_with_sources(app):
    at = app.run()
    assert not at.exception, [e.value for e in at.exception]

    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert not at.exception, [e.value for e in at.exception]

    user, assistant = at.session_state["messages"]
    assert user == {"role": "user", "content": "Why do chunks overlap?"}
    # A str, not write_stream's list form: replay feeds this to st.markdown.
    assert assistant["content"] == CANNED
    assert assistant["sources"], "answer stored without the docs that grounded it"
    assert "error" not in assistant


def test_user_turn_is_echoed_unparsed(app):
    """A question containing Markdown must come back as typed, not rendered."""
    at = app.run()
    at.chat_input[0].set_value("what about `snake_case` and # headings?").run()
    at.run()  # replay from session_state, which is the path that used to differ

    assert "what about `snake_case` and # headings?" in [t.value for t in at.text]


def test_failed_generation_is_recorded_as_an_error_turn(app, monkeypatch):
    """A failure must replay as an error, not as an ordinary answer."""
    monkeypatch.setattr(
        pipeline_mod.RAGPipeline,
        "stream_answer",
        _fail_mid_stream(RuntimeError("Claude API request failed: rate limited")),
    )
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


def test_an_interrupted_run_still_pairs_the_turn(app, monkeypatch):
    """The regression test for a question left permanently unanswered.

    Streamlit's toolbar Stop raises `StopException`, which derives from
    `BaseException` — so `except Exception` in the frontend never sees it and the
    script is torn down mid-answer. The user half of the turn is appended before
    generation starts, so without the `finally` that commits both halves
    together, session_state keeps a question with nothing under it, on every
    later rerun, with no code path that ever fills it in.
    """
    monkeypatch.setattr(
        pipeline_mod.RAGPipeline,
        "stream_answer",
        _fail_mid_stream(StopException("user pressed stop")),
    )
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()

    assert _roles(at) == ["user", "assistant"], (
        "the run was torn down mid-answer and left an unpaired turn"
    )
    reply = at.session_state["messages"][1]
    assert reply["error"] is True
    assert "Interrupted" in reply["content"]


def test_every_turn_stays_paired_across_mixed_outcomes(app, monkeypatch):
    """Success, failure and interruption in one history, still strictly paired.

    Pins the invariant itself rather than one path: whatever happens to a turn,
    a user message is always immediately followed by an assistant message.
    """
    at = app.run()
    at.chat_input[0].set_value("first").run()

    monkeypatch.setattr(
        pipeline_mod.RAGPipeline,
        "stream_answer",
        _fail_mid_stream(RuntimeError("Claude API request failed: boom")),
    )
    at.chat_input[0].set_value("second").run()

    monkeypatch.setattr(
        pipeline_mod.RAGPipeline,
        "stream_answer",
        _fail_mid_stream(StopException("stop")),
    )
    at.chat_input[0].set_value("third").run()

    assert _roles(at) == ["user", "assistant"] * 3
    contents = [m["content"] for m in at.session_state["messages"]]
    assert contents[0::2] == ["first", "second", "third"]


def test_clear_conversation_empties_the_history(app):
    at = app.run()
    at.chat_input[0].set_value("Why do chunks overlap?").run()
    assert at.session_state["messages"]

    at.sidebar.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["messages"] == []


def test_missing_index_is_reported_not_raised(app, settings, monkeypatch):
    """A setup failure must land in the caught union and render as a message."""
    monkeypatch.setenv("PERSIST_DIR", str(settings.persist_dir / "no-such-index"))
    st.cache_resource.clear()

    at = app.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("No index found at" in e.value for e in at.error)
    assert not at.chat_input, "the app must stop before offering an input"
