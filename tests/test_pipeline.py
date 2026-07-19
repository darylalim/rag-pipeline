"""Tests for the query phase: helpers, guards, and retrieval round-trip.

Generation (the Claude call) is not exercised here — it requires a real API
key. These tests cover everything up to it: the source helpers, the setup
guards, and that an ingested chunk can be retrieved by its own text.
"""

from __future__ import annotations

import dataclasses

import anthropic
import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_pipeline import ingest as ingest_mod
from rag_pipeline.pipeline import RAGPipeline, format_docs, unique_sources


def test_unique_sources_dedupes_in_order():
    docs = [
        Document(page_content="1", metadata={"source": "b.md"}),
        Document(page_content="2", metadata={"source": "a.md"}),
        Document(page_content="3", metadata={"source": "b.md"}),
        Document(page_content="4", metadata={}),  # missing source -> "unknown"
    ]
    assert unique_sources(docs) == ["b.md", "a.md", "unknown"]


def test_format_docs_labels_each_source():
    docs = [
        Document(page_content="hello", metadata={"source": "a.md"}),
        Document(page_content="world", metadata={"source": "b.md"}),
    ]
    out = format_docs(docs)
    assert "[Source: a.md]" in out
    assert "hello" in out
    assert "[Source: b.md]" in out
    assert "world" in out


def test_pipeline_requires_index(settings, monkeypatch):
    # Key is present so we know it's the *index* check that fails, not the key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    with pytest.raises(FileNotFoundError):
        RAGPipeline(settings)  # nothing ingested yet


def test_pipeline_requires_api_key(settings, fake_embeddings, monkeypatch):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)  # index now exists
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        RAGPipeline(settings, embeddings=fake_embeddings)


def test_pipeline_rejects_empty_index(settings, fake_embeddings, monkeypatch):
    # persist_dir exists but nothing was ingested -> loud error, not silent empty.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        RAGPipeline(settings, embeddings=fake_embeddings)


def test_pipeline_rejects_mismatched_collection(settings, fake_embeddings, monkeypatch):
    import dataclasses

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # Ingests into settings.collection_name; we then query a different one.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    other = dataclasses.replace(settings, collection_name="different")
    with pytest.raises(FileNotFoundError):
        RAGPipeline(other, embeddings=fake_embeddings)


def test_retrieve_round_trip(settings, fake_embeddings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    # Recreate the exact chunk text that was stored, then query with it.
    docs = ingest_mod.load_documents(settings.data_dir)
    target = ingest_mod.split_documents(docs, settings)[0]

    pipeline = RAGPipeline(settings, embeddings=fake_embeddings)
    results = pipeline.retrieve(target.page_content)

    assert results, "expected at least one retrieved chunk"
    assert results[0].metadata["source"] == target.metadata["source"]


def test_answer_returns_model_output_and_sources(
    settings, fake_embeddings, monkeypatch
):
    # Injecting an llm skips building ChatAnthropic, so no real key is needed.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    canned = "Chunks overlap to preserve context across boundaries. (rag_concepts.md)"
    fake_llm = FakeListChatModel(responses=[canned])
    pipeline = RAGPipeline(settings, embeddings=fake_embeddings, llm=fake_llm)

    result = pipeline.answer("Why do chunks overlap?")

    assert result.text == canned
    assert result.sources, "expected grounding sources"
    assert all("source" in doc.metadata for doc in result.sources)


def test_stream_answer_agrees_with_answer(settings, fake_embeddings, monkeypatch):
    """The two shapes must not drift: `answer()` is a join over `stream_answer()`.

    Both frontends stream; `answer()` is what the tests above and any library
    caller use. Pinning them equal is what keeps the single-code-path refactor
    honest — a future `answer()` that stopped delegating would pass every other
    test in this file.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    canned = "Chunks overlap to preserve context across boundaries. (rag_concepts.md)"
    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=[canned]),
    )

    question = "Why do chunks overlap?"
    _docs, chunks = pipeline.stream_answer(question)
    streamed = "".join(chunks)

    assert streamed == canned
    assert streamed == pipeline.answer(question).text


def test_stream_answer_yields_incrementally(settings, fake_embeddings, monkeypatch):
    """Guards the point of streaming, which no other test would notice losing.

    Rewriting the generator as a single `yield self._chain.invoke(...)` removes
    token-by-token delivery entirely while still passing every other test here —
    the joined text is identical and the error translation still holds. Chunk
    count is the only observable that changes.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    canned = "Chunks overlap to preserve context across boundaries."
    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=[canned]),
    )

    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    pieces = list(chunks)

    assert len(pieces) > 1, "generation arrived as one piece — no longer streaming"
    assert "".join(pieces) == canned


def test_stream_answer_retrieves_before_generating(
    settings, fake_embeddings, monkeypatch
):
    """The docs must be ready on return; only generation stays lazy.

    Frontends put a spinner around the call and render sources from its first
    return value, so retrieval has to have happened by then. If it were deferred
    into the generator, the sources would be empty until the answer was consumed.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    pipeline = RAGPipeline(
        settings, embeddings=fake_embeddings, llm=FakeListChatModel(responses=["ok"])
    )
    docs, chunks = pipeline.stream_answer("Why do chunks overlap?")

    assert docs, "retrieval had not run by the time stream_answer returned"
    assert all("source" in doc.metadata for doc in docs)
    assert "".join(chunks) == "ok"


def test_answer_injects_retrieved_context_into_prompt(
    settings, fake_embeddings, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    captured: dict = {}

    def spy(prompt_value):
        # The chain feeds the rendered prompt into the model; capture it here.
        captured["messages"] = prompt_value.to_messages()
        return AIMessage(content="ok")

    pipeline = RAGPipeline(
        settings, embeddings=fake_embeddings, llm=RunnableLambda(spy)
    )
    result = pipeline.answer("Why do chunks overlap?")

    human = captured["messages"][-1].content
    assert "Question: Why do chunks overlap?" in human
    # Retrieved chunks are stuffed into the prompt as labeled context.
    assert "[Source:" in human
    assert result.sources[0].metadata["source"] in human


def test_collection_mismatch_is_caught_by_the_empty_collection_guard(
    settings, fake_embeddings, monkeypatch
):
    """Pins *which* guard rejects a COLLECTION_NAME mismatch, and that the index
    it rejected was not actually empty.

    Chroma's `get_or_create` hands back a silently-empty collection when the
    name doesn't match what was ingested, so without the guard every question is
    answered "I don't know". `raises(FileNotFoundError)` alone can't tell that
    guard apart from the `persist_dir.exists()` check that runs before it — a
    populated persist_dir plus `match=` does. The `reset_store_cache()` puts the
    re-open in the state a real `rag query` starts from, rather than letting
    chromadb's per-process client cache decide the outcome.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()

    mismatched = dataclasses.replace(settings, collection_name="not_the_ingested_one")
    assert mismatched.persist_dir.exists(), (
        "the index must exist for this to prove anything"
    )

    with pytest.raises(FileNotFoundError, match="is empty"):
        RAGPipeline(mismatched, embeddings=fake_embeddings)

    # And the correctly-named collection still retrieves: the failure above was
    # about the name, not a genuinely empty index.
    ingest_mod.reset_store_cache()
    assert RAGPipeline(settings, embeddings=fake_embeddings).retrieve("apples")


# The exception union both frontends catch (`cli.py`, `app.py`). A failure mode
# outside it escapes as a traceback in the CLI and a Streamlit crash page.
_FRONTEND_EXCEPTIONS = (FileNotFoundError, RuntimeError, ValueError)


def _fail_ingest_missing_data_dir(settings, fake_embeddings, monkeypatch, tmp_path):
    ingest_mod.ingest(
        dataclasses.replace(settings, data_dir=tmp_path / "no-such-dir"),
        embeddings=fake_embeddings,
    )


def _fail_ingest_empty_corpus(settings, fake_embeddings, monkeypatch, tmp_path):
    empty = tmp_path / "empty-data"
    empty.mkdir()
    ingest_mod.ingest(
        dataclasses.replace(settings, data_dir=empty), embeddings=fake_embeddings
    )


def _fail_pipeline_missing_index(settings, fake_embeddings, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    RAGPipeline(
        dataclasses.replace(settings, persist_dir=tmp_path / "no-such-index"),
        embeddings=fake_embeddings,
    )


def _fail_pipeline_missing_api_key(settings, fake_embeddings, monkeypatch, tmp_path):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    # Set-but-empty rather than delenv, matching how config.py's helpers treat a
    # blank var: both must read as "no key".
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    RAGPipeline(settings, embeddings=fake_embeddings)


def _fail_pipeline_empty_collection(settings, fake_embeddings, monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    RAGPipeline(settings, embeddings=fake_embeddings)


def _fail_answer_on_provider_error(settings, fake_embeddings, monkeypatch, tmp_path):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    def explode(_prompt_value):
        # A real provider failure (bad key, rate limit, network) surfaces as an
        # anthropic.APIError subclass; raise one offline instead of calling out.
        raise anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )

    pipeline = RAGPipeline(
        settings, embeddings=fake_embeddings, llm=RunnableLambda(explode)
    )
    pipeline.answer("Why do chunks overlap?")


def _fail_stream_answer_on_provider_error(
    settings, fake_embeddings, monkeypatch, tmp_path
):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    def explode(_prompt_value):
        raise anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )

    pipeline = RAGPipeline(
        settings, embeddings=fake_embeddings, llm=RunnableLambda(explode)
    )
    # Consumed to exhaustion: generation is lazy, so merely calling
    # stream_answer() raises nothing. This is the shape both frontends use, and
    # it is where the translation has to hold.
    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    list(chunks)


@pytest.mark.parametrize(
    ("failing_call", "expected_type", "expected_message"),
    [
        pytest.param(
            _fail_ingest_missing_data_dir,
            FileNotFoundError,
            "Data directory does not exist",
            id="ingest-missing-data-dir",
        ),
        pytest.param(
            _fail_ingest_empty_corpus,
            ValueError,
            "No readable documents found",
            id="ingest-empty-corpus",
        ),
        pytest.param(
            _fail_pipeline_missing_index,
            FileNotFoundError,
            "No index found at",
            id="pipeline-missing-index",
        ),
        pytest.param(
            _fail_pipeline_missing_api_key,
            RuntimeError,
            "ANTHROPIC_API_KEY is not set",
            id="pipeline-missing-api-key",
        ),
        pytest.param(
            _fail_pipeline_empty_collection,
            FileNotFoundError,
            "is empty",
            id="pipeline-empty-collection",
        ),
        pytest.param(
            _fail_answer_on_provider_error,
            RuntimeError,
            "Claude API request failed",
            id="answer-provider-error",
        ),
        pytest.param(
            _fail_stream_answer_on_provider_error,
            RuntimeError,
            "Claude API request failed",
            id="stream-answer-provider-error",
        ),
    ],
)
def test_failure_modes_stay_inside_the_frontend_exception_union(
    failing_call,
    expected_type,
    expected_message,
    settings,
    fake_embeddings,
    monkeypatch,
    tmp_path,
):
    """Every known failure path must land in `FileNotFoundError | RuntimeError |
    ValueError`, the union `cli.py` and `app.py` catch.

    Individual tests above already cover most of these one at a time; this one
    exists to make the *union* the thing under test, so adding a fourth type
    (or letting `anthropic.APIError` escape `answer()` untranslated, which
    would drag the Anthropic SDK into both frontends) fails here rather than at
    a user's terminal. `expected_type` is checked exactly, so a path can't drift
    to a different member of the union unnoticed.
    """
    with pytest.raises(_FRONTEND_EXCEPTIONS, match=expected_message) as excinfo:
        failing_call(settings, fake_embeddings, monkeypatch, tmp_path)

    assert type(excinfo.value) is expected_type
