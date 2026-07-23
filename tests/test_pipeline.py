"""Tests for the query phase: helpers, guards, retrieval, and generation.

Generation *is* exercised here, through an injected fake chat model rather than
a real one — so no API key and no network, per the injection seam described in
CLAUDE.md. That covers both shapes (`stream_answer()` and the `answer()` join
over it), that they cannot drift apart, that streaming stays incremental, and
that every generation failure lands in the union both frontends catch.
"""

from __future__ import annotations

import dataclasses

import anthropic
import httpx
import pytest
import voyageai
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_pipeline import ingest as ingest_mod
from rag_pipeline.pipeline import (
    RAGPipeline,
    build_chat_model,
    build_reranker,
    format_docs,
    source_excerpts,
    unique_sources,
)


def test_unique_sources_dedupes_in_order():
    docs = [
        Document(page_content="1", metadata={"source": "b.md"}),
        Document(page_content="2", metadata={"source": "a.md"}),
        Document(page_content="3", metadata={"source": "b.md"}),
        Document(page_content="4", metadata={}),  # missing source -> "unknown"
    ]
    assert unique_sources(docs) == ["b.md", "a.md", "unknown"]


def test_source_excerpts_keeps_retrieval_order_and_repeats():
    """The panel is a transcript of the prompt, so neither order nor repeats go.

    `format_docs` joins in list order, so reordering here would show a reader a
    prompt the model never saw -- and collapsing two chunks from one file would
    hide that a claim rests on two passages rather than one. Both are the
    opposite of `unique_sources`, which exists to shorten a citation line.
    """
    docs = [
        Document(page_content="first", metadata={"source": "b.md"}),
        Document(page_content="second", metadata={"source": "a.md"}),
        Document(page_content="third", metadata={"source": "b.md"}),
        Document(page_content="fourth", metadata={}),  # missing source
    ]

    assert source_excerpts(docs) == [
        {"source": "b.md", "text": "first"},
        {"source": "a.md", "text": "second"},
        {"source": "b.md", "text": "third"},
        {"source": "unknown", "text": "fourth"},
    ]


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


def test_pipeline_requires_index(settings, fake_embeddings, monkeypatch):
    # Key present, embeddings injected: so it is the *index* check that fails,
    # not the ANTHROPIC or VOYAGE key guard. Nothing ingested -> no vector index.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    with pytest.raises(FileNotFoundError, match="No queryable vector index"):
        RAGPipeline(settings, embeddings=fake_embeddings)  # nothing ingested yet


def test_pipeline_requires_api_key(settings, fake_embeddings, monkeypatch):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)  # index now exists
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        RAGPipeline(settings, embeddings=fake_embeddings)


def test_pipeline_rejects_empty_index(settings, fake_embeddings, monkeypatch):
    # A queryable index over zero of this pipeline's chunks -> loud error, not a
    # silent "I don't know" to every question. Build the index by ingesting, then
    # drop the chunks (keeping the index), reaching exactly that state.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()
    store = ingest_mod.open_store(settings, fake_embeddings)
    store.collection.delete_many({"content_hash": {"$exists": True}})
    ingest_mod.reset_store_cache()
    with pytest.raises(FileNotFoundError, match="is empty"):
        RAGPipeline(settings, embeddings=fake_embeddings)


def test_pipeline_rejects_mismatched_collection(settings, fake_embeddings, monkeypatch):
    import dataclasses

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # Ingests into settings.collection_name; we then query a different one.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    other = dataclasses.replace(settings, collection_name="different")
    with pytest.raises(FileNotFoundError):
        RAGPipeline(other, embeddings=fake_embeddings)


def test_retrieve_round_trip(settings, fake_embeddings, fake_reranker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    # Recreate the exact chunk text that was stored, then query with it.
    docs = ingest_mod.load_documents(settings.data_dir)
    target = ingest_mod.split_documents(docs, settings)[0]

    pipeline = RAGPipeline(settings, embeddings=fake_embeddings, reranker=fake_reranker)
    results = pipeline.retrieve(target.page_content)

    assert results, "expected at least one retrieved chunk"
    assert results[0].metadata["source"] == target.metadata["source"]


def test_answer_returns_model_output_and_sources(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
    # Injecting an llm skips building ChatAnthropic, so no real key is needed.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    canned = "Chunks overlap to preserve context across boundaries. (rag_concepts.md)"
    fake_llm = FakeListChatModel(responses=[canned])
    pipeline = RAGPipeline(
        settings, embeddings=fake_embeddings, llm=fake_llm, reranker=fake_reranker
    )

    result = pipeline.answer("Why do chunks overlap?")

    assert result.text == canned
    assert result.sources, "expected grounding sources"
    assert all("source" in doc.metadata for doc in result.sources)


def test_stream_answer_agrees_with_answer(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
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
        reranker=fake_reranker,
    )

    question = "Why do chunks overlap?"
    _docs, chunks = pipeline.stream_answer(question)
    streamed = "".join(chunks)

    assert streamed == canned
    assert streamed == pipeline.answer(question).text


def test_stream_answer_yields_incrementally(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
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
        reranker=fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    pieces = list(chunks)

    assert len(pieces) > 1, "generation arrived as one piece — no longer streaming"
    assert "".join(pieces) == canned


def test_stream_answer_retrieves_before_generating(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
    """The docs must be ready on return; only generation stays lazy.

    Frontends put a spinner around the call and render sources from its first
    return value, so retrieval has to have happened by then. If it were deferred
    into the generator, the sources would be empty until the answer was consumed.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["ok"]),
        reranker=fake_reranker,
    )
    docs, chunks = pipeline.stream_answer("Why do chunks overlap?")

    assert docs, "retrieval had not run by the time stream_answer returned"
    assert all("source" in doc.metadata for doc in docs)
    assert "".join(chunks) == "ok"


def test_answer_injects_retrieved_context_into_prompt(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    captured: dict = {}

    def spy(prompt_value):
        # The chain feeds the rendered prompt into the model; capture it here.
        captured["messages"] = prompt_value.to_messages()
        return AIMessage(content="ok")

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=RunnableLambda(spy),
        reranker=fake_reranker,
    )
    result = pipeline.answer("Why do chunks overlap?")

    human = captured["messages"][-1].content
    assert "Question: Why do chunks overlap?" in human
    # Retrieved chunks are stuffed into the prompt as labeled context.
    assert "[Source:" in human
    assert result.sources[0].metadata["source"] in human


def test_collection_mismatch_is_caught_by_the_index_guard(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
    """Pins *which* guard rejects a COLLECTION_NAME mismatch, and that the store
    it rejected was not actually broken.

    A wrong COLLECTION_NAME (or MONGODB_DB) points at a different physical
    namespace, which has no vector index — so unlike Chroma's silently-empty
    `get_or_create`, the mismatch surfaces as the *index* guard, not the
    empty-collection one. `match=` pins that, and the correctly-named collection
    still retrieving proves the rejection was about the name, not a genuine
    problem. The `reset_store_cache()` puts each re-open in the state a real
    `rag query` starts from.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()

    mismatched = dataclasses.replace(settings, collection_name="not_the_ingested_one")
    with pytest.raises(FileNotFoundError, match="No queryable vector index"):
        RAGPipeline(mismatched, embeddings=fake_embeddings)

    # And the correctly-named collection still retrieves: the failure above was
    # about the name, not a genuinely missing index.
    ingest_mod.reset_store_cache()
    assert RAGPipeline(
        settings, embeddings=fake_embeddings, reranker=fake_reranker
    ).retrieve("apples")


def test_retrieve_reranks_and_truncates(settings, fake_embeddings, monkeypatch):
    """retrieve() returns the *reranker's* order and count, not vector search's.

    Proves the two-stage wiring: a reranker that reverses its input must flip the
    order retrieve() hands back and cap it at retrieval_k. No other test here
    would notice reranking being dropped from retrieve() — the joined answer and
    the citations would look the same.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    class _Reversing(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            return list(reversed(documents))[: settings.retrieval_k]

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=_Reversing(),
    )
    candidates = pipeline._retriever.invoke("apples")
    reranked = pipeline.retrieve("apples")

    assert reranked == list(reversed(candidates))[: settings.retrieval_k]


def test_build_chat_model_sets_no_sampling_params(settings, monkeypatch):
    """Grounding comes from the retrieved context, so neither is set.

    The behavioral form of what used to be a text rule forbidding `temperature=`
    anywhere in this module. Reading them back off the constructed model is
    stronger: it holds however they might arrive — a keyword here, a dict
    splatted in, a default changed upstream — where matching the assignment only
    caught the one spelling.

    Left as None rather than tuned because some models reject sampling params
    outright: Opus 4.8 errors on a request carrying either.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # The real factory, not a fake: this asserts what production builds. No
    # socket opens — ChatAnthropic constructs its client lazily, and the key is
    # never sent anywhere.
    model = build_chat_model(settings)

    # Narrowed before the reads, so they are type-checked rather than resolved
    # on `BaseChatModel` at runtime — and because "production builds a
    # ChatAnthropic" is half of what this test is for.
    assert isinstance(model, ChatAnthropic)
    assert model.temperature is None
    assert model.top_p is None


# The exception union both frontends catch (`cli.py`, `app.py`). A failure mode
# outside it escapes as a traceback in the CLI and a Streamlit crash page.
_FRONTEND_EXCEPTIONS = (FileNotFoundError, RuntimeError, ValueError)


def _fail_ingest_missing_data_dir(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    ingest_mod.ingest(
        dataclasses.replace(settings, data_dir=tmp_path / "no-such-dir"),
        embeddings=fake_embeddings,
    )


def _fail_ingest_empty_corpus(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    empty = tmp_path / "empty-data"
    empty.mkdir()
    ingest_mod.ingest(
        dataclasses.replace(settings, data_dir=empty), embeddings=fake_embeddings
    )


def _fail_pipeline_missing_index(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # Nothing ingested into this unique namespace, so there is no vector index.
    RAGPipeline(settings, embeddings=fake_embeddings)


def _fail_pipeline_missing_api_key(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    # Set-but-empty rather than delenv, matching how config.py's helpers treat a
    # blank var: both must read as "no key".
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    RAGPipeline(settings, embeddings=fake_embeddings)


def _fail_pipeline_empty_collection(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # A queryable index over zero of this pipeline's chunks: ingest, then drop the
    # chunks but leave the index standing.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()
    store = ingest_mod.open_store(settings, fake_embeddings)
    store.collection.delete_many({"content_hash": {"$exists": True}})
    ingest_mod.reset_store_cache()
    RAGPipeline(settings, embeddings=fake_embeddings)


def _fail_pipeline_bad_mongo_uri(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A malformed MONGODB_URI must translate to RuntimeError, not ValueError, so
    # it lands in the branch app.py catches below its sidebar (keeping the
    # uploader reachable) rather than the one that stops the script above it.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    monkeypatch.setenv("MONGODB_URI", "not-a-valid://uri")
    ingest_mod.reset_store_cache()  # force the client to rebuild with the bad URI
    RAGPipeline(settings, embeddings=fake_embeddings)


def _ingested_pipeline(settings, fake_embeddings, llm, reranker) -> RAGPipeline:
    """A pipeline over a freshly ingested index, generating through `llm`."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    return RAGPipeline(settings, embeddings=fake_embeddings, llm=llm, reranker=reranker)


def _exploding_llm() -> RunnableLambda:
    """A model that fails the way a real provider does.

    A real failure (bad key, rate limit, network) surfaces as an
    anthropic.APIError subclass; raise one offline instead of calling out.
    """

    def explode(_prompt_value):
        raise anthropic.APIConnectionError(
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        )

    return RunnableLambda(explode)


def _exploding_embeddings() -> Embeddings:
    """Embeddings that fail the way Voyage's API does — the embedding-side analog
    of `_exploding_llm()`.

    A real failure (bad key, rate limit, network) surfaces as a voyageai.error
    subclass; raise one offline instead of calling out.
    """

    class _Exploding(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise voyageai.error.RateLimitError("rate limited")

        def embed_query(self, text: str) -> list[float]:
            raise voyageai.error.RateLimitError("rate limited")

    return _Exploding()


def _fail_answer_on_provider_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    pipeline = _ingested_pipeline(
        settings, fake_embeddings, _exploding_llm(), fake_reranker
    )
    pipeline.answer("Why do chunks overlap?")


def _fail_stream_answer_on_provider_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    pipeline = _ingested_pipeline(
        settings, fake_embeddings, _exploding_llm(), fake_reranker
    )
    # Consumed to exhaustion: generation is lazy, so merely calling
    # stream_answer() raises nothing. This is the shape both frontends use, and
    # it is where the translation has to hold.
    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    list(chunks)


def _fail_stream_answer_on_empty_response(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Guarded in the pipeline rather than per-frontend: an answer that arrives
    # empty would otherwise be presented with a full citation list by whichever
    # frontend forgot to check.
    pipeline = _ingested_pipeline(
        settings, fake_embeddings, FakeListChatModel(responses=["   "]), fake_reranker
    )
    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    list(chunks)


def _fail_build_embeddings_missing_voyage_key(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Guards before the client is built, so no socket is opened reaching it.
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    ingest_mod.build_embeddings(settings)


def _fail_build_reranker_missing_voyage_key(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Reranking uses Voyage too, so build_reranker guards VOYAGE_API_KEY the same
    # way build_embeddings does — before the client is built, so no socket opens.
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    build_reranker(settings)


def _fail_ingest_on_voyage_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    ingest_mod.ingest(settings, embeddings=_exploding_embeddings())


def _fail_ingest_dimension_change(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Atlas accepts a wrong-width vector on insert and only fails at query time,
    # so the guard is ingest's own probe: a model whose actual width disagrees
    # with EMBEDDING_DIMENSIONS raises a ValueError before anything is written.
    # (Under Chroma this surfaced as a dimension error at add time instead.)
    ingest_mod.ingest(
        dataclasses.replace(settings, embedding_dimensions=8),
        embeddings=DeterministicFakeEmbedding(size=16),
    )


def _fail_retrieve_on_voyage_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Embedding is a network call at query time too, so a provider error while
    # embedding the question translates the same way as at ingest.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    pipeline = RAGPipeline(
        settings,
        embeddings=_exploding_embeddings(),
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    )
    pipeline.retrieve("apples")


class _ExplodingReranker(BaseDocumentCompressor):
    """A reranker that fails the way Voyage's rerank API does — the rerank analog
    of `_exploding_embeddings()`.

    Reranking is a Voyage API call, so a real failure surfaces as a voyageai.error
    subclass; raise one offline instead of calling out.
    """

    def compress_documents(self, documents, query, callbacks=None):
        raise voyageai.error.RateLimitError("rate limited")


def _fail_retrieve_on_rerank_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Reranking is the third Voyage call, after embedding at ingest and query, so
    # a provider error while reranking translates the same way.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=_ExplodingReranker(),
    ).retrieve("apples")


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
            "No queryable vector index",
            id="pipeline-missing-index",
        ),
        pytest.param(
            _fail_pipeline_bad_mongo_uri,
            RuntimeError,
            "Vector store request failed",
            id="pipeline-mongo-connection-error",
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
        pytest.param(
            _fail_stream_answer_on_empty_response,
            RuntimeError,
            "empty answer",
            id="stream-answer-empty-response",
        ),
        pytest.param(
            _fail_build_embeddings_missing_voyage_key,
            RuntimeError,
            "VOYAGE_API_KEY is not set",
            id="build-embeddings-missing-voyage-key",
        ),
        pytest.param(
            _fail_build_reranker_missing_voyage_key,
            RuntimeError,
            "VOYAGE_API_KEY is not set",
            id="build-reranker-missing-voyage-key",
        ),
        pytest.param(
            _fail_ingest_on_voyage_error,
            RuntimeError,
            "Voyage API request failed",
            id="ingest-voyage-provider-error",
        ),
        pytest.param(
            _fail_ingest_dimension_change,
            ValueError,
            "EMBEDDING_DIMENSIONS",
            id="ingest-dimension-mismatch",
        ),
        pytest.param(
            _fail_retrieve_on_voyage_error,
            RuntimeError,
            "Voyage API request failed",
            id="retrieve-voyage-provider-error",
        ),
        pytest.param(
            _fail_retrieve_on_rerank_error,
            RuntimeError,
            "Voyage API request failed",
            id="retrieve-rerank-provider-error",
        ),
    ],
)
def test_failure_modes_stay_inside_the_frontend_exception_union(
    failing_call,
    expected_type,
    expected_message,
    settings,
    fake_embeddings,
    fake_reranker,
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
        failing_call(settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path)

    assert type(excinfo.value) is expected_type
