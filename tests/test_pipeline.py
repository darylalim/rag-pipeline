"""Tests for the query phase: helpers, guards, retrieval, and generation.

Generation *is* exercised here, through an injected fake chat model rather than
a real one — so no API is called, per the injection seam described in
CLAUDE.md. That covers both shapes (`stream_answer()` and the
`answer()` join over it), that they cannot drift apart, that streaming stays
incremental, and that every failure lands in the union both frontends catch.

The index guards are tested with *no* models injected. conftest removes the API
keys, so a pipeline that built a model before checking the index would fail
with the factory's missing-key RuntimeError instead of the guard's
FileNotFoundError -- which makes each guard test a proof of ordering as well as
of the message.
"""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace

import pytest
import voyageai.error
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_voyageai import VoyageAIEmbeddings, VoyageAIRerank

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.claude_model import ClaudeChatModel
from rag_pipeline.config import Settings
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


# --- the index guards, before any model --------------------------------------


def test_pipeline_requires_index(settings):
    """A fresh checkout is told to run `rag ingest`, without loading a model first.

    No models injected (see the module docstring): the FileNotFoundError is also
    the proof that the check runs before the ~22 GB of local models would load.
    And looking must not create what it looked for -- an empty collection left
    behind would turn the next attempt's message into a different one.
    """
    with pytest.raises(FileNotFoundError, match="nothing was ever ingested"):
        RAGPipeline(settings)

    assert ingest_mod._collection(settings).database.list_collection_names() == []


def _emptied(settings: Settings, embeddings: Embeddings) -> None:
    """Ingested, then every chunk of ours deleted: the collection still exists."""
    ingest_mod.ingest(settings, embeddings=embeddings)
    ingest_mod._collection(settings).delete_many(ingest_mod.OWN_CHUNKS)


def _foreign_only(settings: Settings, embeddings: Embeddings) -> None:
    """A collection of the configured name holding only someone else's records."""
    ingest_mod._collection(settings).insert_one(
        {
            "_id": "theirs",
            "text": "Somebody else's notes.",
            "embedding": embeddings.embed_query("Somebody else's notes."),
            "source": "theirs.md",
        }
    )


@pytest.mark.parametrize(
    "arrange", [_emptied, _foreign_only], ids=["emptied", "foreign-only"]
)
def test_pipeline_rejects_empty_index(arrange, settings, fake_embeddings):
    """A collection with none of this pipeline's chunks is an error, not an index.

    Otherwise every question is answered "I don't know" off zero candidates.
    The foreign-only case is the scoping half: records someone else wrote into
    a shared collection do not pass for an index either, just as retrieval never
    returns them.
    """
    arrange(settings, fake_embeddings)

    with pytest.raises(FileNotFoundError, match="is empty"):
        RAGPipeline(settings)


def test_pipeline_rejects_mismatched_collection(settings, fake_embeddings):
    # Ingests into settings.collection_name; we then query a different one.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    other = dataclasses.replace(settings, collection_name="different")
    with pytest.raises(FileNotFoundError):
        RAGPipeline(other)


def test_collection_mismatch_is_caught_by_the_missing_collection_guard(
    settings, fake_embeddings, fake_reranker
):
    """Pins *which* guard rejects a COLLECTION_NAME mismatch, that rejecting it
    left no trace, and that the store it rejected was not actually broken.

    A wrong COLLECTION_NAME names a collection that does not exist, in a
    database that holds the right one, so `match=` pins the collection guard;
    and the lookup must not conjure the collection into existence -- the query
    path never creates one, or a second attempt would meet the empty-index
    message instead of the one naming the mismatch.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    mismatched = dataclasses.replace(settings, collection_name="not_the_ingested_one")
    with pytest.raises(FileNotFoundError, match="nothing was ever ingested"):
        RAGPipeline(mismatched)
    names = ingest_mod._collection(settings).database.list_collection_names()
    assert "not_the_ingested_one" not in names

    # And the correctly-named collection still retrieves: the failure above was
    # about the name, not a genuinely missing index.
    assert RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    ).retrieve("apples")


def test_a_complete_index_gets_past_the_guards_to_the_models(settings, fake_embeddings):
    """The control for the guard tests above.

    They inject no models and expect FileNotFoundError, which would prove
    nothing about ordering if a pipeline with no models injected never reached
    a model at all. Over a complete index it does: the first factory it reaches
    is Voyage's embedder, which conftest has left without a key.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY is not set"):
        RAGPipeline(settings)


def test_a_collection_that_vanishes_after_the_check_is_not_recreated(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
    """The query path creates nothing, whatever the guard saw.

    In production the embedding model loads between the check and the first
    search, so the collection can go in between; created afresh, it would be an
    empty one that answers every question "I don't know" -- and that a later
    check would pass for the index. The search finds nothing, and nothing is
    left behind for the next attempt's guard to find.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    check = pipeline_mod.require_index

    def check_then_vanish(s: Settings) -> None:
        check(s)
        ingest_mod._collection(s).drop()

    monkeypatch.setattr(pipeline_mod, "require_index", check_then_vanish)

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    )

    assert pipeline.retrieve("apples") == []
    names = ingest_mod._collection(settings).database.list_collection_names()
    assert settings.collection_name not in names
    with pytest.raises(FileNotFoundError, match="nothing was ever ingested"):
        ingest_mod.require_index(settings)


# --- retrieval ---------------------------------------------------------------


def test_retrieve_round_trip(settings, fake_embeddings, fake_reranker):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    # Recreate the exact chunk text that was stored, then query with it.
    docs = ingest_mod.load_documents(settings.data_dir)
    target = ingest_mod.split_documents(docs, settings)[0]

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    )
    results = pipeline.retrieve(target.page_content)

    assert results, "expected at least one retrieved chunk"
    assert results[0].metadata["source"] == target.metadata["source"]


def test_retrieve_reranks_and_truncates(settings, fake_embeddings, tmp_path):
    """retrieve() returns the *reranker's* order and count, over fetch_k candidates.

    Proves the two-stage wiring: vector search must hand the reranker a wide net
    -- fetch_k of the corpus, not retrieval_k and not all of it -- and a
    reranker that reverses its input must flip the order retrieve() hands back
    and cap it at retrieval_k. No other test here would notice reranking being
    dropped from retrieve(), or the net shrinking to what is finally kept: the
    joined answer and the citations would look the same.
    """
    corpus = tmp_path / "wide"
    corpus.mkdir()
    for i in range(8):
        (corpus / f"doc{i}.md").write_text(
            f"Document {i} about apples, orchard number {i}.\n", encoding="utf-8"
        )
    wide = dataclasses.replace(settings, data_dir=corpus, fetch_k=5)
    assert wide.retrieval_k < wide.fetch_k < 8, "the three sizes must be distinct"
    ingest_mod.ingest(wide, embeddings=fake_embeddings)

    seen: list[Document] = []

    class _Reversing(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            seen.extend(documents)
            return list(reversed(documents))[: wide.retrieval_k]

    pipeline = RAGPipeline(
        wide,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=_Reversing(),
    )
    reranked = pipeline.retrieve("apples")

    assert len(seen) == wide.fetch_k
    assert reranked == list(reversed(seen))[: wide.retrieval_k]


def test_retrieve_never_returns_a_foreign_document(
    settings, fake_embeddings, fake_reranker
):
    """A record this pipeline did not write is never retrieved, however well it
    matches -- so data sharing the collection never reaches a prompt or a
    citation.

    The foreign record carries a `source` and a `content_hash`, so looking like
    one of ours is not enough: only the marker every chunk of ours carries
    counts. Queried with its exact text, which an unscoped search ranks first
    (the control, checked first).
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    text = "Zebra facts that no ingested file contains."
    ingest_mod._collection(settings).insert_one(
        {
            "_id": "foreign",
            "text": text,
            "embedding": fake_embeddings.embed_query(text),
            "source": "zebra.md",
            "content_hash": "0" * 64,
        }
    )

    # The control first: an unscoped search must find the record, or the
    # assertion below would pass only because Atlas had not indexed it yet --
    # an insert becomes searchable asynchronously.
    unscoped = ingest_mod.open_store(settings, fake_embeddings)
    deadline = time.monotonic() + 60
    while unscoped.similarity_search(text, k=1)[0].metadata.get("source") != "zebra.md":
        assert time.monotonic() < deadline, "the foreign record never became searchable"
        time.sleep(0.5)

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    )
    retrieved = pipeline.retrieve(text)

    assert retrieved, "the pipeline's own chunks should still be retrieved"
    assert "zebra.md" not in unique_sources(retrieved)


# --- generation --------------------------------------------------------------


def test_answer_returns_model_output_and_sources(
    settings, fake_embeddings, fake_reranker
):
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


def test_stream_answer_agrees_with_answer(settings, fake_embeddings, fake_reranker):
    """The two shapes must not drift: `answer()` is a join over `stream_answer()`.

    Both frontends stream; `answer()` is what the tests above and any library
    caller use. Pinning them equal is what keeps the single-code-path refactor
    honest — a future `answer()` that stopped delegating would pass every other
    test in this file.
    """
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


def test_stream_answer_yields_incrementally(settings, fake_embeddings, fake_reranker):
    """Guards the point of streaming, which no other test would notice losing.

    Rewriting the generator as a single `yield self._chain.invoke(...)` removes
    token-by-token delivery entirely while still passing every other test here —
    the joined text is identical and the empty-answer guard still holds. Chunk
    count is the only observable that changes.
    """
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
    settings, fake_embeddings, fake_reranker
):
    """The docs must be ready on return; only generation stays lazy.

    Frontends put a spinner around the call and render sources from its first
    return value, so retrieval has to have happened by then. If it were deferred
    into the generator, the sources would be empty until the answer was consumed.
    """
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
    settings, fake_embeddings, fake_reranker
):
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


def test_a_model_that_returns_plain_text_is_streamed_as_is(
    settings, fake_embeddings, fake_reranker
):
    """`llm` is any runnable, and not every one returns message chunks.

    The pipeline extracts the text itself rather than through a
    `StrOutputParser` (see `test_closing_the_answer_stream_stops_the_local_model`
    for why), so it has to accept the plain strings a parser would have.
    """
    pipeline = _ingested_pipeline(
        settings,
        fake_embeddings,
        RunnableLambda(lambda _prompt: "A plain-text answer."),
        fake_reranker,
    )

    assert pipeline.answer("Why do chunks overlap?").text == "A plain-text answer."


def test_closing_the_answer_stream_ends_the_models_request(
    settings, fake_embeddings, fake_reranker, fake_claude
):
    """What the app's Stop relies on: closing the stream stops generation there.

    Through the real chat model, over a real client and a stand-in server,
    because the property is about the stack between them: a chain that drained
    the model when closed (a `StrOutputParser` on the end does) would keep
    Claude generating, and billing, to MAX_TOKENS. Closed after one piece, the
    answer's HTTP response must be closed then, with the rest never read.
    """
    fake_claude.pieces = [f"word{i} " for i in range(200)]
    pipeline = _ingested_pipeline(
        settings, fake_embeddings, fake_claude.chat(max_tokens=500), fake_reranker
    )

    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    assert next(chunks) == "word0 "
    chunks.close()

    (body,) = fake_claude.bodies
    assert body.closed, "the model's request was left open after the close"
    assert body.pieces_sent < len(fake_claude.pieces) / 2


def test_an_answer_cut_off_at_max_tokens_says_so(
    settings, fake_embeddings, fake_reranker, fake_claude
):
    """A truncated answer must not read as a finished one.

    The model reports why it stopped only in metadata that the text alone does
    not carry, so without this an answer cut off mid-sentence reaches both
    frontends looking complete. Said in the answer itself because that is the
    one channel both of them show. Through the real chat model, whose
    `max_tokens` stop reason is what the note keys off.
    """
    fake_claude.pieces = ["Chunks overlap ", "so that"]
    fake_claude.stop_reason = "max_tokens"
    pipeline = _ingested_pipeline(
        settings, fake_embeddings, fake_claude.chat(), fake_reranker
    )

    text = pipeline.answer("Why do chunks overlap?").text

    assert text.startswith("Chunks overlap so that")
    assert f"MAX_TOKENS={settings.max_tokens}" in text

    fake_claude.stop_reason = "end_turn"
    assert pipeline.answer("Why do chunks overlap?").text == "Chunks overlap so that"


def test_a_refused_question_is_an_error_not_an_empty_answer(
    settings, fake_embeddings, fake_reranker, fake_claude
):
    """A refusal arrives as a stop reason on an empty answer; read as text, it
    would be the "empty answer" error at best -- a frontend must be told why."""
    fake_claude.pieces, fake_claude.stop_reason = [], "refusal"
    pipeline = _ingested_pipeline(
        settings, fake_embeddings, fake_claude.chat(), fake_reranker
    )

    with pytest.raises(RuntimeError, match="declined to answer"):
        pipeline.answer("Why do chunks overlap?")


# --- the model factories -----------------------------------------------------


def test_build_chat_model_builds_claude_from_settings(settings, monkeypatch):
    """Production builds the Claude adapter from settings, with no sampling
    parameters -- Claude Sonnet 5.5 rejects them -- and the configured cap.

    Distinctive settings, so a factory that ignored them in favour of the
    defaults (or a literal) fails. The request it sends is the adapter's,
    tested in test_claude_model.py.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    configured = dataclasses.replace(settings, chat_model="claude-test", max_tokens=321)

    model = build_chat_model(configured)

    assert isinstance(model, ClaudeChatModel)
    assert (model.model, model.max_tokens) == ("claude-test", 321)
    sampling = {"temperature", "top_p", "top_k", "min_p", "sampler"}
    assert not sampling & set(type(model).model_fields)


def test_build_reranker_caps_at_retrieval_k(settings, monkeypatch):
    """The reranker's own cap is what makes retrieve() return retrieval_k passages.

    retrieve() does no slicing of its own, so a factory that passed FETCH_K (or
    a literal) here would put every candidate into each prompt and citation
    list -- and every other test would miss it, since they all inject a fake
    reranker. Built for real; constructing it makes no call.
    """
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    configured = dataclasses.replace(
        settings, rerank_model="rerank-test", retrieval_k=3, fetch_k=9
    )

    reranker = build_reranker(configured)

    assert isinstance(reranker, VoyageAIRerank)
    assert (reranker.model, reranker.top_k) == ("rerank-test", 3)


def test_the_voyage_clients_retry_and_time_out(settings, monkeypatch):
    """langchain-voyageai builds its clients with one attempt and no timeout: a
    rate-limited call (HTTP 429) would fail the question at once, and a stalled
    one would hang it forever. The factories replace them."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    embeddings = ingest_mod.build_embeddings(settings_at_a_voyage_width(settings))
    reranker = build_reranker(settings)
    assert isinstance(embeddings, VoyageAIEmbeddings)
    assert isinstance(reranker, VoyageAIRerank)

    for client in (embeddings._client, reranker.client):
        assert client.max_retries > 1
        assert client._params["request_timeout"] is not None


def test_the_reranker_returns_the_candidates_themselves_in_voyages_order(
    settings, monkeypatch
):
    """Voyage's order and scores, on the documents as retrieved -- ids kept.

    langchain-voyageai's own reranker rebuilds each document without its id,
    which is what ties a reranked chunk to the stored one in a trace, and adds
    a total_tokens key to its metadata. Voyage's response is faked at the one
    call the subclass makes, so this runs offline.
    """
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    reranker = build_reranker(settings)
    candidates = [
        Document(
            page_content=f"chunk {i}", id=f"a.md:{i}:h", metadata={"source": "a.md"}
        )
        for i in range(3)
    ]
    response = SimpleNamespace(
        results=[
            SimpleNamespace(index=2, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.4),
        ],
        total_tokens=42,
    )
    monkeypatch.setattr(type(reranker), "_rerank", lambda _self, _docs, _q: response)

    ranked = list(reranker.compress_documents(candidates, "which chunk?"))

    assert [d.id for d in ranked] == ["a.md:2:h", "a.md:0:h"]
    assert [d.metadata for d in ranked] == [
        {"source": "a.md", "relevance_score": 0.9},
        {"source": "a.md", "relevance_score": 0.4},
    ]
    assert candidates[2].metadata == {"source": "a.md"}, "a candidate was changed"


def settings_at_a_voyage_width(settings: Settings) -> Settings:
    """The fixture's width is the fake embedder's, which Voyage does not offer."""
    return dataclasses.replace(settings, embedding_dimensions=1024)


def test_build_embeddings_asks_for_the_configured_width(settings, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")

    embeddings = ingest_mod.build_embeddings(
        dataclasses.replace(
            settings, embedding_model="voyage-test", embedding_dimensions=512
        )
    )

    assert isinstance(embeddings, VoyageAIEmbeddings)
    assert (embeddings.model, embeddings.output_dimension) == ("voyage-test", 512)


# --- failures ----------------------------------------------------------------


class _StopScript(BaseException):
    """Stands in for Streamlit's StopException: how a rerun or the Stop button
    ends a script partway through an answer."""


def _failing_embeddings(error: BaseException) -> Embeddings:
    """Embeddings that fail as the local model does: its adapter has already
    turned the failure into `error`."""

    class _Failing(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise error

        def embed_query(self, text: str) -> list[float]:
            raise error

    return _Failing()


def _failing_reranker(error: BaseException) -> BaseDocumentCompressor:
    """The rerank analog of `_failing_embeddings()`."""

    class _Failing(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            raise error

    return _Failing()


def _failing_llm(error: BaseException) -> RunnableLambda:
    """The generation analog of `_failing_embeddings()`. Raised when the chain
    reaches the model, which for a stream is during consumption, not the call."""

    def fail(_prompt_value):
        raise error

    return RunnableLambda(fail)


@pytest.mark.parametrize("error_type", [RuntimeError, _StopScript])
@pytest.mark.parametrize("stage", ["embed", "rerank", "generate"])
def test_model_errors_pass_through_unchanged(
    stage, error_type, settings, fake_embeddings, fake_reranker
):
    """The adapters translate their own failures; the pipeline must not again.

    A model failure already arrives as a RuntimeError naming the model --
    `claude_model` translates where the model runs -- so neither retrieve()'s
    store translation nor `_generate()` may re-wrap it: relabelled "Vector store
    request failed", it would send a reader to the wrong component. A
    BaseException must pass untouched as well: it is how Streamlit stops a
    script, and turning it into an error -- including the empty-answer one,
    since no content had arrived -- would show the Stop button as a failure.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    error = error_type(f"the {stage} model failed")
    pipeline = RAGPipeline(
        settings,
        embeddings=_failing_embeddings(error) if stage == "embed" else fake_embeddings,
        llm=_failing_llm(error)
        if stage == "generate"
        else FakeListChatModel(responses=["unused"]),
        reranker=_failing_reranker(error) if stage == "rerank" else fake_reranker,
    )

    # answer() is stream_answer() consumed, so this covers both halves: embed
    # and rerank fail on the call, generate only once the stream is pulled.
    with pytest.raises(error_type) as info:
        pipeline.answer("Why do chunks overlap?")

    assert info.value is error


# The exception union both frontends catch (`cli.py`, `streamlit_app.py`). A
# failure mode outside it escapes as a traceback in the CLI and a Streamlit crash
# page.
_FRONTEND_EXCEPTIONS = (FileNotFoundError, RuntimeError, ValueError)

# How the local models fail by the time the pipeline sees them: already a
# RuntimeError naming the model, from the adapter.
_MODEL_FAILURE = "Generation with 'some-org/some-model' failed: [metal] out of memory"


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


def _fail_ingest_invalid_collection_name(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # MongoDB refuses the name outright, as pymongo's InvalidName. The
    # translation must make that a RuntimeError -- never pymongo's own type,
    # and never a ValueError, which streamlit_app.py's pipeline-load guard does
    # not catch.
    ingest_mod.ingest(
        dataclasses.replace(settings, collection_name="bad$name"),
        embeddings=fake_embeddings,
    )


def _fail_ingest_dimension_mismatch(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # MongoDB stores a vector of any width, so a model whose actual width
    # disagrees with EMBEDDING_DIMENSIONS would be stored without complaint
    # under a fingerprint that claims the declared one. Ingest's own probe is
    # the guard, and it raises before anything is written.
    ingest_mod.ingest(
        dataclasses.replace(settings, embedding_dimensions=8),
        embeddings=DeterministicFakeEmbedding(size=16),
    )


def _fail_ingest_collection_width_change(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # The index serves one width, so a narrower model cannot be indexed into the
    # collection. Caught before the delete that would otherwise already have
    # removed the chunks being replaced. EMBEDDING_MODEL changes along with the
    # fake, as it would for real: that is what marks every source changed.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.ingest(
        dataclasses.replace(
            settings, embedding_model="some-org/narrower-model", embedding_dimensions=16
        ),
        embeddings=DeterministicFakeEmbedding(size=16),
    )


def _fail_ingest_on_embedding_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    ingest_mod.ingest(
        settings, embeddings=_failing_embeddings(RuntimeError(_MODEL_FAILURE))
    )


def _fail_pipeline_nothing_ingested(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # No models injected: the guard must fire before any would be built.
    RAGPipeline(settings)


def _fail_pipeline_missing_vector_index(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Chunks, but no index to search them with: every question would find
    # nothing, so this is refused before any model loads.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(dataclasses.replace(settings, vector_index_name="no_such_index"))


def _fail_pipeline_without_mongodb_uri(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    monkeypatch.delenv("MONGODB_URI")
    RAGPipeline(settings)


def _fail_pipeline_unreachable_cluster(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A paused cluster, or an IP missing from the access list, looks like this.
    monkeypatch.setenv("MONGODB_URI", "mongodb://127.0.0.1:1/?directConnection=true")
    RAGPipeline(dataclasses.replace(settings, mongodb_timeout_ms=200))


def _fail_pipeline_missing_collection(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A store exists, but nothing was ever ingested under this collection name.
    ingest_mod.ingest(
        dataclasses.replace(settings, collection_name="other_docs"),
        embeddings=fake_embeddings,
    )
    RAGPipeline(settings)


def _fail_pipeline_empty_collection(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    _emptied(settings, fake_embeddings)
    RAGPipeline(settings)


def _fail_pipeline_invalid_collection_name(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A name MongoDB refuses fails on the read path too -- as a RuntimeError,
    # never pymongo's own InvalidName.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(dataclasses.replace(settings, collection_name="bad$name"))


def _fail_pipeline_fetch_k_below_one(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # $vectorSearch would refuse the search, and only on the first question; no
    # models injected, so this is also checked before any would load.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(dataclasses.replace(settings, fetch_k=0))


def _fail_pipeline_fetch_k_above_the_cap(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Ten candidates per result would pass $vectorSearch's 10,000 cap.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(dataclasses.replace(settings, fetch_k=1001))


def _fail_pipeline_without_anthropic_key(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # No chat model injected: the factory needs the key first, and conftest has
    # removed the developer's.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(settings, embeddings=fake_embeddings, reranker=fake_reranker)


def _fail_ingest_without_voyage_key(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # No embeddings injected: production's factory needs the key first.
    ingest_mod.ingest(settings)


def _fail_pipeline_width_voyage_lacks(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # The fixture's 32 is the fake's width, not one Voyage returns; refused when
    # the pipeline loads, not when the first question is embedded.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    RAGPipeline(settings, reranker=fake_reranker)


def _fail_pipeline_retrieval_k_below_one(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    RAGPipeline(
        dataclasses.replace(settings, retrieval_k=0),
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
    )


def _fail_retrieve_on_a_voyage_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # What a Voyage outage, or a rate limit its retries did not outlast, looks
    # like from inside: voyageai's own type, outside the union.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(
        settings,
        embeddings=_failing_embeddings(voyageai.error.RateLimitError("rate limited")),
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    ).retrieve("apples")


def _ingested_pipeline(settings, fake_embeddings, llm, reranker) -> RAGPipeline:
    """A pipeline over a freshly ingested index, generating through `llm`."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    return RAGPipeline(settings, embeddings=fake_embeddings, llm=llm, reranker=reranker)


def _fail_retrieve_on_embedding_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # The question is embedded by the same local model at query time, so its
    # failure has to hold the same way as at ingest.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(
        settings,
        embeddings=_failing_embeddings(RuntimeError(_MODEL_FAILURE)),
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    ).retrieve("apples")


def _fail_retrieve_on_rerank_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=_failing_reranker(RuntimeError(_MODEL_FAILURE)),
    ).retrieve("apples")


def _fail_retrieve_on_dimension_mismatch(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A collection built by one model, queried by a narrower one: Atlas accepts
    # the open and fails the search. The message has to name the remedy (a new
    # COLLECTION_NAME), since nothing about the question was wrong.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(
        settings,
        embeddings=DeterministicFakeEmbedding(size=16),
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    ).retrieve("apples")


def _fail_answer_on_model_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    pipeline = _ingested_pipeline(
        settings,
        fake_embeddings,
        _failing_llm(RuntimeError(_MODEL_FAILURE)),
        fake_reranker,
    )
    pipeline.answer("Why do chunks overlap?")


def _fail_stream_answer_on_model_error(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    pipeline = _ingested_pipeline(
        settings,
        fake_embeddings,
        _failing_llm(RuntimeError(_MODEL_FAILURE)),
        fake_reranker,
    )
    # Consumed to exhaustion: generation is lazy, so merely calling
    # stream_answer() raises nothing. This is the shape both frontends use, and
    # it is where the union has to hold.
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
            _fail_ingest_invalid_collection_name,
            RuntimeError,
            "Vector store request failed",
            id="ingest-invalid-collection-name",
        ),
        pytest.param(
            _fail_ingest_dimension_mismatch,
            ValueError,
            "EMBEDDING_DIMENSIONS",
            id="ingest-dimension-mismatch",
        ),
        pytest.param(
            _fail_ingest_collection_width_change,
            ValueError,
            "COLLECTION_NAME",
            id="ingest-collection-width-change",
        ),
        pytest.param(
            _fail_ingest_on_embedding_error,
            RuntimeError,
            "out of memory",
            id="ingest-embedding-error",
        ),
        pytest.param(
            _fail_pipeline_nothing_ingested,
            FileNotFoundError,
            "nothing was ever ingested",
            id="pipeline-nothing-ingested",
        ),
        pytest.param(
            _fail_pipeline_missing_vector_index,
            FileNotFoundError,
            "VECTOR_INDEX_NAME",
            id="pipeline-missing-vector-index",
        ),
        pytest.param(
            _fail_pipeline_without_mongodb_uri,
            RuntimeError,
            "MONGODB_URI is not set",
            id="pipeline-without-mongodb-uri",
        ),
        pytest.param(
            _fail_pipeline_unreachable_cluster,
            RuntimeError,
            "Vector store request failed",
            id="pipeline-unreachable-cluster",
        ),
        pytest.param(
            _fail_pipeline_missing_collection,
            FileNotFoundError,
            "nothing was ever ingested",
            id="pipeline-missing-collection",
        ),
        pytest.param(
            _fail_pipeline_empty_collection,
            FileNotFoundError,
            "is empty",
            id="pipeline-empty-collection",
        ),
        pytest.param(
            _fail_pipeline_invalid_collection_name,
            RuntimeError,
            "Vector store request failed",
            id="pipeline-invalid-collection-name",
        ),
        pytest.param(
            _fail_pipeline_fetch_k_below_one,
            RuntimeError,
            "FETCH_K",
            id="pipeline-fetch-k-below-one",
        ),
        pytest.param(
            _fail_pipeline_fetch_k_above_the_cap,
            RuntimeError,
            "FETCH_K",
            id="pipeline-fetch-k-above-the-cap",
        ),
        pytest.param(
            _fail_pipeline_without_anthropic_key,
            RuntimeError,
            "ANTHROPIC_API_KEY is not set",
            id="pipeline-without-anthropic-key",
        ),
        pytest.param(
            _fail_ingest_without_voyage_key,
            RuntimeError,
            "VOYAGE_API_KEY is not set",
            id="ingest-without-voyage-key",
        ),
        pytest.param(
            _fail_pipeline_width_voyage_lacks,
            RuntimeError,
            "not a width Voyage offers",
            id="pipeline-width-voyage-lacks",
        ),
        pytest.param(
            _fail_pipeline_retrieval_k_below_one,
            RuntimeError,
            "RETRIEVAL_K",
            id="pipeline-retrieval-k-below-one",
        ),
        pytest.param(
            _fail_retrieve_on_a_voyage_error,
            RuntimeError,
            "Voyage AI request failed",
            id="retrieve-voyage-error",
        ),
        pytest.param(
            _fail_retrieve_on_embedding_error,
            RuntimeError,
            "out of memory",
            id="retrieve-embedding-error",
        ),
        pytest.param(
            _fail_retrieve_on_rerank_error,
            RuntimeError,
            "out of memory",
            id="retrieve-rerank-error",
        ),
        pytest.param(
            _fail_retrieve_on_dimension_mismatch,
            RuntimeError,
            "set a new COLLECTION_NAME",
            id="retrieve-dimension-mismatch",
        ),
        pytest.param(
            _fail_answer_on_model_error,
            RuntimeError,
            "out of memory",
            id="answer-model-error",
        ),
        pytest.param(
            _fail_stream_answer_on_model_error,
            RuntimeError,
            "out of memory",
            id="stream-answer-model-error",
        ),
        pytest.param(
            _fail_stream_answer_on_empty_response,
            RuntimeError,
            "empty answer",
            id="stream-answer-empty-response",
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
    ValueError`, the union `cli.py` and `streamlit_app.py` catch.

    Individual tests above already cover most of these one at a time; this one
    exists to make the *union* the thing under test, so adding a fourth type
    (or letting a pymongo, voyageai or anthropic exception escape untranslated,
    which would drag that library into both frontends) fails here rather than
    at a user's terminal. `expected_type` is checked exactly, so a path can't
    drift to a different member of the union unnoticed -- and nothing on the
    pipeline-load path may become a ValueError, which streamlit_app.py's guard
    there does not catch: a traceback under the sidebar, on every rerun.
    """
    with pytest.raises(_FRONTEND_EXCEPTIONS, match=expected_message) as excinfo:
        failing_call(settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path)

    assert type(excinfo.value) is expected_type
