"""Tests for the query phase: helpers, guards, retrieval, and generation.

Generation *is* exercised here, through an injected fake chat model rather than
a real one — so no weights load and MLX is never imported, per the injection
seam described in CLAUDE.md. That covers both shapes (`stream_answer()` and the
`answer()` join over it), that they cannot drift apart, that streaming stays
incremental, and that every failure lands in the union both frontends catch.

The index guards are tested with *no* models injected. conftest makes MLX
unimportable, so a pipeline that built a model before checking the index would
fail with the loader's RuntimeError instead of the guard's FileNotFoundError --
which makes each guard test a proof of ordering as well as of the message.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from types import SimpleNamespace
from typing import Any

import huggingface_hub.constants as hf_constants
import pytest
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import Settings
from rag_pipeline.mlx_models import MLXChatModel, QwenVLReranker
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
    And looking must not create what it looked for -- an empty persist dir left
    behind would turn the next attempt's message into a different one.
    """
    with pytest.raises(FileNotFoundError, match="No index found at"):
        RAGPipeline(settings)

    assert not settings.persist_dir.exists()


def _emptied(settings: Settings, embeddings: Embeddings) -> None:
    """Ingested, then every chunk of ours deleted: the collection still exists."""
    ingest_mod.ingest(settings, embeddings=embeddings)
    store = ingest_mod.open_store(settings, embeddings)
    store.delete(ids=store.get(where=ingest_mod.OWN_CHUNKS)["ids"])


def _foreign_only(settings: Settings, embeddings: Embeddings) -> None:
    """A collection of the configured name holding only someone else's records."""
    ingest_mod.open_store(settings, embeddings).add_texts(
        ["Somebody else's notes."], metadatas=[{"source": "theirs.md"}], ids=["theirs"]
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
    ingest_mod.reset_store_cache()

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

    A wrong COLLECTION_NAME names a collection that does not exist. The
    `persist_dir` check cannot see that (the directory is populated), so `match=`
    pins the collection guard; and the lookup must not conjure the collection
    into existence -- the query path never creates one, or a second attempt
    would meet the empty-index message instead of the one naming the mismatch.
    The `reset_store_cache()` calls put each re-open in the state a real
    `rag query` starts from.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()

    mismatched = dataclasses.replace(settings, collection_name="not_the_ingested_one")
    with pytest.raises(FileNotFoundError, match="nothing was ever ingested"):
        RAGPipeline(mismatched)
    with pytest.raises(FileNotFoundError):
        ingest_mod.open_store(mismatched, fake_embeddings, create=False)

    # And the correctly-named collection still retrieves: the failure above was
    # about the name, not a genuinely missing index.
    ingest_mod.reset_store_cache()
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
    a model at all. Over a complete index it does, and under conftest's MLX
    block that is the loader's RuntimeError.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()

    with pytest.raises(RuntimeError, match="MLX"):
        RAGPipeline(settings)


def test_a_collection_that_vanishes_after_the_check_is_not_recreated(
    settings, fake_embeddings, fake_reranker, monkeypatch
):
    """The query path opens the store with ``create=False``, whatever the guard saw.

    In production the embedding model loads between the check and the open, so
    the collection can go in between; created afresh, it would be an empty one
    that answers every question "I don't know". Refused instead, as the error
    naming the fix -- and nothing is left behind for the next attempt to find.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()
    check = pipeline_mod.require_index

    def check_then_vanish(s: Settings) -> None:
        check(s)
        ingest_mod.open_store(s, fake_embeddings, create=False).delete_collection()

    monkeypatch.setattr(pipeline_mod, "require_index", check_then_vanish)

    with pytest.raises(FileNotFoundError, match="nothing was ever ingested"):
        RAGPipeline(
            settings,
            embeddings=fake_embeddings,
            llm=FakeListChatModel(responses=["unused"]),
            reranker=fake_reranker,
        )
    with pytest.raises(FileNotFoundError):
        ingest_mod.open_store(settings, fake_embeddings, create=False)


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
    (the control at the end).
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    text = "Zebra facts that no ingested file contains."
    ingest_mod.open_store(settings, fake_embeddings).add_texts(
        [text],
        metadatas=[{"source": "zebra.md", "content_hash": "0" * 64}],
        ids=["foreign"],
    )

    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    )
    retrieved = pipeline.retrieve(text)

    assert retrieved, "the pipeline's own chunks should still be retrieved"
    assert "zebra.md" not in unique_sources(retrieved)
    unscoped = ingest_mod.open_store(settings, fake_embeddings, create=False)
    assert unscoped.similarity_search(text, k=1)[0].metadata["source"] == "zebra.md"


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


def test_closing_the_answer_stream_stops_the_local_model(
    settings, fake_embeddings, fake_reranker, fake_mlx, model_dir
):
    """What the app's Stop relies on: closing the stream stops generation there.

    Through the real local chat model, over a fake MLX, because the property is
    about the stack between them: the model holds a process-wide lock for as
    long as it generates, and a chain that drained the model when closed (a
    `StrOutputParser` on the end does) would keep every other question waiting
    until MAX_TOKENS. Closed after one piece, the model must have produced one
    -- and have finished, lock held, before the lock is let go.
    """
    fake_mlx.pieces = [f"word{i} " for i in range(20)]
    pipeline = _ingested_pipeline(
        settings,
        fake_embeddings,
        MLXChatModel(model_id=model_dir, max_tokens=50),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer("Why do chunks overlap?")
    assert next(chunks) == "word0 "
    chunks.close()

    assert fake_mlx.pieces_generated == 1, "the model ran on after the close"
    assert fake_mlx.lock_held_at_close == [True]
    assert not mlx_models._GENERATION_LOCK.locked()


def test_an_answer_cut_off_at_max_tokens_says_so(
    settings, fake_embeddings, fake_reranker, fake_mlx, model_dir
):
    """A truncated answer must not read as a finished one.

    The model reports why it stopped only in metadata that the text alone does
    not carry, so without this an answer cut off mid-sentence reaches both
    frontends looking complete. Said in the answer itself because that is the
    one channel both of them show.
    """
    fake_mlx.pieces = ["Chunks overlap ", "so that"]
    fake_mlx.finish_reason = "length"
    pipeline = _ingested_pipeline(
        settings,
        fake_embeddings,
        MLXChatModel(model_id=model_dir, max_tokens=50),
        fake_reranker,
    )

    text = pipeline.answer("Why do chunks overlap?").text

    assert text.startswith("Chunks overlap so that")
    assert f"MAX_TOKENS={settings.max_tokens}" in text

    fake_mlx.finish_reason = "stop"
    assert pipeline.answer("Why do chunks overlap?").text == "Chunks overlap so that"


# --- the model factories -----------------------------------------------------


def _stub_weights(monkeypatch, model: Any, tokenizer: Any) -> list[str]:
    """Replace the one weight loader with a stub; return the ids it is asked for.

    The factories are tested for real, down to the adapter they construct, with
    only the load taken out -- the real one is gigabytes, and conftest makes it
    unreachable anyway.
    """
    requested: list[str] = []

    def load(model_id: str) -> tuple[Any, Any]:
        requested.append(model_id)
        return model, tokenizer

    monkeypatch.setattr(mlx_models, "load_mlx_model", load)
    return requested


def test_build_chat_model_sets_no_sampling_params(settings, monkeypatch):
    """Production builds the local chat model from settings, decoding greedily.

    Greedy on purpose: grounding comes from the retrieved context, and the same
    question over the same context should get the same answer to be checkable
    against it. Greedy is the *absence* of sampling parameters, and the model
    drops unknown keywords silently, so a field is the only place one could
    live -- which is why the check reads the constructed model's fields rather
    than the factory's call. Distinctive settings, so a factory that ignored
    them in favour of the defaults (or a literal) fails; MAX_TOKENS matters
    because mlx-lm's own default of 256 would cut a cited answer short.
    """
    requested = _stub_weights(monkeypatch, object(), object())
    configured = dataclasses.replace(
        settings, chat_model="some-org/some-chat-model", max_tokens=321
    )

    model = build_chat_model(configured)

    # Narrowed before the reads, so they are type-checked -- and because
    # "production builds the local MLX model" is half of what this is for.
    assert isinstance(model, MLXChatModel)
    assert (model.model_id, model.max_tokens) == ("some-org/some-chat-model", 321)
    assert requested == ["some-org/some-chat-model"]
    sampling = {"temperature", "top_p", "top_k", "min_p", "sampler"}
    assert not sampling & set(type(model).model_fields)


def test_build_reranker_caps_at_retrieval_k(settings, monkeypatch):
    """The reranker's own cap is what makes retrieve() return retrieval_k passages.

    retrieve() does no slicing of its own, so a factory that passed FETCH_K (or
    a literal) here would put every candidate into each prompt and citation
    list -- and every other test would miss it, since they all inject a fake
    reranker. The stub is the least a Qwen3-VL-Reranker checkpoint provides for
    construction: a backbone, and a vocabulary with "yes" and "no" in it.
    """
    requested = _stub_weights(
        monkeypatch,
        SimpleNamespace(language_model=SimpleNamespace(model=None)),
        SimpleNamespace(get_vocab=lambda: {"yes": 1, "no": 2}, pad_token_id=0),
    )
    configured = dataclasses.replace(
        settings, rerank_model="some-org/some-reranker", retrieval_k=3, fetch_k=9
    )

    reranker = build_reranker(configured)

    assert isinstance(reranker, QwenVLReranker)
    assert (reranker.model_id, reranker.top_n) == ("some-org/some-reranker", 3)
    assert requested == ["some-org/some-reranker"]


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
    `mlx_models` translates where the model runs -- so neither retrieve()'s
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


# The exception union both frontends catch (`cli.py`, `app.py`). A failure mode
# outside it escapes as a traceback in the CLI and a Streamlit crash page.
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
    # Chroma rejects the name when the collection is created. The translation
    # must make that a RuntimeError -- never chromadb's own type, and never a
    # ValueError, which app.py's pipeline-load guard does not catch.
    ingest_mod.ingest(
        dataclasses.replace(settings, collection_name="x"), embeddings=fake_embeddings
    )


def _fail_ingest_dimension_mismatch(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Chroma takes whatever width the first insert has, so a model whose actual
    # width disagrees with EMBEDDING_DIMENSIONS would be indexed without
    # complaint under a fingerprint that claims the declared one. Ingest's own
    # probe is the guard, and it raises before anything is written.
    ingest_mod.ingest(
        dataclasses.replace(settings, embedding_dimensions=8),
        embeddings=DeterministicFakeEmbedding(size=16),
    )


def _fail_ingest_collection_width_change(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A collection keeps its width for good, so a narrower model cannot be
    # indexed into it. Caught before the delete that would otherwise already
    # have removed the chunks being replaced. EMBEDDING_MODEL changes along with
    # the fake, as it would for real: that is what marks every source changed.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()
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


def _fail_pipeline_missing_index(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # No models injected: the guard must fire before any would be built.
    RAGPipeline(dataclasses.replace(settings, persist_dir=tmp_path / "no-such-index"))


def _fail_pipeline_missing_collection(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A store exists, but nothing was ever ingested under this collection name.
    ingest_mod.ingest(
        dataclasses.replace(settings, collection_name="other_docs"),
        embeddings=fake_embeddings,
    )
    ingest_mod.reset_store_cache()
    RAGPipeline(settings)


def _fail_pipeline_empty_collection(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    _emptied(settings, fake_embeddings)
    ingest_mod.reset_store_cache()
    RAGPipeline(settings)


def _fail_pipeline_invalid_collection_name(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # A name Chroma would refuse to create is, on the read path, simply one that
    # was never ingested into -- which must not surface as chromadb's error.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(dataclasses.replace(settings, collection_name="x"))


def _fail_pipeline_fetch_k_below_one(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # Chroma would refuse the search with a builtins TypeError, and only on the
    # first question; no models injected, so this is also checked before any
    # would load.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(dataclasses.replace(settings, fetch_k=0))


def _fail_pipeline_without_mlx(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # conftest's MLX block is exactly what a machine without MLX looks like.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(settings, embeddings=fake_embeddings, reranker=fake_reranker)


def _fail_pipeline_model_not_cached(
    settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path
):
    # MLX present -- a stand-in, since the loader only needs the import to
    # succeed before it looks in the cache -- but the chat model never
    # downloaded, against an empty cache so the developer's own is not read.
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    RAGPipeline(
        dataclasses.replace(settings, chat_model="some-org/never-downloaded"),
        embeddings=fake_embeddings,
        reranker=fake_reranker,
    )


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
    # A collection built by one model, queried by a narrower one: Chroma accepts
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
            _fail_pipeline_missing_index,
            FileNotFoundError,
            "No index found at",
            id="pipeline-missing-index",
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
            FileNotFoundError,
            "COLLECTION_NAME",
            id="pipeline-invalid-collection-name",
        ),
        pytest.param(
            _fail_pipeline_fetch_k_below_one,
            RuntimeError,
            "FETCH_K",
            id="pipeline-fetch-k-below-one",
        ),
        pytest.param(
            _fail_pipeline_without_mlx,
            RuntimeError,
            "Apple Silicon",
            id="pipeline-without-mlx",
        ),
        pytest.param(
            _fail_pipeline_model_not_cached,
            FileNotFoundError,
            "hf download some-org/never-downloaded",
            id="pipeline-model-not-cached",
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
    ValueError`, the union `cli.py` and `app.py` catch.

    Individual tests above already cover most of these one at a time; this one
    exists to make the *union* the thing under test, so adding a fourth type
    (or letting a chromadb or huggingface_hub exception escape untranslated,
    which would drag that library into both frontends) fails here rather than
    at a user's terminal. `expected_type` is checked exactly, so a path can't
    drift to a different member of the union unnoticed -- and nothing on the
    pipeline-load path may become a ValueError, which app.py's guard there does
    not catch: a traceback under the sidebar, on every rerun.
    """
    with pytest.raises(_FRONTEND_EXCEPTIONS, match=expected_message) as excinfo:
        failing_call(settings, fake_embeddings, fake_reranker, monkeypatch, tmp_path)

    assert type(excinfo.value) is expected_type
