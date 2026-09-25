"""Query phase: embed question -> search -> rerank -> generate a grounded answer.

``RAGPipeline`` opens the persisted Chroma collection and the local models
once, then answers questions against them. Both the CLI and the Streamlit app
build a single pipeline and reuse it across queries.
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import closing
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    RerankerAttributes,
    SpanAttributes,
)
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from rag_pipeline.config import Settings
from rag_pipeline.ingest import (
    OWN_CHUNKS,
    open_store,
    require_index,
    store_errors_as_runtime,
)
from rag_pipeline.mlx_models import MLXChatModel, QwenVLReranker

# Grounding prompt: the model must answer from the retrieved context only, and
# admit when the context does not contain the answer. This is what turns a
# general chat model into a document-grounded question-answerer.
_SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using only the provided "
    "context. Follow these rules:\n"
    "- Base your answer solely on the context below. Do not use outside "
    "knowledge.\n"
    "- If the context does not contain the answer, say you don't know based on "
    "the provided documents.\n"
    "- Be concise, and cite the source file(s) you used in parentheses."
)

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)


@dataclass
class Answer:
    """An answer plus the source chunks that grounded it."""

    text: str
    sources: list[Document]


class Excerpt(TypedDict):
    """One retrieved passage, as a frontend stores and replays it.

    A TypedDict rather than a dataclass because the chat history that holds
    these is itself plain dicts, and because it must survive a round trip
    through Streamlit's session state; the annotation is what keeps the two key
    names checked rather than spelled from memory at each use.
    """

    source: str
    text: str


def _source_of(doc: Document) -> str:
    """The citation label for one chunk.

    One spelling of the fallback, because three functions below put this string
    in front of a reader -- in the prompt, in a citation line, and in the panel
    that is supposed to prove the first two agree. Copies of it would be exactly
    the drift they exist to prevent.
    """
    return doc.metadata.get("source", "unknown")


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into a single context string, labeled by source."""
    return "\n\n".join(
        f"[Source: {_source_of(doc)}]\n{doc.page_content}" for doc in docs
    )


def unique_sources(docs: list[Document]) -> list[str]:
    """Distinct source files across the retrieved chunks, in retrieval order."""
    # dict.fromkeys preserves insertion order while dropping duplicates.
    return list(dict.fromkeys(_source_of(doc) for doc in docs))


def source_excerpts(docs: list[Document]) -> list[Excerpt]:
    """The retrieved passages in a form a frontend can store and replay.

    Retrieval order is preserved and repeated sources are not collapsed:
    ``format_docs`` joins in list order, so this order *is* the order the model
    read them in, and two chunks from one file are two pieces of evidence rather
    than one repeated citation.
    """
    return [Excerpt(source=_source_of(doc), text=doc.page_content) for doc in docs]


# --- tracing -----------------------------------------------------------------
#
# Through the OpenTelemetry API only, so all of this is a no-op until a frontend
# installs a provider (rag_pipeline.tracing); the attribute names are
# OpenInference's, which is what Phoenix renders a trace from.

# The metadata key a LangChain reranker scores its documents under -- the
# convention QwenVLReranker follows too.
_SCORE_KEY = "relevance_score"

_TEXT = OpenInferenceMimeTypeValues.TEXT.value


def _tracer() -> trace.Tracer:
    """This module's tracer, looked up per question rather than held.

    A tracer resolves against the provider installed when it is first used,
    and keeps that answer: held at import, it would keep the first one for good,
    and the tests install a fresh provider for each test that records spans.
    """
    return trace.get_tracer(__name__)


def _span_kind(kind: OpenInferenceSpanKindValues) -> dict[str, AttributeValue]:
    # Upper case, as the enum's values are: Phoenix's span filters are written
    # that way.
    return {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value}


def _documents(key: str, docs: list[Document]) -> dict[str, AttributeValue]:
    """``docs`` as OpenInference's flattened document list under ``key``."""
    attributes: dict[str, AttributeValue] = {}
    for index, doc in enumerate(docs):
        prefix = f"{key}.{index}."
        attributes[prefix + DocumentAttributes.DOCUMENT_CONTENT] = doc.page_content
        attributes[prefix + DocumentAttributes.DOCUMENT_METADATA] = json.dumps(
            doc.metadata, default=str
        )
        if doc.id is not None:
            attributes[prefix + DocumentAttributes.DOCUMENT_ID] = doc.id
        score = doc.metadata.get(_SCORE_KEY)
        if isinstance(score, float):
            attributes[prefix + DocumentAttributes.DOCUMENT_SCORE] = score
    return attributes


def _finish(span: Span, error: BaseException | None) -> None:
    """End a question's root span with the status its outcome earned.

    A failure is an error, with the exception recorded. A question stopped
    part-way -- the app's Stop, Ctrl-C at the terminal, a caller that closes the
    stream early -- is not, so its status is left unset and the stop is an event
    instead; otherwise Phoenix would count every Stop as a failed question. (The
    model's own span still ends in error on a Stop: LangChain reports a closed
    stream to its tracer as one, and nothing here sees it first.)
    """
    if isinstance(error, Exception):
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, f"{type(error).__name__}: {error}"))
    elif error is not None:
        span.add_event("stopped", {"reason": type(error).__name__})
    else:
        span.set_status(Status(StatusCode.OK))
    span.end()


def _traced(
    span: Span, pieces: Generator[str, None, None]
) -> Generator[str, None, None]:
    """``pieces``, generated inside ``span``, which ends when the stream does.

    The span is made current around each step rather than across a yield.
    OpenTelemetry keeps the current span in a context variable, and one held
    across a yield leaks into whatever the consumer does between pieces -- and
    fails to detach ("Failed to detach context") when the stream is closed from
    another context, as a Stop in the app can close it. One step is one
    synchronous frame, so attach and detach always pair. The first step is the
    one that counts: LangChain parents its run to whichever span is current
    when the chain starts, and the chain starts on the first pull.

    Primed: the first ``yield`` gives nothing, comes before any generation, and
    is consumed by ``stream_answer``. A generator's ``finally`` exists only once
    its body has started, so unprimed, a stream closed before its first piece --
    a Stop that lands between retrieval and generation -- would end nothing,
    and that question's trace would never be exported. The model still does not
    start until the caller asks for a piece.
    """
    context = trace.set_span_in_context(span)
    answer: list[str] = []
    error: BaseException | None = None
    try:
        yield ""
        while True:
            token = otel_context.attach(context)
            try:
                piece = next(pieces, None)
            finally:
                otel_context.detach(token)
            if piece is None:
                break
            answer.append(piece)
            yield piece
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            # Closed at once, not left to the garbage collector: this is what
            # stops the model when the caller closes the stream early.
            pieces.close()
        finally:
            # What was generated, however it ended: the partial answer is the
            # useful part of a stopped or failed question's trace.
            span.set_attributes(
                {
                    SpanAttributes.OUTPUT_VALUE: "".join(answer),
                    SpanAttributes.OUTPUT_MIME_TYPE: _TEXT,
                }
            )
            _finish(span, error)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the local chat model used for generation.

    No temperature/top_p/top_k: the model decodes greedily, so the same question
    over the same retrieved context gets the same answer — grounding comes from
    the context, and an answer that changes from one run to the next is harder to
    check against it. ``max_tokens`` is passed through because mlx-lm's own
    default (256) would cut a cited answer short. Cheap to call again: the
    weights load once per process, so the app's rebuild after every ingest wraps
    the model it already holds rather than loading a second copy.
    """
    return MLXChatModel(model_id=settings.chat_model, max_tokens=settings.max_tokens)


def build_reranker(settings: Settings) -> BaseDocumentCompressor:
    """Construct the local reranker.

    Here, not in ingest.py: reranking is a query-only stage with no ingest-side
    counterpart, so the shared-factory reason that keeps build_embeddings in
    ingest.py doesn't apply — it sits beside build_chat_model, both query-time
    model factories. ``top_n`` is the reranker's own cap, so it returns exactly
    retrieval_k docs and ``retrieve()`` needs no manual slice.
    """
    return QwenVLReranker(model_id=settings.rerank_model, top_n=settings.retrieval_k)


class RAGPipeline:
    """Loads the persisted index and answers questions against it."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = None,
        reranker: BaseDocumentCompressor | None = None,
    ) -> None:
        # Chroma refuses a search for fewer than one result with a builtins
        # TypeError -- outside the union, and only on the first question, after
        # every model had loaded. RuntimeError, not ValueError: this runs on the
        # app's pipeline-load path, whose handler sits below the sidebar.
        if settings.fetch_k < 1:
            raise RuntimeError(f"FETCH_K must be at least 1, not {settings.fetch_k}.")
        # Before any model is built: a missing, misnamed or empty index is
        # reported without first loading ~22 GB of weights to find out.
        require_index(settings)

        self.settings = settings

        # Reopen the existing store via the shared factory, so the same
        # embedding model that indexed the documents also embeds queries.
        # `embeddings` and `llm` are injectable for tests; production leaves
        # both as None and gets the local models. `create=False`, so the query
        # path never creates a collection, even if one vanished since the check.
        vectorstore = open_store(settings, embeddings, create=False)
        # Retrieve a wide candidate set (fetch_k); the reranker below narrows it
        # to retrieval_k. Filtered to this pipeline's own chunks, like every
        # read at ingest, so a foreign record sharing the collection is never
        # retrieved or cited. `reranker` is injectable for tests alongside
        # `embeddings`/`llm`; production leaves it None and builds the real one.
        self._retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.fetch_k, "filter": OWN_CHUNKS}
        )
        self._reranker = reranker or build_reranker(settings)

        # The model's own message chunks, not a StrOutputParser's strings:
        # closing a parser's stream does not stop the model -- langchain-core
        # catches the GeneratorExit and drains the parser's input to the end --
        # so a Stop in the app would keep generating, under the process-wide
        # generation lock, until MAX_TOKENS. _generate() extracts the text
        # itself instead.
        self._chain = _PROMPT | (llm or build_chat_model(settings))

    def retrieve(self, question: str) -> list[Document]:
        """Return the reranked top chunks for the question.

        Vector search casts a wide net (fetch_k); the reranker — scoring each
        candidate against the question jointly — narrows it to retrieval_k. Both
        calls are wrapped so a store failure, such as a query-time dimension
        mismatch against a collection built with another model, surfaces as the
        RuntimeError both frontends catch rather than a raw chromadb exception.
        The models need no wrapping: their adapters already raise inside the
        union.

        The search is traced by LangChain's instrumentation, as a retriever run;
        the rerank is not -- a compressor is not a Runnable -- so it gets a span
        of its own here. That span is where a trace shows what the reranker was
        given and what it kept, with their scores: the step whose choices
        decide what the model is shown.
        """
        with store_errors_as_runtime():
            candidates = self._retriever.invoke(question)
            with _tracer().start_as_current_span(
                type(self._reranker).__name__,
                attributes=_span_kind(OpenInferenceSpanKindValues.RERANKER),
            ) as span:
                # Guarded: with tracing off, nothing is serialized for a span
                # that records nothing.
                if span.is_recording():
                    span.set_attributes(
                        {
                            RerankerAttributes.RERANKER_QUERY: question,
                            RerankerAttributes.RERANKER_MODEL_NAME: (
                                self.settings.rerank_model
                            ),
                            RerankerAttributes.RERANKER_TOP_K: self.settings.retrieval_k,
                            **_documents(
                                RerankerAttributes.RERANKER_INPUT_DOCUMENTS, candidates
                            ),
                        }
                    )
                ranked = list(self._reranker.compress_documents(candidates, question))
                if span.is_recording():
                    span.set_attributes(
                        _documents(RerankerAttributes.RERANKER_OUTPUT_DOCUMENTS, ranked)
                    )
                # OK, as LangChain's own spans end: left unset, it is the one
                # step in a successful trace that does not read as a success.
                span.set_status(Status(StatusCode.OK))
            return ranked

    def _generate(
        self, question: str, docs: list[Document]
    ) -> Generator[str, None, None]:
        """Yield the grounded answer in pieces, as the model produces them.

        The single generation path, so a generation-level check lives here once
        rather than in each frontend. A model failure already arrives as a
        RuntimeError — the adapter translates it where the model runs — so the
        checks left for this layer are about the response: one that arrives
        empty, and one the model cut off at MAX_TOKENS. Both surface while the
        generator is being consumed, not when it is created: `.stream()` is
        lazy, and the model does not start until the first piece is pulled.

        Closing this generator closes the model's stream at once (see
        `stream_answer`), because the chain's stream is closed with it rather
        than left for the garbage collector.
        """
        produced_content = False
        truncated = False
        # A RunnableSequence's stream is a generator, typed only as an Iterator;
        # the cast is what lets it be closed explicitly.
        stream = cast(
            Generator[Any, None, None],
            self._chain.stream(
                {"context": format_docs(docs), "question": question},
                # How the step reads in a trace; otherwise "RunnableSequence".
                config={"run_name": "generate"},
            ),
        )
        with closing(stream):
            for message in stream:
                # A str from a plain runnable, else a message chunk: `.text`
                # joins structured content blocks as well as a plain string.
                if isinstance(message, str):
                    piece = message
                else:
                    piece = str(message.text)
                    finish = message.response_metadata.get("finish_reason")
                    truncated = truncated or finish == "length"
                # Skipped when empty: the model's closing chunks carry only
                # metadata, and the app's first-token spinner must wait for text.
                if piece:
                    produced_content = produced_content or bool(piece.strip())
                    yield piece
        if not produced_content:
            # Otherwise each frontend presents nothing as a cited answer — a
            # blank chat bubble above a full panel of passages, or the CLI's
            # "Sources:" block under an empty line — claiming the strongest
            # possible grounding for no content at all.
            raise RuntimeError("The chat model returned an empty answer")
        if truncated:
            # Said in the answer itself, the one channel both frontends show:
            # otherwise an answer cut off mid-sentence reads as a complete one.
            yield (
                f"\n\n[Answer cut off at MAX_TOKENS={self.settings.max_tokens}; "
                "raise it for a longer one.]"
            )

    def stream_answer(
        self, question: str
    ) -> tuple[list[Document], Generator[str, None, None]]:
        """Search, then hand back the sources and a lazy stream of the answer.

        Both halves in one call because every frontend needs both, and splitting
        them made each frontend re-implement the same three steps. Returning the
        docs alongside the stream also means the citations shown are provably the
        ones the answer was generated from, not a second search that could drift.

        Note the two halves evaluate at different times: retrieval has already
        run when this returns (so a caller can put a spinner around just this
        call), while generation has not started and will not until the iterator
        is consumed.

        A caller that stops reading early must `close()` the stream rather than
        drop it. The local model holds a process-wide lock for as long as its
        stream is open, and a dropped one is freed only when the garbage
        collector gets to it -- which, for a stream a Streamlit script holds in
        a global, can be never: every later answer would wait on that lock.

        With tracing on, the question is one trace: a root span opened here,
        current while retrieval runs so the search and rerank nest under it,
        and ended by the stream -- however the stream ends (see `_traced`).
        Its end is why a stream that is never read should still be closed:
        until it is, the trace is not sent.
        """
        span = _tracer().start_span(
            "RAGPipeline",
            attributes={
                **_span_kind(OpenInferenceSpanKindValues.CHAIN),
                SpanAttributes.INPUT_VALUE: question,
                SpanAttributes.INPUT_MIME_TYPE: _TEXT,
            },
        )
        try:
            # Made current only for this synchronous stretch. Statuses are left
            # to _finish, which tells a failure from a stop.
            with trace.use_span(
                span, record_exception=False, set_status_on_exception=False
            ):
                docs = self.retrieve(question)
        except BaseException as exc:
            _finish(span, exc)
            raise
        answer = _traced(span, self._generate(question, docs))
        next(answer)  # primes it; see _traced
        return docs, answer

    def answer(self, question: str) -> Answer:
        """Retrieve context, then generate a grounded answer with sources.

        The all-at-once shape, for library callers that just want the finished
        string; both frontends stream instead. A join over the same path rather
        than a second call into the chain.
        """
        docs, chunks = self.stream_answer(question)
        return Answer(text="".join(chunks), sources=docs)
