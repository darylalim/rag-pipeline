"""Query phase: embed question -> search -> rerank -> generate a grounded answer.

``RAGPipeline`` opens the Atlas Vector Search index and the models once, then
answers questions against them. Both the CLI and the Streamlit app
build a single pipeline and reuse it across queries.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from typing import Any, TypedDict, cast

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_voyageai import VoyageAIRerank
from langsmith import Client, trace, tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree
from langsmith.utils import tracing_is_enabled
from pydantic import SecretStr

from rag_pipeline.claude_model import ClaudeChatModel
from rag_pipeline.config import Settings, require_env_key
from rag_pipeline.ingest import (
    OWN_CHUNKS,
    open_store,
    provider_errors_as_runtime,
    require_index,
    voyage_clients,
)
from rag_pipeline.tracing import tracing_client

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
# One LangSmith trace per question, built here by hand. Its root run is opened
# by stream_answer and ended by the answer stream, however that ends; the
# search and the model are LangChain runs, which nest under whatever run is
# current when they start; the rerank is a run of its own (see retrieve()).

# The metadata key a LangChain reranker scores its documents under, Voyage's
# included.
_SCORE_KEY = "relevance_score"


@contextmanager
def _tracing(root: RunTree | None) -> Iterator[None]:
    """Trace what runs inside under ``root`` -- or, with no root, nothing.

    Explicit both ways. On, ``root`` is the parent every run inside nests
    under. Off, tracing is disabled outright rather than left to the
    environment: LangSmith also switches itself on from variables of its own
    (an old ``LANGCHAIN_TRACING_V2=true``, say) and from any run already
    current, and either would upload a question this pipeline was told not to.
    """
    if root is None:
        with tracing_context(enabled=False, parent=False):
            yield
    else:
        with tracing_context(enabled=True, parent=root):
            yield


def _finish(root: RunTree, answer: str, error: BaseException | None) -> None:
    """End a question's root run with what its outcome earned, and send it.

    A failure is an error, with its message. A question stopped part-way --
    the app's Stop, Ctrl-C at the terminal, a caller that closes the stream
    early -- is not, or LangSmith would count every Stop as a failed question:
    it is tagged ``stopped`` instead. Either way the partial answer is kept as
    the output, the useful part of such a trace. (The model's own run still
    ends in error on a Stop: LangChain reports a closed stream to its tracer as
    one, and nothing here sees it first.)
    """
    if isinstance(error, Exception):
        root.end(outputs={"answer": answer}, error=f"{type(error).__name__}: {error}")
    else:
        if error is not None:
            root.add_tags(["stopped"])
            root.add_metadata({"stopped_by": type(error).__name__})
        root.end(outputs={"answer": answer})
    root.patch()


def _traced(
    root: RunTree | None, pieces: Generator[str, None, None]
) -> Generator[str, None, None]:
    """``pieces``, generated inside ``root``, which ends when the stream does.

    The tracing context is entered around each step rather than across a
    yield. LangSmith keeps it in context variables, and one held across a
    yield leaks into whatever the consumer does between pieces -- and cannot be
    reset when the stream is closed from another context, as a Stop in the
    app can close it. One step is one synchronous frame, so entry and exit
    always pair. The first step is the one that counts: LangChain parents its
    run to whichever run is current when the chain starts, and the chain
    starts on the first pull. With tracing off, each step is still wrapped, in
    a context that disables it.

    Primed: the first ``yield`` gives nothing, comes before any generation, and
    is consumed by ``stream_answer``. A generator's ``finally`` exists only once
    its body has started, so unprimed, a stream closed before its first piece --
    a Stop that lands between retrieval and generation -- would end nothing,
    and that question's trace would never be sent. The model still does not
    start until the caller asks for a piece.
    """
    answer: list[str] = []
    error: BaseException | None = None
    try:
        yield ""
        while True:
            with _tracing(root):
                piece = next(pieces, None)
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
            if root is not None:
                _finish(root, "".join(answer), error)


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the chat model used for generation: Claude, over its API.

    The request it sends -- thinking at its lowest, a fixed effort, no sampling
    parameters -- is the adapter's, in claude_model.py. ``max_tokens`` is passed
    through as the answer's cap. Cheap to call again: it builds a client and
    loads nothing.
    """
    return ClaudeChatModel(model=settings.chat_model, max_tokens=settings.max_tokens)


class _VoyageRerank(VoyageAIRerank):
    """Voyage's reranker, returning the candidates themselves, ids included.

    langchain-voyageai rebuilds each document it returns from its text and
    metadata alone, so the id -- ``source:index:content_hash``, what ties a
    reranked chunk to the stored one in a trace -- is lost, and a
    ``total_tokens`` key is added to every chunk's metadata. Here each kept
    document is the candidate as retrieved, with only the score added, under
    the key every reranker uses.
    """

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        ranked = self._rerank(documents, query)
        return [
            documents[result.index].model_copy(
                update={
                    "metadata": {
                        **documents[result.index].metadata,
                        _SCORE_KEY: result.relevance_score,
                    }
                }
            )
            for result in ranked.results
        ]


def build_reranker(settings: Settings) -> BaseDocumentCompressor:
    """Construct the reranker, Voyage AI's.

    Here, not in ingest.py: reranking is a query-only stage with no ingest-side
    counterpart, so the shared-factory reason that keeps build_embeddings in
    ingest.py doesn't apply — it sits beside build_chat_model, both query-time
    model factories. ``top_k`` is the reranker's own cap, so it returns exactly
    retrieval_k docs and ``retrieve()`` needs no manual slice. RuntimeError,
    never ValueError, for a missing key or a cap below one: this runs on the
    app's pipeline-load path, and Voyage would refuse the cap only at the first
    question.
    """
    key = require_env_key("VOYAGE_API_KEY", "Reranking uses Voyage AI")
    if settings.retrieval_k < 1:
        raise RuntimeError(
            f"RETRIEVAL_K must be at least 1, not {settings.retrieval_k}."
        )
    reranker = _VoyageRerank(
        model=settings.rerank_model,
        top_k=settings.retrieval_k,
        voyage_api_key=SecretStr(key),
    )
    reranker.client, reranker.aclient = voyage_clients(key)
    return reranker


# The largest FETCH_K a search accepts: $vectorSearch caps its candidates at
# 10,000, and langchain-mongodb asks for ten per result.
_MAX_FETCH_K = 1000


class RAGPipeline:
    """Loads the persisted index and answers questions against it."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = None,
        reranker: BaseDocumentCompressor | None = None,
    ) -> None:
        # $vectorSearch refuses a limit below 1, and considers ten candidates
        # per result (langchain-mongodb's oversampling) up to its cap of 10,000
        # -- but only when the first question is searched, after every model
        # had loaded. RuntimeError, not ValueError: this runs on the app's
        # pipeline-load path, whose handler sits below the sidebar.
        if not 1 <= settings.fetch_k <= _MAX_FETCH_K:
            raise RuntimeError(
                f"FETCH_K must be between 1 and {_MAX_FETCH_K}, not {settings.fetch_k}."
            )
        # Before any model is built: a missing, misnamed or empty index is
        # reported before any model client is made.
        require_index(settings)

        self.settings = settings

        # Reopen the existing store via the shared factory, so the same
        # embedding model that indexed the documents also embeds queries.
        # `embeddings` and `llm` are injectable for tests; production leaves
        # both as None and gets the real models. Opening it creates nothing.
        vectorstore = open_store(settings, embeddings)
        # Retrieve a wide candidate set (fetch_k); the reranker below narrows it
        # to retrieval_k. Filtered to this pipeline's own chunks, like every
        # read at ingest, so a foreign record sharing the collection is never
        # retrieved or cited. `reranker` is injectable for tests alongside
        # `embeddings`/`llm`; production leaves it None and builds the real one.
        self._retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.fetch_k, "pre_filter": OWN_CHUNKS}
        )
        self._reranker = reranker or build_reranker(settings)

        # The model's own message chunks, not a StrOutputParser's strings:
        # closing a parser's stream does not stop the model -- langchain-core
        # catches the GeneratorExit and drains the parser's input to the end --
        # so a Stop in the app would keep Claude generating, and billing, until
        # MAX_TOKENS. _generate() extracts the text
        # itself instead.
        self._chain = _PROMPT | (llm or build_chat_model(settings))
        # None while tracing is off. Built with the models, on the load path,
        # so a missing LANGSMITH_API_KEY is reported before the first question.
        self._tracing_client: Client | None = tracing_client(settings)

    def retrieve(self, question: str) -> list[Document]:
        """Return the reranked top chunks for the question.

        Vector search casts a wide net (fetch_k); the reranker — scoring each
        candidate against the question jointly — narrows it to retrieval_k. Both
        calls are wrapped so a store failure, such as a query-time dimension
        mismatch against a collection built with another model, surfaces as the
        RuntimeError both frontends catch rather than a raw pymongo exception.
        The models need no wrapping: their adapters already raise inside the
        union.

        Traced, the search is LangChain's own retriever run; the rerank is not
        a Runnable, so it gets a run of its own here -- the step whose choices
        decide what the model is shown, with what it was given and what it kept,
        scored.
        """
        with provider_errors_as_runtime():
            candidates = self._retriever.invoke(question)
            if not tracing_is_enabled():
                return list(self._reranker.compress_documents(candidates, question))
            with trace(
                type(self._reranker).__name__,
                run_type="retriever",
                inputs={"query": question, "documents": candidates},
                metadata={
                    "model": self.settings.rerank_model,
                    "top_k": self.settings.retrieval_k,
                },
            ) as run:
                ranked = list(self._reranker.compress_documents(candidates, question))
                run.end(outputs={"documents": ranked})
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
        drop it. The model's request stays open for as long as its stream does,
        and a dropped one is closed only when the garbage collector gets to it
        -- which, for a stream a Streamlit script holds in a global, can be
        never: Claude would generate, and bill, on to MAX_TOKENS.

        With tracing on, the question is one trace: a root run opened here,
        current while retrieval runs so the search and rerank nest under it,
        and ended by the stream -- however the stream ends (see `_traced`).
        Its end is why a stream that is never read should still be closed:
        until it is, the root run is not finished.
        """
        root = self._open_root(question)
        try:
            with _tracing(root):
                docs = self.retrieve(question)
        except BaseException as exc:
            if root is not None:
                _finish(root, "", exc)
            raise
        answer = _traced(root, self._generate(question, docs))
        next(answer)  # primes it; see _traced
        return docs, answer

    def _open_root(self, question: str) -> RunTree | None:
        """Start this question's root run and send it, or None if not traced.

        Inside a run that is already current -- `rag eval`'s, which traces each
        question it asks -- the root nests under that run, in its project, so
        an experiment shows each answer's whole trace. Otherwise it starts a
        trace of its own in LANGSMITH_PROJECT.
        """
        if self._tracing_client is None:
            return None
        inputs = {"question": question}
        parent = get_current_run_tree()
        if parent is not None:
            root = parent.create_child(
                name="RAGPipeline", run_type="chain", inputs=inputs
            )
        else:
            root = RunTree(
                name="RAGPipeline",
                run_type="chain",
                inputs=inputs,
                ls_client=self._tracing_client,
                project_name=self.settings.langsmith_project,
            )
        root.post()
        return root

    def answer(self, question: str) -> Answer:
        """Retrieve context, then generate a grounded answer with sources.

        The all-at-once shape, for library callers that just want the finished
        string; both frontends stream instead. A join over the same path rather
        than a second call into the chain.
        """
        docs, chunks = self.stream_answer(question)
        return Answer(text="".join(chunks), sources=docs)
