"""Query phase: embed question -> search -> rerank -> generate a grounded answer.

``RAGPipeline`` opens the persisted Chroma collection and the local models
once, then answers questions against them. Both the CLI and the Streamlit app
build a single pipeline and reuse it across queries.
"""

from __future__ import annotations

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
        """
        with store_errors_as_runtime():
            candidates = self._retriever.invoke(question)
            return list(self._reranker.compress_documents(candidates, question))

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
            self._chain.stream({"context": format_docs(docs), "question": question}),
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
        """
        docs = self.retrieve(question)
        return docs, self._generate(question, docs)

    def answer(self, question: str) -> Answer:
        """Retrieve context, then generate a grounded answer with sources.

        The all-at-once shape, for library callers that just want the finished
        string; both frontends stream instead. A join over the same path rather
        than a second call into the chain.
        """
        docs, chunks = self.stream_answer(question)
        return Answer(text="".join(chunks), sources=docs)
