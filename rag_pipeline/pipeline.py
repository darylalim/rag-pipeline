"""Query phase: embed question -> search -> rerank -> generate a grounded answer.

``RAGPipeline`` opens the Atlas Vector Search collection and a Claude chat model
once, then answers questions against it. Both the CLI and the Streamlit app build
a single pipeline and reuse it across queries.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypedDict

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_voyageai import VoyageAIRerank

from rag_pipeline.config import Settings, require_env_key
from rag_pipeline.ingest import open_store, provider_errors_as_runtime

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
    """Construct the Claude chat model used for generation.

    No temperature/top_p: grounding comes from the retrieved context, and
    omitting sampling params keeps this safe across models — Opus 4.8, for
    instance, rejects them outright. Reads ANTHROPIC_API_KEY from the
    environment.
    """
    return ChatAnthropic(model=settings.chat_model, max_tokens=settings.max_tokens)


def build_reranker(settings: Settings) -> BaseDocumentCompressor:
    """Construct the Voyage AI reranker.

    Here, not in ingest.py: reranking is a query-only stage with no ingest-side
    counterpart, so the shared-factory reason that keeps build_embeddings in
    ingest.py doesn't apply — it sits beside build_chat_model, both query-time
    model factories. ``top_k`` is the reranker's own cap, so it returns exactly
    retrieval_k docs and ``retrieve()`` needs no manual slice.

    Guards VOYAGE_API_KEY up front like build_embeddings, so a missing key fails
    fast inside the caught union rather than as a raw validation error escaping to
    a frontend.
    """
    require_env_key("VOYAGE_API_KEY", "Reranking uses Voyage AI")
    return VoyageAIRerank(model=settings.rerank_model, top_k=settings.retrieval_k)


class RAGPipeline:
    """Loads the persisted index and answers questions against it."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = None,
        reranker: BaseDocumentCompressor | None = None,
    ) -> None:
        # Fast-fail on a missing key *before* loading the embedding model — but
        # only when we're going to build the real Claude client (an injected
        # `llm`, as in tests, needs no key).
        if llm is None:
            require_env_key("ANTHROPIC_API_KEY", "Generation uses Claude")

        self.settings = settings

        # Reopen the existing store via the shared factory, so the same
        # embedding model that indexed the documents also embeds queries.
        # `embeddings` and `llm` are injectable for tests; production leaves
        # both as None and gets the embedding model + ChatAnthropic. Opening
        # the store pings the cluster, so an unreachable/paused one fails here.
        vectorstore = open_store(settings, embeddings)
        collection = vectorstore.collection
        with provider_errors_as_runtime():
            index_info = list(
                collection.list_search_indexes(settings.vector_index_name)
            )
            has_documents = collection.count_documents(
                {"content_hash": {"$exists": True}}, limit=1
            )

        # Two distinct failures, checked in order. A full collection with no
        # (or a not-yet-built) vector index answers every question "I don't know"
        # off zero rows — and the empty-collection guard below would not catch
        # it, because the documents are there. Unlike Chroma, Atlas builds no
        # index implicitly on write, so this is the query-side half of what
        # `rag ingest` promises.
        if not index_info or not index_info[0].get("queryable"):
            raise FileNotFoundError(
                f"No queryable vector index '{settings.vector_index_name}' on "
                f"{settings.mongodb_db}.{settings.collection_name}. Run `rag ingest` "
                "first (index builds are asynchronous)."
            )
        # An empty (or wrong) namespace yields a silently-empty result set —
        # Mongo creates a namespace implicitly on first write and returns zero
        # documents, with no error, for one that was never ingested into. Scoped
        # to this pipeline's own chunks, so a collection holding only unrelated
        # documents (or only the version marker) reads as empty-for-this-pipeline.
        if has_documents == 0:
            raise FileNotFoundError(
                f"Namespace {settings.mongodb_db}.{settings.collection_name} is "
                "empty. Run `rag ingest` first, and check MONGODB_DB/COLLECTION_NAME "
                "match the ones used to ingest."
            )
        # Retrieve a wide candidate set (fetch_k); the reranker below narrows it
        # to retrieval_k. `exact` runs exact (ENN) search, correct for a corpus
        # under ~10k chunks and free of numCandidates tuning. `reranker` is
        # injectable for tests alongside `embeddings`/`llm`; production leaves it
        # None and builds the real one.
        self._retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.fetch_k, "exact": True}
        )
        self._reranker = reranker or build_reranker(settings)

        # StrOutputParser extracts plain text whether the model returns a string
        # or structured content blocks.
        self._chain = _PROMPT | (llm or build_chat_model(settings)) | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        """Return the reranked top chunks for the question.

        Vector search casts a wide net (fetch_k); the Voyage reranker — a
        cross-encoder scoring each candidate against the question jointly —
        narrows it to retrieval_k. Both calls are wrapped so a Voyage provider
        error (embedding *or* reranking), or a query-time dimension mismatch
        against a stale index, surfaces as the RuntimeError both frontends catch
        rather than a raw voyageai/pymongo/bson exception.
        """
        with provider_errors_as_runtime():
            candidates = self._retriever.invoke(question)
            return list(self._reranker.compress_documents(candidates, question))

    def _generate(self, question: str, docs: list[Document]) -> Iterator[str]:
        """Yield the grounded answer in pieces, as the model produces them.

        The single generation path, so every generation-level failure is raised
        here rather than in each frontend: a provider error, and a response that
        arrives empty. The provider translation must wrap the *iteration* —
        `.stream()` is lazy, so a failed request surfaces while the generator is
        being consumed, not when it is created.
        """
        produced_content = False
        try:
            for chunk in self._chain.stream(
                {"context": format_docs(docs), "question": question}
            ):
                produced_content = produced_content or bool(chunk.strip())
                yield chunk
        except anthropic.APIError as exc:
            # Translate provider errors (bad/expired key, rate limit, network)
            # into a generic RuntimeError, so frontends handle a failed
            # generation uniformly without depending on the Anthropic SDK.
            raise RuntimeError(f"Claude API request failed: {exc}") from exc
        if not produced_content:
            # Otherwise each frontend presents nothing as a cited answer — a
            # blank chat bubble above a full Sources expander, or the CLI's
            # "Sources:" block under an empty line — claiming the strongest
            # possible grounding for no content at all.
            raise RuntimeError("Claude returned an empty answer")

    def stream_answer(self, question: str) -> tuple[list[Document], Iterator[str]]:
        """Search, then hand back the sources and a lazy stream of the answer.

        Both halves in one call because every frontend needs both, and splitting
        them made each frontend re-implement the same three steps. Returning the
        docs alongside the stream also means the citations shown are provably the
        ones the answer was generated from, not a second search that could drift.

        Note the two halves evaluate at different times: retrieval has already
        run when this returns (so a caller can put a spinner around just this
        call), while generation has not started and will not until the iterator
        is consumed.
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
