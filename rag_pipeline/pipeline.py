"""Query phase: embed question -> search -> generate a grounded answer.

``RAGPipeline`` loads the persisted Chroma index and a Claude chat model once,
then answers questions against it. Both the CLI and the Streamlit app build a
single pipeline and reuse it across queries.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag_pipeline.config import Settings
from rag_pipeline.ingest import open_store

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


def format_docs(docs: list[Document]) -> str:
    """Render retrieved chunks into a single context string, labeled by source."""
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def unique_sources(docs: list[Document]) -> list[str]:
    """Distinct source files across the retrieved chunks, in retrieval order."""
    # dict.fromkeys preserves insertion order while dropping duplicates.
    return list(dict.fromkeys(doc.metadata.get("source", "unknown") for doc in docs))


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the Claude chat model used for generation.

    No temperature/top_p: grounding comes from the retrieved context, and
    omitting sampling params keeps this safe across models — Opus 4.8, for
    instance, rejects them outright. Reads ANTHROPIC_API_KEY from the
    environment.
    """
    return ChatAnthropic(model=settings.chat_model, max_tokens=settings.max_tokens)


class RAGPipeline:
    """Loads the persisted index and answers questions against it."""

    def __init__(
        self,
        settings: Settings,
        embeddings: Embeddings | None = None,
        llm: Runnable | None = None,
    ) -> None:
        if not settings.persist_dir.exists():
            raise FileNotFoundError(
                f"No index found at {settings.persist_dir}. "
                "Run `rag ingest` (or `uv run rag ingest`) first."
            )
        # Fast-fail on a missing key *before* loading the embedding model — but
        # only when we're going to build the real Claude client (an injected
        # `llm`, as in tests, needs no key).
        if llm is None and not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Generation uses Claude; set the "
                "key in your environment or a .env file (see .env.example). "
                "Embedding/ingest runs locally and needs no key."
            )

        self.settings = settings

        # Reload the existing store via the shared factory, so the same
        # embedding model that indexed the documents also embeds queries.
        # `embeddings` and `llm` are injectable for tests; production leaves
        # both as None and gets local embeddings + ChatAnthropic.
        vectorstore = open_store(settings, embeddings)
        # `persist_dir.exists()` alone is too weak: an empty directory or a
        # COLLECTION_NAME that doesn't match what was ingested yields a
        # silently-empty collection (get_or_create), so every question would be
        # answered "I don't know". Fail loudly instead.
        if not vectorstore.get(limit=1)["ids"]:
            raise FileNotFoundError(
                f"Index at {settings.persist_dir} (collection "
                f"'{settings.collection_name}') is empty. Run `rag ingest` first, "
                "and check COLLECTION_NAME matches the one used to ingest."
            )
        self._retriever = vectorstore.as_retriever(
            search_kwargs={"k": settings.retrieval_k}
        )

        # StrOutputParser extracts plain text whether the model returns a string
        # or structured content blocks.
        self._chain = _PROMPT | (llm or build_chat_model(settings)) | StrOutputParser()

    def retrieve(self, question: str) -> list[Document]:
        """Return the chunks most relevant to the question."""
        return self._retriever.invoke(question)

    def _generate(self, question: str, docs: list[Document]) -> Iterator[str]:
        """Yield the grounded answer in pieces, as the model produces them.

        The single generation path, so the provider-error translation lives here
        only. It must wrap the *iteration*: `.stream()` is lazy, so a failed
        request surfaces while the generator is being consumed, not when it is
        created.
        """
        try:
            yield from self._chain.stream(
                {"context": format_docs(docs), "question": question}
            )
        except anthropic.APIError as exc:
            # Translate provider errors (bad/expired key, rate limit, network)
            # into a generic RuntimeError, so frontends handle a failed
            # generation uniformly without depending on the Anthropic SDK.
            raise RuntimeError(f"Claude API request failed: {exc}") from exc

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
