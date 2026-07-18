"""Query phase: embed question -> search -> generate a grounded answer.

``RAGPipeline`` loads the persisted Chroma index and a Claude chat model once,
then answers questions against it. Both the CLI and the Streamlit app build a
single pipeline and reuse it across queries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag_pipeline.config import Settings
from rag_pipeline.ingest import build_embeddings

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
    seen: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.append(src)
    return seen


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

        # Reload the existing store; the same embedding model that indexed the
        # documents must embed the queries, so we reuse the shared factory.
        # `embeddings` and `llm` are injectable for tests; production leaves
        # both as None and gets local embeddings + ChatAnthropic.
        vectorstore = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings or build_embeddings(settings),
            persist_directory=str(settings.persist_dir),
        )
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

    def answer(self, question: str) -> Answer:
        """Retrieve context, then generate a grounded answer with sources."""
        docs = self.retrieve(question)
        text = self._chain.invoke(
            {"context": format_docs(docs), "question": question}
        )
        return Answer(text=text, sources=docs)
