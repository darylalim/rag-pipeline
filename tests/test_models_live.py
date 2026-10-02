"""Live tests: the real local chat model and Voyage AI's real API.

Deselected by default -- pyproject's ``addopts`` carries ``-m "not models"`` --
and run with ``uv run pytest -m models``. Each test skips, rather than fails,
where it cannot run: without MLX (anything but Apple Silicon macOS) or the chat
model downloaded, without ``VOYAGE_API_KEY``, or -- the ingest-and-answer pass
over ``data/`` -- without the sample document its question is about, since
``data/`` is the user's to replace.

The Voyage tests check what the fakes cannot: that the factories ask for
documents and questions the right way round (a query embedded as a document
still retrieves something, just worse), at the configured width, and that the
reranker's order is the one ``retrieve()`` relies on. They cost a fraction of
a cent. These are the only tests allowed onto the network, and the only ones
that keep ``VOYAGE_API_KEY`` (conftest's ``_offline`` and ``_no_real_store``
exempt the ``models`` mark); ``MONGODB_URI`` is still the test container's.

The chat model loads once per process (``load_mlx_model`` memoizes it), so the
fixture below is cheap after the first test that needs it.
"""

from __future__ import annotations

import dataclasses
import math
import os

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline.config import Settings
from rag_pipeline.mlx_models import MLXChatModel, resolve_model_path
from rag_pipeline.pipeline import RAGPipeline, build_reranker

pytestmark = pytest.mark.models

_DEFAULTS = Settings()

_FACTS = [
    "Paris is the capital and most populous city of France, on the Seine river.",
    "The Great Barrier Reef lies off the coast of Queensland in north-eastern Australia.",
    "Berlin is the capital and largest city of Germany by both area and population.",
]


def _require(model_id: str) -> None:
    """Skip unless MLX is importable and ``model_id`` is fully downloaded."""
    pytest.importorskip("mlx_lm", reason="MLX needs Apple Silicon macOS")
    try:
        resolve_model_path(model_id)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _require_voyage() -> None:
    if not os.environ.get("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY is not set")


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture
def embedder() -> Embeddings:
    _require_voyage()
    return ingest_mod.build_embeddings(_DEFAULTS)


@pytest.fixture
def chat() -> MLXChatModel:
    _require(_DEFAULTS.chat_model)
    # Small, to keep the run short; every expected answer fits well inside it.
    return MLXChatModel(model_id=_DEFAULTS.chat_model, max_tokens=64)


# --- Voyage AI ----------------------------------------------------------------


def test_embeddings_are_unit_vectors_at_the_configured_width(embedder):
    vectors = [embedder.embed_query("What is RAG?"), *embedder.embed_documents(["x"])]

    for v in vectors:
        assert len(v) == _DEFAULTS.embedding_dimensions
        assert math.sqrt(_dot(v, v)) == pytest.approx(1.0, abs=1e-3)


def test_a_question_retrieves_its_passage(embedder):
    """Each question, embedded as a query, is nearest its own passage, embedded
    as a document -- Berlin a hard negative for France's capital."""
    docs = embedder.embed_documents(_FACTS)
    for i, question in enumerate(
        [
            "What is the capital of France?",
            "Where is the Great Barrier Reef?",
            "Which city is Germany's capital?",
        ]
    ):
        q = embedder.embed_query(question)
        scores = [_dot(q, d) for d in docs]
        assert scores.index(max(scores)) == i


def test_the_reranker_puts_the_relevant_passage_first():
    _require_voyage()
    reranker = build_reranker(dataclasses.replace(_DEFAULTS, retrieval_k=3))
    docs = [Document(page_content=text, id=str(i)) for i, text in enumerate(_FACTS)]

    ranked = list(reranker.compress_documents(docs, "What is the capital of France?"))

    assert ranked[0].id == "0"
    scores = [d.metadata["relevance_score"] for d in ranked]
    assert scores == sorted(scores, reverse=True)
    assert len(ranked) == 3


# --- generation ---------------------------------------------------------------

# The same shape as the pipeline's prompt: grounding rules as the system turn,
# context and question as the human turn.
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Answer using only the provided context. If the context does not "
                "contain the answer, say you don't know based on the provided "
                "documents. Be concise, and cite the source file in parentheses."
            ),
        ),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)
_CONTEXT = (
    "[overlap.md]\nChunk overlap repeats the last 200 characters of one chunk at "
    "the start of the next, so a sentence that straddles a boundary is not lost."
)


def _stream(chat: MLXChatModel, question: str) -> list[str]:
    # The pipeline's own chain shape: prompt then model, with no output parser.
    chain = _PROMPT | chat
    return [
        str(message.text)
        for message in chain.stream({"context": _CONTEXT, "question": question})
    ]


def test_the_chat_model_streams_a_grounded_answer(chat):
    pieces = _stream(chat, "How many characters does chunk overlap repeat?")
    answer = "".join(pieces)

    assert len([p for p in pieces if p]) > 1  # streamed, not delivered whole
    assert "200" in answer
    # Thinking left on streams the model's reasoning into the answer instead.
    assert "<think>" not in answer
    assert "</think>" not in answer
    assert not mlx_models._GENERATION_LOCK.locked()


def test_the_chat_model_declines_what_the_context_does_not_say(chat):
    answer = "".join(_stream(chat, "What is the capital of Australia?")).lower()

    assert answer.strip()
    assert "canberra" not in answer
    assert any(
        phrase in answer
        for phrase in (
            "don't know",
            "do not know",
            "not contain",
            "does not",
            "no information",
        )
    )


def test_the_chat_model_reports_why_it_stopped(chat):
    """The metadata the MAX_TOKENS note is read from, as mlx-lm really sends it.

    The pipeline says an answer was cut off only when the model reports
    "length", and the fake that CI runs against cannot follow mlx-lm: were an
    upgrade to rename that value or move it, the note would vanish with every
    other test still green. An answer that fits must stop on its own, and one
    given four tokens must be cut off at exactly four.
    """
    prompt = _PROMPT.invoke(
        {
            "context": _CONTEXT,
            "question": "How many characters does chunk overlap repeat?",
        }
    )
    finished = chat.invoke(prompt)
    cut_off = MLXChatModel(model_id=_DEFAULTS.chat_model, max_tokens=4).invoke(prompt)

    assert finished.response_metadata["finish_reason"] == "stop"
    assert finished.usage_metadata is not None
    assert 0 < finished.usage_metadata["output_tokens"] < chat.max_tokens
    assert cut_off.response_metadata["finish_reason"] == "length"
    assert cut_off.usage_metadata is not None
    assert cut_off.usage_metadata["output_tokens"] == 4


# --- the whole pipeline -------------------------------------------------------


def test_ingest_then_answer_over_the_repo_corpus(atlas):
    """Voyage and the chat model together, through the real factories, over
    data/, in the test's own database on the atlas-local container."""
    _require(_DEFAULTS.chat_model)
    _require_voyage()
    # The question and both assertions are about the sample corpus, and data/ is
    # the user's to replace: a replaced corpus says nothing about whether the
    # models are driven correctly, so it is a skip, not a failure.
    if not (_DEFAULTS.data_dir / "rag_concepts.md").is_file():
        pytest.skip("data/rag_concepts.md is gone; this asks what only it answers")
    settings = dataclasses.replace(_DEFAULTS, mongodb_db=atlas)

    assert ingest_mod.ingest(settings) > 0
    result = RAGPipeline(settings).answer(
        "What chunk overlap is typically recommended?"
    )

    assert "rag_concepts.md" in {d.metadata["source"] for d in result.sources}
    assert "20" in result.text
