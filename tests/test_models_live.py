"""Live tests: the real local models, loaded from the Hugging Face cache.

Deselected by default -- pyproject's ``addopts`` carries ``-m "not models"`` --
and run with ``uv run pytest -m models``. Each test skips, rather than fails,
where it cannot run: without MLX (anything but Apple Silicon macOS), without its
model downloaded, or -- the ingest-and-answer pass over ``data/`` -- without the
sample document its question is about, since ``data/`` is the user's to replace.

These are the only tests that notice a wrong recipe. The embedder and reranker
implement their model family's prompt format and pooling by hand, and a recipe
that is subtly wrong -- a missing appended token, a space in the prompt --
still produces unit vectors and plausible rankings. Reproducing the model cards'
published scores is what pins the recipe down. The rest checks the properties
the pipeline relies on: batching changes no result, a narrower embedding is the
normalized prefix of the full one, and the chat model streams a grounded answer
with no reasoning in it and reports why it stopped.

The models load once per process (``load_mlx_model`` memoizes them), so the
fixtures below are cheap after the first test that needs each one, and every
load happens under conftest's socket block -- which is itself the check that
loading never reaches the network.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline.config import Settings
from rag_pipeline.mlx_models import MLXChatModel, QwenVLReranker, resolve_model_path
from rag_pipeline.pipeline import RAGPipeline

pytestmark = pytest.mark.models

# The default checkpoints: the card values below are theirs.
_DEFAULTS = Settings()

# Qwen3-VL-Embedding-2B's model card: four queries against one document, with
# the default instruction on both sides, and the cosine similarities it prints.
_CARD_QUERIES = [
    "A woman playing with her dog on a beach at sunset.",
    "Pet owner training dog outdoors near water.",
    "Woman surfing on waves during a sunny day.",
    "City skyline view from a high-rise building at night.",
]
_CARD_DOC = (
    "A woman shares a joyful moment with her golden retriever on a sun-drenched "
    "beach at sunset, as the dog offers its paw in a heartwarming display of "
    "companionship and trust."
)
_CARD_SIMILARITIES = [0.8158, 0.5195, 0.3884, 0.1093]

# Qwen3-VL-Reranker-2B's model card: its text example scores this pair (under
# the card's own instruction) 0.8613.
_CARD_RERANK_INSTRUCTION = "Retrieve images or text relevant to the user's query."
_CARD_RERANK_SCORE = 0.8613

_FACTS = [
    "Paris is the capital and most populous city of France, on the Seine river.",
    "The Great Barrier Reef lies off the coast of Queensland in north-eastern Australia.",
    "Berlin is the capital and largest city of Germany by both area and population.",
    "Sourdough bread is leavened with a starter of wild yeast and lactic acid bacteria.",
    (
        "Chunk overlap repeats the tail of one chunk at the head of the next, so a "
        "sentence that straddles a boundary is still retrievable in full."
    ),
]

# Mixed lengths, so a batch holds padding: that is what batching could get wrong.
_MIXED = [
    "hi",
    *_FACTS,
    _CARD_DOC,
    " ".join(_FACTS) * 3,
    "A reranker reads the question and a passage together. " * 12,
    "Voyager 1 crossed the heliopause in 2012.",
]


def _require(model_id: str) -> None:
    """Skip unless MLX is importable and ``model_id`` is fully downloaded."""
    pytest.importorskip("mlx_lm", reason="MLX needs Apple Silicon macOS")
    try:
        resolve_model_path(model_id)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


def _dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def _norm(v: list[float]) -> float:
    return math.sqrt(_dot(v, v))


@pytest.fixture
def embedder() -> Embeddings:
    _require(_DEFAULTS.embedding_model)
    return ingest_mod.build_embeddings(_DEFAULTS)


def _reranker(top_n: int) -> QwenVLReranker:
    _require(_DEFAULTS.rerank_model)
    return QwenVLReranker(model_id=_DEFAULTS.rerank_model, top_n=top_n)


@pytest.fixture
def chat() -> MLXChatModel:
    _require(_DEFAULTS.chat_model)
    # Small, to keep the run short; every expected answer fits well inside it.
    return MLXChatModel(model_id=_DEFAULTS.chat_model, max_tokens=64)


# --- embeddings ---------------------------------------------------------------


def test_the_embedder_reproduces_the_model_card(embedder):
    """Without the appended <|endoftext|> these come out 0.7786/0.5503/0.5241/
    0.2859 -- plausible numbers, and wrong by up to 0.18."""
    queries = embedder.embed_documents(_CARD_QUERIES)
    (doc,) = embedder.embed_documents([_CARD_DOC])

    similarities = [_dot(q, doc) for q in queries]

    assert similarities == pytest.approx(_CARD_SIMILARITIES, abs=0.01)


def test_embedding_vectors_are_unit_length_at_the_configured_width(embedder):
    vectors = [embedder.embed_query("What is RAG?"), *embedder.embed_documents(["x"])]

    for v in vectors:
        assert len(v) == _DEFAULTS.embedding_dimensions
        assert _norm(v) == pytest.approx(1.0, abs=1e-4)


def test_batching_does_not_change_an_embedding(embedder):
    """Right padding under a causal mask cannot affect a real token; this is
    the check that the pooled position is still each text's own last token."""
    batched = embedder.embed_documents(_MIXED)
    single = [embedder.embed_documents([text])[0] for text in _MIXED]

    for b, s in zip(batched, single, strict=True):
        assert _dot(b, s) >= 0.999


def test_a_question_retrieves_its_passage(embedder):
    """``embed_query`` takes a different instruction from the card's, so the
    card test does not cover it."""
    docs = embedder.embed_documents(_FACTS[:3])
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


def test_a_narrower_embedding_is_the_normalized_prefix_of_the_full_one(embedder):
    """Matryoshka: EMBEDDING_DIMENSIONS below the native width keeps the leading
    values and re-normalizes, rather than needing another model."""
    width = 256
    narrow_embedder = ingest_mod.build_embeddings(
        dataclasses.replace(_DEFAULTS, embedding_dimensions=width)
    )

    (full,) = embedder.embed_documents([_CARD_DOC])
    (narrow,) = narrow_embedder.embed_documents([_CARD_DOC])

    prefix_norm = _norm(full[:width])
    assert len(narrow) == width
    assert _norm(narrow) == pytest.approx(1.0, abs=1e-4)
    assert _dot(narrow, [x / prefix_norm for x in full[:width]]) >= 0.9999


# --- reranking ----------------------------------------------------------------


def _docs(texts: list[str]) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": f"{i}.md"}, id=str(i))
        for i, t in enumerate(texts)
    ]


def test_the_reranker_reproduces_the_model_card(monkeypatch):
    """Under the card's instruction, not the adapter's default; the prompt
    format, the yes/no token ids and the score formula all show up here."""
    reranker = _reranker(top_n=1)
    monkeypatch.setattr(mlx_models, "_RERANK_INSTRUCTION", _CARD_RERANK_INSTRUCTION)

    (scored,) = reranker.compress_documents(_docs([_CARD_DOC]), _CARD_QUERIES[0])

    assert scored.metadata["relevance_score"] == pytest.approx(
        _CARD_RERANK_SCORE, abs=0.01
    )


def test_the_score_head_reads_a_quantized_embedding():
    """A quantized reranker checkpoint scores off real weights, not packed ones.

    In a quantized conversion ``embed_tokens`` is a ``QuantizedEmbedding``, whose
    ``.weight`` is packed uint32 words -- indexed, the yes-minus-no vector is the
    wrong width and every rerank fails. A small layer rather than a checkpoint,
    so this needs MLX but no download; and the unquantized case must come out
    bit for bit as indexing would, or the default's card scores move.
    """
    pytest.importorskip("mlx.core", reason="MLX needs Apple Silicon macOS")
    import mlx.core as mx
    from mlx import nn

    mx.random.seed(0)
    plain = nn.Embedding(32, 64)
    exact = (plain.weight[3] - plain.weight[5]).astype(mx.float32)
    quantized = nn.QuantizedEmbedding.from_embedding(plain, group_size=32, bits=8)

    assert mx.array_equal(mlx_models._yes_minus_no(plain, (3, 5)), exact)
    vector = mlx_models._yes_minus_no(quantized, (3, 5))
    assert vector.shape == exact.shape
    assert mx.allclose(vector, exact, atol=0.02)


def test_a_relevant_passage_outranks_irrelevant_ones():
    """Berlin is the hard negative: a capital city, just not France's."""
    ranked = _reranker(top_n=3).compress_documents(
        _docs(_FACTS[:3]), "What is the capital of France?"
    )

    assert ranked[0].id == "0"
    scores = [d.metadata["relevance_score"] for d in ranked]
    assert scores[0] > 0.5 > scores[1] >= scores[2]


def test_batching_does_not_change_a_rerank_score():
    """More candidates than one micro-batch, of mixed lengths: each score must
    not depend on which other candidates shared its padded batch -- in pure bf16
    it did, by enough to change the top four."""
    question = "How does chunk overlap help a text splitter?"
    reranker = _reranker(top_n=len(_MIXED))

    batched = {
        d.id: d.metadata["relevance_score"]
        for d in reranker.compress_documents(_docs(_MIXED), question)
    }
    single = {
        doc.id: reranker.compress_documents([doc], question)[0].metadata[
            "relevance_score"
        ]
        for doc in _docs(_MIXED)
    }

    assert batched == pytest.approx(single, abs=1e-3)


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


def test_ingest_then_answer_over_the_repo_corpus(tmp_path):
    """The three models together, through the real factories, over data/."""
    for model_id in (
        _DEFAULTS.embedding_model,
        _DEFAULTS.rerank_model,
        _DEFAULTS.chat_model,
    ):
        _require(model_id)
    # The question and both assertions are about the sample corpus, and data/ is
    # the user's to replace: a replaced corpus says nothing about whether the
    # models are driven correctly, so it is a skip, not a failure.
    if not (_DEFAULTS.data_dir / "rag_concepts.md").is_file():
        pytest.skip("data/rag_concepts.md is gone; this asks what only it answers")
    settings = dataclasses.replace(_DEFAULTS, persist_dir=tmp_path / "chroma")

    assert ingest_mod.ingest(settings) > 0
    ingest_mod.reset_store_cache()
    result = RAGPipeline(settings).answer(
        "What chunk overlap is typically recommended?"
    )

    assert "rag_concepts.md" in {d.metadata["source"] for d in result.sources}
    assert "20" in result.text
