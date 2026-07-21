"""Shared pytest fixtures.

Tests use a deterministic fake embedding model instead of the real Voyage AI
one, so the suite runs fast and fully offline (no API calls, no network). The
fake still round-trips text->vector->store->retrieve, which is what the
ingest/pipeline tests exercise.
"""

from __future__ import annotations

import socket

import pytest
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import ENV_VARS, Settings
from rag_pipeline.ingest import reset_store_cache

# Exposed as fixtures below rather than imported: `tests/` has no __init__.py,
# so `from tests.conftest import ...` would load this file a second time under a
# different module name. Fixtures are how pytest shares these.
_CANNED_ANSWER = "Chunks overlap to preserve context across boundaries. (a.md)"
_PARTIAL_ANSWER = "a partial ans"


@pytest.fixture
def canned_answer() -> str:
    """What the faked chat model answers with, for tests that assert on it."""
    return _CANNED_ANSWER


@pytest.fixture
def partial_answer() -> str:
    """What `fail_mid_stream` emits before raising."""
    return _PARTIAL_ANSWER


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Deterministic, offline embeddings.

    Same text -> same vector, so querying with a chunk's exact text retrieves
    that chunk. Good enough to test the store/retrieve wiring without the real
    embedding model.
    """
    return DeterministicFakeEmbedding(size=32)


class _SliceReranker(BaseDocumentCompressor):
    """Offline stand-in for VoyageAIRerank: keep retrieval order, cap at top_k.

    Enough to exercise the retrieve->rerank wiring and the top_k contract with no
    network call. A test that must prove reranking *reorders* builds a reversing
    variant of its own instead.
    """

    top_k: int

    def compress_documents(self, documents, query, callbacks=None):
        return documents[: self.top_k]


@pytest.fixture
def fake_reranker(settings) -> BaseDocumentCompressor:
    """Deterministic, offline reranker (identity + truncate to retrieval_k)."""
    return _SliceReranker(top_k=settings.retrieval_k)


@pytest.fixture
def sample_data_dir(tmp_path):
    """A data directory that exercises the loader: a nested subdirectory, a
    whitespace-only file, and an unsupported extension (both must be skipped).

    `a.md` opens with a Markdown heading because the corpus is Markdown and one
    frontend renders retrieved passages back to the user: without syntax in the
    fixture, nothing distinguishes displaying a passage as text from parsing it.
    """
    root = tmp_path / "data"
    files = {
        "a.md": "# Alpha\n\nAlpha topic about apples and orchards.\n",
        "sub/b.txt": "Beta topic about bicycles and boats.\n",
        "empty.md": "   \n",  # whitespace only -> skipped
        "notes.rst": "unsupported extension -> skipped",  # bad suffix -> skipped
    }
    for name, content in files.items():
        # Create parents per entry, so adding a new nested path above just works.
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path, sample_data_dir) -> Settings:
    """Settings pointed at the sample data and an isolated persist directory."""
    return Settings(
        data_dir=sample_data_dir,
        persist_dir=tmp_path / "chroma",
        collection_name="test_docs",
        chunk_size=200,
        chunk_overlap=40,
        retrieval_k=2,
    )


@pytest.fixture
def wired_env(settings, fake_embeddings, fake_reranker, monkeypatch) -> Settings:
    """A frontend's view of the world: fixture settings in the environment, fakes
    behind both factories.

    Neither frontend takes injected models — `app.py` is a script and `cli.py`
    builds its own `Settings.from_env()` — so the environment is how a fixture's
    temp index reaches them, and the two factories are where they reach a model.
    That is the same seam for both, which is why this is one fixture rather than
    a copy in each frontend's test file.

    Derived from ENV_VARS rather than spelled out: a hand-kept list would
    silently stop covering a new setting, and config.py's import-time
    load_dotenv() means the developer's own .env would answer whichever name was
    missed. The API key is set because RAGPipeline fast-fails without one
    whenever it builds the model itself; the factory is patched below, so nothing
    ever authenticates with it.
    """
    for var in ENV_VARS:
        monkeypatch.setenv(var, str(getattr(settings, var.lower())))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")

    monkeypatch.setattr(ingest_mod, "build_embeddings", lambda _s: fake_embeddings)
    monkeypatch.setattr(
        pipeline_mod,
        "build_chat_model",
        lambda _s: FakeListChatModel(responses=[_CANNED_ANSWER]),
    )
    monkeypatch.setattr(pipeline_mod, "build_reranker", lambda _s: fake_reranker)
    return settings


@pytest.fixture
def fail_mid_stream(monkeypatch):
    """Make generation emit, then fail — as a real one would.

    Patched at `_generate` rather than `stream_answer` so real retrieval still
    runs and the frontend still receives real sources. Failing partway rather
    than at the call is the honest shape: generation is lazy, so a provider
    error lands while the frontend is already rendering.

    Here rather than in one frontend's test file because both frontends have to
    survive it, and a dependency on a private method is worth declaring once.
    """

    def arrange(exc: BaseException) -> None:
        def generate(_self, _question, _docs):
            yield _PARTIAL_ANSWER
            raise exc

        monkeypatch.setattr(pipeline_mod.RAGPipeline, "_generate", generate)

    return arrange


@pytest.fixture(autouse=True)
def _reset_chroma_client_cache():
    """chromadb caches its client per directory within a process. Clear it at
    each test boundary so tests don't see another test's client, and so an
    in-process re-ingest behaves like a fresh CLI run."""
    reset_store_cache()
    yield
    reset_store_cache()


@pytest.fixture(autouse=True)
def _offline_only(monkeypatch):
    """Fail any test that opens a network socket.

    The offline guarantee rests on tests injecting fakes for the embedding model
    and the LLM. That is a convention, and a test that simply forgets to pass
    ``embeddings=`` falls back to the real Voyage AI path, which makes an HTTP
    request to embed. Blocking the socket catches every spelling of that
    mistake, including ones no grep would find, because the failure is defined
    by behavior rather than by the name of a class.
    """

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "test opened a network socket -- the suite must run offline; "
            "inject fake embeddings/LLM instead of the real model"
        )

    # Both are load-bearing: httpcore reaches the network via create_connection.
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
