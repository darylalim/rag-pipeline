"""Shared pytest fixtures.

Tests run against a real Chroma store -- an in-process ``PersistentClient``
under a per-test temp directory, so there is no server, no container and no
network. The models are the part that is never real: a deterministic fake
embedding model, a fake reranker and a fake chat model are injected, because the
real ones are multi-gigabyte MLX checkpoints (about 22 GB together) that only
run on Apple Silicon.

Two autouse guards back that injection convention, since forgetting it is
silent otherwise:

- ``_no_real_models`` makes MLX unimportable, so a test that forgets
  ``embeddings=``/``reranker=``/``llm=`` fails with the loader's RuntimeError
  instead of quietly loading weights from the local Hugging Face cache. With the
  models cached, no socket is involved, so a network block alone would not
  notice. Tests marked ``models`` (deselected by default) opt out of it.
- ``_offline`` blocks every socket, so nothing -- a model download, telemetry,
  Chroma's default ONNX embedder -- reaches the network. ``_no_tracing`` stops
  the one thing that would otherwise try on every chain: LangSmith tracing
  switched on by a developer's .env.
"""

from __future__ import annotations

import socket
import sys

import pytest
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import ENV_VARS, Settings
from rag_pipeline.ingest import reset_store_cache
from tests.fake_mlx import FakeMLX, write_model
from tests.fake_mlx import modules as fake_mlx_modules

# The fake embedding width. The `settings` fixture pins EMBEDDING_DIMENSIONS to
# it, or ingest's probe-vs-declared-width check would raise on every run.
_EMBED_SIZE = 32

_CANNED_ANSWER = "Chunks overlap to preserve context across boundaries. (a.md)"
_PARTIAL_ANSWER = "a partial ans"

# The modules whose import means "a real model is about to run". `mlx` covers
# `mlx.core`: a None parent makes every submodule import fail too.
_MLX_MODULES = ("mlx", "mlx_lm")


# --- the injection seam ------------------------------------------------------


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
    embedding model. Its width is `_EMBED_SIZE`, which the settings fixture
    declares as EMBEDDING_DIMENSIONS.
    """
    return DeterministicFakeEmbedding(size=_EMBED_SIZE)


class _SliceReranker(BaseDocumentCompressor):
    """Offline stand-in for the local reranker: keep retrieval order, cap at top_n.

    Enough to exercise the retrieve->rerank wiring and the top_n contract with no
    model. A test that must prove reranking *reorders* builds a reversing variant
    of its own instead.
    """

    top_n: int

    def compress_documents(self, documents, query, callbacks=None):
        return documents[: self.top_n]


@pytest.fixture
def fake_reranker(settings) -> BaseDocumentCompressor:
    """Deterministic, offline reranker (identity + truncate to retrieval_k)."""
    return _SliceReranker(top_n=settings.retrieval_k)


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
def settings(sample_data_dir, tmp_path) -> Settings:
    """Settings pointed at the sample data and an isolated, per-test index.

    EMBEDDING_DIMENSIONS matches the fake embedding width, or ingest's probe
    check would reject every run.
    """
    return Settings(
        data_dir=sample_data_dir,
        persist_dir=tmp_path / "chroma",
        collection_name="test_docs",
        embedding_dimensions=_EMBED_SIZE,
        chunk_size=200,
        chunk_overlap=40,
        retrieval_k=2,
    )


@pytest.fixture
def wired_env(settings, fake_embeddings, fake_reranker, monkeypatch) -> Settings:
    """A frontend's view of the world: fixture settings in the environment, fakes
    behind the three model factories.

    Neither frontend takes injected models -- `app.py` is a script and `cli.py`
    builds its own `Settings.from_env()` -- so the environment is how a fixture's
    temp index reaches them, and the factories are where they reach a model.
    That is the same seam for both, which is why this is one fixture rather than
    a copy in each frontend's test file.

    Derived from ENV_VARS rather than spelled out: a hand-kept list would
    silently stop covering a new setting, and config.py's import-time
    load_dotenv() means the developer's own .env would answer whichever name was
    missed.
    """
    for var in ENV_VARS:
        monkeypatch.setenv(var, str(getattr(settings, var.lower())))

    monkeypatch.setattr(ingest_mod, "build_embeddings", lambda _s: fake_embeddings)
    monkeypatch.setattr(
        pipeline_mod,
        "build_chat_model",
        lambda _s: FakeListChatModel(responses=[_CANNED_ANSWER]),
    )
    monkeypatch.setattr(pipeline_mod, "build_reranker", lambda _s: fake_reranker)
    return settings


@pytest.fixture
def fake_mlx(monkeypatch) -> FakeMLX:
    """A fake MLX stack in ``sys.modules``, and a fresh model memo.

    For driving the real adapters -- down to the generation lock -- with no MLX
    and no weights. Set after ``_no_real_models`` has put ``None`` there, so it
    wins for this test only; the memo is swapped rather than cleared so no fake
    model outlives the test that made it.
    """
    fake = FakeMLX()
    for name, module in fake_mlx_modules(fake).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(mlx_models, "_LOADED", {})
    return fake


@pytest.fixture
def model_dir(tmp_path) -> str:
    """A complete local model directory, as a model id, for ``fake_mlx`` to load."""
    return str(write_model(tmp_path / "model"))


@pytest.fixture
def fail_mid_stream(monkeypatch):
    """Make generation emit, then fail -- as a real one would.

    Patched at `_generate` rather than `stream_answer` so real retrieval still
    runs and the frontend still receives real sources. Failing partway rather
    than at the call is the honest shape: generation is lazy, so a model error
    lands while the frontend is already rendering.

    Here rather than in one frontend's test file because both frontends have to
    survive it, and a dependency on a private method is worth declaring once.
    """

    def arrange(exc: BaseException) -> None:
        def generate(_self, _question, _docs):
            yield _PARTIAL_ANSWER
            raise exc

        monkeypatch.setattr(pipeline_mod.RAGPipeline, "_generate", generate)

    return arrange


# --- the guards --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_store_client():
    """Chroma caches one client System per persist directory per process, and a
    cached System's vector view does not see writes another process made after
    it was opened. Dropping the cache at each test boundary gives every test a
    fresh System and lets an in-process re-ingest behave like a fresh CLI run."""
    reset_store_cache()
    yield
    reset_store_cache()


def _hide_mlx(node, monkeypatch) -> None:
    """Make MLX unimportable for ``node``'s test, unless it is marked ``models``."""
    if node.get_closest_marker("models"):
        return
    for name in _MLX_MODULES:
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def hide_mlx():
    """`_no_real_models`' decision, callable on any node.

    Exposed so test_offline_guard can check the ``models`` exemption from both
    sides: a test that is actually marked is deselected from the default run,
    so the exemption could otherwise only be seen from a run that loads models.
    """
    return _hide_mlx


@pytest.fixture(autouse=True)
def _no_real_models(request, monkeypatch):
    """Make MLX unimportable, so no test can load a real model by accident.

    A ``None`` entry in ``sys.modules`` makes ``import mlx_lm`` (and
    ``mlx.core``) raise ImportError, which the model loader reports as a
    RuntimeError before it ever looks in the Hugging Face cache. That catches
    every route to a real model -- a forgotten ``embeddings=``, a factory called
    directly, an import added somewhere new -- identically on a Mac with MLX
    installed and on the Linux CI legs without it. Tests marked ``models`` are
    the deliberate exception.
    """
    _hide_mlx(request.node, monkeypatch)


@pytest.fixture(autouse=True, scope="session")
def _no_tracing():
    """Keep LangSmith tracing off, whatever the developer's .env says.

    config.py loads .env at import time, and ``LANGSMITH_TRACING=true`` there
    makes every chain a test runs queue a trace for a background thread to
    upload. `_offline` blocks that upload only while a test is running; a flush
    that lands after its monkeypatch is undone, or at interpreter exit, would
    send test traces out for real. Session-scoped because langsmith reads these
    once per process and caches the answer, so they must be in place before the
    first chain runs. Every spelling is set, because the first one found wins
    and "false" also keeps langchain-core's legacy ``LANGCHAIN_TRACING`` check
    quiet.
    """
    with pytest.MonkeyPatch.context() as mp:
        for var in (
            "LANGSMITH_TRACING_V2",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING",
        ):
            mp.setenv(var, "false")
        yield


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Fail any test that opens a socket, to any host.

    Chroma runs in-process and the models load from local files, so nothing in
    the suite has a reason to connect anywhere: a connection means something is
    downloading (a model, Chroma's default ONNX embedder) or phoning home, and
    either should fail the test that caused it.
    """

    def blocked(*_args, **_kwargs):
        raise RuntimeError("test opened a network socket; the suite must stay offline")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
