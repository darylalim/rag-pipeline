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
  Chroma's default ONNX embedder -- reaches the network. ``_no_tracing`` keeps
  tracing off whatever a developer's .env says, and ``_no_tracer_left_on``
  fails a test that leaves it on: an exporter is the one route out that the
  socket block cannot stop, because the tracing SDK swallows the error.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator

import pytest
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from openinference.instrumentation import TracerProvider
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.test.globals_test import reset_trace_globals

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import ENV_VARS, Settings
from rag_pipeline.ingest import reset_store_cache
from tests.fake_mlx import FakeMLX, write_model
from tests.fake_mlx import modules as fake_mlx_modules

# pytest's own harness for running a pytest session inside a test: how
# test_offline_guard shows what the tracing guard leaves for the test after it.
pytest_plugins = ("pytester",)

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
def spans() -> Iterator[InMemorySpanExporter]:
    """Every span the test produces, recorded in memory as tracing-on would.

    The provider production installs -- OpenInference's, whose attribute cap a
    reranker span over a large FETCH_K needs -- with LangChain instrumented
    against it, but a synchronous processor into memory in place of the batched
    exporter, so a span can be asserted on the moment it ends. All of it is
    undone afterwards: OpenTelemetry allows one provider per process, and
    LangChain's hook is process-wide, so either left in place would trace every
    later test.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    LangChainInstrumentor().instrument(tracer_provider=provider)
    try:
        yield exporter
    finally:
        LangChainInstrumentor().uninstrument()
        provider.shutdown()
        reset_trace_globals()


def _switch_tracing_off() -> None:
    """Undo whatever of tracing is on: LangChain's hook, then the provider.

    One undo for both callers -- `undo_tracing` after a test that set tracing
    up on purpose, `_no_tracer_left_on` after one that left it on by mistake --
    so the two cannot drift apart as setup comes to install more.
    """
    if LangChainInstrumentor().is_instrumented_by_opentelemetry:
        LangChainInstrumentor().uninstrument()
    provider = trace.get_tracer_provider()
    if isinstance(provider, SDKTracerProvider):
        provider.shutdown()
    reset_trace_globals()


@pytest.fixture
def undo_tracing() -> Iterator[None]:
    """For a test that runs `setup_tracing` for real: switch tracing back off.

    Everything setup installs is process-wide, and `_no_tracer_left_on` fails
    the test that leaves any of it behind.
    """
    yield
    _switch_tracing_off()


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


@pytest.fixture
def fresh_interpreter(tmp_path) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Run ``code`` in a new interpreter, with ``env`` over a scrubbed environment.

    For what only a fresh process can show: some libraries read their settings
    from the environment once, as they are first imported, and this process
    imported all of them before the first test ran. Left out of the child are
    the developer's own settings -- this repo's (config.py's load_dotenv() has
    put .env's in os.environ), OpenTelemetry's and chromadb's -- and .env is
    switched off, or config.py would read it straight back in. The working
    directory is the test's own, because chromadb reads a .env from there for
    itself. MLX is made unimportable and PERSIST_DIR names an index that does
    not exist, so the child can load no model and open no store, which is also
    what keeps it offline: ``_offline`` cannot reach into another process.

    Here rather than in one frontend's test file because both frontends have to
    survive what it shows.
    """

    def run(code: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        child = {
            name: value
            for name, value in os.environ.items()
            if name not in ENV_VARS and not name.startswith(("OTEL_", "CHROMA_"))
        }
        child |= {
            "PYTHON_DOTENV_DISABLED": "1",
            "PERSIST_DIR": str(tmp_path / "no-index"),
            "DATA_DIR": str(tmp_path / "data"),
            **env,
        }
        prelude = 'import sys\nsys.modules["mlx"] = sys.modules["mlx_lm"] = None\n'
        return subprocess.run(
            [sys.executable, "-c", prelude + code, *args],
            env=child,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

    return run


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
    """Keep tracing off, whatever the developer's .env says.

    config.py loads .env at import time, so a ``PHOENIX_COLLECTOR_ENDPOINT``
    there would reach every test that builds its settings from the
    environment -- both frontends -- and each would install a real exporter.
    ``_offline`` does not stop one: the tracing SDK catches the socket block's
    error and logs it, and a batch still queued at exit is sent after the block
    is undone, into the developer's own Phoenix.

    LangSmith is switched off as well. The pipeline no longer uses it, but
    langsmith still ships inside langchain-core and still acts on
    ``LANGSMITH_TRACING=true``, uploading every chain from a background thread.
    Every spelling is set, because the first one found wins and "false" also
    keeps langchain-core's legacy ``LANGCHAIN_TRACING`` check quiet.
    Session-scoped because langsmith reads these once per process and caches
    the answer, so they must be in place before the first chain runs.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
        for var in (
            "LANGSMITH_TRACING_V2",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING",
        ):
            mp.setenv(var, "false")
        yield


def _tracing_left_on() -> list[str]:
    """What of tracing is switched on process-wide right now, if anything."""
    left = []
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        left.append("a tracer provider is installed")
    if LangChainInstrumentor().is_instrumented_by_opentelemetry:
        left.append("LangChain is instrumented")
    return left


@pytest.fixture
def tracing_left_on() -> Callable[[], list[str]]:
    """`_no_tracer_left_on`'s check, callable: test_offline_guard trips it, and
    test_tracing shows it staying quiet."""
    return _tracing_left_on


@pytest.fixture(autouse=True)
def _no_tracer_left_on():
    """Fail a test that leaves tracing switched on behind it.

    `setup_tracing` installs a provider for the life of the process -- that is
    its job -- so a test that reaches it with an endpoint would trace every test
    after it, and export them. Checked after the test, and after `spans` has
    undone its own provider: an autouse fixture is torn down last.

    Switched off before failing, so the failure is that test's alone. Left on,
    the tests after it would fail for it too, far from the cause: each trips
    this check again until one happens to undo tracing, and one that sets
    tracing up finds it already done -- setup returns early, installing and
    raising nothing.
    """
    yield
    left = _tracing_left_on()
    if left:
        _switch_tracing_off()
    assert not left, f"the test left tracing on: {', '.join(left)}"


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
