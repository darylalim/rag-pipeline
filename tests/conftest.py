"""Shared pytest fixtures.

Tests run against a real Atlas Vector Search: MongoDB's ``mongodb-atlas-local``
image (mongod plus mongot, the search process, so ``$vectorSearch`` and search
indexes behave as on Atlas), one container for the session, started through
Docker, and one database per test, dropped after it. A fake would have to
reproduce exactly the behaviour that matters -- an index built asynchronously,
writes that become searchable later, pre-filters on declared fields -- so the
store is the one part that is real. The models are the part that is never
real: a deterministic fake embedding model, a fake reranker and a fake chat
model are injected, because the real ones are paid APIs -- Voyage AI's and
Anthropic's.

The developer's own ``MONGODB_URI`` -- a real cluster -- is removed for every
test (``_no_real_store``), and only the container's is ever set.

Two autouse guards back that injection convention, since forgetting it is
silent otherwise:

- ``_no_real_store`` removes the developer's credentials -- the API keys as well
  as ``MONGODB_URI`` -- so a test that forgets ``embeddings=``/``reranker=``/
  ``llm=`` reaches a model factory that stops at the missing key, before any
  request. Tests marked ``live`` (deselected by default) keep the API keys.
- ``_offline`` blocks every socket to a host other than this machine, so
  nothing -- a model download, telemetry, a real Atlas cluster -- is reached;
  the container is on loopback. ``_no_tracing`` keeps
  tracing off whatever a developer's .env says, and ``_no_tracer_left_on``
  fails a test that leaves it on: an exporter is the one route out that the
  socket block cannot stop, because the tracing stack swallows the error.
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
from langsmith.run_helpers import _set_tracing_context, get_tracing_context
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import ENV_VARS, Settings
from rag_pipeline.ingest import reset_store_cache
from tests.fake_claude import FakeClaude
from tests.fake_langsmith import TraceRecorder

# pytest's own harness for running a pytest session inside a test: how
# test_offline_guard shows what the tracing guard leaves for the test after it.
pytest_plugins = ("pytester",)

# The fake embedding width. The `settings` fixture pins EMBEDDING_DIMENSIONS to
# it, or ingest's probe-vs-declared-width check would raise on every run.
_EMBED_SIZE = 32

_CANNED_ANSWER = "Chunks overlap to preserve context across boundaries. (a.md)"
_PARTIAL_ANSWER = "a partial ans"


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


# --- the store: Atlas in a local container -----------------------------------

# Pinned, so a new image cannot change the suite's results unannounced.
_ATLAS_IMAGE = "mongodb/mongodb-atlas-local:8.0.17"
_START_ATTEMPTS = 3
_START_RETRY_S = 5.0
# Credentials and other variables a developer's environment (or .env) holds
# that no test may inherit.
_CREDENTIALS = (
    "MONGODB_URI",
    "ANTHROPIC_API_KEY",
    "VOYAGE_API_KEY",
    "LANGSMITH_API_KEY",
)
# What a test marked `live` keeps: the live tests call Anthropic's and Voyage's
# APIs. Never LangSmith's: nothing live is traced.
_LIVE_CREDENTIALS = ("VOYAGE_API_KEY", "ANTHROPIC_API_KEY")


def _await_search_ready(uri: str, timeout_s: float = 180.0) -> None:
    """Block until the container's search process accepts and builds an index.

    mongod answers first and mongot warms up seconds later; until then creating
    a search index fails ("Error connecting to Search Index Management
    service"). Building and querying a throwaway index is the readiness check:
    the image logs no documented marker for it.
    """
    import time

    client: MongoClient = MongoClient(uri, serverSelectionTimeoutMS=30000)
    try:
        probe = client["readiness_probe"]["probe"]
        probe.insert_one({"_id": "p", "embedding": [1.0, 0.0]})
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                probe.create_search_index(
                    SearchIndexModel(
                        definition={
                            "fields": [
                                {
                                    "type": "vector",
                                    "path": "embedding",
                                    "numDimensions": 2,
                                    "similarity": "cosine",
                                }
                            ]
                        },
                        name="probe",
                        type="vectorSearch",
                    )
                )
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)
        while not any(ix.get("queryable") for ix in probe.list_search_indexes("probe")):
            if time.monotonic() > deadline:
                raise RuntimeError("atlas-local never built its readiness index")
            time.sleep(1)
        client.drop_database("readiness_probe")
    finally:
        client.close()


@pytest.fixture(scope="session")
def atlas_uri() -> Iterator[str]:
    """One atlas-local container for the session; yields its URI.

    Session-scoped, so the start and mongot's warm-up are paid once.
    ``directConnection=true`` because the image is a one-node replica set whose
    member advertises a container-internal hostname a client outside cannot
    resolve. Without Docker, every test that needs the store errors here, with
    this message, rather than being skipped: a skipped store test is not a
    passed one.
    """
    import time

    from docker.errors import DockerException
    from testcontainers.core.container import DockerContainer

    # Retried: Docker Desktop's Resource Saver pauses its VM after a few idle
    # minutes, and the first start after that can fail while the VM wakes --
    # every store test in the session then errors at setup, though a rerun a
    # moment later passes. A Docker that is not running fails every attempt.
    failures: list[str] = []
    for attempt in range(_START_ATTEMPTS):
        container = DockerContainer(_ATLAS_IMAGE).with_exposed_ports(27017)
        try:
            container.start()
            break
        except DockerException as exc:
            failures.append(f"attempt {attempt + 1}: {exc}")
            time.sleep(_START_RETRY_S)
    else:
        pytest.fail(
            f"The store tests need Docker, to run {_ATLAS_IMAGE}, and it did not "
            f"start ({'; '.join(failures)}). Start Docker (Docker Desktop on a "
            "Mac) and run them again.",
            pytrace=False,
        )
    try:
        uri = (
            f"mongodb://{container.get_container_host_ip()}:"
            f"{container.get_exposed_port(27017)}/?directConnection=true"
        )
        _await_search_ready(uri)
        yield uri
    finally:
        container.stop()


@pytest.fixture
def atlas(atlas_uri, monkeypatch, request) -> Iterator[str]:
    """The container as this test's ``MONGODB_URI``; yields a fresh database name.

    A database per test, dropped afterwards, so tests share the container but
    nothing in it: each starts with no collection, no index and no lock.
    """
    database = f"test_{request.node.name[:40]}_{os.getpid()}_{id(request)}"
    database = "".join(c if c.isalnum() or c == "_" else "_" for c in database)[:63]
    monkeypatch.setenv("MONGODB_URI", atlas_uri)
    yield database
    reset_store_cache()
    client: MongoClient = MongoClient(atlas_uri)
    try:
        client.drop_database(database)
    finally:
        client.close()


@pytest.fixture
def settings(sample_data_dir, atlas) -> Settings:
    """Settings pointed at the sample data and an isolated, per-test database.

    EMBEDDING_DIMENSIONS matches the fake embedding width, or ingest's probe
    check would reject every run.
    """
    return Settings(
        data_dir=sample_data_dir,
        mongodb_db=atlas,
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

    Neither frontend takes injected models -- `streamlit_app.py` is a script and
    `cli.py` builds its own `Settings.from_env()` -- so the environment is how a
    fixture's temp index reaches them, and the factories are where they reach a
    model.
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
def fake_claude() -> FakeClaude:
    """A stand-in Anthropic server, for driving the real ``ClaudeChatModel`` --
    down to whether closing an answer closes its HTTP response -- offline."""
    return FakeClaude()


@pytest.fixture
def traces(monkeypatch) -> Iterator[TraceRecorder]:
    """Every run a test with tracing on would send to LangSmith, recorded.

    Production's own path, with only the client swapped: the pipeline asks
    `tracing_client` for one exactly as it does for real, and gets a client
    that records instead of sending (``fake_langsmith.py``). A test turns
    tracing on in its settings, as a user would -- ``langsmith_tracing=True``.
    """
    recorder = TraceRecorder()
    monkeypatch.setattr(
        pipeline_mod,
        "tracing_client",
        lambda s: recorder.client if s.langsmith_tracing else None,
    )
    try:
        yield recorder
    finally:
        recorder.client.close()


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
    put .env's in os.environ, credentials included) and LangSmith's -- and
    .env is switched off, or config.py would read it straight back in. No
    credential is set, so the child can reach no model, open no store and
    trace nothing, which is also what keeps it offline: ``_offline`` cannot
    reach into another process.

    Here rather than in one frontend's test file because both frontends have to
    survive what it shows.
    """

    def run(code: str, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
        child = {
            name: value
            for name, value in os.environ.items()
            if name not in ENV_VARS
            and name not in _CREDENTIALS
            and not name.startswith(("LANGSMITH_", "LANGCHAIN_"))
        }
        child |= {
            "PYTHON_DOTENV_DISABLED": "1",
            "DATA_DIR": str(tmp_path / "data"),
            **env,
        }
        return subprocess.run(
            [sys.executable, "-c", code, *args],
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
    """Close the process's MongoDB clients at each test boundary, so every test
    starts as a fresh process would and none inherits another's connection."""
    reset_store_cache()
    yield
    reset_store_cache()


@pytest.fixture(autouse=True)
def _no_real_store(request, monkeypatch):
    """Remove the developer's credentials -- a real ``MONGODB_URI`` above all.

    config.py loads .env at import, so a developer's real cluster is in
    os.environ for every test; one that forgot ``atlas`` would otherwise write
    into it. Only the ``atlas`` fixture sets ``MONGODB_URI``, to the container.
    ``_offline`` is the second guard: a real cluster is not on loopback.

    The API keys are the other half of the injection guard: a test that forgets
    to inject a fake reaches a model factory, which stops at the missing key
    before any request. Tests marked ``live`` keep them: they call the real
    APIs, by hand, on purpose. Never ``MONGODB_URI``.
    """
    for name in _CREDENTIALS:
        if is_live(request.node) and name in _LIVE_CREDENTIALS:
            continue
        monkeypatch.delenv(name, raising=False)


def is_live(node) -> bool:
    """Whether ``node``'s test is exempt from the API-key and network guards.

    One function for both guards, and for test_offline_guard to check from both
    sides: a test really marked ``live`` is deselected from the default run.
    """
    return node.get_closest_marker("live") is not None


@pytest.fixture(autouse=True, scope="session")
def _no_tracing():
    """Keep tracing off, whatever the developer's .env says.

    config.py loads .env at import time, so a ``LANGSMITH_TRACING=true`` there
    would reach every test that builds its settings from the environment --
    both frontends -- and each would build a real client and send its
    questions. ``_offline`` does not stop one: the client sends from a
    background thread, which logs the socket block's error and carries on, and
    a batch still queued at exit is sent after the block is undone, into the
    developer's own LangSmith project.

    Every spelling is set, not only the one ``Settings`` reads: LangSmith also
    switches itself on from the others for any chain run outside the
    pipeline's explicit context, the first one found wins, and "false" keeps
    langchain-core's legacy ``LANGCHAIN_TRACING`` check quiet. Session-scoped
    because langsmith reads these once per process and caches the answer, so
    they must be in place before the first chain runs.
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


def _tracing_left_on() -> list[str]:
    """What of a LangSmith tracing context is still in place, if anything."""
    context = get_tracing_context()
    left = []
    if context["parent"] is not None:
        left.append(f"run {context['parent'].name!r} is current")
    if context["enabled"] is not None:
        left.append(f"tracing is set to {context['enabled']!r}")
    return left


@pytest.fixture
def tracing_left_on() -> Callable[[], list[str]]:
    """`_no_tracer_left_on`'s check, callable: test_offline_guard trips it, and
    test_tracing checks it between the pieces of an answer."""
    return _tracing_left_on


@pytest.fixture(autouse=True)
def _no_tracer_left_on():
    """Fail a test that leaves a tracing context behind it.

    LangSmith keeps the current run, and whether tracing is on, in context
    variables, which every later test on this thread would inherit: one left
    set would trace those tests, under a run that has ended. The pipeline
    enters its context only around synchronous steps, never across a yield, so
    nothing should ever be left -- this is what notices if that changes.

    Cleared before failing, so the failure is that test's alone. Left set, the
    tests after it would fail for it too, far from the cause.
    """
    yield
    left = _tracing_left_on()
    if left:
        _set_tracing_context(None)
    assert not left, f"the test left tracing on: {', '.join(left)}"


@pytest.fixture(autouse=True)
def _offline(request, monkeypatch):
    """Fail any test that opens a socket to a host other than this machine.

    The store is the atlas-local container, reached on loopback, and the models
    load from local files, so nothing in the suite has a reason to connect
    anywhere else: such a connection means something is downloading (a model),
    phoning home, or reaching a real cluster, and it fails the test that made
    it. Unix sockets -- Docker's own API -- stay open. Tests marked ``live`` are
    exempt: they call the real APIs, and run only by hand (``-m live``).
    """
    if is_live(request.node):
        return
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex
    create_connection = socket.create_connection

    def local(address) -> bool:
        return not isinstance(address, tuple) or address[0] in _LOOPBACK

    def refuse(address):
        raise RuntimeError(
            f"test opened a network socket to {address[0]!r}; the suite may "
            "only reach this machine"
        )

    def guarded_connect(self, address):
        if not local(address):
            refuse(address)
        return connect(self, address)

    def guarded_connect_ex(self, address):
        if not local(address):
            refuse(address)
        return connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not local(address):
            refuse(address)
        return create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)


# The hosts `_offline` lets a test reach: this machine, however it is spelled.
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
