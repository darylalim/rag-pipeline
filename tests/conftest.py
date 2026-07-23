"""Shared pytest fixtures.

Tests run against a real MongoDB Atlas Vector Search, served locally by the
`mongodb/mongodb-atlas-local` container (mongod + mongot, so `$vectorSearch`
works with no cloud cluster and no secret). The dependency-injection seam is
unchanged: a deterministic fake embedding model and a fake chat model are still
injected, so only the *store* is real — the suite makes no paid API call. The
socket guard is relaxed from a blanket block to a loopback-only allowlist, so
the container is reachable while Voyage/Anthropic still are not.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from uuid import uuid4

import pytest
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
from testcontainers.core.container import DockerContainer

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import ENV_VARS, Settings
from rag_pipeline.ingest import reset_store_cache

# The fake embedding width. The `settings` fixture pins EMBEDDING_DIMENSIONS to
# it, or ingest's probe-vs-declared-width check would raise on every run.
_EMBED_SIZE = 32

_CANNED_ANSWER = "Chunks overlap to preserve context across boundaries. (a.md)"
_PARTIAL_ANSWER = "a partial ans"

# Hosts the loopback allowlist permits: the local container, nothing external.
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0"}


# --- the local Atlas container -----------------------------------------------


def _await_search_ready(uri: str, timeout_s: float = 180.0) -> None:
    """Block until mongod answers AND mongot can build and serve a vector index.

    atlas-local starts mongod first and warms mongot up a few seconds later, so
    `create_search_index` raises "Error connecting to Search Index Management
    service" for a moment after the first ping. Building and querying a throwaway
    index is the robust readiness gate — the image logs no documented marker for
    it — and it means every test after this can assume mongot is warm.
    """
    client: MongoClient[dict] = MongoClient(uri, serverSelectionTimeoutMS=5000)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            client.admin.command("ping")
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(1)

    probe = client["_readiness"]["_probe"]
    probe.replace_one(
        {"_id": "p"}, {"_id": "p", "embedding": [1.0] * _EMBED_SIZE}, upsert=True
    )
    while True:
        try:
            if not list(probe.list_search_indexes("probe_index")):
                probe.create_search_index(
                    model=SearchIndexModel(
                        definition={
                            "fields": [
                                {
                                    "type": "vector",
                                    "path": "embedding",
                                    "numDimensions": _EMBED_SIZE,
                                    "similarity": "cosine",
                                }
                            ]
                        },
                        name="probe_index",
                        type="vectorSearch",
                    )
                )
            break
        except Exception:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)

    while True:
        info = list(probe.list_search_indexes("probe_index"))
        if info and info[0].get("queryable"):
            break
        if time.monotonic() > deadline:
            raise RuntimeError("atlas-local search index never became queryable")
        time.sleep(1)
    client.close()


@pytest.fixture(scope="session")
def atlas_uri() -> Iterator[str]:
    """Start one atlas-local container for the whole session; yield its URI.

    Session-scoped so the image start and the mongot warm-up are paid once.
    `directConnection=true` is required — atlas-local is a single-node replica
    set the driver would otherwise try to discover.
    """
    container = DockerContainer("mongodb/mongodb-atlas-local:8.0")
    container.with_exposed_ports(27017)
    container.start()
    try:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(27017)
        uri = f"mongodb://{host}:{port}/?directConnection=true"
        _await_search_ready(uri)
        yield uri
    finally:
        container.stop()


@pytest.fixture(autouse=True)
def _mongo_uri(atlas_uri, monkeypatch):
    """Point every test at the container. Autouse because dozens of tests build a
    bare ``Settings`` and reach ``require_env_key("MONGODB_URI")`` in the client
    factory, which would otherwise fast-fail with the wrong error."""
    monkeypatch.setenv("MONGODB_URI", atlas_uri)


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
def settings(sample_data_dir, atlas_uri) -> Iterator[Settings]:
    """Settings pointed at the sample data and an isolated Atlas namespace.

    A unique database per test keeps them from seeing each other's vectors and
    indexes; it is dropped on teardown. EMBEDDING_DIMENSIONS matches the fake
    embedding width, or ingest's probe check would reject every run.
    """
    resolved = Settings(
        data_dir=sample_data_dir,
        mongodb_db=f"test_{uuid4().hex}",
        collection_name="test_docs",
        vector_index_name="vector_index",
        embedding_dimensions=_EMBED_SIZE,
        chunk_size=200,
        chunk_overlap=40,
        retrieval_k=2,
    )
    yield resolved
    cleanup: MongoClient[dict] = MongoClient(atlas_uri, serverSelectionTimeoutMS=5000)
    cleanup.drop_database(resolved.mongodb_db)
    cleanup.close()


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
    missed. MONGODB_URI is set by the autouse `_mongo_uri` fixture (it is a key,
    not a Settings field). The API key is set because RAGPipeline fast-fails
    without one whenever it builds the model itself; the factory is patched
    below, so nothing ever authenticates with it.
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
def _reset_store_client():
    """The store keeps one MongoClient per process. Drop it at each test boundary
    so a test reconnects fresh and an in-process re-ingest behaves like a fresh
    CLI run — the connection-hygiene inverse of the old chromadb cache-clear."""
    reset_store_cache()
    yield
    reset_store_cache()


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch):
    """Fail any test that opens a socket to a non-loopback host.

    The offline guarantee still rests on tests injecting fakes for the embedding
    model, the reranker and the LLM. A test that forgets ``embeddings=`` falls
    back to the real Voyage AI path, which connects to api.voyageai.com — a
    public address — and is blocked here, catching every spelling of that mistake
    including ones no grep would find. The store is the deliberate exception: it
    reaches the atlas-local container on loopback, which is allowed. AF_UNIX
    addresses (the Docker socket) are strings, not (host, port) tuples, so they
    fall through the check untouched.
    """
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def check(address):
        if isinstance(address, tuple) and address and address[0] not in _LOCAL_HOSTS:
            raise RuntimeError(
                f"test opened a non-loopback socket to {address[0]!r} -- inject "
                "fake embeddings/LLM/reranker; the suite must not reach a paid API"
            )

    def connect(self, address):
        check(address)
        return real_connect(self, address)

    def connect_ex(self, address):
        check(address)
        return real_connect_ex(self, address)

    def create_connection(address, *args, **kwargs):
        check(address)
        return real_create_connection(address, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
