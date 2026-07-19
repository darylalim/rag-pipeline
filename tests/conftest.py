"""Shared pytest fixtures.

Tests use a deterministic fake embedding model instead of the real
sentence-transformers one, so the suite runs fast and fully offline (no torch,
no model download). The fake still round-trips text->vector->store->retrieve,
which is what the ingest/pipeline tests exercise.
"""

from __future__ import annotations

import socket

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_pipeline.config import Settings
from rag_pipeline.ingest import reset_store_cache


@pytest.fixture
def fake_embeddings() -> DeterministicFakeEmbedding:
    """Deterministic, torch-free embeddings.

    Same text -> same vector, so querying with a chunk's exact text retrieves
    that chunk. Good enough to test the store/retrieve wiring without the real
    embedding model.
    """
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def sample_data_dir(tmp_path):
    """A data directory that exercises the loader: a nested subdirectory, a
    whitespace-only file, and an unsupported extension (both must be skipped)."""
    root = tmp_path / "data"
    files = {
        "a.md": "Alpha topic about apples and orchards.\n",
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
    ``embeddings=`` falls back to the real sentence-transformers path — either
    downloading a model, or (worse, because it still passes) loading a warm
    cache and running for minutes in CI. Blocking the socket catches every
    spelling of that mistake, including ones no grep would find, because the
    failure is defined by behavior rather than by the name of a class.

    Deliberately NOT paired with HF_HUB_OFFLINE=1: that makes huggingface_hub
    skip its revision check, so a cached model loads with no socket at all and
    this fixture goes blind. The two are antagonistic, not complementary —
    blocking the socket is strictly stronger alone. The delenv defends that
    precondition against an ambient export.

    Caveat: a warm-cache load is caught only because huggingface_hub currently
    revision-checks over the network. If that gains a local TTL, coverage
    quietly narrows to cold-cache runs.
    """
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def blocked(*args, **kwargs):
        raise RuntimeError(
            "test opened a network socket -- the suite must run offline; "
            "inject fake embeddings/LLM instead of the real model"
        )

    # Both are load-bearing: httpcore reaches the network via create_connection.
    monkeypatch.setattr(socket, "socket", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
