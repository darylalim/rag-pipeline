"""Shared pytest fixtures.

Tests use a deterministic fake embedding model instead of the real
sentence-transformers one, so the suite runs fast and fully offline (no torch,
no model download). The fake still round-trips text->vector->store->retrieve,
which is what the ingest/pipeline tests exercise.
"""

from __future__ import annotations

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.embeddings import DeterministicFakeEmbedding

from rag_pipeline.config import Settings


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
    (root / "sub").mkdir(parents=True)
    (root / "a.md").write_text("Alpha topic about apples and orchards.\n", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("Beta topic about bicycles and boats.\n", encoding="utf-8")
    (root / "empty.md").write_text("   \n", encoding="utf-8")  # whitespace only -> skipped
    (root / "notes.rst").write_text("unsupported extension -> skipped", encoding="utf-8")
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
    SharedSystemClient.clear_system_cache()
    yield
    SharedSystemClient.clear_system_cache()
