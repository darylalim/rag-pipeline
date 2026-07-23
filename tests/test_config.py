"""Tests for Settings defaults and environment-variable overrides."""

from __future__ import annotations

import pytest

from rag_pipeline.config import ENV_VARS, Settings


def test_defaults():
    s = Settings()
    assert s.chat_model == "claude-haiku-4-5"
    assert s.embedding_model == "voyage-4-lite"
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.retrieval_k == 4
    assert s.collection_name == "rag_docs"
    assert s.data_dir.name == "data"
    # The Atlas store fields.
    assert s.mongodb_db == "rag_db"
    assert s.vector_index_name == "vector_index"
    assert s.embedding_dimensions == 1024
    assert s.atlas_similarity == "cosine"
    assert s.mongodb_timeout_ms == 10000


def test_from_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("RETRIEVAL_K", "7")
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("COLLECTION_NAME", "custom")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ATLAS_SIMILARITY", "dotProduct")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")

    s = Settings.from_env()

    assert s.chat_model == "claude-opus-4-8"
    assert s.retrieval_k == 7
    assert isinstance(s.retrieval_k, int)
    assert s.chunk_size == 512
    assert s.collection_name == "custom"
    assert s.data_dir == tmp_path.resolve()
    assert s.atlas_similarity == "dotProduct"
    assert s.embedding_dimensions == 256


def test_from_env_rejects_an_unknown_similarity(monkeypatch):
    # An unusable metric must raise the ValueError both frontends catch, rather
    # than reach Atlas and fail opaquely at index-build time.
    monkeypatch.setenv("ATLAS_SIMILARITY", "manhattan")
    with pytest.raises(ValueError, match="ATLAS_SIMILARITY"):
        Settings.from_env()


def test_from_env_uses_defaults_when_unset(monkeypatch):
    # ENV_VARS, not a hand-kept list: config.py loads .env at import time, so a
    # name missing here would be answered by the developer's own .env and this
    # test would keep passing while no longer covering that default.
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    s = Settings.from_env()

    assert s.chat_model == "claude-haiku-4-5"
    assert s.retrieval_k == 4
    assert s.chunk_size == 1000


def test_from_env_empty_string_falls_back_to_default(monkeypatch):
    # A set-but-empty var should fall back to the default, not pass "" through.
    monkeypatch.setenv("CHAT_MODEL", "")
    monkeypatch.setenv("COLLECTION_NAME", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "")

    s = Settings.from_env()

    assert s.chat_model == "claude-haiku-4-5"
    assert s.collection_name == "rag_docs"
    assert s.embedding_model == "voyage-4-lite"
