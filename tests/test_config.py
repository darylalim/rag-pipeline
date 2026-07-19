"""Tests for Settings defaults and environment-variable overrides."""

from __future__ import annotations

from rag_pipeline.config import ENV_VARS, Settings


def test_defaults():
    s = Settings()
    assert s.chat_model == "claude-haiku-4-5"
    assert "MiniLM" in s.embedding_model
    assert s.chunk_size == 1000
    assert s.chunk_overlap == 200
    assert s.retrieval_k == 4
    assert s.collection_name == "rag_docs"
    assert s.data_dir.name == "data"


def test_from_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_MODEL", "claude-opus-4-8")
    monkeypatch.setenv("RETRIEVAL_K", "7")
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("COLLECTION_NAME", "custom")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    s = Settings.from_env()

    assert s.chat_model == "claude-opus-4-8"
    assert s.retrieval_k == 7
    assert isinstance(s.retrieval_k, int)
    assert s.chunk_size == 512
    assert s.collection_name == "custom"
    assert s.data_dir == tmp_path.resolve()


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
    assert "MiniLM" in s.embedding_model
