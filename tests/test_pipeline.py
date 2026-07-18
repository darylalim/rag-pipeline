"""Tests for the query phase: helpers, guards, and retrieval round-trip.

Generation (the Claude call) is not exercised here — it requires a real API
key. These tests cover everything up to it: the source helpers, the setup
guards, and that an ingested chunk can be retrieved by its own text.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag_pipeline import ingest as ingest_mod
from rag_pipeline.pipeline import RAGPipeline, format_docs, unique_sources


def test_unique_sources_dedupes_in_order():
    docs = [
        Document(page_content="1", metadata={"source": "b.md"}),
        Document(page_content="2", metadata={"source": "a.md"}),
        Document(page_content="3", metadata={"source": "b.md"}),
        Document(page_content="4", metadata={}),  # missing source -> "unknown"
    ]
    assert unique_sources(docs) == ["b.md", "a.md", "unknown"]


def test_format_docs_labels_each_source():
    docs = [
        Document(page_content="hello", metadata={"source": "a.md"}),
        Document(page_content="world", metadata={"source": "b.md"}),
    ]
    out = format_docs(docs)
    assert "[Source: a.md]" in out and "hello" in out
    assert "[Source: b.md]" in out and "world" in out


def test_pipeline_requires_index(settings, monkeypatch):
    # Key is present so we know it's the *index* check that fails, not the key.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    with pytest.raises(FileNotFoundError):
        RAGPipeline(settings)  # nothing ingested yet


def test_pipeline_requires_api_key(settings, fake_embeddings, monkeypatch):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)  # index now exists
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        RAGPipeline(settings, embeddings=fake_embeddings)


def test_retrieve_round_trip(settings, fake_embeddings, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    # Recreate the exact chunk text that was stored, then query with it.
    docs = ingest_mod.load_documents(settings.data_dir)
    target = ingest_mod.split_documents(docs, settings)[0]

    pipeline = RAGPipeline(settings, embeddings=fake_embeddings)
    results = pipeline.retrieve(target.page_content)

    assert results, "expected at least one retrieved chunk"
    assert results[0].metadata["source"] == target.metadata["source"]
