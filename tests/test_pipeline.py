"""Tests for the query phase: helpers, guards, and retrieval round-trip.

Generation (the Claude call) is not exercised here — it requires a real API
key. These tests cover everything up to it: the source helpers, the setup
guards, and that an ingested chunk can be retrieved by its own text.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

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
    assert "[Source: a.md]" in out
    assert "hello" in out
    assert "[Source: b.md]" in out
    assert "world" in out


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


def test_pipeline_rejects_empty_index(settings, fake_embeddings, monkeypatch):
    # persist_dir exists but nothing was ingested -> loud error, not silent empty.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        RAGPipeline(settings, embeddings=fake_embeddings)


def test_pipeline_rejects_mismatched_collection(settings, fake_embeddings, monkeypatch):
    import dataclasses

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    # Ingests into settings.collection_name; we then query a different one.
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    other = dataclasses.replace(settings, collection_name="different")
    with pytest.raises(FileNotFoundError):
        RAGPipeline(other, embeddings=fake_embeddings)


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


def test_answer_returns_model_output_and_sources(
    settings, fake_embeddings, monkeypatch
):
    # Injecting an llm skips building ChatAnthropic, so no real key is needed.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    canned = "Chunks overlap to preserve context across boundaries. (rag_concepts.md)"
    fake_llm = FakeListChatModel(responses=[canned])
    pipeline = RAGPipeline(settings, embeddings=fake_embeddings, llm=fake_llm)

    result = pipeline.answer("Why do chunks overlap?")

    assert result.text == canned
    assert result.sources, "expected grounding sources"
    assert all("source" in doc.metadata for doc in result.sources)


def test_answer_injects_retrieved_context_into_prompt(
    settings, fake_embeddings, monkeypatch
):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    captured: dict = {}

    def spy(prompt_value):
        # The chain feeds the rendered prompt into the model; capture it here.
        captured["messages"] = prompt_value.to_messages()
        return AIMessage(content="ok")

    pipeline = RAGPipeline(
        settings, embeddings=fake_embeddings, llm=RunnableLambda(spy)
    )
    result = pipeline.answer("Why do chunks overlap?")

    human = captured["messages"][-1].content
    assert "Question: Why do chunks overlap?" in human
    # Retrieved chunks are stuffed into the prompt as labeled context.
    assert "[Source:" in human
    assert result.sources[0].metadata["source"] in human
