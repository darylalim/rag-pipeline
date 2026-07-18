"""Tests for the indexing phase: loading, splitting, and building the store."""

from __future__ import annotations

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag_pipeline import ingest as ingest_mod
from rag_pipeline.config import Settings


def test_load_documents_reads_supported_and_skips_others(sample_data_dir):
    docs = ingest_mod.load_documents(sample_data_dir)

    sources = sorted(d.metadata["source"] for d in docs)
    assert sources == ["a.md", "sub/b.txt"]  # empty.md and notes.rst skipped
    assert all(d.page_content.strip() for d in docs)


def test_load_documents_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_mod.load_documents(tmp_path / "does-not-exist")


def test_load_documents_skips_unreadable_file(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "good.md").write_text("readable content", encoding="utf-8")
    # Invalid UTF-8 bytes: must be skipped, not abort the whole load.
    (root / "bad.txt").write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["good.md"]


def test_ingest_preserves_unrelated_files_in_persist_dir(settings, fake_embeddings):
    settings.persist_dir.mkdir(parents=True, exist_ok=True)
    sentinel = settings.persist_dir / "KEEP_ME.txt"
    sentinel.write_text("do not delete", encoding="utf-8")

    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert sentinel.exists(), "ingest must not delete unrelated files in persist_dir"


def test_split_preserves_source_and_bounds_chunk_size(settings):
    long_text = "sentence. " * 400  # ~4000 chars -> many 200-char chunks
    doc = Document(page_content=long_text, metadata={"source": "big.md"})

    chunks = ingest_mod.split_documents([doc], settings)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "big.md" for c in chunks)
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)


def test_ingest_empty_dir_raises(tmp_path, fake_embeddings):
    empty = tmp_path / "data"
    empty.mkdir()
    s = Settings(data_dir=empty, persist_dir=tmp_path / "chroma")

    with pytest.raises(ValueError):
        ingest_mod.ingest(s, embeddings=fake_embeddings)


def test_ingest_is_idempotent(settings, fake_embeddings):
    n1 = ingest_mod.ingest(settings, embeddings=fake_embeddings)

    # Emulate a fresh CLI process, then re-ingest the same data.
    ingest_mod.reset_store_cache()
    n2 = ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert n1 == n2 >= 2

    # The rebuilt store holds n2 vectors, not 2*n2 — no duplicate append.
    ingest_mod.reset_store_cache()
    store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=fake_embeddings,
        persist_directory=str(settings.persist_dir),
    )
    assert len(store.get()["ids"]) == n2
