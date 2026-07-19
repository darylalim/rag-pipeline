"""Tests for the indexing phase: loading, splitting, and building the store."""

from __future__ import annotations

from pathlib import Path

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

    # `match` pins this to the empty-corpus ValueError; without it the test
    # would also pass on an unrelated ValueError (e.g. a bad numeric env var).
    with pytest.raises(ValueError, match="No readable documents found"):
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


def test_every_source_is_a_relative_posix_path_that_resolves_under_data_dir(tmp_path):
    """`source` is what citations key off and what both frontends print, so its
    shape is a contract, not an implementation detail.

    An absolute path would leak the indexing machine's layout into answers; a
    Windows-style path would not round-trip back to the file. Stated as a
    property over *every* returned document — rather than a list of expected
    filenames — so a loader added later for a new suffix has to satisfy it too.
    """
    root = tmp_path / "data"
    files = {
        "top.md": "Top-level document.",
        "deep/mid.txt": "One directory down.",
        "deep/nested/dir/leaf.md": "Three directories down.",
    }
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    docs = ingest_mod.load_documents(root)

    assert len(docs) == len(files), "expected one document per readable file"
    for doc in docs:
        source = doc.metadata["source"]
        assert not Path(source).is_absolute(), f"{source!r} must be data_dir-relative"
        assert "\\" not in source, f"{source!r} must use POSIX separators"
        assert (root / source).is_file(), f"{source!r} must resolve back to its file"

    # The subdirectory has to survive into `source`: two files named the same in
    # different directories must stay distinguishable in a citation.
    assert {d.metadata["source"] for d in docs} == set(files)
