"""Tests for the indexing phase: loading, splitting, and building the store."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import itertools
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pymongo.errors
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import FakeListChatModel
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel
from pypdf import PageObject, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_pipeline import ingest as ingest_mod
from rag_pipeline.config import ENV_VARS, Settings
from rag_pipeline.ingest import OWN_CHUNKS
from rag_pipeline.pipeline import RAGPipeline


def minimal_pdf(pages: list[str]) -> bytes:
    """A real PDF carrying `pages` as extractable text, one string per page.

    Generated rather than committed as a fixture: a binary blob in the tree is
    unreviewable, and the thing under test is text extraction, which needs a
    genuine content stream -- an annotation or a blank page would not exercise
    it. `add_blank_page` gives the page; the content stream and the font
    resource are attached directly, because pypdf writes PDFs but has no API
    for drawing text into one, and `extract_text` returns nothing without a font
    to resolve `/F1` against.
    """
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    for text in pages:
        page = writer.add_blank_page(width=200, height=200)
        contents = DecodedStreamObject()
        contents.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = contents
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )

    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def test_load_documents_reads_supported_and_skips_others(sample_data_dir):
    docs = ingest_mod.load_documents(sample_data_dir)

    sources = sorted(d.metadata["source"] for d in docs)
    assert sources == ["a.md", "sub/b.txt"]  # empty.md and notes.rst skipped
    assert all(d.page_content.strip() for d in docs)


def test_load_documents_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingest_mod.load_documents(tmp_path / "does-not-exist")


def test_load_documents_skips_unreadable_file(tmp_path, capsys):
    root = tmp_path / "data"
    root.mkdir()
    (root / "good.md").write_text("readable content", encoding="utf-8")
    # Invalid UTF-8 bytes: must be skipped, not abort the whole load.
    (root / "bad.txt").write_bytes(b"\xff\xfe not valid utf-8 \x80\x81")

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["good.md"]
    # Skipping silently would make a half-indexed corpus indistinguishable from
    # a complete one, so the warning naming the file is part of the contract --
    # and stderr specifically, so `rag ingest > log` still surfaces it.
    err = capsys.readouterr().err
    assert "bad.txt" in err
    assert "good.md" not in err


# --- the PDF loader ----------------------------------------------------------


def test_load_documents_reads_a_pdf_joining_its_pages(tmp_path):
    """`.pdf` is advertised in SUPPORTED_SUFFIXES, so extraction is a contract.

    One Document per file rather than per page, which is what makes `source`
    mean the same thing for a PDF as for a markdown file -- the citation names
    the document either way.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "manual.pdf").write_bytes(
        minimal_pdf(["Chunking splits documents", "Overlap preserves context"])
    )

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["manual.pdf"]
    assert "Chunking splits documents" in docs[0].page_content
    assert "Overlap preserves context" in docs[0].page_content


def test_load_documents_skips_a_pdf_with_no_extractable_text(tmp_path):
    """A scanned PDF is the realistic case: pages exist, text does not.

    It must be skipped like a whitespace-only markdown file rather than indexed
    as an empty document, which would occupy a retrieval slot with nothing in
    it and cite a source that says nothing.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "scanned.pdf").write_bytes(minimal_pdf(["", ""]))
    (root / "real.md").write_text("actual content", encoding="utf-8")

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["real.md"]


def test_load_documents_skips_a_corrupt_pdf_without_aborting(tmp_path, capsys):
    """One unreadable PDF must not cost the whole ingest.

    The same resilience the encoding case above asserts, through the other
    loader -- pypdf raises its own exception types, which `_read_pdf` translates
    into the ValueError the loader skips a file for.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a real pdf\n")
    (root / "fine.md").write_text("readable content", encoding="utf-8")

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["fine.md"]
    assert "broken.pdf" in capsys.readouterr().err


def test_load_documents_skips_a_pdf_that_pypdf_fails_on_with_a_builtins_error(
    tmp_path, capsys, monkeypatch
):
    """Whatever pypdf raises on a malformed file, that file is skipped.

    Its parser does not keep to its own exception types: a `/Font` resource
    that is a number makes `extract_text` raise a builtins TypeError. The loader
    catches only OSError and ValueError, so this rests on `_read_pdf`
    translating everything pypdf raises -- without that, one such file aborts
    the whole ingest. Raised through a patch rather than that file, so the test
    does not depend on pypdf keeping that particular bug.
    """

    def choke(_page, *_args, **_kwargs):
        raise TypeError("'NumberObject' object is not iterable")

    monkeypatch.setattr(PageObject, "extract_text", choke)
    root = tmp_path / "data"
    root.mkdir()
    (root / "odd.pdf").write_bytes(minimal_pdf(["never extracted"]))
    (root / "fine.md").write_text("readable content", encoding="utf-8")

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["fine.md"]
    err = capsys.readouterr().err
    assert "odd.pdf" in err
    # Named, because a builtins error's message rarely says what it is.
    assert "TypeError" in err


# --- accepting an uploaded file ----------------------------------------------


def test_save_upload_lands_a_file_the_loader_then_reads(tmp_path):
    """The contract that matters: what is saved must come back out of the loader.

    Asserted through `load_documents` rather than against the path, because
    `save_upload` exists to produce input for it — a file written somewhere the
    loader does not look would satisfy every other assertion here.
    """
    root = tmp_path / "data"

    name = ingest_mod.save_upload(root, "guide.md", b"# Guide\n\nUploaded content.\n")

    assert name == "guide.md"
    docs = ingest_mod.load_documents(root)
    assert [d.metadata["source"] for d in docs] == ["guide.md"]
    assert "Uploaded content." in docs[0].page_content


def test_save_upload_creates_a_missing_data_dir(tmp_path):
    """Bootstrapping an empty checkout is the case the uploader is most needed in."""
    root = tmp_path / "nowhere" / "data"

    ingest_mod.save_upload(root, "first.md", b"first document")

    assert (root / "first.md").is_file()


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.md",
        "../../../../etc/escape.md",
        "/absolute/escape.md",
        "sub/dir/escape.md",
        "..\\..\\windows\\escape.md",
        "C:\\Users\\evil\\escape.md",
    ],
)
def test_save_upload_cannot_write_outside_data_dir(tmp_path, filename):
    """A filename arrives from a browser, so it is input, not a fact.

    Every case here must land flat inside `data_dir` rather than anywhere the
    name asked for. The Windows spellings are listed because a POSIX server does
    not treat a backslash as a separator: without folding them first, the whole
    string stays one filename and the traversal is preserved verbatim.
    """
    root = tmp_path / "data"
    root.mkdir()

    name = ingest_mod.save_upload(root, filename, b"payload")

    assert name == "escape.md", f"{filename!r} kept a directory component"
    written = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert written == [root / "escape.md"], "a file was written outside data_dir"


@pytest.mark.parametrize(
    "filename", ["notes.rst", "archive.zip", "noextension", "..", ""]
)
def test_save_upload_rejects_what_the_loader_would_skip(tmp_path, filename):
    """Rejected at the door rather than written and silently ignored.

    `load_documents` skips an unsupported suffix, so saving one would leave a
    file in `data_dir` that never reaches an answer and never explains why.
    ValueError specifically: it is inside the union both frontends catch.
    """
    root = tmp_path / "data"
    root.mkdir()

    with pytest.raises(ValueError, match="Cannot index"):
        ingest_mod.save_upload(root, filename, b"payload")

    assert not list(root.iterdir()), "a rejected upload must leave nothing behind"


def test_save_upload_replaces_a_file_of_the_same_name(tmp_path):
    """Re-uploading a corrected document must update it, not duplicate it.

    The alternative — uniquifying the name — would leave the stale copy indexed
    and retrievable, so a question could still be answered from the version the
    user just replaced.
    """
    root = tmp_path / "data"
    root.mkdir()
    ingest_mod.save_upload(root, "doc.md", b"first version")

    ingest_mod.save_upload(root, "doc.md", b"second version")

    assert (root / "doc.md").read_bytes() == b"second version"
    assert len(list(root.iterdir())) == 1


def test_save_upload_returns_the_name_the_loader_will_report(tmp_path):
    """The name returned is the file's `source`, even when the disk chose it.

    macOS's default volume matches names regardless of case, so `Notes.md`
    uploaded beside `notes.md` replaces that file and keeps its spelling. The
    app looks the returned name up among the index's sources to decide whether
    an upload was indexed, and given the upload's own spelling it reported a
    file it had just indexed as one with no text. Only a case-insensitive
    volume -- the Mac the app runs on -- can show that; on a case-sensitive one
    the two names are two files, and the same contract holds.

    The hard link is that file under another name, one sorting first and never
    a source: found by what the file is, it could be returned in its place.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "notes.md").write_bytes(b"first version")
    os.link(root / "notes.md", root / "a-backup")

    name = ingest_mod.save_upload(root, "Notes.md", b"second version")

    docs = ingest_mod.load_documents(root)
    sources = {d.metadata["source"]: d.page_content for d in docs}
    assert name in sources, f"{name!r} is not among the sources {sorted(sources)}"
    assert sources[name] == "second version"


def test_an_uppercase_suffix_is_accepted_and_read(tmp_path):
    """`NOTES.MD` and `SCAN.PDF` are how some tools and scanners name files.

    The suffix is matched regardless of case both where an upload is accepted
    and where the loader walks `data_dir`: accepted at one and skipped at the
    other, a file would be saved and then never indexed, with nothing said.
    """
    root = tmp_path / "data"

    ingest_mod.save_upload(root, "NOTES.MD", b"Notes under an uppercase suffix.")
    ingest_mod.save_upload(root, "SCAN.PDF", minimal_pdf(["A scan, uppercase suffix"]))

    docs = ingest_mod.load_documents(root)
    sources = {d.metadata["source"]: d.page_content for d in docs}
    assert sorted(sources) == ["NOTES.MD", "SCAN.PDF"]
    assert "uppercase suffix" in sources["NOTES.MD"]
    assert "uppercase suffix" in sources["SCAN.PDF"]


def test_save_upload_writes_bytes_unchanged(tmp_path):
    """A PDF is binary, so the bytes must survive verbatim.

    Round-tripped through the real PDF loader rather than compared as bytes: an
    encode/decode step inserted here would corrupt the stream in a way only
    extraction notices.
    """
    root = tmp_path / "data"

    ingest_mod.save_upload(root, "manual.pdf", minimal_pdf(["Uploaded page text"]))

    docs = ingest_mod.load_documents(root)
    assert "Uploaded page text" in docs[0].page_content


def test_split_preserves_source_and_bounds_chunk_size(settings):
    long_text = "sentence. " * 400  # ~4000 chars -> many 200-char chunks
    doc = Document(page_content=long_text, metadata={"source": "big.md"})

    chunks = ingest_mod.split_documents([doc], settings)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "big.md" for c in chunks)
    assert all(len(c.page_content) <= settings.chunk_size for c in chunks)


# --- building the index ------------------------------------------------------


class _CountingEmbeddings(Embeddings):
    """Records the texts it was asked to embed, so a *skipped* file is visible.

    The whole point of the incremental path is a call that does not happen, and
    a chunk count cannot see that: the index holds the same vectors either way,
    whether they were just re-embedded or left alone. Only the embedder knows.
    Queries are recorded apart from documents, because ingest's one query is
    the width probe, and a run with nothing to embed should not make even that.
    """

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.embedded: list[str] = []
        self.queried: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.queried.append(text)
        return self.inner.embed_query(text)


@pytest.fixture
def counting_embeddings(fake_embeddings):
    return _CountingEmbeddings(fake_embeddings)


def chunks_of(settings) -> Collection:
    """The chunks collection, read directly -- what any other client would see."""
    return ingest_mod._collection(settings)


def own_ids(settings) -> list[str]:
    """The ids of this pipeline's chunks in the collection, sorted."""
    return sorted(
        str(row["_id"]) for row in chunks_of(settings).find(OWN_CHUNKS, {"_id": 1})
    )


def sources_in(settings) -> set[str]:
    """Every `source` the collection currently holds this pipeline's chunks for."""
    return set(chunks_of(settings).distinct("source", OWN_CHUNKS))


def search(settings, embeddings, query: str, k: int) -> list[Document]:
    """Search as the pipeline does: a retriever pre-filtered to this pipeline's chunks."""
    store = ingest_mod.open_store(settings, embeddings)
    return store.as_retriever(search_kwargs={"k": k, "pre_filter": OWN_CHUNKS}).invoke(
        query
    )


def test_ingest_empty_dir_raises(settings, tmp_path, fake_embeddings):
    empty = tmp_path / "empty"
    empty.mkdir()

    # `match` pins this to the empty-corpus ValueError; without it the test
    # would also pass on an unrelated ValueError (e.g. a bad numeric env var).
    with pytest.raises(ValueError, match="No readable documents found"):
        ingest_mod.ingest(
            dataclasses.replace(settings, data_dir=empty), embeddings=fake_embeddings
        )


def test_ingest_is_idempotent(settings, fake_embeddings):
    n1 = ingest_mod.ingest(settings, embeddings=fake_embeddings)
    held = own_ids(settings)

    # Emulate a fresh CLI process, then re-ingest the same data.
    ingest_mod.reset_store_cache()
    n2 = ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert n1 == n2 >= 2
    # The collection holds n2 chunks, not 2*n2, and the same ones -- no
    # duplicating append.
    assert own_ids(settings) == held
    assert len(held) == n2


def test_chunk_ids_are_derived_from_the_content(settings, fake_embeddings):
    """The same corpus gets the same ids, whichever index it goes into.

    That is what makes adding a chunk an upsert that replaces it rather than an
    append that duplicates it: with random ids, any add of a chunk the
    collection already holds -- by whatever path -- would leave two copies
    competing for the same retrieval slots. Two independent indexes stand in
    for "the same chunk added twice", since an ordinary re-ingest never re-adds
    anything.
    """
    elsewhere = dataclasses.replace(settings, collection_name="elsewhere")

    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.ingest(elsewhere, embeddings=fake_embeddings)

    assert own_ids(settings) == own_ids(elsewhere)


def test_ingest_reports_the_chunks_the_index_holds(settings, counting_embeddings):
    """Not the number re-embedded: after a one-file edit those differ.

    Both frontends print this as "Indexed N chunks", which describes the index;
    counting the work instead would report a handful after an edit, and zero
    for an unchanged corpus that is fully indexed.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    (settings.data_dir / "a.md").write_text("# Alpha\nrewritten.\n", encoding="utf-8")
    counting_embeddings.embedded.clear()

    n = ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert n == len(own_ids(settings))
    assert n > len(counting_embeddings.embedded), "b.txt kept its chunks"


def test_chunks_are_added_in_slices_renewing_the_lock_between_them(
    settings, fake_embeddings, monkeypatch
):
    """No slice outlasts the writer lock's lease, however large the corpus.

    langchain-mongodb embeds and writes whatever it is handed in one call, so
    unsliced, a corpus that takes longer to embed than the lease runs would
    lose the lock part-way -- and an interrupted run would lose everything it
    had embedded. The slice is shrunk here so a small corpus spans several.
    """
    monkeypatch.setattr(ingest_mod, "_ADD_SLICE", 3)
    sizes: list[int] = []
    add = MongoDBAtlasVectorSearch.add_documents
    renewals: list[int] = []
    renew = ingest_mod._WriterLock.renew

    def spy_add(self, documents, *args, **kwargs):
        sizes.append(len(documents))
        return add(self, documents, *args, **kwargs)

    def spy_renew(self):
        renewals.append(len(sizes))
        return renew(self)

    monkeypatch.setattr(MongoDBAtlasVectorSearch, "add_documents", spy_add)
    monkeypatch.setattr(ingest_mod._WriterLock, "renew", spy_renew)
    (settings.data_dir / "a.md").write_text(
        "\n\n".join(
            f"Alpha paragraph {i} about apples, at some length." for i in range(12)
        ),
        encoding="utf-8",
    )

    n = ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert n > 3
    assert len(sizes) > 1
    assert max(sizes) <= 3
    # Renewed after every slice: each renewal sees one more slice written.
    assert renewals[-len(sizes) :] == list(range(1, len(sizes) + 1))
    assert len(own_ids(settings)) == n


# --- never a wipe ------------------------------------------------------------


def test_ingest_preserves_other_collections_in_the_database(settings, fake_embeddings):
    """ingest() is a scoped collection rebuild, never a database wipe.

    A surviving neighbour proves the property however the deletion was written
    -- a dropped collection, a dropped database, a delete without a filter --
    where a text rule forbidding one spelling would only catch that spelling.
    Two neighbours: another tool's collection, and another of this pipeline's
    (COLLECTION_NAME is the documented way out of a width change, which leaves
    the old collection beside the new one).
    """
    database = chunks_of(settings).database
    database["notes"].insert_one({"_id": "keep-me", "text": "do not delete"})
    neighbour = dataclasses.replace(settings, collection_name="neighbour_docs")
    ingest_mod.ingest(neighbour, embeddings=fake_embeddings)
    held = own_ids(neighbour)

    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    (settings.data_dir / "a.md").unlink()
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert database["notes"].find_one({"_id": "keep-me"}) is not None
    assert own_ids(neighbour) == held


def _add_foreign_record(settings, embeddings) -> None:
    """Another tool's record, sharing the collection: everything but the marker.

    It names a file this pipeline indexes, carries a ``content_hash`` and a
    vector of the index's width, so only the ``ingested_by`` marker tells it
    apart -- a filter on ``source``, or on "has a content_hash", matches it.
    """
    chunks_of(settings).insert_one(
        {
            "_id": "foreign:1",
            "text": "Another tool's notes on apples, stored alongside.",
            "embedding": embeddings.embed_query("Another tool's notes on apples."),
            "source": "a.md",
            "content_hash": "theirs",
        }
    )


def test_ingest_preserves_foreign_documents_in_a_shared_collection(
    settings, fake_embeddings
):
    """ingest() is a scoped rebuild, never a collection wipe.

    A record this pipeline did not write must survive a rebuild that *does*
    delete -- here one deleting the very source the record names, which is the
    case every filter short of the marker gets wrong. Surviving, it must still
    not pass for ours: `indexed_sources` is what the app asks whether an upload
    was indexed, and here the record is all that is left under a.md.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    _add_foreign_record(settings, fake_embeddings)

    # Force a real deletion: remove the file, so the re-ingest deletes a.md.
    (settings.data_dir / "a.md").unlink()
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert chunks_of(settings).find_one({"_id": "foreign:1"}) is not None, (
        "ingest must not delete documents it did not write"
    )
    assert "a.md" not in sources_in(settings)
    assert "a.md" not in ingest_mod.indexed_sources(settings), (
        "a foreign record was reported as indexed"
    )


def test_a_foreign_record_does_not_change_what_ingest_decides(
    settings, counting_embeddings
):
    """The read half of the scoping: a foreign record is never taken for ours.

    Read unscoped, its ``content_hash`` would stand in for a.md's, a.md would
    look changed, and an untouched corpus would be re-embedded on every run.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    _add_foreign_record(settings, counting_embeddings)
    counting_embeddings.embedded.clear()

    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert counting_embeddings.embedded == []


# --- incremental re-indexing -------------------------------------------------


def test_reingest_embeds_nothing_when_no_document_changed(
    settings, counting_embeddings
):
    """The saving that justifies the fingerprint at all.

    Embedding is a forward pass of a local model for every chunk, and the app
    re-ingests on every upload, so re-ingesting an unchanged corpus must cost
    nothing -- not even the width probe. Asserted on the embedder rather than
    the returned count, which is deliberately the same both times.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    assert counting_embeddings.embedded, "the first ingest must embed everything"

    counting_embeddings.embedded.clear()
    counting_embeddings.queried.clear()
    ingest_mod.reset_store_cache()
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert counting_embeddings.embedded == []
    assert counting_embeddings.queried == []


def test_only_the_edited_document_is_re_embedded(settings, counting_embeddings):
    """A one-file edit costs one file, not the corpus."""
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    counting_embeddings.embedded.clear()

    (settings.data_dir / "a.md").write_text("# Alpha\nrewritten.\n", encoding="utf-8")
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert counting_embeddings.embedded, "the edited file must be re-embedded"
    # b.txt's text is untouched, so none of it may appear in what was embedded.
    b_text = (settings.data_dir / "sub" / "b.txt").read_text(encoding="utf-8")
    assert not any(b_text.strip() in text for text in counting_embeddings.embedded)


def test_a_partly_indexed_document_is_re_embedded(settings, counting_embeddings):
    """An add that died part-way must not be vouched for by its fingerprint.

    Its surviving chunks carry the current fingerprint, so a check on the hash
    alone reads the source as current and leaves it short of chunks for good.
    Simulated by deleting one of a source's chunks behind ingest's back, which
    is the state an interrupted add leaves.
    """
    paragraph = "Alpha topic about apples and orchards, at some length. " * 3
    (settings.data_dir / "a.md").write_text(
        "# Alpha\n\n" + "\n\n".join([paragraph] * 4), encoding="utf-8"
    )
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    before = own_ids(settings)
    a_ids = [chunk_id for chunk_id in before if chunk_id.startswith("a.md:")]
    assert len(a_ids) > 1, "the fixture must split a.md into several chunks"

    chunks_of(settings).delete_one({"_id": a_ids[-1]})
    counting_embeddings.embedded.clear()
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert own_ids(settings) == before
    # Only the damaged source is redone; the intact one keeps its vectors.
    b_text = (settings.data_dir / "sub" / "b.txt").read_text(encoding="utf-8")
    assert counting_embeddings.embedded
    assert not any(b_text.strip() in text for text in counting_embeddings.embedded)


def test_a_new_document_joins_the_existing_index(settings, counting_embeddings):
    """Adding a file must not cost, or disturb, the documents already indexed.

    This is the case a naive "index only what was just uploaded" would get
    wrong in the other direction -- it is asserted from both ends, that the new
    file is present *and* that the old ones survived, because an implementation
    that rebuilt from the upload alone would still pass the first half.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    before = sources_in(settings)
    counting_embeddings.embedded.clear()

    (settings.data_dir / "c.md").write_text("# Gamma\nbrand new.\n", encoding="utf-8")
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert sources_in(settings) == before | {"c.md"}
    assert all("brand new" in text for text in counting_embeddings.embedded)


def test_documents_are_embedded_in_the_same_order_every_run(
    settings, counting_embeddings
):
    """The same corpus reaches the embedder in the same order, run after run.

    The order decides which chunks share a padded batch, and the local model's
    bf16 arithmetic varies in the last bits with a batch's shape. In a set's
    order -- which moves with each process's hash seed -- rebuilding the same
    corpus would not reproduce its own vectors. Ten sources, so an unordered
    run matching by chance is out of the question.
    """
    for i in range(8):
        (settings.data_dir / f"doc{i}.md").write_text(
            f"Document number {i}, about topic {i}.\n", encoding="utf-8"
        )

    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    documents = ingest_mod.load_documents(settings.data_dir)  # sorted by path
    expected = ingest_mod.split_documents(documents, settings)
    assert counting_embeddings.embedded == [c.page_content for c in expected]


def test_a_removed_document_loses_its_chunks(settings, fake_embeddings):
    """The index tracks `data_dir`, so a deleted file must not answer questions.

    The half of "only embed what changed" that is easy to skip: a file that is
    gone has no fresh chunks to add, so nothing about the add path would ever
    notice it. Its vectors would linger and stay retrievable.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    assert "a.md" in sources_in(settings)

    (settings.data_dir / "a.md").unlink()
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert "a.md" not in sources_in(settings)
    hits = search(settings, fake_embeddings, "Alpha topic about apples", k=5)
    assert "a.md" not in {doc.metadata["source"] for doc in hits}


@pytest.mark.parametrize(
    "change",
    [{"chunk_size": 80}, {"chunk_overlap": 10}],
    ids=["chunk_size", "chunk_overlap"],
)
def test_changing_the_chunking_re_embeds_everything(
    settings, counting_embeddings, change
):
    """Chunk boundaries are part of what the stored vectors represent.

    Content-only fingerprinting would leave every existing chunk in place under
    a new CHUNK_SIZE, so the index would keep vectors the current settings could
    not have produced -- stale in a way no file inspection would reveal.

    One setting at a time, each of which must reach the fingerprint on its own:
    changed together, dropping either went unnoticed. And the documents split
    the same way under both, so the fingerprint is all that can tell the runs
    apart: ingest's chunk-count check re-embeds a source whose number of chunks
    changed whatever the fingerprint says, but one whose chunks only moved is
    the fingerprint's alone to catch.
    """
    rechunked = dataclasses.replace(settings, **change)
    documents = ingest_mod.load_documents(settings.data_dir)
    before = ingest_mod.split_documents(documents, settings)
    after = ingest_mod.split_documents(documents, rechunked)
    assert [c.page_content for c in before] == [c.page_content for c in after]
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    counting_embeddings.embedded.clear()

    n = ingest_mod.ingest(rechunked, embeddings=counting_embeddings)

    assert len(counting_embeddings.embedded) == n


def test_changing_the_embedding_model_re_embeds_everything(
    settings, counting_embeddings
):
    """A vector means nothing except with respect to the model that made it.

    The dangerous case of a stale skip, because nothing looks wrong: same
    files, same chunks, same width -- a model swap is now just a different repo
    id -- so every later query would be compared against vectors from a model
    that is no longer configured. The fake is unchanged on purpose; only the
    setting moves, which is all a skip would have to go on.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    counting_embeddings.embedded.clear()

    n = ingest_mod.ingest(
        dataclasses.replace(settings, embedding_model="mlx-community/another-embedder"),
        embeddings=counting_embeddings,
    )

    assert len(counting_embeddings.embedded) == n


# --- the vector index and its width ------------------------------------------


def test_ingest_builds_the_vector_index_searches_filter_on(settings, fake_embeddings):
    """`rag ingest` is the whole setup: it creates the collection and the index.

    The index declares the two fields searches pre-filter on -- $vectorSearch
    refuses a filter on an undeclared field -- at the configured width.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    (index,) = chunks_of(settings).list_search_indexes(settings.vector_index_name)
    fields = index["latestDefinition"]["fields"]
    assert {"type": "filter", "path": "ingested_by"} in fields
    assert {"type": "filter", "path": "source"} in fields
    (vector,) = [f for f in fields if f["type"] == "vector"]
    assert vector["numDimensions"] == settings.embedding_dimensions
    assert index["queryable"]


def test_ingested_chunks_are_searchable_when_ingest_returns(settings, fake_embeddings):
    """Atlas indexes a write asynchronously; ingest waits it out.

    The app answers about an upload on the same run that ingested it, so a
    chunk that is stored but not yet searchable would be an upload reported
    added and then not found. Asked for by its exact text, it is its own
    nearest neighbour.
    """
    text = "Okapis are forest giraffids from the Congo basin."
    (settings.data_dir / "okapi.md").write_text(f"{text}\n", encoding="utf-8")

    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    hits = search(settings, fake_embeddings, text, k=1)
    assert [doc.metadata["source"] for doc in hits] == ["okapi.md"]


def _drop_index(collection: Collection, name: str) -> None:
    """Drop a search index and wait until it is gone: dropping is asynchronous."""
    collection.drop_search_index(name)
    deadline = time.monotonic() + 60
    while list(collection.list_search_indexes(name)):
        assert time.monotonic() < deadline, "the index never finished dropping"
        time.sleep(0.5)


def test_an_index_dropped_by_hand_is_rebuilt_by_the_next_ingest(
    settings, fake_embeddings
):
    """Ensured on every run, not only when something changed: an unchanged
    corpus whose index was dropped would otherwise never be searchable again."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    _drop_index(chunks_of(settings), settings.vector_index_name)

    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert search(settings, fake_embeddings, "apples", k=1)


def test_a_model_that_contradicts_embedding_dimensions_is_refused_first(
    settings, fake_embeddings
):
    """EMBEDDING_DIMENSIONS is checked against the live model before any write.

    The declared width is what every fingerprint records, so vectors of another
    width must never be stored under it -- and caught only at the add, the
    mismatch would surface after the delete had already dropped the chunks
    being replaced. Nothing may change.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    held = own_ids(settings)
    (settings.data_dir / "a.md").write_text("# Alpha\nrewritten.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS"):
        ingest_mod.ingest(settings, embeddings=DeterministicFakeEmbedding(size=16))

    assert own_ids(settings) == held


def test_a_new_width_is_refused_before_anything_is_deleted(settings, fake_embeddings):
    """A collection's chunks and its index hold one width.

    A different EMBEDDING_DIMENSIONS changes every fingerprint, so a re-ingest
    is about to replace every chunk -- and the index could not serve the new
    vectors. It must be refused up front, with the remedy (a new collection),
    leaving the index exactly as it was. (Were the width not in the
    fingerprint, nothing would be refused: every chunk would be skipped as
    current, and the first query would be the one to fail.)
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    held = own_ids(settings)
    narrower = dataclasses.replace(settings, embedding_dimensions=16)

    with pytest.raises(ValueError, match="COLLECTION_NAME"):
        ingest_mod.ingest(narrower, embeddings=DeterministicFakeEmbedding(size=16))

    assert own_ids(settings) == held


def test_an_index_of_another_width_is_refused_with_none_of_our_chunks_left(
    settings, fake_embeddings
):
    """The index's own width is checked, not only the stored chunks'.

    With this pipeline's chunks all gone -- deleted by hand, or never written
    under this name -- the chunks give the width check nothing to compare, but
    the index still serves one width only. Refused before anything is written,
    as a ValueError naming the fix.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    chunks_of(settings).delete_many(OWN_CHUNKS)
    narrower = dataclasses.replace(settings, embedding_dimensions=16)

    with pytest.raises(ValueError, match="COLLECTION_NAME"):
        ingest_mod.ingest(narrower, embeddings=DeterministicFakeEmbedding(size=16))

    assert own_ids(settings) == []


def test_an_index_without_the_filter_fields_is_refused_before_anything_is_deleted(
    settings, fake_embeddings
):
    """An index made by another tool may lack the fields every search filters
    on, and a search through it would fail. Refused while nothing is deleted,
    rather than rebuilt: the index is not this pipeline's to change."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    held = own_ids(settings)
    collection = chunks_of(settings)
    _drop_index(collection, settings.vector_index_name)
    collection.create_search_index(
        SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": settings.embedding_dimensions,
                        "similarity": "cosine",
                    }
                ]
            },
            name=settings.vector_index_name,
            type="vectorSearch",
        )
    )
    (settings.data_dir / "a.md").unlink()

    with pytest.raises(ValueError, match="filter fields"):
        ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert own_ids(settings) == held


def test_a_search_of_another_width_is_a_runtime_error_with_the_fix(
    settings, fake_embeddings, fake_reranker
):
    """Atlas's own refusal, reached for real: RuntimeError, never ValueError.

    A query embedded at another width -- a model swapped without re-ingesting
    -- fails at the search, as pymongo's OperationFailure, outside the union
    both frontends catch. It must arrive as a RuntimeError, which
    ``streamlit_app.py`` handles below its sidebar, naming the remedy.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    pipeline = RAGPipeline(
        settings,
        embeddings=DeterministicFakeEmbedding(size=16),
        llm=FakeListChatModel(responses=["unused"]),
        reranker=fake_reranker,
    )

    with pytest.raises(RuntimeError) as excinfo:
        pipeline.retrieve("apples")

    assert excinfo.type is RuntimeError
    assert "set a new COLLECTION_NAME" in str(excinfo.value)


@pytest.mark.parametrize(
    "change",
    [{"collection_name": "bad$name"}, {"mongodb_db": "bad/db"}],
    ids=["collection", "database"],
)
def test_an_invalid_name_is_a_runtime_error_without_the_width_hint(
    settings, fake_embeddings, change
):
    """MongoDB refuses some names outright, as pymongo's InvalidName -- raised
    when the handle is made, before any operation. It must still arrive inside
    the union, and without the width hint, which would send the user after a
    problem they do not have."""
    with pytest.raises(RuntimeError) as excinfo:
        ingest_mod.ingest(
            dataclasses.replace(settings, **change), embeddings=fake_embeddings
        )

    assert excinfo.type is RuntimeError
    assert "set a new COLLECTION_NAME" not in str(excinfo.value)


# --- reaching the cluster ----------------------------------------------------


def test_a_missing_mongodb_uri_is_a_runtime_error_naming_it(
    settings, fake_embeddings, monkeypatch
):
    monkeypatch.delenv("MONGODB_URI")

    with pytest.raises(RuntimeError, match=r"^MONGODB_URI is not set"):
        ingest_mod.ingest(settings, embeddings=fake_embeddings)


@pytest.mark.parametrize(
    "uri",
    [
        pytest.param("not-a-uri", id="malformed"),
        # A port nothing listens on, on loopback (so _offline allows it).
        pytest.param("mongodb://127.0.0.1:1/?directConnection=true", id="unreachable"),
    ],
)
def test_an_unusable_cluster_is_a_runtime_error_each_time(
    settings, fake_embeddings, monkeypatch, uri
):
    """A paused cluster, an IP missing from the access list, a mistyped URI.

    pymongo's errors sit outside the union; they arrive as RuntimeError, and a
    client whose first contact failed is not kept -- every later attempt
    reports the same failure, and one that later succeeds is not handed a
    broken client.
    """
    monkeypatch.setenv("MONGODB_URI", uri)
    quick = dataclasses.replace(settings, mongodb_timeout_ms=200)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="Vector store request failed"):
            ingest_mod.index_version(quick)
    assert ingest_mod._clients == {}


def test_concurrent_first_opens_share_one_client(settings):
    """Every Streamlit session's first read creates the client at once; one
    client results, not one per session with all but one leaked."""
    barrier = threading.Barrier(4)

    def first_open():
        barrier.wait()
        return ingest_mod._client(settings)

    with _workers(4) as pool:
        clients = [
            f.result(timeout=30) for f in [pool.submit(first_open) for _ in range(4)]
        ]

    assert len({id(client) for client in clients}) == 1
    assert len(ingest_mod._clients) == 1


# --- index_version -----------------------------------------------------------


def test_index_version_is_empty_before_any_ingest_and_creates_nothing(settings):
    """A read creates nothing.

    The app calls this on every rerun, a fresh setup's first included. A
    collection appearing before anything was ingested would pass for an index,
    and the app would report it empty instead of missing.
    """
    assert ingest_mod.index_version(settings) == ""
    assert ingest_mod.indexed_sources(settings) == set()
    with pytest.raises(FileNotFoundError, match="Run `rag ingest` first"):
        ingest_mod.require_index(settings)

    assert chunks_of(settings).database.list_collection_names() == []


def test_index_version_is_empty_for_a_collection_never_ingested_into(
    settings, fake_embeddings
):
    """A mistyped COLLECTION_NAME is "nothing ingested", not an error -- the
    pipeline's own guard is what names the fix."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    other = dataclasses.replace(settings, collection_name="never_ingested")

    assert ingest_mod.index_version(other) == ""
    assert ingest_mod.indexed_sources(other) == set()
    # ...and reading it did not create it.
    assert "never_ingested" not in chunks_of(settings).database.list_collection_names()


def test_index_version_changes_on_exactly_the_corpus_changes(settings, fake_embeddings):
    """The app's pipeline cache is keyed on this, so it must move on an edit, an
    add or a removal -- or the app keeps answering from the old corpus -- and on
    nothing else, or every rerun after an unchanged ingest reloads for nothing.
    """

    def version_after_ingest() -> str:
        ingest_mod.ingest(settings, embeddings=fake_embeddings)
        return ingest_mod.index_version(settings)

    first = version_after_ingest()
    assert first
    assert version_after_ingest() == first, "an unchanged re-ingest moved it"

    (settings.data_dir / "a.md").write_text("edited", encoding="utf-8")
    edited = version_after_ingest()
    (settings.data_dir / "c.md").write_text("added", encoding="utf-8")
    added = version_after_ingest()
    (settings.data_dir / "sub" / "b.txt").unlink()
    removed = version_after_ingest()

    assert len({first, edited, added, removed}) == 4


def _spy_meta_writes(monkeypatch, settings) -> list[object]:
    """Record every replace_one on the bookkeeping collection, then perform it."""
    replace_one = Collection.replace_one
    writes: list[object] = []

    def spy(self, filter, *args, **kwargs):
        if self.name == ingest_mod._META_COLLECTION:
            writes.append(filter)
        return replace_one(self, filter, *args, **kwargs)

    monkeypatch.setattr(Collection, "replace_one", spy)
    return writes


def test_an_unchanged_reingest_writes_no_version_stamp(
    settings, fake_embeddings, monkeypatch
):
    """The stamp is reconciled on every run, but written only when it differs."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    writes = _spy_meta_writes(monkeypatch, settings)

    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert writes == []


def test_a_rerun_repairs_a_version_stamp_a_failed_run_missed(
    settings, fake_embeddings, monkeypatch
):
    """A run whose chunks landed but whose stamp did not must be finished by the
    next one.

    After such a run every source looks current, so a stamp written only when
    something changed would never be written: index_version would keep naming
    the old corpus for good, and the app -- keyed on it -- would keep answering
    from its old pipeline until restarted, however often `rag ingest` succeeded.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    before = ingest_mod.index_version(settings)
    (settings.data_dir / "a.md").write_text("# Alpha\nrewritten.\n", encoding="utf-8")
    replace_one = Collection.replace_one

    def fail(self, *args, **kwargs):
        if self.name == ingest_mod._META_COLLECTION:
            raise pymongo.errors.OperationFailure("disk I/O error")
        return replace_one(self, *args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(Collection, "replace_one", fail)
        with pytest.raises(RuntimeError, match="disk I/O error"):
            ingest_mod.ingest(settings, embeddings=fake_embeddings)
    assert ingest_mod.index_version(settings) == before, "the stamp was not missed"

    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    elsewhere = dataclasses.replace(settings, collection_name="elsewhere")
    ingest_mod.ingest(elsewhere, embeddings=fake_embeddings)
    assert ingest_mod.index_version(settings) == ingest_mod.index_version(elsewhere)


def test_the_version_stamp_is_never_a_chunk(settings, fake_embeddings):
    """Bookkeeping lives beside the chunks, never among them, so nothing that
    reads, counts, deletes or searches chunks can meet it."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert chunks_of(settings).count_documents({}) == len(own_ids(settings))
    assert ingest_mod.index_version(settings)


def test_a_version_stamp_that_is_not_a_string_reads_as_no_version(
    settings, fake_embeddings
):
    """A stamp edited by hand into a number is "no version", not a cache key of
    the wrong type."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod._meta(settings).update_one(
        {"_id": ingest_mod._version_id(settings)}, {"$set": {"digest": 7}}
    )

    assert ingest_mod.index_version(settings) == ""


# --- other processes ---------------------------------------------------------


def _ingest_in_another_process(settings: Settings) -> None:
    """Run `ingest` in a separate interpreter, as a terminal `rag ingest` would.

    Configured the way the CLI is -- through the environment, every variable
    derived from ENV_VARS, and MONGODB_URI the container's -- with .env off so
    the developer's cannot answer, and with MLX hidden and a fake injected,
    since conftest's guards do not reach a child process.
    """
    env = {
        **os.environ,
        **{var: str(getattr(settings, var.lower())) for var in ENV_VARS},
        "PYTHON_DOTENV_DISABLED": "1",
    }
    code = textwrap.dedent(
        f"""
        import sys
        sys.modules["mlx"] = sys.modules["mlx_lm"] = None
        from langchain_core.embeddings import DeterministicFakeEmbedding
        from rag_pipeline.config import Settings
        from rag_pipeline.ingest import ingest
        ingest(
            Settings.from_env(),
            embeddings=DeterministicFakeEmbedding(size={settings.embedding_dimensions}),
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_a_store_held_by_this_process_sees_another_processs_ingest(
    settings, fake_embeddings
):
    """Why nothing resets the store before a pipeline is rebuilt.

    The app holds its pipeline -- and so its store -- while a terminal `rag
    ingest` rewrites the index. Atlas is a live server, so the held store
    searches the new index with no reset at all, and the version the app keys
    its cache on moves too. (Chroma's cached view did neither, which is what
    the reset the app no longer makes was for.)
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    held = ingest_mod.open_store(settings, fake_embeddings)
    retriever = held.as_retriever(search_kwargs={"k": 1, "pre_filter": OWN_CHUNKS})
    retriever.invoke("apples")  # a search now, before the other process writes
    before = ingest_mod.index_version(settings)

    new_text = "Okapis are forest giraffids from the Congo basin."
    (settings.data_dir / "okapi.md").write_text(f"{new_text}\n", encoding="utf-8")
    _ingest_in_another_process(settings)

    assert ingest_mod.index_version(settings) != before
    hits = retriever.invoke(new_text)
    assert [doc.metadata["source"] for doc in hits] == ["okapi.md"]


def test_an_ingest_leaves_a_held_store_searching(settings, fake_embeddings):
    """The client is shared by every pipeline in the process, so ingest must
    never close it: the app ingests an upload while another session's pipeline
    is still answering through the same client."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    held = ingest_mod.open_store(settings, fake_embeddings)
    retriever = held.as_retriever(search_kwargs={"k": 1, "pre_filter": OWN_CHUNKS})
    assert retriever.invoke("apples")

    (settings.data_dir / "c.md").write_text("# Gamma\nuploaded.\n", encoding="utf-8")
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert retriever.invoke("apples")


# --- one writer at a time ----------------------------------------------------


@contextlib.contextmanager
def _workers(count: int) -> Iterator[ThreadPoolExecutor]:
    """A thread pool whose exit does not wait for its workers.

    `with ThreadPoolExecutor()` joins every worker on the way out, unbounded,
    so a run that never finishes -- a lock never released, the very bug the
    tests below exist for -- would hang the suite without naming the test. Each
    test waits in `result(timeout=)` instead: that fails, or raises what the
    worker failed with. A stuck worker still holds the interpreter open at exit,
    as any non-daemon thread does, but only after the failure is reported.
    """
    pool = ThreadPoolExecutor(max_workers=count)
    try:
        yield pool
    finally:
        pool.shutdown(wait=False)


@contextlib.contextmanager
def _held_by_another_writer(settings) -> Iterator[None]:
    """Hold the collection's ingest lock the way another process's ingest would."""
    other = ingest_mod._WriterLock(settings)
    assert other.try_acquire()
    try:
        yield
    finally:
        other.release()


def test_concurrent_ingests_take_turns(settings, fake_embeddings, monkeypatch):
    """Two ingests at once run one after the other, not interleaved.

    The app re-ingests on upload from whichever session asked, so two can
    overlap; interleaved, each would decide what to delete and add from a state
    the other is changing. Reading data_dir -- the first thing done under the
    lock -- is slowed and recorded by thread, so overlapping runs would show up
    as the threads alternating. The order is the proof; a consistent index
    afterwards is the point.
    """
    calls: list[int] = []
    load = ingest_mod.load_documents

    def slow_load(data_dir):
        calls.append(threading.get_ident())
        time.sleep(0.3)  # a window wide enough for an unguarded run to enter
        calls.append(threading.get_ident())
        return load(data_dir)

    monkeypatch.setattr(ingest_mod, "load_documents", slow_load)
    start = threading.Barrier(2)

    def run() -> int:
        start.wait()
        return ingest_mod.ingest(settings, embeddings=fake_embeddings)

    with _workers(2) as pool:
        counts = [f.result(timeout=120) for f in [pool.submit(run) for _ in range(2)]]

    switches = sum(a != b for a, b in itertools.pairwise(calls))
    assert switches == 1, "the two ingests ran at once"
    assert counts == [len(own_ids(settings))] * 2


def test_ingest_waits_for_a_writer_holding_the_lock(settings, fake_embeddings):
    """The lock is held in Atlas, so it binds every process and machine.

    Held here the way another process's ingest would hold it: nothing may be
    written until it is released, and the ingest must then complete.
    """
    with _workers(1) as pool, _held_by_another_writer(settings):
        run = pool.submit(ingest_mod.ingest, settings, embeddings=fake_embeddings)
        wait([run], timeout=2)
        waited = not run.done()
        written_meanwhile = ingest_mod.index_version(settings)
    run.result(timeout=120)  # it completes once released, or raises what it failed with

    assert waited, "ingest ran while another writer held the lock"
    assert written_meanwhile == ""
    assert ingest_mod.index_version(settings) != ""


def test_a_run_that_waited_for_the_lock_indexes_data_dir_as_it_is_now(
    settings, fake_embeddings
):
    """data_dir is read under the lock, not before it.

    A run that read it and then waited would apply that older snapshot after
    the writer it waited on: anything that writer had just indexed -- another
    session's upload -- would be taken for a removed file and deleted, with the
    user already told it was added. Here a file lands while the run waits.
    """
    with _workers(1) as pool, _held_by_another_writer(settings):
        run = pool.submit(ingest_mod.ingest, settings, embeddings=fake_embeddings)
        wait([run], timeout=2)
        assert not run.done(), "ingest ran while another writer held the lock"
        (settings.data_dir / "late.md").write_text(
            "Written while the ingest waited.\n", encoding="utf-8"
        )
    run.result(timeout=120)

    assert "late.md" in sources_in(settings)


def test_a_dead_writers_lock_expires(settings, fake_embeddings):
    """A writer that died -- killed, its machine gone -- holds the lock only
    until its lease runs out; then the next ingest takes it over."""
    ingest_mod._meta(settings).insert_one(
        {
            "_id": f"ingest-lock:{settings.collection_name}",
            "owner": "a-writer-that-died",
            "expires_at": datetime.now(UTC) - timedelta(seconds=1),
        }
    )

    assert ingest_mod.ingest(settings, embeddings=fake_embeddings) > 0


def test_a_writer_whose_lease_was_taken_over_stops(settings):
    """A writer stalled past its lease has lost the lock to another; its next
    renewal -- made after every slice -- stops it before it writes again,
    rather than letting two writers interleave."""
    stalled = ingest_mod._WriterLock(settings)
    assert stalled.try_acquire()
    ingest_mod._meta(settings).update_one(
        {"_id": stalled._id},
        [{"$set": {"expires_at": {"$subtract": ["$$NOW", 1000]}}}],
    )
    successor = ingest_mod._WriterLock(settings)
    assert successor.try_acquire()

    with pytest.raises(RuntimeError, match="Lost the ingest lock"):
        stalled.renew()
    assert not stalled.try_acquire(), "the successor's live lease must hold"


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
