"""Tests for the indexing phase: loading, splitting, and building the store."""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from rag_pipeline import ingest as ingest_mod
from rag_pipeline.config import Settings


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
    loader -- pypdf raises its own exception types, which the loader's broad
    `except` is there to absorb.
    """
    root = tmp_path / "data"
    root.mkdir()
    (root / "broken.pdf").write_bytes(b"%PDF-1.4\nthis is not a real pdf\n")
    (root / "fine.md").write_text("readable content", encoding="utf-8")

    docs = ingest_mod.load_documents(root)

    assert [d.metadata["source"] for d in docs] == ["fine.md"]
    assert "broken.pdf" in capsys.readouterr().err


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


def test_ingest_preserves_foreign_documents_in_a_shared_collection(
    settings, fake_embeddings
):
    """ingest() is a scoped rebuild, never a collection wipe.

    The document-level successor to the old persist-dir guard, and stronger,
    because Atlas has no separate directory to preserve -- the collection itself
    may be shared. A document this pipeline did not write (no `content_hash`)
    must survive a rebuild that *does* delete, however the deletion is written.
    Proves the {"content_hash": {"$exists": True}} scoping on every read and
    delete: without it, ingest's deletion set would sweep up foreign documents,
    which under Chroma's exclusive persist dir could never happen.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    ingest_mod.reset_store_cache()
    store = ingest_mod.open_store(settings, fake_embeddings)
    # No content_hash, no source, no embedding: unrelated application data.
    store.collection.insert_one({"_id": "external:1", "text": "unrelated app data"})

    # Force a real deletion path: remove a file so the re-ingest runs delete_many.
    (settings.data_dir / "a.md").unlink()
    ingest_mod.reset_store_cache()
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    ingest_mod.reset_store_cache()
    store = ingest_mod.open_store(settings, fake_embeddings)
    assert store.collection.find_one({"_id": "external:1"}) is not None, (
        "ingest must not delete documents it did not write"
    )
    assert "a.md" not in sources_in(settings, fake_embeddings)


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
    s = Settings(data_dir=empty)

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

    # The collection holds n2 chunks, not 2*n2 — deterministic _ids make the
    # re-add an idempotent upsert-replace, not a duplicating append.
    ingest_mod.reset_store_cache()
    store = ingest_mod.open_store(settings, fake_embeddings)
    held = store.collection.count_documents({"content_hash": {"$exists": True}})
    assert held == n2


# --- incremental re-indexing -------------------------------------------------


class _CountingEmbeddings(Embeddings):
    """Records the texts it was asked to embed, so a *skipped* file is visible.

    The whole point of the incremental path is a call that does not happen, and
    a chunk count cannot see that: the index holds the same vectors either way,
    whether they were just re-embedded or left alone. Only the embedder knows.
    """

    def __init__(self, inner: Embeddings) -> None:
        self.inner = inner
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


@pytest.fixture
def counting_embeddings(fake_embeddings):
    return _CountingEmbeddings(fake_embeddings)


def sources_in(settings, embeddings) -> set[str]:
    """Every `source` the collection currently holds this pipeline's chunks for.

    Reads the raw pymongo collection — MongoDBAtlasVectorSearch has no enumerate
    — scoped to this pipeline's own documents, and flattened top-level metadata
    means `source` is a plain field.
    """
    ingest_mod.reset_store_cache()
    store = ingest_mod.open_store(settings, embeddings)
    return {
        doc["source"]
        for doc in store.collection.find(
            {"content_hash": {"$exists": True}}, {"source": 1}
        )
    }


def test_reingest_embeds_nothing_when_no_document_changed(
    settings, counting_embeddings
):
    """The saving that justifies the fingerprint at all.

    Embedding is a billed Voyage call, so re-ingesting an unchanged corpus must
    cost nothing. Asserted on the embedder rather than the returned count, which
    is deliberately the same both times.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    assert counting_embeddings.embedded, "the first ingest must embed everything"

    counting_embeddings.embedded.clear()
    ingest_mod.reset_store_cache()
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert counting_embeddings.embedded == []


def test_only_the_edited_document_is_re_embedded(settings, counting_embeddings):
    """A one-file edit costs one file, not the corpus."""
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    counting_embeddings.embedded.clear()

    (settings.data_dir / "a.md").write_text("# Alpha\nrewritten.\n", encoding="utf-8")
    ingest_mod.reset_store_cache()
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert counting_embeddings.embedded, "the edited file must be re-embedded"
    # b.txt's text is untouched, so none of it may appear in what was embedded.
    b_text = (settings.data_dir / "sub" / "b.txt").read_text(encoding="utf-8")
    assert not any(b_text.strip() in text for text in counting_embeddings.embedded)


def test_a_new_document_joins_the_existing_index(settings, counting_embeddings):
    """Adding a file must not cost, or disturb, the documents already indexed.

    This is the case a naive "index only what was just uploaded" would get
    wrong in the other direction — it is asserted from both ends, that the new
    file is present *and* that the old ones survived, because an implementation
    that rebuilt from the upload alone would still pass the first half.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    before = sources_in(settings, counting_embeddings)
    counting_embeddings.embedded.clear()

    (settings.data_dir / "c.md").write_text("# Gamma\nbrand new.\n", encoding="utf-8")
    ingest_mod.reset_store_cache()
    ingest_mod.ingest(settings, embeddings=counting_embeddings)

    assert sources_in(settings, counting_embeddings) == before | {"c.md"}
    assert all("brand new" in text for text in counting_embeddings.embedded)


def test_a_removed_document_loses_its_chunks(settings, fake_embeddings):
    """The index tracks `data_dir`, so a deleted file must not answer questions.

    The half of "only embed what changed" that is easy to skip: a file that is
    gone has no fresh chunks to add, so nothing about the add path would ever
    notice it. Its vectors would linger and stay retrievable.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    assert "a.md" in sources_in(settings, fake_embeddings)

    (settings.data_dir / "a.md").unlink()
    ingest_mod.reset_store_cache()
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    assert "a.md" not in sources_in(settings, fake_embeddings)


def test_changing_the_chunking_re_embeds_everything(settings, counting_embeddings):
    """Chunk boundaries are part of what the stored vectors represent.

    Content-only fingerprinting would leave every existing chunk in place under
    a new CHUNK_SIZE, so the index would keep vectors the current settings could
    not have produced — stale in a way no file inspection would reveal.
    """
    ingest_mod.ingest(settings, embeddings=counting_embeddings)
    counting_embeddings.embedded.clear()

    ingest_mod.reset_store_cache()
    ingest_mod.ingest(
        dataclasses.replace(settings, chunk_size=80, chunk_overlap=10),
        embeddings=counting_embeddings,
    )

    assert counting_embeddings.embedded != []


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
