"""Indexing phase: load -> split -> embed -> store.

Run once (via ``rag ingest``) whenever the documents in ``data/`` change. The
expensive embedding step happens here; querying later just reloads the persisted
Chroma index from disk.
"""

from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import chromadb.errors
import voyageai
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings

from rag_pipeline.config import Settings, require_env_key

# File extensions we know how to read into text.
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the embedding model.

    Defined here (not in the pipeline) because it is the one component that
    *must* be identical for indexing and querying — vectors from different
    models are not comparable. Both stages import this single factory.

    Voyage embeds queries and documents asymmetrically; the wrapper handles
    that itself (input_type="document" from embed_documents at ingest,
    "query" from embed_query at retrieval), so neither stage passes it here.
    """
    # Fast-fail before the first API call — mirroring RAGPipeline's
    # ANTHROPIC_API_KEY guard. Unlike the old on-device model, embedding now
    # needs a key at ingest as well as query.
    require_env_key("VOYAGE_API_KEY", "Embedding uses Voyage AI")
    return VoyageAIEmbeddings(model=settings.embedding_model)


def open_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    """Open the persisted Chroma collection.

    The store's identity (collection name, persist dir, embedding function) must
    match between indexing and querying, so both stages open it through this one
    factory. ``embeddings`` is injectable for tests; production leaves it None
    and builds the embedding model.

    The translation is here rather than left to each caller because ``Chroma(...)``
    creates and validates the collection eagerly: an invalid ``COLLECTION_NAME``,
    or an index file that cannot be read, raises a ``ChromaError`` *at
    construction* — above whatever block a caller wraps its own store ops in, and
    so outside the union both frontends catch. Failing to establish the store's
    identity belongs with the code that establishes it. ``build_embeddings()``'s
    missing-key ``RuntimeError`` passes through unchanged; it is neither a Voyage
    nor a Chroma error.
    """
    with voyage_errors_as_runtime():
        return Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings or build_embeddings(settings),
            persist_directory=str(settings.persist_dir),
        )


@contextmanager
def voyage_errors_as_runtime() -> Iterator[None]:
    """Translate Voyage/Chroma failures into the RuntimeError the frontends catch.

    Wraps every Voyage AI call — embedding at ingest and query, and reranking at
    query — plus the Chroma store ops around them. Neither voyageai's nor
    chromadb's exception type is in the ``FileNotFoundError | RuntimeError |
    ValueError`` union both frontends handle, and neither belongs in a frontend.
    This is the retrieval-side parallel to ``_generate()``'s ``anthropic.APIError``
    translation, covering all three Voyage calls: ``ingest()`` here and
    ``RAGPipeline.retrieve()`` in pipeline.py (which reranks as well as embeds).
    ``open_store()`` is the third wrapping site, for the store errors that land
    before either of those — including the bad ``COLLECTION_NAME`` the comment
    below keys off, which is raised at construction and reachable no other way.
    """
    try:
        yield
    except voyageai.error.VoyageError as exc:
        raise RuntimeError(f"Voyage API request failed: {exc}") from exc
    except chromadb.errors.ChromaError as exc:
        # Translate any store-layer failure into the caught union. Add the
        # "rebuild your index" hint only for a dimension mismatch, keyed off the
        # message rather than the exception type: chromadb raises the same type
        # for unrelated validation (e.g. a bad COLLECTION_NAME) that the hint
        # would misdiagnose.
        hint = (
            " The index was likely built with a different EMBEDDING_MODEL; delete "
            "the persist directory (or change COLLECTION_NAME), then run `rag ingest`."
            if "dimension" in str(exc).lower()
            else ""
        )
        raise RuntimeError(f"{exc}.{hint}") from exc


def reset_store_cache() -> None:
    """Drop chromadb's per-process client cache.

    chromadb caches one client per persist directory within a process. A caller
    that re-opens a store after it was rebuilt on disk (e.g. the Streamlit app
    after `rag ingest`) must clear this first, or it reuses the stale client and
    reads the old index. Centralized here so callers don't reach into chromadb
    internals themselves.
    """
    SharedSystemClient.clear_system_cache()


def index_version(settings: Settings) -> float:
    """A value that changes whenever the on-disk index is rebuilt.

    The Streamlit app keys its pipeline cache on this so a `rag ingest` is
    picked up automatically. Reads one sentinel file's mtime (chromadb bumps it
    on every write) rather than walking the whole index directory.
    """
    sentinel = settings.persist_dir / "chroma.sqlite3"
    return sentinel.stat().st_mtime if sentinel.exists() else 0.0


def _read_pdf(path: Path) -> str:
    """Extract text from every page of a PDF and join it."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def save_upload(data_dir: Path, filename: str, data: bytes) -> str:
    """Write one uploaded file into ``data_dir``; return the name it landed under.

    Here rather than in the frontend that calls it because the name it produces
    has to satisfy ``load_documents``' contract — a supported suffix, and a
    relative POSIX path that resolves back under ``data_dir``. ``index_version``
    sits in this module for the same reason: serving one frontend is not the
    same as belonging to it.

    The name is reduced to its final component, and *that* is what keeps an
    upload from writing outside ``data_dir`` — a browser supplies this string,
    so ``../../.ssh/authorized_keys`` arrives as an ordinary value rather than
    an attack the caller has to notice. Backslashes are folded first, because a
    POSIX server does not read a Windows separator as one and would otherwise
    keep ``C:\\Users\\evil.md`` as a single filename. Flattening also discards
    any directory the uploader meant to keep, which is the safe direction to be
    wrong in and is recoverable by writing to ``data_dir`` directly.

    Where that boundary stops: this does write through a symlink already sitting
    in ``data_dir``, as any program would. Putting one there needs the access it
    would grant, so it is not a boundary this is trying to hold — the untrusted
    input here is the *name*, not the directory's existing contents.

    Bytes are written through unchanged rather than decoded and re-encoded: a
    ``.pdf`` is binary, and a mis-encoded ``.txt`` should meet the same warn-and
    -skip path in ``load_documents`` that one copied in by hand does.

    Raises ``ValueError`` for a name this pipeline cannot index, which is inside
    the union both frontends already catch.
    """
    path = PurePosixPath(filename.replace("\\", "/"))
    name = path.name
    # An empty name (``..``, ``/``, ``""``) has an empty suffix, so this one
    # check rejects it too — there is no name to write it under either way.
    # `.suffix` reads the final component, so it is the same on the whole path
    # as on ``name`` — one object, not two.
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Cannot index {filename!r}: expected one of "
            f"{', '.join(sorted(SUPPORTED_SUFFIXES))}."
        )

    # Created here so an upload can bootstrap an empty checkout, rather than
    # failing on the one path where the app has nothing else to offer.
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / name).write_bytes(data)
    return name


def load_documents(data_dir: Path) -> list[Document]:
    """Walk ``data_dir`` and read supported files into LangChain Documents.

    Each document records its file path (relative to ``data_dir``) under the
    ``source`` metadata key so answers can cite where evidence came from.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in SUPPORTED_SUFFIXES:
            continue

        source = path.relative_to(data_dir).as_posix()
        try:
            if suffix == ".pdf":
                text = _read_pdf(path)
            else:
                text = path.read_text(encoding="utf-8")
        except Exception as exc:
            # One unreadable file (bad encoding, corrupt PDF, permissions)
            # must not abort the whole ingest — skip it with a warning.
            print(
                f"Warning: skipping unreadable file {source!r}: {exc}", file=sys.stderr
            )
            continue

        if not text.strip():
            continue  # skip empty files

        documents.append(Document(page_content=text, metadata={"source": source}))

    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Recursive character splitting breaks on the most natural boundary that fits
    (paragraph, then line, then space), so chunks stay coherent. The overlap
    carries a little context across boundaries so a sentence split between two
    chunks is not lost to either.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def _fingerprint(text: str, settings: Settings) -> str:
    """What must be unchanged for a document's existing chunks to still be valid.

    The extracted text, and every setting the stored vectors depend on: the
    splitter's, because the same file under a new ``CHUNK_SIZE`` is cut into
    different chunks, and ``EMBEDDING_MODEL``, because a vector means nothing
    except with respect to the model that produced it. Content alone would let a
    re-ingest keep vectors the current settings would never have produced — and
    a changed model is the dangerous half: the chunks still *look* current, so
    the skip is silent and every later query compares against vectors from a
    model that is no longer configured.

    Hashed rather than compared against mtime or size. mtime moves when nothing
    changed (a checkout, a copy, `touch`) and stands still when something did (a
    write that preserves it), and either error is silent — one re-embeds the
    corpus for nothing, the other serves answers from a file's previous
    contents. This reads the bytes we already read.
    """
    payload = (
        f"{settings.embedding_model}:{settings.chunk_size}:"
        f"{settings.chunk_overlap}:{text}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ingest(settings: Settings, embeddings: Embeddings | None = None) -> int:
    """Bring the index in line with ``data_dir``, embedding only what changed.

    Afterwards the collection holds exactly the chunks for the documents
    currently in ``data_dir`` — nothing stale, nothing missing — which is the
    property every caller depends on: the app rebuilds after an upload and
    expects the sample corpus to still be answerable, and a file edited by hand
    between runs must be picked up without being announced.

    Reaching that state costs one embedding call per *changed* document rather
    than per document. Embedding is a paid, rate-limited Voyage API call, so a
    corpus that is re-ingested whenever one file is added would otherwise charge
    for the whole of it every time. Unchanged documents keep the vectors they
    already have; changed and removed ones have their chunks dropped first, so a
    file's old text can never outlive it in the index.

    Scoped to this collection's own ids throughout — the persist directory may
    hold unrelated data and is never deleted. Returns the number of chunks the
    index now holds, not the number re-embedded: it describes the index, which
    is what makes re-ingesting the same corpus report the same number.
    ``embeddings`` is injectable so tests can substitute a lightweight fake;
    production callers leave it as None.
    """
    documents = load_documents(settings.data_dir)
    if not documents:
        raise ValueError(
            f"No readable documents found in {settings.data_dir} "
            f"(looked for {', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )

    for document in documents:
        document.metadata["content_hash"] = _fingerprint(
            document.page_content, settings
        )
    # Chunks inherit their parent's metadata, so each carries the source's
    # fingerprint and the comparison below needs no second pass over the files.
    chunks = split_documents(documents, settings)
    fresh = {doc.metadata["source"]: doc.metadata["content_hash"] for doc in documents}

    store = open_store(settings, embeddings)
    with voyage_errors_as_runtime():
        # Metadata, not ids alone: the fingerprints are what decide the work.
        # Documents are still whole here, so one chunk per source answers for it.
        stored = store.get(include=["metadatas"])
        ids_by_source: dict[str, list[str]] = defaultdict(list)
        indexed: dict[str, str | None] = {}
        for chunk_id, metadata in zip(stored["ids"], stored["metadatas"], strict=True):
            source = str(metadata.get("source", "unknown"))
            ids_by_source[source].append(chunk_id)
            # `.get`, because an index built before fingerprints existed has no
            # such key -- it reads as changed and is re-embedded once.
            fingerprint = metadata.get("content_hash")
            indexed[source] = str(fingerprint) if fingerprint is not None else None

        # Deleted first, and computed over the *indexed* sources, so a source
        # that is gone from data_dir is dropped rather than merely not refreshed.
        superseded = [
            chunk_id
            for source, fingerprint in indexed.items()
            if fresh.get(source) != fingerprint
            for chunk_id in ids_by_source[source]
        ]
        if superseded:
            store.delete(ids=superseded)

        changed = {
            source
            for source, fingerprint in fresh.items()
            if indexed.get(source) != fingerprint
        }
        if changed:
            store.add_documents(
                [chunk for chunk in chunks if chunk.metadata["source"] in changed]
            )
    return len(chunks)
