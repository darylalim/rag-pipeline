"""Indexing phase: load -> split -> embed -> store.

Run once (via ``rag ingest``) whenever the documents in ``data/`` change. The
expensive embedding step happens here; querying later just reloads the persisted
Chroma index from disk.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath

from chromadb.api.shared_system_client import SharedSystemClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings

from rag_pipeline.config import Settings

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
    # Fast-fail with a message in the union both frontends catch, before the
    # first API call — mirroring RAGPipeline's ANTHROPIC_API_KEY guard. Unlike
    # the old on-device model, embedding now needs a key at ingest as well.
    if not os.getenv("VOYAGE_API_KEY"):
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Embedding uses Voyage AI; set the key "
            "in your environment or a .env file (see .env.example)."
        )
    return VoyageAIEmbeddings(model=settings.embedding_model)


def open_store(settings: Settings, embeddings: Embeddings | None = None) -> Chroma:
    """Open the persisted Chroma collection.

    The store's identity (collection name, persist dir, embedding function) must
    match between indexing and querying, so both stages open it through this one
    factory. ``embeddings`` is injectable for tests; production leaves it None
    and builds the local model.
    """
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings or build_embeddings(settings),
        persist_directory=str(settings.persist_dir),
    )


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


def ingest(settings: Settings, embeddings: Embeddings | None = None) -> int:
    """Run the full indexing phase and persist the vector store.

    Only this collection's existing vectors are cleared before the fresh chunks
    are added, so re-ingesting is idempotent (no duplicate chunks) *without*
    deleting the persist directory — which may hold unrelated data. Returns the
    number of chunks indexed. ``embeddings`` is injectable so tests can
    substitute a lightweight fake; production callers leave it as None.
    """
    documents = load_documents(settings.data_dir)
    if not documents:
        raise ValueError(
            f"No readable documents found in {settings.data_dir} "
            f"(looked for {', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )

    chunks = split_documents(documents, settings)

    store = open_store(settings, embeddings)
    # Rebuild this collection in place: drop its existing vectors (if any), then
    # add the fresh chunks. Scoped to the collection, so re-ingest never touches
    # other files in the persist directory. `include=[]` fetches only the ids,
    # not every chunk's text and metadata.
    existing_ids = store.get(include=[])["ids"]
    if existing_ids:
        store.delete(ids=existing_ids)
    store.add_documents(chunks)
    return len(chunks)
