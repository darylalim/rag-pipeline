"""Indexing phase: load -> split -> embed -> store.

Run once (via ``rag ingest``) whenever the documents in ``data/`` change. The
expensive embedding step happens here; querying later just reopens the
MongoDB Atlas collection and its vector index.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

import bson.errors
import pymongo.errors
import voyageai
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from rag_pipeline.config import Settings, require_env_key

# File extensions we know how to read into text.
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

# The reserved document that records the corpus fingerprint the Streamlit app
# keys its cache on (see index_version). It carries neither `content_hash` nor
# `embedding`, so it is invisible to every {"content_hash": {"$exists": True}}
# scan below and to $vectorSearch — it is bookkeeping, not a chunk.
_VERSION_ID = "__rag_ingest_version__"

# One MongoClient per process, reused across every open_store call. Unlike
# Chroma's per-directory client (which cached a stale on-disk snapshot), a
# MongoClient is a connection pool that always reads current server state — so
# the hazard inverts: the client must be created ONCE and reused, not cleared.
# reset_store_cache() closes and drops it (connection hygiene + fresh-process
# emulation for tests). The lock makes the lazy create/close atomic: Streamlit
# can build two pipelines concurrently, and an unguarded check-then-set would
# let both create a client and leak one.
_client: MongoClient[dict[str, Any]] | None = None
_client_lock = threading.Lock()

# Ceiling and interval for the search-index readiness polls below. Not a Setting
# (an internal build-latency bound, not user config), but named rather than
# inlined so the two related timeouts stay one value.
_INDEX_POLL_TIMEOUT_S = 120.0
_INDEX_POLL_INTERVAL_S = 1.0


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the embedding model.

    Defined here (not in the pipeline) because it is the one component that
    *must* be identical for indexing and querying -- vectors from different
    models are not comparable. Both stages import this single factory.

    Voyage embeds queries and documents asymmetrically; the wrapper handles
    that itself (input_type="document" from embed_documents at ingest,
    "query" from embed_query at retrieval), so neither stage passes it here.
    """
    # Fast-fail before the first API call -- mirroring RAGPipeline's
    # ANTHROPIC_API_KEY guard. Unlike the old on-device model, embedding now
    # needs a key at ingest as well as query.
    require_env_key("VOYAGE_API_KEY", "Embedding uses Voyage AI")
    return VoyageAIEmbeddings(model=settings.embedding_model)


def _get_client(settings: Settings) -> MongoClient[dict[str, Any]]:
    """The process-wide MongoClient, created once and pinged for fail-fast.

    ``MongoClient(...)`` is lazy -- it opens no connection and validates nothing
    until the first operation -- so a paused cluster, an IP not on the Atlas
    access list, or a bad URI would otherwise surface much later, deep inside an
    ingest or a query. The ``ping`` here is the one eager check that replaces the
    validation Chroma did at construction. ``MONGODB_URI`` carries credentials,
    so it is guarded like the provider API keys rather than living in Settings.
    """
    global _client
    # Double-checked: the common case (client already built) never takes the
    # lock, and the create is serialized so two concurrent callers can't each
    # build one and leak the loser.
    if _client is None:
        with _client_lock:
            if _client is None:
                require_env_key("MONGODB_URI", "The vector store is MongoDB Atlas")
                with provider_errors_as_runtime():
                    client: MongoClient[dict[str, Any]] = MongoClient(
                        os.environ["MONGODB_URI"],
                        serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
                    )
                    client.admin.command("ping")
                _client = client
    return _client


def _collection(settings: Settings) -> Collection[dict[str, Any]]:
    """The pymongo collection holding this pipeline's chunks."""
    return _get_client(settings)[settings.mongodb_db][settings.collection_name]


def open_store(
    settings: Settings, embeddings: Embeddings | None = None
) -> MongoDBAtlasVectorSearch:
    """Open the Atlas Vector Search store over this pipeline's collection.

    The store's identity -- (connection URI, database, collection, vector index
    name, embedding function) -- must match between indexing and querying, so
    both stages open it through this one factory. ``embeddings`` is injectable
    for tests; production leaves it None and builds the embedding model.

    Construction is deliberately inert: with the default ``auto_create_index``
    and ``dimensions`` langchain-mongodb creates nothing and makes no call, so
    the query path never builds an index (``ingest()`` owns that, via
    ``_ensure_vector_index``). ``.collection`` exposes the raw pymongo handle the
    incremental bookkeeping needs -- MongoDBAtlasVectorSearch has no enumerate.
    """
    return MongoDBAtlasVectorSearch(
        collection=_collection(settings),
        embedding=embeddings or build_embeddings(settings),
        index_name=settings.vector_index_name,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn=settings.atlas_similarity,
    )


@contextmanager
def provider_errors_as_runtime() -> Iterator[None]:
    """Translate Voyage/MongoDB failures into the RuntimeError the frontends catch.

    Wraps every provider call the retrieval side makes -- Voyage embedding at
    ingest and query and Voyage reranking at query, plus every MongoDB store op
    around them. None of voyageai's, pymongo's or bson's exception types is in
    the ``FileNotFoundError | RuntimeError | ValueError`` union both frontends
    handle, and none belongs in a frontend. This is the retrieval-side parallel
    to ``_generate()``'s ``anthropic.APIError`` translation.

    ``bson.errors.BSONError`` sits *outside* ``pymongo.errors.PyMongoError``, so
    it needs its own arm. Everything here becomes a RuntimeError, never a
    ValueError -- a bad ``MONGODB_URI`` (``ConfigurationError``) must land in the
    branch ``app.py`` catches *below* its sidebar, keeping the uploader reachable,
    rather than the ``ValueError`` branch that stops the script above it.
    """
    try:
        yield
    except voyageai.error.VoyageError as exc:
        raise RuntimeError(f"Voyage API request failed: {exc}") from exc
    except (pymongo.errors.PyMongoError, bson.errors.BSONError) as exc:
        # The dimension hint keys off the message, not the type: a wrong-width
        # index surfaces at *query* time (Atlas accepts a wrong-width vector on
        # insert and only fails at $vectorSearch), as an ordinary OperationFailure.
        hint = (
            " The vectors were likely built at a different EMBEDDING_MODEL/"
            "EMBEDDING_DIMENSIONS; run `rag ingest` to re-embed and rebuild the index."
            if "dimension" in str(exc).lower()
            else ""
        )
        raise RuntimeError(f"Vector store request failed: {exc}.{hint}") from exc


def reset_store_cache() -> None:
    """Close and drop the process-wide MongoClient.

    The inverse of the old chromadb cache-clear: there is no stale on-disk
    snapshot to discard (a live server is always current), so production never
    needs this -- a rebuilt pipeline reuses the live pooled client. It exists for
    the tests, which call it at every boundary and between in-process re-ingests
    to emulate a fresh CLI process. Locked, so it cannot race a concurrent
    ``_get_client`` create.
    """
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None


def index_version(settings: Settings) -> str:
    """A value that changes whenever the on-disk corpus is rebuilt.

    The Streamlit app keys its pipeline cache on this so a `rag ingest` is
    picked up automatically. Reads the digest ingest() stamps over the corpus
    fingerprints (see _write_index_version) -- stable across an unchanged
    re-ingest, so it does not needlessly bust the cache. ``""`` means nothing has
    been ingested yet. Can raise RuntimeError (e.g. a paused cluster); the app
    reads it inside the guard that already catches that.
    """
    with provider_errors_as_runtime():
        doc = _collection(settings).find_one({"_id": _VERSION_ID}, {"digest": 1})
    # `.get`, not `doc["digest"]`: a marker present but missing the field (hand
    # edited, a partial write) reads as "no version" rather than a KeyError that
    # would escape the caught union into a crash page.
    return doc.get("digest", "") if doc else ""


def _read_pdf(path: Path) -> str:
    """Extract text from every page of a PDF and join it."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def save_upload(data_dir: Path, filename: str, data: bytes) -> str:
    """Write one uploaded file into ``data_dir``; return the name it landed under.

    Here rather than in the frontend that calls it because the name it produces
    has to satisfy ``load_documents``' contract -- a supported suffix, and a
    relative POSIX path that resolves back under ``data_dir``. ``index_version``
    sits in this module for the same reason: serving one frontend is not the
    same as belonging to it.

    The name is reduced to its final component, and *that* is what keeps an
    upload from writing outside ``data_dir`` -- a browser supplies this string,
    so ``../../.ssh/authorized_keys`` arrives as an ordinary value rather than
    an attack the caller has to notice. Backslashes are folded first, because a
    POSIX server does not read a Windows separator as one and would otherwise
    keep ``C:\\Users\\evil.md`` as a single filename. Flattening also discards
    any directory the uploader meant to keep, which is the safe direction to be
    wrong in and is recoverable by writing to ``data_dir`` directly.

    Where that boundary stops: this does write through a symlink already sitting
    in ``data_dir``, as any program would. Putting one there needs the access it
    would grant, so it is not a boundary this is trying to hold -- the untrusted
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
    # check rejects it too -- there is no name to write it under either way.
    # `.suffix` reads the final component, so it is the same on the whole path
    # as on ``name`` -- one object, not two.
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
            # must not abort the whole ingest -- skip it with a warning.
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
    different chunks; ``EMBEDDING_MODEL``, because a vector means nothing except
    with respect to the model that produced it; and ``EMBEDDING_DIMENSIONS``,
    because the Atlas index fixes ``numDimensions`` at create time, so a
    different-width model produces vectors the existing index cannot serve.
    Content alone would let a re-ingest keep vectors the current settings would
    never have produced -- and a changed model is the dangerous half: the chunks
    still *look* current, so the skip is silent and every later query compares
    against vectors from a model that is no longer configured.

    Hashed rather than compared against mtime or size. mtime moves when nothing
    changed (a checkout, a copy, `touch`) and stands still when something did (a
    write that preserves it), and either error is silent -- one re-embeds the
    corpus for nothing, the other serves answers from a file's previous
    contents. This reads the bytes we already read.
    """
    payload = (
        f"{settings.embedding_model}:{settings.embedding_dimensions}:"
        f"{settings.chunk_size}:{settings.chunk_overlap}:{text}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ensure_vector_index(
    collection: Collection[dict[str, Any]], settings: Settings
) -> None:
    """Create the Atlas Vector Search index if it is absent, then wait for it.

    Programmatic creation works on the free tier, so ``rag ingest`` stays a
    complete setup step -- there is no separate "create the index" instruction.
    Create-if-missing only: a change to ``EMBEDDING_DIMENSIONS``/
    ``ATLAS_SIMILARITY`` needs the index dropped (or ``VECTOR_INDEX_NAME``
    changed), since editing width in place is a larger operation than a demo
    warrants. The build is asynchronous and a query against a not-yet-ready
    index returns *zero rows with no error*, so the queryable poll is
    load-bearing, not politeness.
    """
    if not list(collection.list_search_indexes(settings.vector_index_name)):
        collection.create_search_index(
            model=SearchIndexModel(
                definition={
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": settings.embedding_dimensions,
                            "similarity": settings.atlas_similarity,
                        },
                        {"type": "filter", "path": "source"},
                    ]
                },
                name=settings.vector_index_name,
                type="vectorSearch",
            )
        )
    _await_queryable(collection, settings.vector_index_name)


def _await_queryable(
    collection: Collection[dict[str, Any]],
    index_name: str,
    timeout_s: float = _INDEX_POLL_TIMEOUT_S,
) -> None:
    """Block until the named search index reports ``queryable`` is True.

    The ceiling is generous because the poll returns the instant the index is
    ready: it only bites when the search process is genuinely lagging (a busy
    shared mongot, a cold cluster), where a short timeout would surface a false
    "index never built" for what a moment's wait would have resolved.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        info = list(collection.list_search_indexes(index_name))
        if info and info[0].get("queryable"):
            return
        time.sleep(_INDEX_POLL_INTERVAL_S)
    raise RuntimeError(
        f"Vector index {index_name!r} did not become queryable within {timeout_s:.0f}s."
    )


def _await_searchable(
    collection: Collection[dict[str, Any]],
    settings: Settings,
    ids: list[str],
    timeout_s: float = _INDEX_POLL_TIMEOUT_S,
) -> None:
    """Block until freshly-inserted chunks are returned by ``$vectorSearch``.

    Atlas Vector Search indexes asynchronously off the collection's change
    stream, so a document is in the collection (``find`` sees it) before it is
    searchable. Every caller that ingests and then retrieves in the same process
    -- the app answering a question about a file just uploaded, the round-trip
    test -- depends on this, so the wait lives in ingest rather than in each of
    them, and ``retrieve()`` stays free of a poll on the hot path.

    Probes with one inserted chunk's own vector: under cosine it is its own
    nearest neighbour, so it surfaces the moment the index has caught up. The
    search is pre-filtered to that chunk's own ``source`` (an indexed filter
    field), which both keeps the exact (brute-force) scan off the rest of the
    collection and bounds the tie set: a repetitive corpus embeds distinct chunks
    to *identical* vectors, so a top-1 query could stably return a tie-mate rather
    than the probe itself and never match; asking for ``len(ids)`` results (capped
    at the $vectorSearch maximum) within the one source always includes the probe.
    """
    probe = collection.find_one({"_id": ids[0]}, {"embedding": 1, "source": 1})
    if not probe or "embedding" not in probe:
        return
    stage: dict[str, Any] = {
        "index": settings.vector_index_name,
        "path": "embedding",
        "queryVector": probe["embedding"],
        "exact": True,
        "limit": min(len(ids), 10000),
    }
    if probe.get("source") is not None:
        stage["filter"] = {"source": probe["source"]}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        hits = collection.aggregate(
            [{"$vectorSearch": stage}, {"$project": {"_id": 1}}]
        )
        if any(hit["_id"] == ids[0] for hit in hits):
            return
        time.sleep(_INDEX_POLL_INTERVAL_S)
    raise RuntimeError("Newly ingested documents did not become searchable in time.")


def _write_index_version(
    collection: Collection[dict[str, Any]], fresh: dict[str, str]
) -> None:
    """Stamp a digest of the corpus fingerprints, for index_version to read.

    A reserved document with no ``content_hash`` or ``embedding`` field, so it is
    invisible to the incremental scans and to $vectorSearch. Digesting the
    fingerprints (not a counter or timestamp) keeps it stable across an unchanged
    re-ingest, so the Streamlit cache is busted on exactly the events an edit,
    add, or removal changes -- and no others.
    """
    digest = hashlib.sha256(
        "".join(f"{source}:{h};" for source, h in sorted(fresh.items())).encode("utf-8")
    ).hexdigest()
    collection.replace_one(
        {"_id": _VERSION_ID}, {"_id": _VERSION_ID, "digest": digest}, upsert=True
    )


def ingest(settings: Settings, embeddings: Embeddings | None = None) -> int:
    """Bring the index in line with ``data_dir``, embedding only what changed.

    Afterwards the collection holds exactly the chunks for the documents
    currently in ``data_dir`` -- nothing stale, nothing missing -- which is the
    property every caller depends on: the app rebuilds after an upload and
    expects the sample corpus to still be answerable, and a file edited by hand
    between runs must be picked up without being announced.

    Reaching that state costs one embedding call per *changed* document rather
    than per document. Embedding is a paid, rate-limited Voyage API call, so a
    corpus that is re-ingested whenever one file is added would otherwise charge
    for the whole of it every time. Unchanged documents keep the vectors they
    already have; changed and removed ones have their chunks dropped first, so a
    file's old text can never outlive it in the index.

    Scoped throughout to the documents this pipeline wrote -- every read and
    delete is filtered to ``{"content_hash": {"$exists": True}}`` -- so a
    collection shared with unrelated data is never read, counted, or deleted from
    (``ingest`` never wipes anything wholesale). Chunks are keyed by a
    deterministic ``_id`` (``source:index:content_hash``), so re-adding is an
    idempotent upsert-replace rather than a duplicating append. Returns the
    number of chunks the index now holds, not the number re-embedded: it
    describes the index, which is what makes re-ingesting the same corpus report
    the same number. ``embeddings`` is injectable so tests can substitute a
    lightweight fake; production callers leave it as None.
    """
    documents = load_documents(settings.data_dir)
    if not documents:
        raise ValueError(
            f"No readable documents found in {settings.data_dir} "
            f"(looked for {', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )

    embedder = embeddings or build_embeddings(settings)

    for document in documents:
        document.metadata["content_hash"] = _fingerprint(
            document.page_content, settings
        )
    # Chunks inherit their parent's metadata, so each carries the source's
    # fingerprint and the comparison below needs no second pass over the files.
    chunks = split_documents(documents, settings)
    fresh = {doc.metadata["source"]: doc.metadata["content_hash"] for doc in documents}

    store = open_store(settings, embedder)
    collection = store.collection
    with provider_errors_as_runtime():
        # Read the state SCOPED to this pipeline's chunks. langchain-mongodb
        # flattens Document metadata to top-level fields, so `source` and
        # `content_hash` are queryable directly; one row per source answers for
        # all its chunks, so the aggregation groups them server-side.
        indexed = {
            row["_id"]: row["hash"]
            for row in collection.aggregate(
                [
                    {"$match": {"content_hash": {"$exists": True}}},
                    {"$group": {"_id": "$source", "hash": {"$first": "$content_hash"}}},
                ]
            )
        }

        # Group fresh chunks by source, then split the sources into what to drop
        # and what to (re-)embed. Deletion is computed over the *indexed* sources,
        # so a source gone from data_dir is dropped rather than merely not added.
        chunks_by_source: dict[str, list[Document]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_source[chunk.metadata["source"]].append(chunk)

        changed = {
            source
            for source, fingerprint in fresh.items()
            if indexed.get(source) != fingerprint
        }
        new_chunks: list[Document] = []
        ids: list[str] = []
        for source in changed:
            for i, chunk in enumerate(chunks_by_source[source]):
                new_chunks.append(chunk)
                ids.append(f"{source}:{i}:{chunk.metadata['content_hash']}")

        # Validate the declared width against the live model *before* any write --
        # Atlas would otherwise accept a wrong-width vector on insert and only fail
        # at query time. Only when there is something to embed, so an unchanged
        # re-ingest still makes no embedding call at all (this embed_query is the
        # sole exception, and it runs only on a run about to embed documents anyway).
        if new_chunks:
            probe_dims = len(embedder.embed_query("dimension probe"))
            if probe_dims != settings.embedding_dimensions:
                raise ValueError(
                    f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} but "
                    f"{settings.embedding_model} produced {probe_dims}-wide vectors. "
                    f"Set EMBEDDING_DIMENSIONS={probe_dims}."
                )

        # Delete superseded chunks, then add changed ones, then build the index.
        # The order is forced by Atlas: MongoDB creates the collection implicitly
        # on the first insert, and create_search_index over a namespace that does
        # not exist yet fails -- so the index can only be ensured once documents
        # are present.
        superseded = [
            source for source in indexed if fresh.get(source) != indexed[source]
        ]
        if superseded:
            collection.delete_many(
                {"source": {"$in": superseded}, "content_hash": {"$exists": True}}
            )
        if new_chunks:
            store.add_documents(new_chunks, ids=ids)

        _ensure_vector_index(collection, settings)

        if new_chunks:
            _await_searchable(collection, settings, ids)

        # Only when the corpus actually changed: on a no-op re-ingest the stored
        # digest already equals hash(fresh), so re-writing it is a needless write.
        if new_chunks or superseded:
            _write_index_version(collection, fresh)
    return len(chunks)
