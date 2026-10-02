"""Indexing phase: load -> split -> embed -> store.

Run (via ``rag ingest``) whenever the documents in ``data/`` change. The
expensive embedding step happens here; querying later searches the MongoDB
Atlas collection and the vector index this builds.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import time
import unicodedata
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Any

import bson.errors
import pymongo.errors
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection
from pymongo.operations import SearchIndexModel

from rag_pipeline.config import Settings, require_env_key
from rag_pipeline.mlx_models import QwenVLEmbeddings

# File extensions we know how to read into text.
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

# Every chunk this pipeline writes carries `ingested_by: "rag-pipeline"`, and
# every read, delete and search it makes is filtered on it -- so a collection
# shared with unrelated records is never read, counted, deleted from or
# retrieved from. A dedicated marker matched by equality rather than "has a
# content_hash": a foreign record could carry a field of that name, and a
# scoped delete built on the guess would remove it. Spelled with `$eq` because
# that is the form both a find and $vectorSearch's pre-filter accept.
_OWNER = "rag-pipeline"
OWN_CHUNKS: dict[str, Any] = {"ingested_by": {"$eq": _OWNER}}

# The fields of a stored chunk, beside its metadata (which langchain-mongodb
# writes as top-level fields: `source`, `content_hash`, `ingested_by`).
_TEXT_KEY = "text"
_EMBEDDING_KEY = "embedding"

# Bookkeeping that is not a chunk -- the corpus digest and the writer lock --
# lives in a collection of its own beside the chunks, keyed by the chunks'
# collection name, so it can never be retrieved, counted or deleted as one.
_META_COLLECTION = "rag_pipeline_meta"

# Not settings: internal bounds on waits for Atlas's own asynchronous work,
# named so the related values stay one value each.
_INDEX_POLL_TIMEOUT_S = 180.0
_INDEX_POLL_INTERVAL_S = 0.25
# The writer lock's lease. Renewed after every slice of adds, so it only has to
# outlast one slice (and the index build); a writer that dies holds the lock no
# longer than this.
_LOCK_LEASE_S = 300
_LOCK_POLL_S = 1.0
# Chunks embedded and written per slice: small enough that an interrupted run
# keeps most of its progress and the lock is renewed often.
_ADD_SLICE = 256


def build_embeddings(settings: Settings) -> Embeddings:
    """Construct the embedding model.

    Defined here (not in the pipeline) because it is the one component that
    *must* be identical for indexing and querying -- vectors from different
    models are not comparable. Both stages import this single factory.

    Qwen3-VL-Embedding is trained asymmetrically -- documents and questions are
    embedded under different instructions -- and the adapter applies them itself
    (``embed_documents`` at ingest, ``embed_query`` at retrieval), so neither
    stage passes one here. Cheap to call again: the weights load once per
    process, so the app's ingest after an upload reuses the model its pipeline
    already holds instead of loading a second copy.
    """
    return QwenVLEmbeddings(
        settings.embedding_model, dimensions=settings.embedding_dimensions
    )


# One MongoClient per (URI, timeout) for the life of the process. A client is a
# connection pool that always reads the server's current state, so -- unlike
# a cached on-disk store -- it is never stale and must not be rebuilt per
# pipeline: closing one closes it under every pipeline still using it.
# reset_store_cache() exists for the tests. The lock makes the lazy create
# atomic, so two Streamlit sessions cannot each build one and leak the loser.
_clients: dict[tuple[str, int], MongoClient[dict[str, Any]]] = {}
_clients_lock = threading.Lock()


def _client(settings: Settings) -> MongoClient[dict[str, Any]]:
    """The process's client for ``MONGODB_URI``; every one is built here.

    ``MongoClient(...)`` connects lazily, so a paused cluster, an IP missing
    from the Atlas access list, or wrong credentials would otherwise surface
    deep inside an ingest or a question. The ``ping`` on creation reports them
    where the store is first opened, and a client whose ping failed is not kept.
    """
    uri = require_env_key("MONGODB_URI", "The index is stored in MongoDB Atlas")
    key = (uri, settings.mongodb_timeout_ms)
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            with store_errors_as_runtime():
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
                    appname="rag-pipeline",
                )
                try:
                    client.admin.command("ping")
                except BaseException:
                    client.close()
                    raise
            _clients[key] = client
    return client


def _collection(settings: Settings) -> Collection[dict[str, Any]]:
    """The collection holding the chunks, for bookkeeping that needs no model.

    Getting a handle creates nothing: MongoDB creates a collection only on its
    first write, and only ``ingest`` writes. Getting one can still fail -- a
    database or collection name MongoDB refuses is pymongo's ``InvalidName``,
    raised here rather than at the first operation -- so it is translated here,
    where every caller gets its handle.
    """
    client = _client(settings)
    with store_errors_as_runtime():
        return client[settings.mongodb_db][settings.collection_name]


def _meta(settings: Settings) -> Collection[dict[str, Any]]:
    client = _client(settings)
    with store_errors_as_runtime():
        return client[settings.mongodb_db][_META_COLLECTION]


def _no_collection(settings: Settings) -> FileNotFoundError:
    """The error for a query against a collection that was never ingested into."""
    return FileNotFoundError(
        f"No index in MongoDB Atlas at {settings.mongodb_db}."
        f"{settings.collection_name} -- nothing was ever ingested under that "
        "name. Run `rag ingest` first, and check MONGODB_DB and COLLECTION_NAME "
        "match the ones used to ingest."
    )


def open_store(
    settings: Settings, embeddings: Embeddings | None = None
) -> MongoDBAtlasVectorSearch:
    """Open the Atlas Vector Search store this pipeline indexes into and searches.

    The store's identity -- (``MONGODB_URI``, database, collection, vector
    index, embedding function) -- must match between indexing and querying, so
    both stages open it through this one factory. ``embeddings`` is injectable
    for tests; production leaves it None and builds the embedding model.

    Construction is inert: with ``auto_create_index=False`` langchain-mongodb
    creates no index and makes no call, so the query path never builds
    anything. ``ingest`` owns the index (``_ensure_vector_index``).
    """
    return MongoDBAtlasVectorSearch(
        collection=_collection(settings),
        embedding=embeddings or build_embeddings(settings),
        index_name=settings.vector_index_name,
        text_key=_TEXT_KEY,
        embedding_key=_EMBEDDING_KEY,
        relevance_score_fn="cosine",
        auto_create_index=False,
    )


@contextmanager
def store_errors_as_runtime() -> Iterator[None]:
    """Translate MongoDB failures into the RuntimeError the frontends catch.

    Wraps every store op on both sides: connecting, ingest's reads, deletes,
    adds and index management, and the search at query. pymongo's and bson's
    exception types sit outside the ``FileNotFoundError | RuntimeError |
    ValueError`` union both frontends handle. Everything becomes a
    RuntimeError, never a ValueError -- a store failure while the app loads its
    pipeline must land in the branch ``streamlit_app.py`` catches *below* its
    sidebar, keeping the uploader reachable, rather than the ``ValueError``
    branch that stops the script above it. A malformed ``MONGODB_URI`` is
    pymongo's ``ConfigurationError``, so it lands there too.

    ``bson.errors.BSONError`` is not a ``PyMongoError``, so it has its own arm.
    Model failures need no arm: the adapters in ``mlx_models`` already raise
    inside the union.
    """
    try:
        yield
    except (pymongo.errors.PyMongoError, bson.errors.BSONError) as exc:
        # Keyed off the message, not the type: a vector of the wrong width is an
        # ordinary OperationFailure, raised at search time.
        hint = (
            " The index was built for a different EMBEDDING_MODEL/"
            "EMBEDDING_DIMENSIONS; set a new COLLECTION_NAME (or drop that "
            "collection's vector index) and run `rag ingest`."
            if "dimension" in str(exc).lower()
            else ""
        )
        raise RuntimeError(f"Vector store request failed: {exc}.{hint}") from exc


def reset_store_cache() -> None:
    """Close and drop every client this process opened.

    For the tests, which call it at every boundary so each starts as a fresh
    process would. Production never needs it: a client always reads the
    server's current state, and closing one would break every pipeline still
    holding it -- an outgoing one answering in another Streamlit session.
    """
    with _clients_lock:
        clients = list(_clients.values())
        _clients.clear()
    for client in clients:
        client.close()


def _version_id(settings: Settings) -> str:
    return f"version:{settings.collection_name}"


def index_version(settings: Settings) -> str:
    """A value that changes whenever the indexed corpus changes.

    The Streamlit app keys its pipeline cache on this so a `rag ingest` is
    picked up automatically. Reads the digest ingest() stamps over the corpus
    fingerprints (see _write_index_version) -- stable across an unchanged
    re-ingest, so it does not needlessly bust the cache.

    ``""`` means nothing has been ingested yet. A read, so it creates nothing,
    which matters because the app calls it on every rerun. Can raise
    RuntimeError -- an unreachable cluster, a missing ``MONGODB_URI`` -- which
    the app reads inside the guard that already catches it.
    """
    with store_errors_as_runtime():
        stamp = _meta(settings).find_one({"_id": _version_id(settings)})
    # `.get` plus a type check: a stamp edited by hand reads as "no version"
    # rather than as a KeyError escaping the caught union into a crash page.
    version = (stamp or {}).get("digest", "")
    return version if isinstance(version, str) else ""


def require_index(settings: Settings) -> None:
    """Fail with the fix when there is nothing to query, before any model loads.

    The query path's guards, here rather than in the pipeline so they can use
    the raw collection, which needs no embedding model: a fresh setup should be
    told to run ``rag ingest`` without first loading ~22 GB of local models to
    find that out. Three cases, each a ``FileNotFoundError`` naming the fix, and
    none of them creates anything:

    - no collection of that name -- a ``COLLECTION_NAME`` that differs from the
      one ingested into is a *different* collection, and one that silently
      searched empty would answer every question "I don't know";
    - a collection holding none of this pipeline's chunks (emptied, or only
      foreign records) -- scoped like every other read, so unrelated data does
      not pass for an index;
    - no vector index of that name, so no search could find anything.

    An unreachable cluster is the RuntimeError ``_client`` raises.
    """
    collection = _collection(settings)
    with store_errors_as_runtime():
        if settings.collection_name not in collection.database.list_collection_names(
            filter={"name": settings.collection_name}
        ):
            raise _no_collection(settings)
        if collection.find_one(OWN_CHUNKS, {"_id": 1}) is None:
            raise FileNotFoundError(
                f"The index at {settings.mongodb_db}.{settings.collection_name} "
                "is empty. Run `rag ingest` first, and check COLLECTION_NAME "
                "matches the one used to ingest."
            )
        if not list(collection.list_search_indexes(settings.vector_index_name)):
            raise FileNotFoundError(
                f"{settings.mongodb_db}.{settings.collection_name} has no vector "
                f"index named '{settings.vector_index_name}'. Run `rag ingest` to "
                "build it, and check VECTOR_INDEX_NAME matches the one used to "
                "ingest."
            )


def indexed_sources(settings: Settings) -> set[str]:
    """The sources this pipeline's chunks in the index come from.

    Lets a frontend report what ``ingest`` actually indexed rather than what it
    was handed: ``load_documents`` skips a file it cannot read, and one with no
    text -- a scanned PDF has none -- without failing the run, so a file saved
    into ``data_dir`` is not thereby answerable. Read-only like the other read
    paths: needs no model, creates nothing (``distinct`` over a collection that
    does not exist is simply empty).
    """
    with store_errors_as_runtime():
        sources = _collection(settings).distinct("source", OWN_CHUNKS)
    return {str(source) for source in sources}


def _read_pdf(path: Path) -> str:
    """Extract text from every page of a PDF and join it.

    Read and parsed as two steps, because they fail differently. Reading the
    file is the filesystem's part, and its failure stays the ``OSError`` it is.
    Parsing is pypdf reading untrusted bytes, and a malformed file makes it
    raise whatever its parser trips over -- its own ``PdfReadError``, but as
    readily a builtins ``TypeError`` (for a ``/Font`` resource that is a number,
    say). So everything it raises is translated here, where it runs, into the
    ``ValueError`` that ``load_documents`` skips a file for, rather than that
    loop catching every exception there is, a bug of its own included.
    """
    from pypdf import PdfReader

    data = path.read_bytes()  # what pypdf does itself when handed a path
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise ValueError(f"{type(exc).__name__}: {exc}") from exc


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

    The name returned is the one the directory holds, which is not always the
    one uploaded: macOS's default volume matches names regardless of case (and
    of Unicode normalization), so ``Notes.md`` uploaded beside ``notes.md``
    replaces that file and keeps *its* spelling. That spelling is the ``source``
    the loader reports, and the app decides whether an upload was indexed by
    finding its name among the sources -- given the upload's own, it reported a
    file it had just indexed as one with no text.

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
    target = data_dir / name
    target.write_bytes(data)
    entries = {entry.name for entry in data_dir.iterdir()}
    if name in entries:
        return name
    # Found by what it is rather than by folding case, since which names the
    # volume treats as one is its own rule; lstat, so a symlink to the file is
    # not mistaken for it. A hard link *is* the file, under another name, so the
    # names that fold to the upload's are tried first.
    written = target.lstat()
    folded = _fold(name)
    return next(
        (
            entry
            for entry in sorted(entries, key=lambda e: (_fold(e) != folded, e))
            if os.path.samestat((data_dir / entry).lstat(), written)
        ),
        name,
    )


def _fold(name: str) -> str:
    """A name as a case- and normalization-insensitive volume would compare it."""
    return unicodedata.normalize("NFD", name).casefold()


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
        except (OSError, ValueError) as exc:
            # One unreadable file must not abort the whole ingest -- skip it
            # with a warning. OSError is the file itself (permissions, removed
            # since the walk); ValueError is its contents: a bad encoding
            # (UnicodeDecodeError), or a PDF pypdf could not parse.
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
    because a truncated vector is a different vector, and a vector index serves
    one width only, so a different-width run must reach the width checks in
    ``ingest`` rather than be skipped as current. Content alone
    would let a re-ingest keep vectors the current settings would never have
    produced -- and a changed model is the dangerous half: the chunks still
    *look* current, so the skip is silent and every later query compares
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


def _write_index_version(settings: Settings, fresh: dict[str, str]) -> None:
    """Stamp a digest of the corpus fingerprints, for index_version to read.

    Digesting the fingerprints (not a counter or timestamp) keeps it stable
    across an unchanged re-ingest, so the Streamlit cache is busted on exactly
    the events an edit, add, or removal changes -- and no others.

    Called on every run and compared before it writes, rather than called only
    when something changed. A run that died after its deletes and adds but
    before this stamp leaves every source looking current to the next one, so a
    stamp gated on "something changed" would never be repaired -- and the app,
    keyed on it, would keep serving its stale pipeline until restarted. An
    unchanged corpus still writes nothing.
    """
    digest = hashlib.sha256(
        "".join(f"{source}:{h};" for source, h in sorted(fresh.items())).encode("utf-8")
    ).hexdigest()
    meta = _meta(settings)
    stamp = meta.find_one({"_id": _version_id(settings)})
    if (stamp or {}).get("digest") == digest:
        return
    meta.replace_one(
        {"_id": _version_id(settings)},
        {"_id": _version_id(settings), "digest": digest},
        upsert=True,
    )


def _index_definition(settings: Settings) -> dict[str, Any]:
    """The vector index: the embeddings, plus the two fields searches filter on.

    ``ingested_by`` scopes every search to this pipeline's chunks (OWN_CHUNKS);
    ``source`` lets ``_await_searchable`` probe one source's chunks. $vectorSearch
    refuses a pre-filter on a field the index does not declare.
    """
    return {
        "fields": [
            {
                "type": "vector",
                "path": _EMBEDDING_KEY,
                "numDimensions": settings.embedding_dimensions,
                "similarity": "cosine",
            },
            {"type": "filter", "path": "ingested_by"},
            {"type": "filter", "path": "source"},
        ]
    }


def _check_vector_index(
    collection: Collection[dict[str, Any]], settings: Settings
) -> None:
    """Refuse, before any write, an existing index this pipeline cannot use.

    An index serves vectors of one width, and a search pre-filtered on a field
    the index does not declare fails. Neither is fixed by re-embedding into the
    same index, so both are a ValueError naming the way out -- a new
    COLLECTION_NAME, or that index dropped -- raised while nothing has been
    deleted yet. Not repaired in place: the index may be another tool's.
    """
    found = list(collection.list_search_indexes(settings.vector_index_name))
    if not found:
        return
    definition = found[0].get("latestDefinition") or {}
    fields = definition.get("fields", [])
    widths = [f.get("numDimensions") for f in fields if f.get("type") == "vector"]
    filters = {f.get("path") for f in fields if f.get("type") == "filter"}
    where = (
        f"Vector index '{settings.vector_index_name}' on "
        f"{settings.mongodb_db}.{settings.collection_name}"
    )
    fix = (
        "Set a new COLLECTION_NAME (or drop that index in Atlas) and run `rag ingest`."
    )
    if widths != [settings.embedding_dimensions]:
        raise ValueError(
            f"{where} indexes {widths or 'no'}-wide vectors, but "
            f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions}. {fix}"
        )
    if not {"ingested_by", "source"} <= filters:
        raise ValueError(
            f"{where} does not declare `ingested_by` and `source` as filter "
            f"fields, which every search filters on. {fix}"
        )


def _ensure_vector_index(
    collection: Collection[dict[str, Any]], settings: Settings
) -> None:
    """Create the collection and its vector index if absent, then wait for it.

    Atlas refuses to create a search index on a collection that does not exist
    yet, so the collection is created first. Programmatic creation works on the
    free tier, so ``rag ingest`` stays the whole setup. The build is
    asynchronous, and a search against an index that is not yet queryable
    returns no results and no error -- which is why this waits.
    """
    database = collection.database
    if collection.name not in database.list_collection_names(
        filter={"name": collection.name}
    ):
        # CollectionInvalid: created by another writer since the check.
        with suppress(pymongo.errors.CollectionInvalid):
            database.create_collection(collection.name)
    if not list(collection.list_search_indexes(settings.vector_index_name)):
        collection.create_search_index(
            model=SearchIndexModel(
                definition=_index_definition(settings),
                name=settings.vector_index_name,
                type="vectorSearch",
            )
        )
    deadline = time.monotonic() + _INDEX_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        found = list(collection.list_search_indexes(settings.vector_index_name))
        if found and found[0].get("queryable"):
            return
        time.sleep(_INDEX_POLL_INTERVAL_S)
    raise RuntimeError(
        f"Vector index '{settings.vector_index_name}' did not become queryable "
        f"within {_INDEX_POLL_TIMEOUT_S:.0f}s."
    )


def _await_searchable(
    collection: Collection[dict[str, Any]], settings: Settings, chunk_id: str
) -> None:
    """Wait until a chunk just written is returned by ``$vectorSearch``.

    Atlas indexes a write asynchronously, so a chunk is in the collection (a
    find sees it) before a search can find it. A caller that ingests and then
    asks in the same process -- the app answering about a file just uploaded --
    depends on this, so the wait lives here rather than on the query path.

    Probes with the chunk's own vector, exactly (no approximation), within its
    own source and this pipeline's chunks: it is its own nearest neighbour, and
    asking for every chunk of that source keeps a chunk with an identical
    vector (repeated text) from crowding it out of the results.
    """
    probe = collection.find_one({"_id": chunk_id}, {_EMBEDDING_KEY: 1, "source": 1})
    if not probe or _EMBEDDING_KEY not in probe:
        return
    in_source = collection.count_documents({**OWN_CHUNKS, "source": probe["source"]})
    stage = {
        "index": settings.vector_index_name,
        "path": _EMBEDDING_KEY,
        "queryVector": probe[_EMBEDDING_KEY],
        "exact": True,
        "limit": max(1, min(in_source, 10000)),
        "filter": {"$and": [OWN_CHUNKS, {"source": {"$eq": probe["source"]}}]},
    }
    deadline = time.monotonic() + _INDEX_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        hits = collection.aggregate(
            [{"$vectorSearch": stage}, {"$project": {"_id": 1}}]
        )
        if any(hit["_id"] == chunk_id for hit in hits):
            return
        time.sleep(_INDEX_POLL_INTERVAL_S)
    raise RuntimeError(
        f"Newly ingested chunks did not become searchable within "
        f"{_INDEX_POLL_TIMEOUT_S:.0f}s."
    )


class _WriterLock:
    """A lease on one collection's ingest, held in Atlas, timed by its clock.

    Two writers on one collection at once -- a terminal `rag ingest`
    overlapping an upload in the app, or two machines -- would each apply its
    own reading of the index: one deletes what the other just added, and the
    stamp names a corpus neither wrote. A file lock covered one machine; the
    store is shared, so the lock is too.

    A lease rather than a flag, so a writer that dies -- killed, or its machine
    gone -- frees the lock when the lease runs out instead of holding it
    forever. Its expiry is computed by the server (``$$NOW``), so machines with
    different clocks agree on it. The holder renews it after every slice of
    adds, and a renewal that finds the lock taken over -- the lease ran out
    under a stalled writer -- stops that writer before it writes again.
    """

    def __init__(self, settings: Settings) -> None:
        self._meta = _meta(settings)
        self._id = f"ingest-lock:{settings.collection_name}"
        self._owner = uuid.uuid4().hex
        self._where = f"{settings.mongodb_db}.{settings.collection_name}"
        expiry = {"$add": ["$$NOW", _LOCK_LEASE_S * 1000]}
        # Free: no lease yet (the upsert just created the document), or one that
        # ran out. Evaluated against the document as it was before this update,
        # since every expression in one $set stage reads the input document.
        free = {"$lt": [{"$ifNull": ["$expires_at", None]}, "$$NOW"]}
        self._take = [
            {
                "$set": {
                    "owner": {"$cond": [free, self._owner, "$owner"]},
                    "expires_at": {"$cond": [free, expiry, "$expires_at"]},
                }
            }
        ]
        self._extend = [{"$set": {"expires_at": expiry}}]

    def try_acquire(self) -> bool:
        """Take the lock if it is free, in one atomic write; report who holds it.

        The write matches the lock by `_id` alone and decides inside the update
        whether to take it -- MongoDB refuses `$expr` in an upsert's query, so
        the "is it free" test cannot go in the filter. A single-document update
        is atomic, so of two writers racing for a free lock exactly one finds
        itself the owner afterwards.
        """
        held = self._meta.find_one_and_update(
            {"_id": self._id},
            self._take,
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        return bool(held) and held.get("owner") == self._owner

    def acquire(self) -> None:
        announced = False
        while not self.try_acquire():
            if not announced:
                print(
                    f"Waiting for another ingest into {self._where} to finish...",
                    file=sys.stderr,
                )
                announced = True
            time.sleep(_LOCK_POLL_S)

    def renew(self) -> None:
        renewed = self._meta.update_one(
            {"_id": self._id, "owner": self._owner}, self._extend
        )
        if renewed.matched_count != 1:
            raise RuntimeError(
                f"Lost the ingest lock on {self._where}: this ingest stalled for "
                f"over {_LOCK_LEASE_S}s and another took over. Run `rag ingest` "
                "again."
            )

    def release(self) -> None:
        self._meta.delete_one({"_id": self._id, "owner": self._owner})


@contextmanager
def _writer_lock(settings: Settings) -> Iterator[_WriterLock]:
    """Hold the collection's ingest lock for the body, failing inside the union."""
    lock = _WriterLock(settings)
    with store_errors_as_runtime():
        lock.acquire()
    try:
        yield lock
    finally:
        with store_errors_as_runtime():
            lock.release()


def _stored_width(collection: Collection[dict[str, Any]]) -> int | None:
    """The width of the vectors this pipeline's chunks already hold, if any."""
    rows = collection.aggregate(
        [
            {"$match": OWN_CHUNKS},
            {"$limit": 1},
            {
                "$project": {
                    "_id": 0,
                    "width": {
                        "$cond": [
                            {"$isArray": f"${_EMBEDDING_KEY}"},
                            {"$size": f"${_EMBEDDING_KEY}"},
                            None,
                        ]
                    },
                }
            },
        ]
    )
    return next((row.get("width") for row in rows), None)


def ingest(settings: Settings, embeddings: Embeddings | None = None) -> int:
    """Bring the index in line with ``data_dir``, embedding only what changed.

    Afterwards the collection holds exactly the chunks for the documents
    currently in ``data_dir`` -- nothing stale, nothing missing -- which is the
    property every caller depends on: the app rebuilds after an upload and
    expects the sample corpus to still be answerable, and a file edited by hand
    between runs must be picked up without being announced.

    Reaching that state costs one embedding pass per *changed* document rather
    than per document. Embedding runs locally at roughly nine chunks a second,
    and the app re-ingests on every upload, so embedding the whole corpus each
    time would make adding one file cost as much as indexing all of them.
    Unchanged documents keep the vectors they already have; changed and removed
    ones have their chunks dropped first, so a file's old text can never outlive
    it in the index.

    Scoped throughout to the chunks this pipeline wrote -- every read and delete
    is filtered to ``OWN_CHUNKS`` -- so a collection shared with unrelated data
    is never read, counted, or deleted from (``ingest`` never wipes anything
    wholesale). Chunks are keyed by a deterministic id
    (``source:index:content_hash``) and written by langchain-mongodb's
    upsert-replace, so re-adding is idempotent rather than a duplicating append.
    Returns the number of chunks the index now holds, not the number
    re-embedded: it describes the index, which is what makes re-ingesting the
    same corpus report the same number. ``embeddings`` is injectable so tests can
    substitute a lightweight fake; production callers leave it as None.
    """
    # Built before the lock: loading the model takes seconds, which no other
    # writer should wait on, and a model that is missing or fails to load then
    # fails before anything is written. Cheap when this process already holds
    # it (the app, after its first load).
    embedder = embeddings or build_embeddings(settings)

    # One writer at a time per collection (see _WriterLock), held across the
    # whole read -> delete -> add -> stamp sequence *and* the read of data_dir
    # that decides it: a run that read data_dir and then waited here would
    # otherwise apply that older snapshot after the writer it waited on --
    # deleting what that writer had just indexed. Readers need no lock.
    with _writer_lock(settings) as lock:
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
            document.metadata["ingested_by"] = _OWNER
        # Chunks inherit their parent's metadata, so each carries the source's
        # fingerprint and the scope marker, and the comparison below needs no
        # second pass over the files.
        chunks = split_documents(documents, settings)
        fresh = {
            doc.metadata["source"]: doc.metadata["content_hash"] for doc in documents
        }
        chunks_by_source: dict[str, list[Document]] = defaultdict(list)
        for chunk in chunks:
            chunks_by_source[chunk.metadata["source"]].append(chunk)

        store = open_store(settings, embedder)
        collection = _collection(settings)
        with store_errors_as_runtime():
            # The fingerprints are what decide the work, so they are what is read.
            indexed: dict[str, set[str]] = defaultdict(set)
            chunk_counts: Counter[str] = Counter()
            for row in collection.find(
                OWN_CHUNKS, {"_id": 0, "source": 1, "content_hash": 1}
            ):
                source = str(row.get("source"))
                indexed[source].add(str(row.get("content_hash")))
                chunk_counts[source] += 1

            # A source is current only if every chunk it should have is there
            # under its present fingerprint. The fingerprint alone would vouch
            # for a source whose add died part-way (a killed process, a failed
            # slice): its surviving chunks carry the right hash, so it would be
            # skipped as current -- missing chunks -- on every run after.
            current = {
                source
                for source, fingerprint in fresh.items()
                if indexed.get(source) == {fingerprint}
                and chunk_counts[source] == len(chunks_by_source[source])
            }
            # Split the rest into what to drop and what to (re-)embed. Deletion
            # is computed over the *indexed* sources, so a source gone from
            # data_dir is dropped rather than merely not added. Sorted, so the
            # same corpus is embedded in the same order -- and so in the same
            # padded batches -- on every run: in a set's order it varies with
            # the process's hash seed, and so, in the last bits, do the vectors.
            changed = sorted(set(fresh) - current)
            superseded = [source for source in indexed if source not in current]
            new_chunks: list[Document] = []
            ids: list[str] = []
            for source in changed:
                for i, chunk in enumerate(chunks_by_source[source]):
                    new_chunks.append(chunk)
                    ids.append(f"{source}:{i}:{chunk.metadata['content_hash']}")

            # Every width check before any write, so a refused run has deleted
            # nothing. Only when there is something to embed, so an unchanged
            # re-ingest still makes no embedding call at all (this embed_query
            # is the sole exception, and it runs only on a run about to embed
            # documents anyway).
            if new_chunks:
                probe_dims = len(embedder.embed_query("dimension probe"))
                if probe_dims != settings.embedding_dimensions:
                    raise ValueError(
                        f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} but "
                        f"{settings.embedding_model} produced {probe_dims}-wide "
                        f"vectors. Set EMBEDDING_DIMENSIONS={probe_dims}."
                    )
                held = _stored_width(collection)
                if held is not None and held != probe_dims:
                    raise ValueError(
                        f"{settings.mongodb_db}.{settings.collection_name} holds "
                        f"{held}-wide vectors, but {settings.embedding_model} now "
                        f"produces {probe_dims}-wide ones, which its vector index "
                        "cannot serve. Set a new COLLECTION_NAME (or drop that "
                        "collection) and run `rag ingest`."
                    )
            _check_vector_index(collection, settings)

            if superseded:
                collection.delete_many(
                    {"$and": [OWN_CHUNKS, {"source": {"$in": superseded}}]}
                )
            # Before the adds, so the new chunks are indexed as they land, and
            # on every run, so an index dropped by hand is rebuilt.
            _ensure_vector_index(collection, settings)
            lock.renew()

            # In slices, each embedded and written before the next, renewing
            # the lock between them: an interrupted run keeps its progress (the
            # chunk-count check above re-embeds a source a failure cut in
            # half), and no slice outlasts the lease.
            for start in range(0, len(new_chunks), _ADD_SLICE):
                store.add_documents(
                    new_chunks[start : start + _ADD_SLICE],
                    ids=ids[start : start + _ADD_SLICE],
                    batch_size=_ADD_SLICE,
                )
                lock.renew()
            if ids:
                _await_searchable(collection, settings, ids[-1])

            # Last, and on every run; a no-op when the stored digest already
            # matches (see _write_index_version).
            _write_index_version(settings, fresh)
    return len(chunks)
