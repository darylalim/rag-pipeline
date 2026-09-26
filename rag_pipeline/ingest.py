"""Indexing phase: load -> split -> embed -> store.

Run (via ``rag ingest``) whenever the documents in ``data/`` change. The
expensive embedding step happens here; querying later just reopens the
persisted Chroma collection.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import chromadb
import chromadb.errors
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings as ChromaSettings
from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.config import Settings
from rag_pipeline.mlx_models import QwenVLEmbeddings

# File extensions we know how to read into text.
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}

# The marker every chunk this pipeline writes carries, and the where-filter for
# every read, delete and search it makes -- so a collection shared with
# unrelated records is never read, counted, deleted from or retrieved from. A
# dedicated key matched by equality because Chroma has no `$exists`, and the
# obvious stand-in, `{"content_hash": {"$ne": ""}}`, also matches records that
# lack the key: a scoped delete built on it removes other people's documents.
OWN_CHUNKS: chromadb.Where = {"ingested_by": "rag-pipeline"}

# The settings that choose which implementation chromadb builds for each part of
# a client, pinned to chromadb's own defaults: the in-process store on SQLite.
# Left to chromadb, each is read from the environment, where a CHROMA_API_IMPL
# left over from another project swaps the store for an HTTP client -- every
# ingest and question would then go to whatever server CHROMA_SERVER_HOST names
# -- and one naming nothing importable fails as a builtins ValueError or
# ImportError, outside every union the frontends catch. Read from chromadb's
# model rather than listed here, so a field an upgrade adds is pinned too.
_PINNED_IMPLS = {
    name: field.default
    for name, field in ChromaSettings.model_fields.items()
    if name.endswith("_impl")
}

# The collection-metadata key holding the corpus digest (see index_version).
# Metadata rather than a reserved record, because Chroma cannot store a record
# without an embedding -- and one with a made-up vector could be retrieved.
_VERSION_KEY = "rag_index_version"


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


# chromadb's per-directory System cache is a class-level dict with no lock of
# its own. Building a client inserts a System, starts it, then reads it back
# from that dict several times; a clear from another thread in between -- one
# Streamlit session's pipeline rebuild while another session opens a client --
# surfaces as a builtins KeyError, or an AttributeError off a System that thread
# never finished starting, both outside the caught union. Every construction
# goes through _client() (the store-factory rule) and every clear through
# reset_store_cache(), so holding this around both is complete.
_system_cache_lock = threading.Lock()


def _client(settings: Settings) -> chromadb.ClientAPI:
    """A Chroma client on ``persist_dir``; every one this package opens is built here.

    Cheap to call repeatedly: chromadb shares one underlying System per persist
    directory within a process, provided every client asks for it with equal
    settings (unequal ones are a builtins ValueError, outside every union we
    catch), which is what routing them all through here guarantees. The settings
    object is nonetheless new each call, because PersistentClient mutates the
    one it is handed. Telemetry is off so an in-process store has no reason to
    reach the network, and ``_PINNED_IMPLS`` keeps it in-process whatever the
    environment says.

    A failed open empties chromadb's cache before re-raising. chromadb caches a
    directory's System *before* starting it, so one whose start failed (a
    corrupt or unreadable ``chroma.sqlite3``) would stay cached half-built: the
    next client on that path would be handed it, and chromadb's own cleanup of
    it raises a builtins AttributeError -- in the app, a crash page on every
    other rerun where the store error belongs.

    Opening a client creates ``persist_dir``, and a database in it, if either is
    missing -- which is why every read path checks ``_has_store`` first.
    """
    with _system_cache_lock:
        try:
            return chromadb.PersistentClient(
                path=str(settings.persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, **_PINNED_IMPLS),
            )
        except Exception:
            SharedSystemClient.clear_system_cache()
            raise


def _has_store(settings: Settings) -> bool:
    """Whether ``persist_dir`` already holds a Chroma database.

    The read paths' guard, checked before any client opens. The directory alone
    is not enough: opening a client writes a fresh ``chroma.sqlite3`` into
    whatever directory it is given, so a read path that checked only for the
    directory would still create a store in one that exists for another reason
    -- pre-created, or a PERSIST_DIR pointed at an unrelated folder -- and then
    report it "empty" rather than missing. The file name is chromadb's own,
    fixed for every persistent client.
    """
    return (settings.persist_dir / "chroma.sqlite3").is_file()


def _collection(settings: Settings) -> chromadb.Collection:
    """The raw collection, for bookkeeping that needs no embedding model.

    ``get_collection`` never creates, so a read cannot conjure an empty
    collection into existence -- a missing one raises
    ``chromadb.errors.NotFoundError``. No embedding function: chromadb's default
    is an ONNX model it downloads on first use, and nothing done through this
    handle should ever embed.
    """
    return _client(settings).get_collection(
        settings.collection_name, embedding_function=None
    )


def _no_collection(settings: Settings) -> FileNotFoundError:
    """The error for a query against a collection that was never ingested into."""
    return FileNotFoundError(
        f"Index at {settings.persist_dir} (collection "
        f"'{settings.collection_name}') is empty -- nothing was ever ingested "
        "under that name. Run `rag ingest` first, and check COLLECTION_NAME "
        "matches the one used to ingest."
    )


def open_store(
    settings: Settings, embeddings: Embeddings | None = None, *, create: bool = True
) -> Chroma:
    """Open the Chroma collection this pipeline indexes into and searches.

    The store's identity -- (persist directory, collection name, embedding
    function) -- must match between indexing and querying, so both stages open
    it through this one factory. ``embeddings`` is injectable for tests;
    production leaves it None and builds the embedding model.

    ``create`` is for ingest. Construction is eager -- langchain-chroma gets or
    creates the collection on the spot -- so the query path passes False, and a
    ``COLLECTION_NAME`` that was never ingested into is a ``FileNotFoundError``
    naming the fix rather than a freshly created empty collection. Cosine is set
    explicitly because Chroma's default space is L2, and a collection keeps the
    space it was created with. The error translation is here too, rather than
    left to each caller, because a bad ``COLLECTION_NAME`` raises a
    ``ChromaError`` *at construction* -- above whatever block a caller wraps its
    own store ops in.
    """
    with store_errors_as_runtime():
        try:
            return Chroma(
                client=_client(settings),
                collection_name=settings.collection_name,
                embedding_function=embeddings or build_embeddings(settings),
                collection_configuration={"hnsw": {"space": "cosine"}},
                create_collection_if_not_exists=create,
            )
        except chromadb.errors.NotFoundError as exc:
            raise _no_collection(settings) from exc


@contextmanager
def store_errors_as_runtime() -> Iterator[None]:
    """Translate Chroma failures into the RuntimeError the frontends catch.

    Wraps every store op on both sides: opening the collection, the reads,
    deletes and adds at ingest, and the search at query. chromadb's exception
    types sit outside the ``FileNotFoundError | RuntimeError | ValueError`` union
    both frontends handle, and none belongs in a frontend. Everything becomes a
    RuntimeError, never a ValueError -- a store failure while the app loads its
    pipeline must land in the branch ``app.py`` catches *below* its sidebar,
    keeping the uploader reachable, rather than the ``ValueError`` branch that
    stops the script above it.

    ``ChromaError`` only, deliberately. chromadb also raises builtins
    ``ValueError``/``TypeError`` from its own argument checks (an empty ``$in``,
    a one-clause ``$and``, ``hnsw:space`` passed to ``modify``, a search for
    fewer than one result), but catching
    those here would also swallow the ``ValueError`` ingest raises on purpose
    inside this block -- so the calls below avoid them by construction instead.
    Model failures need no arm: the adapters in ``mlx_models`` already raise
    inside the union.
    """
    try:
        yield
    except chromadb.errors.ChromaError as exc:
        # The hint keys off the message, not the type: chromadb raises the same
        # InvalidArgumentError for unrelated validation (a bad COLLECTION_NAME)
        # that the hint would misdiagnose. The remedy is a new collection, never
        # a wiped persist directory: a collection keeps its width even after
        # every row is deleted, and the directory may hold other collections.
        hint = (
            " The collection was built with a different EMBEDDING_MODEL/"
            "EMBEDDING_DIMENSIONS; set a new COLLECTION_NAME (or delete that "
            "collection) and run `rag ingest`."
            if "dimension" in str(exc).lower()
            else ""
        )
        raise RuntimeError(f"Vector store request failed: {exc}.{hint}") from exc


def reset_store_cache() -> None:
    """Drop chromadb's per-process client cache.

    chromadb shares one System per persist directory within a process, and that
    System's vector search does not see writes another process made after it
    was opened: after a terminal ``rag ingest`` it keeps returning chunks that
    were deleted, or raises ``InternalError`` under a where-filter. Counts, gets
    and metadata *are* current, so ``index_version`` sees the new corpus while a
    search still serves the old one. A caller that reopens the store after an
    out-of-process rebuild -- the app, before building a new pipeline -- must
    clear this first.

    Dropped, not closed: closing stops the System under every client still
    holding it -- an outgoing pipeline answering in another session -- whose
    next call then fails with an ``AttributeError`` outside the caught union. A
    dropped System merely keeps its old view, and what that view raises is a
    ``ChromaError``, which ``store_errors_as_runtime`` already turns into a
    RuntimeError. Tests call this at every boundary to emulate a fresh process.

    A writer must start from a fresh System too, which is why ``ingest`` calls
    this under its lock. A stale System does not only read the old view, it
    persists it: its vector index, written back over the other process's, keeps
    the chunks that process deleted, which then take the places of live ones in
    every search -- and past a few thousand chunks make every filtered search
    fail -- for good, since later ingests inherit the damage.
    """
    with _system_cache_lock:
        SharedSystemClient.clear_system_cache()


def index_version(settings: Settings) -> str:
    """A value that changes whenever the indexed corpus changes.

    The Streamlit app keys its pipeline cache on this so a `rag ingest` is
    picked up automatically. Reads the digest ingest() stamps over the corpus
    fingerprints (see _write_index_version) -- stable across an unchanged
    re-ingest, so it does not needlessly bust the cache. Collection metadata is
    read fresh even through this process's cached client (it is not subject to
    the stale vector view ``reset_store_cache`` exists for), which is what lets
    this notice an ingest run from another process at all.

    ``""`` means nothing has been ingested yet. Creates nothing -- the database
    is checked for before a client is opened (which would create one), since
    the app calls this on every rerun of a fresh checkout. Can raise
    RuntimeError; the app reads it inside the guard that already catches that.
    """
    if not _has_store(settings):
        return ""
    with store_errors_as_runtime():
        try:
            metadata = _collection(settings).metadata or {}
        except chromadb.errors.NotFoundError:
            return ""
    # `.get` plus a type check: metadata edited by hand, or by another tool
    # sharing the collection, reads as "no version" rather than as a KeyError
    # that would escape the caught union into a crash page.
    version = metadata.get(_VERSION_KEY, "")
    return version if isinstance(version, str) else ""


def require_index(settings: Settings) -> None:
    """Fail with the fix when there is nothing to query, before any model loads.

    The query path's guards, here rather than in the pipeline so they can use
    the raw collection, which needs no embedding model: a fresh checkout should
    be told to run ``rag ingest`` without first loading ~22 GB of local models
    to find that out. Three cases, each a ``FileNotFoundError`` naming the fix:

    - no database in the persist directory -- checked first, because opening a
      client would create one (see ``_has_store``);
    - no collection of that name -- a ``COLLECTION_NAME`` that differs from the
      one ingested into is a *different* collection, and one that silently
      searched empty would answer every question "I don't know";
    - a collection holding none of this pipeline's chunks (emptied, or only
      foreign records) -- scoped like every other read, so unrelated data does
      not pass for an index.
    """
    if not _has_store(settings):
        raise FileNotFoundError(
            f"No index found at {settings.persist_dir}. Run `rag ingest` first."
        )
    with store_errors_as_runtime():
        try:
            collection = _collection(settings)
        except chromadb.errors.NotFoundError as exc:
            raise _no_collection(settings) from exc
        ids = collection.get(where=OWN_CHUNKS, limit=1, include=[])["ids"]
    if not ids:
        raise FileNotFoundError(
            f"Index at {settings.persist_dir} (collection "
            f"'{settings.collection_name}') is empty. Run `rag ingest` first, "
            "and check COLLECTION_NAME matches the one used to ingest."
        )


def indexed_sources(settings: Settings) -> set[str]:
    """The sources this pipeline's chunks in the index come from.

    Lets a frontend report what ``ingest`` actually indexed rather than what it
    was handed: ``load_documents`` skips a file it cannot read, and one with no
    text -- a scanned PDF has none -- without failing the run, so a file saved
    into ``data_dir`` is not thereby answerable. Read-only like the other read
    paths: needs no model, creates nothing.
    """
    if not _has_store(settings):
        return set()
    with store_errors_as_runtime():
        try:
            got = _collection(settings).get(where=OWN_CHUNKS, include=["metadatas"])
        except chromadb.errors.NotFoundError:
            return set()
    return {
        str(metadata["source"])
        for metadata in got["metadatas"] or []
        if metadata and "source" in metadata
    }


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
    because a truncated vector is a different vector, and a Chroma collection
    fixes its width at the first insert, so a different-width run must reach the
    width check in ``ingest`` rather than be skipped as current. Content alone
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
    before this stamp (killed, or this very ``modify`` failing) leaves every
    source looking current to the next one, so a stamp gated on "something
    changed" would never be repaired -- and the app, keyed on it, would keep
    serving its stale view until restarted. An unchanged corpus still writes
    nothing.

    ``modify`` replaces the collection's whole metadata dict, so the existing
    keys are carried over (another tool sharing the collection may keep its own
    there) -- minus any ``hnsw:*`` key. Those are index settings, which a
    collection takes from its configuration at creation, and ``modify`` refuses
    ``hnsw:space`` outright, with a builtins ValueError, even unchanged: a
    collection created the common LangChain way carries it.
    """
    digest = hashlib.sha256(
        "".join(f"{source}:{h};" for source, h in sorted(fresh.items())).encode("utf-8")
    ).hexdigest()
    collection = _collection(settings)
    metadata = collection.metadata or {}
    if metadata.get(_VERSION_KEY) == digest:
        return
    kept = {
        key: value for key, value in metadata.items() if not key.startswith("hnsw:")
    }
    collection.modify(metadata={**kept, _VERSION_KEY: digest})


@contextmanager
def _writer_lock(settings: Settings) -> Iterator[None]:
    """Hold the cross-process ingest lock, failing inside the caught union.

    The directory is created first because the lock file lives in it. A
    PERSIST_DIR that cannot be created or written -- an existing file, a
    read-only parent or directory -- raises FileExistsError, NotADirectoryError
    or PermissionError: builtins OSErrors outside the union ``rag ingest``
    reports, so they would surface as a traceback. Translated here, around only
    these two steps, so an OSError from inside the critical section keeps its
    own type (the FileNotFoundError for a missing data_dir among them).
    RuntimeError rather than ValueError, like every store failure.
    """
    lock = FileLock(str(settings.persist_dir / ".ingest.lock"))
    try:
        settings.persist_dir.mkdir(parents=True, exist_ok=True)
        lock.acquire()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot write the index at {settings.persist_dir}: {exc}. Check that "
            "PERSIST_DIR names a directory this user can write."
        ) from exc
    try:
        yield
    finally:
        lock.release()


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
    (``source:index:content_hash``) and added through langchain-chroma's upsert,
    so re-adding is an idempotent replace rather than a duplicating append.
    Returns the number of chunks the index now holds, not the number
    re-embedded: it describes the index, which is what makes re-ingesting the
    same corpus report the same number. ``embeddings`` is injectable so tests can
    substitute a lightweight fake; production callers leave it as None.
    """
    # Built before the lock, and before the persist directory exists: loading
    # the model takes seconds, which no other writer should wait on, and a model
    # that is missing or fails to load then fails before anything is written.
    # Cheap when this process already holds it (the app, after its first load).
    embedder = embeddings or build_embeddings(settings)

    # One writer at a time, across processes: two writers on one persist
    # directory -- a terminal `rag ingest` overlapping an upload in the app --
    # corrupt it permanently, and neither sees an error; only every later query
    # does. Held across the whole read -> delete -> add -> stamp sequence *and*
    # the read of data_dir that decides it: a run that read data_dir and then
    # waited here would otherwise apply that older snapshot after the writer it
    # waited on -- deleting what that writer had just indexed (another session's
    # upload), and stamping a digest without it. Read under the lock, whichever
    # run goes last applies the newest data_dir. Readers need no lock.
    with _writer_lock(settings):
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
            document.metadata.update(OWN_CHUNKS)
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

        # A fresh System for the writer (see reset_store_cache): a cached one,
        # opened before another process's ingest, would write its stale view of
        # the index back over that ingest's. Here, under the lock, the view is
        # taken after any other writer has finished.
        reset_store_cache()
        store = open_store(settings, embedder)
        with store_errors_as_runtime():
            # Metadata, not ids alone: the fingerprints are what decide the work.
            stored = store.get(where=OWN_CHUNKS, include=["metadatas"])
            indexed: dict[str, set[str]] = defaultdict(set)
            chunk_counts: Counter[str] = Counter()
            for metadata in stored["metadatas"]:
                source = str(metadata["source"])
                indexed[source].add(str(metadata["content_hash"]))
                chunk_counts[source] += 1

            # A source is current only if every chunk it should have is there
            # under its present fingerprint. The fingerprint alone would vouch
            # for a source whose add died part-way (a killed process, a failed
            # batch): its surviving chunks carry the right hash, so it would be
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

            # Validate the width against the live model and the collection
            # *before* any write. Chroma fixes a collection's width at its first
            # insert -- and keeps it after every row is deleted -- so a
            # different-width model would otherwise fail at the add, after the
            # delete below had already removed the chunks it was replacing. Only
            # when there is something to embed, so an unchanged re-ingest still
            # makes no embedding call at all (this embed_query is the sole
            # exception, and it runs only on a run about to embed documents
            # anyway).
            if new_chunks:
                probe_dims = len(embedder.embed_query("dimension probe"))
                if probe_dims != settings.embedding_dimensions:
                    raise ValueError(
                        f"EMBEDDING_DIMENSIONS={settings.embedding_dimensions} but "
                        f"{settings.embedding_model} produced {probe_dims}-wide "
                        f"vectors. Set EMBEDDING_DIMENSIONS={probe_dims}."
                    )
                # len(), not truthiness: chromadb hands back a numpy array here,
                # and the truth value of one is itself a ValueError.
                held = store.get(where=OWN_CHUNKS, limit=1, include=["embeddings"])
                vectors = held["embeddings"]
                if (
                    vectors is not None
                    and len(vectors)
                    and len(vectors[0]) != probe_dims
                ):
                    raise ValueError(
                        f"Collection '{settings.collection_name}' holds "
                        f"{len(vectors[0])}-wide vectors, but "
                        f"{settings.embedding_model} now produces {probe_dims}-wide "
                        "ones, and a collection cannot change width. Set a new "
                        "COLLECTION_NAME (or delete that collection) and run "
                        "`rag ingest`."
                    )

            # Guarded, because chromadb rejects an empty `$in` outright rather
            # than matching nothing.
            if superseded:
                store.delete(
                    where={"$and": [OWN_CHUNKS, {"source": {"$in": superseded}}]}
                )
            # Added in slices no larger than the backend's own cap. langchain's
            # add embeds everything and then upserts it in one call, which
            # chromadb refuses above get_max_batch_size() -- 5461 records, a
            # few megabytes of text -- so unsliced, a first ingest of a larger
            # corpus could never succeed, and a re-chunk would fail *after* the
            # delete above, leaving the index empty. Each slice is written as
            # soon as it is embedded, so an interrupted run keeps its progress,
            # and the chunk-count check above re-embeds any source a failure cut
            # in half.
            if new_chunks:
                step = _client(settings).get_max_batch_size()
                for start in range(0, len(new_chunks), step):
                    store.add_documents(
                        new_chunks[start : start + step], ids=ids[start : start + step]
                    )

            # Last, and on every run; a no-op when the stored digest already
            # matches (see _write_index_version).
            _write_index_version(settings, fresh)
    return len(chunks)
