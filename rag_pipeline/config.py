"""Central configuration for the RAG pipeline.

Every tunable lives here and is sourced from environment variables (loaded from
a local ``.env`` if present). Both the CLI and the Streamlit app build their
``Settings`` from :meth:`Settings.from_env`, so they always agree on which Atlas
collection holds the index, which models to call, and how documents are
chunked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time, so `os.environ` is populated before any
# Settings are built. `override=False` means a real environment variable always
# wins over the .env file.
load_dotenv(override=False)

# Repository root = two levels up from this file (rag_pipeline/config.py).
_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    # pathlib signals an unusable path with RuntimeError: expanduser() for a
    # `~user` with no home directory, resolve() for a symlink loop (on 3.11 and
    # 3.12). That is a malformed setting, so it is raised as the ValueError
    # every other one is -- the type streamlit_app.py stops on above its sidebar with
    # "Fix it" -- rather than escaping that guard as an uncaught RuntimeError.
    try:
        return Path(value).expanduser().resolve()
    except RuntimeError as exc:
        raise ValueError(f"{name}={value!r} is not a usable path: {exc}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_str(name: str, default: str) -> str:
    # A set-but-empty var (e.g. `CHAT_MODEL=`) falls back to the default, so
    # string settings behave like the int/path helpers rather than passing ""
    # straight through to the model/store.
    value = os.getenv(name)
    return value if value else default


def _env_bool(name: str, default: bool) -> bool:
    # Only "true" or "false", in any case: a value this cannot read is a
    # malformed setting, not a quiet False. LangSmith's own reading of the same
    # variable accepts nothing but "true", so a looser one here -- "1", "yes"
    # -- would be a switch the two read differently.
    value = os.getenv(name)
    if not value:
        return default
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise ValueError(f"{name}={value!r} must be true or false")


def require_env_key(name: str, used_for: str) -> str:
    """Return the credential in ``name``, or raise ``RuntimeError`` naming it.

    Credentials are deliberately not ``Settings`` fields. A field needs a
    literal default that the README and ``.env.example`` can state, and a key
    has none; and a field is a value the frontends display and a traceback can
    print, which a key must never be. So each stage that needs a key reads it
    where it is used, through this one function, and they all agree on what a
    missing key is -- set-but-empty counts as unset, as it does for the
    ``_env_*`` helpers -- and on what the user is told.

    ``RuntimeError`` rather than the ``ValueError`` a malformed setting raises:
    a key is first needed while the pipeline loads, below the Streamlit
    sidebar, where only ``FileNotFoundError`` and ``RuntimeError`` are caught.
    The message is built here rather than by each caller, so the variable it
    names is always the one that was checked. ``used_for`` names the stage
    that needs the key ("Answers come from Claude") and is spliced in ahead of
    a semicolon, so it takes no trailing punctuation.
    """
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. {used_for}; set it in your environment or in "
            "a .env file (see .env.example)."
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of pipeline configuration."""

    # Where source documents (.md/.txt/.pdf) are read from during ingest.
    data_dir: Path = _ROOT / "data"

    # The MongoDB Atlas database and collection holding the chunks, and the
    # Atlas Vector Search index over them. With MONGODB_URI (a credential, so
    # not a setting: see config.require_env_key) and the embedding model these
    # are the store's identity: ingest and query must agree on all of them, or
    # a query reads a wrong or empty collection. `rag ingest` creates the
    # collection and the index; nothing else does.
    mongodb_db: str = "rag_db"
    collection_name: str = "rag_docs"
    vector_index_name: str = "vector_index"

    # How long the MongoDB client waits to reach the cluster before an
    # operation fails. Generous because a free cluster resumes slowly after
    # being paused for inactivity, and a short wait would report "unreachable"
    # for what another few seconds would have connected.
    mongodb_timeout_ms: int = 10000

    # Voyage AI embedding model, called over its API at both ingest and query --
    # the same model must embed documents and questions for their vectors to
    # compare. Any Voyage text embedding model that accepts an output dimension
    # works.
    embedding_model: str = "voyage-4-large"

    # The width of those vectors: 256, 512, 1024 or 2048, the widths Voyage
    # returns. Folded into the chunk fingerprint, and it sets the vector
    # index's numDimensions: an index serves vectors of one width only.
    embedding_dimensions: int = 1024

    # The Claude model that writes the answer, over the Anthropic API. The
    # request sets thinking to `between_tools`, which only Claude Sonnet 5.5
    # accepts (claude_model.py), so another model needs that changed too.
    chat_model: str = "claude-sonnet-5-5"
    max_tokens: int = 1024

    # Splitter: 1000-char chunks with 200-char (20%) overlap keeps enough
    # context per chunk while preserving continuity across chunk boundaries.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Number of chunks kept after reranking and stuffed into the prompt. Each
    # one is about 250 more input tokens on every question, paid for and read
    # by the model before it writes.
    retrieval_k: int = 4

    # Candidates pulled from vector search before the reranker narrows them to
    # retrieval_k. Wider than retrieval_k so the reranker has room to rescue a
    # relevant chunk the embedding search ranked just out of the top few.
    fetch_k: int = 20

    # Voyage AI reranker. It scores each (question, candidate) pair jointly,
    # which embedding similarity only approximates.
    rerank_model: str = "rerank-3"

    # Tracing to LangSmith, off unless set to true: on, each question is sent
    # to LangSmith's cloud as one trace -- the question, every passage
    # retrieved, the prompt and the answer -- filed under this project, with
    # LANGSMITH_API_KEY. The names are the ones LangSmith's own SDK reads, so
    # its docs on them apply; the pipeline passes both to it explicitly for
    # every question, so this switch, and not anything else in the
    # environment, decides whether a question is traced.
    langsmith_tracing: bool = False
    langsmith_project: str = "rag-pipeline"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings, letting environment variables override defaults."""
        return cls(
            data_dir=_env_path("DATA_DIR", cls.data_dir),
            mongodb_db=_env_str("MONGODB_DB", cls.mongodb_db),
            collection_name=_env_str("COLLECTION_NAME", cls.collection_name),
            vector_index_name=_env_str("VECTOR_INDEX_NAME", cls.vector_index_name),
            mongodb_timeout_ms=_env_int("MONGODB_TIMEOUT_MS", cls.mongodb_timeout_ms),
            embedding_model=_env_str("EMBEDDING_MODEL", cls.embedding_model),
            embedding_dimensions=_env_int(
                "EMBEDDING_DIMENSIONS", cls.embedding_dimensions
            ),
            chat_model=_env_str("CHAT_MODEL", cls.chat_model),
            max_tokens=_env_int("MAX_TOKENS", cls.max_tokens),
            chunk_size=_env_int("CHUNK_SIZE", cls.chunk_size),
            chunk_overlap=_env_int("CHUNK_OVERLAP", cls.chunk_overlap),
            retrieval_k=_env_int("RETRIEVAL_K", cls.retrieval_k),
            fetch_k=_env_int("FETCH_K", cls.fetch_k),
            rerank_model=_env_str("RERANK_MODEL", cls.rerank_model),
            langsmith_tracing=_env_bool("LANGSMITH_TRACING", cls.langsmith_tracing),
            langsmith_project=_env_str("LANGSMITH_PROJECT", cls.langsmith_project),
        )


# Every field's environment variable, derived rather than restated. Tests clear
# these before asserting on defaults, and a hand-kept list would drift silently:
# this module calls load_dotenv() at import time, so a name missing from that
# list is answered by the developer's own .env and its default stops being
# tested. Deriving costs one line and makes the drift inexpressible.
ENV_VARS = tuple(field.name.upper() for field in fields(Settings))
