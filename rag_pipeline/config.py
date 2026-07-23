"""Central configuration for the RAG pipeline.

Every tunable lives here and is sourced from environment variables (loaded from
a local ``.env`` if present). Both the CLI and the Streamlit app build their
``Settings`` from :meth:`Settings.from_env`, so they always agree on which Atlas
namespace holds the index, which models to use, and how documents are chunked.
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

# The similarity metrics Atlas Vector Search accepts in an index definition.
# Validated in from_env so a typo fails at construction — inside the ValueError
# both frontends catch — rather than as an opaque Atlas error at index-build time.
_VALID_SIMILARITIES = frozenset({"cosine", "euclidean", "dotProduct"})


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def _env_str(name: str, default: str) -> str:
    # A set-but-empty var (e.g. `CHAT_MODEL=`) falls back to the default, so
    # string settings behave like the int/path helpers rather than passing ""
    # straight through to the model/store.
    value = os.getenv(name)
    return value if value else default


def require_env_key(name: str, used_for: str) -> None:
    """Raise ``RuntimeError`` if an API-key variable is unset or set-but-empty.

    Shared by the provider-key guards — ``VOYAGE_API_KEY`` before embedding and
    reranking, ``ANTHROPIC_API_KEY`` before generation, ``MONGODB_URI`` before
    opening the store — so they agree on the set-but-empty treatment (matching
    the ``_env_*`` helpers) and all land in the exception union the frontends
    catch.

    The message is built here rather than passed in, so those guards also agree
    on what they *tell* the user. The variable named in the message is then the
    one that was actually checked — these guards are copied from each other, and
    a copy that updates the name but not the text sends a reader off to set the
    wrong variable — and the remediation has one spelling to keep current when
    ``.env.example`` or the setup instructions change. Where the key goes is a
    fact about this module, the only one that loads ``.env``, rather than about
    the stage that needs it. ``used_for`` is the only part that varies: a clause
    naming that stage ("Embedding uses Voyage AI"), spliced in ahead of a
    semicolon, so it takes no trailing punctuation.
    """
    if not os.getenv(name):
        raise RuntimeError(
            f"{name} is not set. {used_for}; set the key in your environment "
            "or a .env file (see .env.example)."
        )


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of pipeline configuration."""

    # Where source documents (.md/.txt/.pdf) are read from during ingest.
    data_dir: Path = _ROOT / "data"

    # The MongoDB Atlas namespace holding the vectors — database + collection —
    # and the name of the Atlas Vector Search index over them. Together with
    # MONGODB_URI and the embedding function these are the store's identity:
    # ingest and query must agree on all of them, or a query reads a wrong or
    # silently-empty index. MONGODB_URI carries credentials, so it is guarded
    # via require_env_key rather than living here — a Settings field must have a
    # documentable literal default, and a secret has none.
    mongodb_db: str = "rag_db"
    collection_name: str = "rag_docs"
    vector_index_name: str = "vector_index"

    # Voyage AI embedding model, called over the API at both ingest and query.
    # voyage-4-lite -> 1024-dim vectors; the cost/latency-optimized tier, and
    # the same model must embed documents and questions for vectors to compare.
    embedding_model: str = "voyage-4-lite"

    # The width of those vectors. Pins the Atlas index's numDimensions, and is
    # folded into the chunk fingerprint, because a model of a different width
    # produces vectors the existing index cannot serve. Validated against the
    # live model at ingest — Atlas (unlike Chroma) accepts a wrong-width vector
    # on insert and only fails at query time, so nothing else would catch it.
    embedding_dimensions: int = 1024

    # Generation model. Haiku 4.5 is a fast, low-cost default for demos;
    # override via CHAT_MODEL (e.g. claude-opus-4-8) for higher-quality answers.
    chat_model: str = "claude-haiku-4-5"
    max_tokens: int = 1024

    # Splitter: 1000-char chunks with 200-char (20%) overlap keeps enough
    # context per chunk while preserving continuity across chunk boundaries.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Number of chunks kept after reranking and stuffed into the prompt.
    retrieval_k: int = 4

    # Candidates pulled from vector search before the reranker narrows them to
    # retrieval_k. Wider than retrieval_k so the cross-encoder has room to rescue
    # a relevant chunk the embedding search ranked just out of the top few.
    fetch_k: int = 20

    # Voyage AI reranker (cross-encoder). Rescores the fetch_k candidates against
    # the question jointly, which embedding cosine similarity only approximates.
    # rerank-2.5-lite is the latency/cost-optimized tier (same 32K context and
    # instruction-following as rerank-2.5), matching the voyage-4-lite /
    # claude-haiku-4-5 defaults; override via RERANK_MODEL for higher-quality
    # ranking (e.g. rerank-2.5).
    rerank_model: str = "rerank-2.5-lite"

    # Atlas Vector Search similarity metric — feeds both the index definition's
    # `similarity` and langchain-mongodb's `relevance_score_fn`. Validated in
    # from_env against _VALID_SIMILARITIES.
    atlas_similarity: str = "cosine"

    # `serverSelectionTimeoutMS` for the Mongo client. Generous by default: a
    # free Atlas cluster auto-pauses after 30 idle days and a cold resume can be
    # slow, so a short timeout would misread "resuming" as "unreachable" and
    # surface a RuntimeError where a brief wait would have connected.
    mongodb_timeout_ms: int = 10000

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings, letting environment variables override defaults."""
        # Validated up front so an unusable metric raises the ValueError both
        # frontends catch, rather than reaching Atlas and failing at index build.
        similarity = _env_str("ATLAS_SIMILARITY", cls.atlas_similarity)
        if similarity not in _VALID_SIMILARITIES:
            raise ValueError(
                f"ATLAS_SIMILARITY={similarity!r} is not one of "
                f"{', '.join(sorted(_VALID_SIMILARITIES))}."
            )
        return cls(
            data_dir=_env_path("DATA_DIR", cls.data_dir),
            mongodb_db=_env_str("MONGODB_DB", cls.mongodb_db),
            collection_name=_env_str("COLLECTION_NAME", cls.collection_name),
            vector_index_name=_env_str("VECTOR_INDEX_NAME", cls.vector_index_name),
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
            atlas_similarity=similarity,
            mongodb_timeout_ms=_env_int("MONGODB_TIMEOUT_MS", cls.mongodb_timeout_ms),
        )


# Every field's environment variable, derived rather than restated. Tests clear
# these before asserting on defaults, and a hand-kept list would drift silently:
# this module calls load_dotenv() at import time, so a name missing from that
# list is answered by the developer's own .env and its default stops being
# tested. Deriving costs one line and makes the drift inexpressible.
ENV_VARS = tuple(field.name.upper() for field in fields(Settings))
