"""Central configuration for the RAG pipeline.

Every tunable lives here and is sourced from environment variables (loaded from
a local ``.env`` if present). Both the CLI and the Streamlit app build their
``Settings`` from :meth:`Settings.from_env`, so they always agree on which Chroma
collection holds the index, which local models to load, and how documents are
chunked.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

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
    # every other one is -- the type app.py stops on above its sidebar with
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


def _env_url(name: str, default: str) -> str:
    # Checked here because nothing downstream checks it: the span exporter
    # accepts any string, and one it cannot send to fails only when a trace is
    # sent -- a line in a log, every trace lost -- rather than as a setting the
    # app stops on and names.
    value = os.getenv(name)
    if not value:
        return default
    try:
        url = urlsplit(value)
        # `.port` raises ValueError for a port that is not a number in range.
        usable = (
            url.scheme in ("http", "https") and bool(url.hostname) and url.port != 0
        )
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not a usable URL: {exc}") from exc
    if not usable:
        raise ValueError(
            f"{name}={value!r} must be an http(s) URL, such as http://localhost:6006"
        )
    return value


@dataclass(frozen=True)
class Settings:
    """Immutable bundle of pipeline configuration."""

    # Where source documents (.md/.txt/.pdf) are read from during ingest.
    data_dir: Path = _ROOT / "data"

    # Where Chroma persists the index on disk, and the collection inside it.
    # Together with the embedding model these are the store's identity: ingest
    # and query must agree on all of them, or a query reads a wrong or empty
    # collection.
    persist_dir: Path = _ROOT / "chroma_db"
    collection_name: str = "rag_docs"

    # Local embedding model (a Hugging Face repo id resolved from the local
    # cache, or a path to a model directory), run with MLX at both ingest and
    # query -- the same model must embed documents and questions for their
    # vectors to compare. The adapter implements Qwen3-VL-Embedding's own
    # prompt format and pooling, so another checkpoint of that family (e.g. the
    # 8B) is a drop-in; an unrelated embedding model is not.
    embedding_model: str = "mlx-community/Qwen3-VL-Embedding-2B-bf16"

    # The width of those vectors. Qwen3-VL-Embedding is Matryoshka-trained, so
    # any width up to the model's native 2048 is a valid prefix of the full
    # vector (re-normalized). Folded into the chunk fingerprint, because a
    # Chroma collection fixes its width at the first insert and cannot serve
    # vectors of another.
    embedding_dimensions: int = 2048

    # Local generation model, run with mlx-lm (thinking disabled, greedy
    # decoding). Any mlx-lm chat checkpoint whose chat template accepts a system
    # turn works -- the grounding rules are sent as one, and a template that
    # rejects it (Gemma 2's) fails every question, not the load. The default is
    # a 27B 4-bit model that needs about 16 GB of unified memory on its own.
    chat_model: str = "mlx-community/Qwen3.8-27B-4bit"
    max_tokens: int = 1024

    # Splitter: 1000-char chunks with 200-char (20%) overlap keeps enough
    # context per chunk while preserving continuity across chunk boundaries.
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Number of chunks kept after reranking and stuffed into the prompt. It is
    # also what sets time-to-first-token: the local model reads the whole prompt
    # (at roughly 100-150 tokens/s) before it writes, so every extra chunk adds
    # a couple of seconds.
    retrieval_k: int = 4

    # Candidates pulled from vector search before the reranker narrows them to
    # retrieval_k. Wider than retrieval_k so the reranker has room to rescue a
    # relevant chunk the embedding search ranked just out of the top few.
    fetch_k: int = 20

    # Local reranker. Qwen3-VL-Reranker scores each (question, candidate) pair
    # jointly from its yes/no logits, which embedding similarity only
    # approximates. Like the embedder, the adapter is specific to this model
    # family -- and, reading those logits off the tied embedding matrix, to its
    # tied-embedding checkpoints: the 2B, in any quantization.
    rerank_model: str = "mlx-community/Qwen3-VL-Reranker-2B-bf16"

    # Tracing, off while the endpoint is empty. Set, it is the base URL of a
    # self-hosted Phoenix (http://localhost:6006 for `phoenix serve`), and each
    # question is sent there as one trace, over OTLP/HTTP to its /v1/traces,
    # filed under this project. The names are the ones Phoenix's own clients
    # read, so its docs on these two apply -- except that unset means off here,
    # where Phoenix's clients would assume localhost. Its other client
    # settings (an API key among them) are not read: no credentials are sent.
    phoenix_collector_endpoint: str = ""
    phoenix_project: str = "rag-pipeline"

    @classmethod
    def from_env(cls) -> Settings:
        """Build settings, letting environment variables override defaults."""
        return cls(
            data_dir=_env_path("DATA_DIR", cls.data_dir),
            persist_dir=_env_path("PERSIST_DIR", cls.persist_dir),
            collection_name=_env_str("COLLECTION_NAME", cls.collection_name),
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
            phoenix_collector_endpoint=_env_url(
                "PHOENIX_COLLECTOR_ENDPOINT", cls.phoenix_collector_endpoint
            ),
            phoenix_project=_env_str("PHOENIX_PROJECT", cls.phoenix_project),
        )


# Every field's environment variable, derived rather than restated. Tests clear
# these before asserting on defaults, and a hand-kept list would drift silently:
# this module calls load_dotenv() at import time, so a name missing from that
# list is answered by the developer's own .env and its default stops being
# tested. Deriving costs one line and makes the drift inexpressible.
ENV_VARS = tuple(field.name.upper() for field in fields(Settings))
