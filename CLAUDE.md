# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv)
uv run rag ingest                    # rebuild the Chroma index from data/
uv run rag query "your question"     # ask from the terminal
uv run streamlit run app.py          # chat UI over the same pipeline
uv run pytest                        # full suite (~0.5s, fully offline)
uv run pytest tests/test_config.py::test_defaults   # single test
uv run pytest -k idempotent -v                      # by keyword
```

No linter or formatter is configured; CI (`.github/workflows/ci.yml`) runs only
`uv run pytest`, on Python 3.11 and 3.13. Don't add lint commands to CI without
being asked.

`README.md` covers setup, configuration variables, and usage in detail — consult
it rather than duplicating that material here.

## Architecture

Two phases with a hard boundary between them, and one shared config object:

```
ingest (rag_pipeline/ingest.py)    load → split → embed → store (Chroma on disk)
query  (rag_pipeline/pipeline.py)  embed question → search → stuff prompt → Claude
```

`Settings` (`config.py`) is a frozen dataclass built via `Settings.from_env()`.
Both frontends — `rag_pipeline/cli.py` and `app.py` — construct it the same way,
which is what keeps them agreeing on index location, models, and chunking. All
tunables belong here, not scattered as literals.

### Why the store factories live in `ingest.py`

`build_embeddings()` and `open_store()` are defined in `ingest.py` and imported
*by* `pipeline.py`, not the reverse. This is deliberate: vectors from different
embedding models are not comparable, and a Chroma collection's identity is
(persist dir, collection name, embedding function). Indexing and querying must
therefore go through one factory each. **Never construct `Chroma(...)` or
`HuggingFaceEmbeddings(...)` inline** — route through these factories.

### chromadb's per-process client cache

chromadb caches one client per persist directory *within a process*. Any code
that re-opens a store after it was rebuilt on disk will otherwise reuse the stale
client and silently read the old index. `reset_store_cache()` wraps this
(`SharedSystemClient.clear_system_cache()`) so callers never touch chromadb
internals directly. Two places depend on it:

- `app.py` calls it before rebuilding the cached pipeline.
- `tests/conftest.py` clears it autouse at every test boundary, so an in-process
  re-ingest behaves like a fresh CLI run.

The Streamlit cache key is `index_version()` — the mtime of `chroma.sqlite3`,
which chromadb bumps on every write. That's how the running app picks up a
`rag ingest` without a restart.

### Ingest is a scoped rebuild, not a directory wipe

`ingest()` deletes only the current collection's ids, then adds fresh chunks.
It must never `rmtree` the persist directory — that dir may hold unrelated data
(`test_ingest_preserves_unrelated_files_in_persist_dir` guards this). Re-ingest
is idempotent: same input → same chunk count, no duplicate append.

### Error contract

`FileNotFoundError | RuntimeError | ValueError` is the union both frontends catch
and render as a friendly message (`cli.py`, `app.py`) — roughly: missing/empty
index, missing API key, malformed numeric env var. Keep new failure modes inside
this union rather than introducing a fourth type.

`pipeline.answer()` translates `anthropic.APIError` into `RuntimeError` on
purpose, so frontends handle failed generation without importing the Anthropic
SDK. Preserve that translation.

Setup guards fail loudly and early, in this order: persist dir exists → API key
present (only when building a real client) → collection non-empty. The empty-
collection check matters because Chroma's `get_or_create` silently yields an
empty collection on a `COLLECTION_NAME` mismatch, which would answer every
question with "I don't know."

### Dependency injection is the test seam

`ingest()`, `open_store()`, and `RAGPipeline.__init__` all accept optional
`embeddings` / `llm`. **Production always passes `None`**; the parameters exist
so tests can inject `DeterministicFakeEmbedding` and `FakeListChatModel`. This is
why the suite needs no torch, no model download, and no API key. Any new code
path touching an embedding model or the LLM should thread these through rather
than constructing them unconditionally.

## Conventions

- `build_chat_model()` sets no `temperature`/`top_p` — grounding comes from
  retrieved context, and some models (Opus 4.8) reject sampling params outright.
  Don't add them.
- Env-var helpers in `config.py` treat set-but-empty (`CHAT_MODEL=`) as unset and
  fall back to the default. Match that behavior for new settings.
- `load_documents()` skips unreadable and whitespace-only files with a warning
  instead of aborting the ingest. Preserve that resilience.
- Document `source` metadata (path relative to `data_dir`, POSIX-style) is what
  citations key off. Any new loader must set it.
- Module and function docstrings explain *why*, not what. Match that register.

## Gotchas

- `config.py` calls `load_dotenv(override=False)` at **import time**. A real
  environment variable wins over `.env`, but a developer's local `.env` will leak
  into test runs — config tests must `monkeypatch.setenv`/`delenv` explicitly.
- `cli.py` imports `ingest`/`pipeline` lazily inside the command functions. This
  is load-bearing: importing them pulls in sentence-transformers/torch, measured
  at ~4.3s versus ~0.08s for `rag --help` today. Keep those imports local.
- `chroma_db/` and `.env` are gitignored; the index is a build artifact,
  regenerated by `rag ingest`.
