# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv)
uv run rag ingest                    # rebuild the Chroma index from data/
uv run rag query "your question"     # ask from the terminal
uv run streamlit run app.py          # chat UI over the same pipeline
uv run pytest                        # full suite (~7s wall: ~4s torch import, ~0.5s tests; offline)
uv run pytest tests/test_config.py::test_defaults   # single test
uv run pytest -k idempotent -v                      # by keyword
uv run ruff check --fix .            # lint (fix before formatting — fixes reorder code)
uv run ruff format .                 # format
uv run ty check                      # type check
```

When working with Python, invoke the relevant `/astral:<skill>` — `/astral:uv`,
`/astral:ty`, `/astral:ruff` — to ensure best practices are followed.

CI (`.github/workflows/ci.yml`) has two jobs: `lint` runs
`ruff check` + `ruff format --check` + `ty check` once on 3.13, and `test` runs
pytest on 3.11 and 3.13. Both must stay green.

Ruff and ty are configured in `pyproject.toml` and pinned in the dev group, so
local runs match CI — prefer `uv run ruff`/`uv run ty` over `uvx`. The lint
select list is deliberately broad (`E,W,F,I,UP,B,SIM,C4,PT,RUF`) and the tree is
clean against it; `E501` is off because line length is `ruff format`'s job.
Fix findings rather than adding `# noqa` / `# ty: ignore`.

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
which is what keeps them agreeing on index location, models, and chunking.

All tunables live here — never inline a literal at a call site. Adding one is a
**three-file change**: the field plus its `_env_*` line in `config.py`, a
commented default in `.env.example`, and a row in the README config table.
Leaving either of the latter two stale is a bug.

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
internals directly. Callers (`grep reset_store_cache`):

- `app.py` calls it before rebuilding the cached pipeline.
- `tests/conftest.py` clears it autouse at every test boundary, so an in-process
  re-ingest behaves like a fresh CLI run.
- `tests/test_ingest.py::test_ingest_is_idempotent` calls it directly between
  ingests to emulate a fresh CLI process.

The Streamlit cache key is `index_version()` — the mtime of `chroma.sqlite3`,
which chromadb bumps on every write. That's how the running app picks up a
`rag ingest` without a restart.

### Ingest is a scoped rebuild, not a directory wipe

`ingest()` deletes only the current collection's ids, then adds fresh chunks.
It must never `rmtree` the persist directory — that dir may hold unrelated data
(`test_ingest_preserves_unrelated_files_in_persist_dir` guards this). Re-ingest
is idempotent: same input → same chunk count, no duplicate append.

### Dependency injection is the test seam

`ingest()`, `open_store()`, and `RAGPipeline.__init__` all accept optional
`embeddings` / `llm`. **Production always passes `None`**; the parameters exist
so tests can inject `DeterministicFakeEmbedding` and `FakeListChatModel`. This is
why the suite needs no model download, no network, and no API key. (torch is
still imported transitively via `langchain_huggingface`; it just never runs a
model — that import is the ~4s in `uv run pytest`.) Any new code path touching an
embedding model or the LLM should thread these through rather than constructing
them unconditionally.

## Gotchas

- `config.py` calls `load_dotenv(override=False)` at **import time**. A real
  environment variable wins over `.env`, but a developer's local `.env` will leak
  into test runs — config tests must `monkeypatch.setenv`/`delenv` explicitly.
- `cli.py` imports `ingest`/`pipeline` lazily inside the command functions. This
  is load-bearing: importing them pulls in sentence-transformers/torch, measured
  at ~4.3s versus ~0.08s for `rag --help` today. Keep those imports local.

## Conventions

- New failure modes must fit `FileNotFoundError | RuntimeError | ValueError` —
  the union both frontends catch (`cli.py:65`, `app.py:48`; see the comment at
  `app.py:49-50` for the precise per-type mapping). Don't add a fourth type.
- Preserve `answer()`'s `anthropic.APIError` → `RuntimeError` translation, so
  frontends never import the Anthropic SDK.
- Keep the empty-collection guard in `RAGPipeline.__init__`: Chroma's
  `get_or_create` silently returns an *empty* collection on a `COLLECTION_NAME`
  mismatch, so without it every question is answered "I don't know."
- `build_chat_model()` sets no `temperature`/`top_p` — grounding comes from
  retrieved context, and some models (Opus 4.8) reject sampling params outright.
  Don't add them.
- Env-var helpers in `config.py` treat set-but-empty (`CHAT_MODEL=`) as unset and
  fall back to the default. Match that behavior for new settings.
- `load_documents()` warns on stderr for unreadable files and *silently* skips
  whitespace-only ones, rather than aborting the ingest. Preserve that resilience.
- Document `source` metadata (path relative to `data_dir`, POSIX-style) is what
  citations key off. Any new loader must set it.
- Module and function docstrings explain *why*, not what. Match that register.
