# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv)
uv run rag ingest                    # rebuild the Chroma index from data/
uv run rag query "your question"     # ask from the terminal
uv run streamlit run app.py          # chat UI over the same pipeline
uv run pytest                        # full suite (~7s wall: ~4s torch import, ~2.5s tests; offline)
uv run pytest tests/test_config.py::test_defaults   # single test
uv run pytest -k idempotent -v                      # by keyword
uv run ruff check --fix . && uv run ruff format .   # lint, then format (order matters)
uv run ty check                      # type check
uv sync --locked && uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest   # every check CI runs
```

When working with Python, invoke the relevant `/astral:<skill>` — `/astral:uv`,
`/astral:ty`, `/astral:ruff` — to ensure best practices are followed.

`.python-version` pins local work to 3.13. It does *not* weaken the test matrix:
`setup-uv`'s `python-version:` input sets `UV_PYTHON`, which takes precedence
over the file, so the 3.11 CI leg really does run on 3.11.

To reproduce that leg locally, send it to a *separate* environment:

```bash
UV_PROJECT_ENVIRONMENT=.venv311 uv run -p 3.11 pytest
```

Plain `uv run -p 3.11` would recreate `.venv` itself at 3.11, and the next
ordinary `uv run` would rebuild it at 3.13 — two full torch reinstalls.

More generally: probe uv/tool behaviour in a throwaway project elsewhere, never
here. This venv is 135 packages and multi-GB, and several uv commands rebuild it
without asking.

Add dependencies with `uv add` / `uv add --dev` rather than hand-editing
`pyproject.toml`, so constraints and `uv.lock` stay derived rather than invented.
`uv add` silently no-ops if the current constraint already allows the resolved
version — pass an explicit floor (`uv add --dev "ruff>=0.15.22"`) to tighten one.

The lint select list is broad and the tree is clean against it. **Fix findings
rather than adding `# noqa` / `# ty: ignore`.** Prefer `uv run ruff`/`uv run ty`
over `uvx`, so versions match the lock. Ruff's line length and ty's target
version are both inherited (from the default and from `requires-python`) — don't
re-pin them in `pyproject.toml`.

`README.md` covers setup, configuration variables, usage, and what CI runs;
`ci.yml`'s own comments cover why its steps are ordered as they are. Consult
both rather than duplicating that material here. Every CI job must stay green.

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
**four-file change**:

1. the field plus its `_env_*` line in `config.py`,
2. a commented default in `.env.example`,
3. a row in the README config table,
4. the variable name in the `delenv` tuple in
   `tests/test_config.py::test_from_env_uses_defaults_when_unset`.

Leaving any of the latter three stale is a bug. The fourth is the one that hides:
because `config.py` loads `.env` at import time (see Gotchas), a variable missing
from that tuple is resolved from *the developer's own environment* rather than
its default, so the test keeps passing while silently no longer covering that
default. Nothing else in the repo catches this — `ruff`, `ty`, and the full
suite are all green with a stale `.env.example`, README, or tuple.

The `settings-triad` Stop hook (see below) enforces all four sites.

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
  This is why the `delenv` tuple is the fourth site of the four-file Settings
  change above: a variable omitted from it is quietly sourced from the
  developer's environment, and its default stops being tested.
- `cli.py` imports `ingest`/`pipeline` lazily inside the command functions. This
  is load-bearing: importing them pulls in sentence-transformers/torch, measured
  at ~4.3s versus ~0.08s for `rag --help` today. Keep those imports local.

## Enforcement hooks

Two hooks in `.claude/` mechanize the invariants above, so they fail loudly
rather than rotting. Both are committed, so they apply to everyone.

| Hook | Event | Enforces |
| ---- | ----- | -------- |
| `.claude/hooks/invariant-guard.py` | `PreToolUse` on `Edit`\|`Write` | no inline `Chroma(`/`HuggingFaceEmbeddings(` outside the factories; no top-level `ingest`/`pipeline` import in `cli.py`; no `# noqa`/`# ty: ignore`; no `rmtree` in `ingest.py`; no `temperature=`/`top_p=` in `pipeline.py` |
| `.claude/hooks/settings-triad.py` | `Stop` | all four sites of a Settings change are present |

Two design points worth preserving if you edit them:

- `invariant-guard` inspects **only the text being written**, never the file on
  disk, so it cannot fire on pre-existing code. It exempts `.claude/` — its own
  error messages quote the banned patterns, so without that guard it would make
  itself uneditable. `tests/` and `ingest.py` are exempt from the factory rule
  (`test_ingest.py` opens a `Chroma` collection directly on purpose).
- `settings-triad` runs on `Stop`, not `PostToolUse`. When `config.py` is
  written the other three sites legitimately do not have their entry yet, so a
  per-edit check would fire on every *correct* change. It honors
  `stop_hook_active` so it nudges once rather than looping.
- `settings-triad` only runs when the working tree has touched one of the four
  sites, so a turn about something else is never blocked by drift it did not
  cause. Within that scope it validates *all* settings, not just the changed
  one. The tradeoff: drift that is already committed goes unreported until
  someone next touches one of the four files. If git cannot answer, the check
  runs unconditionally — the failure direction is "enforce anyway".

Both hooks exit 1, not 2, on an unparseable payload: a shape change in a future
Claude Code release must not wedge every turn, but it must not disable
enforcement silently either, so they print one line saying they are not
enforcing. Treat that line as a bug report about the hook.

These scripts are linted by CI like any other file — `ruff check .` does not
exclude dot-directories, so a `# noqa`-free, formatted hook is not optional.

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
