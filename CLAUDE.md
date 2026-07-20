# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv)
uv run rag ingest                    # rebuild the Chroma index from data/
uv run rag query "your question"     # ask from the terminal
uv run streamlit run app.py          # chat UI over the same pipeline
uv run pytest                        # full suite (~9s warm: ~3s tests, rest torch import + startup; offline)
uv run pytest tests/test_config.py::test_defaults   # single test
uv run pytest -k idempotent -v                      # by keyword
uv run pytest --cov=rag_pipeline --cov=app --cov-report=term-missing   # coverage, on demand
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
**three-file change**:

1. the field plus its `_env_*` line in `config.py`,
2. a commented default in `.env.example`,
3. a row in the README config table.

Leaving either of the latter two stale is a bug, and nothing else in the repo
catches it — `ruff`, `ty` and the full suite are all green against a stale
README. `test_every_setting_is_documented` is what catches it; the
`derived-docs` hook reports the same thing sooner.

There is no fourth site. `config.ENV_VARS` derives every variable name from the
dataclass fields, and `tests/test_config.py` clears *that* rather than a
hand-kept list. This matters because `config.py` loads `.env` at import time
(see Gotchas): a name missing from a hand-kept list would be answered by the
developer's own `.env`, so its default would silently stop being tested. Derived,
that drift is not merely detected — it is inexpressible.

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
- `tests/test_pipeline.py` does the same around the collection-mismatch tests,
  which re-open a store after it was written and so need the state a real
  `rag query` starts from.

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

Injection is a convention, so `conftest.py` backs it with an autouse fixture that
blocks `socket.socket`/`create_connection`. A test that simply forgets
`embeddings=` names no banned symbol and no grep would find it — but it opens a
socket, and that fails. Do not pair it with `HF_HUB_OFFLINE=1`: that makes
`huggingface_hub` skip its revision check, so a cached model loads with no socket
at all and the fixture goes blind. The two are antagonistic, not complementary.

`app.py` takes no such parameters — it is a script, not a function — so
`test_app.py` reaches the same seam through the factories instead, patching
`ingest.build_embeddings` and `pipeline.build_chat_model` on their modules. That
works only because both are looked up as module globals at call time, which is a
second reason the never-construct-inline rule above is load-bearing: inline a
`HuggingFaceEmbeddings(...)` anywhere and the frontend stops being testable
offline, not just inconsistent. `st.cache_resource` is cleared per test, since
its key deliberately ignores `_settings` and would otherwise serve one test's
pipeline to the next.

## Gotchas

- `config.py` calls `load_dotenv(override=False)` at **import time**. A real
  environment variable wins over `.env`, but a developer's local `.env` will leak
  into test runs — config tests must `monkeypatch.setenv`/`delenv` explicitly.
  This is why `test_config.py` clears `config.ENV_VARS` rather than a
  hand-written list — see the Settings rule above.
- `cli.py` imports `ingest`/`pipeline` lazily inside the command functions. This
  is load-bearing: importing them pulls in sentence-transformers/torch, measured
  at ~4.3s versus ~0.08s for `rag --help` today. Keep those imports local.

## Enforcing the invariants

The mechanically checkable rules above live in `tests/invariants.py` as data,
and `tests/test_invariants.py` enforces them across every tracked `.py` file.
**That test is the enforcement** — it runs in CI, for every contributor and
every PR from a fork, whoever wrote the code and whatever editor they used.

The two hooks in `.claude/` load the same module and answer the same questions
at write time instead of at review time. They are a latency optimization, not a
second source of truth: delete `.claude/` and nothing stops being enforced.

| Layer | Covers | When |
| ----- | ------ | ---- |
| `tests/test_invariants.py` | the whole tree, everyone | CI and `uv run pytest` |
| `.claude/hooks/invariant-guard.py` (`PreToolUse` on `Edit`\|`Write`) | the text about to be written | during a Claude session |
| `.claude/hooks/derived-docs.py` (`Stop`) | Settings documented at all three sites; every rule documented in the README rule table | end of a Claude turn |

Adding a rule means adding a `Rule` to `RULES`, a case in each direction in
`test_invariants.py`, and a row in the README rule table —
`test_every_rule_is_documented` is what catches the last one, and it exists
because the README's prose had already fallen two rules behind `RULES` with the
whole suite green. Two properties are load-bearing and easy to break:

- `tests/invariants.py` must stay **standard-library only**. The hooks exec on
  `#!/usr/bin/env python3` — the system interpreter, not the venv — so an import
  from `rag_pipeline` pulls in `dotenv` and crashes the hook rather than checking
  anything. It is why `settings_defaults` reads config.py's AST instead of
  `Settings`. `uv run pytest` cannot catch a violation: under uv those imports
  resolve. Check with
  `/usr/bin/env python3 -c "import ast; ast.parse(open('tests/invariants.py').read())"`
  and a hook run.
- Rules match a **masked** copy of the text: string literals are blanked for
  every rule, comments too for all but the suppression rule. Without that, a
  comment describing a rule is blocked by the rule it describes.
- The masking alternation must stay **linear**. An earlier form let two branches
  both match a backslash, and an unterminated quote — the normal case in an Edit
  fragment — took 6.5s at 8 lines and never finished at 12. A hook that hangs
  wedges the edit.

Not every invariant belongs there. The exception union, the empty-collection
guard, and `source` metadata on loaders are enforced by ordinary tests in
`test_pipeline.py` and `test_ingest.py`, which assert what the code *does*
rather than how it is spelled — a better layer whenever it is available.

Design rationale for the hooks themselves (Stop vs PostToolUse, exit 1 vs 2,
the working-tree gate) lives in each hook's module docstring. Both are linted by
CI like any other file: `ruff check .` does not exclude dot-directories.

## Conventions

- New failure modes must fit `FileNotFoundError | RuntimeError | ValueError` —
  the union both frontends catch (`cli.py:77`, `app.py:65`; see the comment at
  `app.py:66-67` for the precise per-type mapping). Don't add a fourth type.
- Generation-level failures are raised in `_generate()` and nowhere else — the
  `anthropic.APIError` → `RuntimeError` translation that keeps the SDK out of
  both frontends, and the empty-response guard that stops a frontend presenting
  no content under a full citation list. `stream_answer()` wraps `_generate()`
  and `answer()` joins over that, so every shape inherits both. A new failure
  belongs here rather than in a frontend: the one that goes in `app.py` is the
  one `cli.py` silently doesn't get. Note the translation wraps the *iteration*,
  not the `.stream()` call: the chain is lazy, so a provider error surfaces
  during consumption, and a `try` around the call alone would catch nothing.
- `stream_answer()` returns `(docs, chunks)` because every frontend needs both,
  and handing back the docs the answer was actually generated from is what stops
  displayed citations from drifting via a second search. Its two halves settle at
  different times — retrieval has run when it returns, generation has not — which
  is what lets a caller wrap a spinner around just the call.
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
