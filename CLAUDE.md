# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv)
uv run rag ingest                    # embed data/ into the Atlas collection (+ create the index)
uv run rag query "your question"     # ask from the terminal
uv run streamlit run app.py          # chat UI over the same pipeline
uv run pytest                        # full suite (runs a local Atlas container; needs Docker, no secrets)
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
ordinary `uv run` would rebuild it at 3.13 — two full environment reinstalls.

More generally: probe uv/tool behaviour in a throwaway project elsewhere, never
here. This venv is ~125 packages and ~670 MB, and several uv commands rebuild it
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
ingest (rag_pipeline/ingest.py)    load → split → embed → store (MongoDB Atlas Vector Search)
query  (rag_pipeline/pipeline.py)  embed question → search → rerank → stuff prompt → Claude
```

`Settings` (`config.py`) is a frozen dataclass built via `Settings.from_env()`.
Both frontends — `rag_pipeline/cli.py` and `app.py` — construct it the same way,
which is what keeps them agreeing on the Atlas namespace, models, and chunking.
`MONGODB_URI` carries credentials, so it is a `require_env_key` guard like the
API keys, not a Settings field (a field needs a documentable literal default).

All tunables live here — never inline a literal at a call site. Adding one is a
**three-file change**:

1. the field plus its `_env_*` line in `config.py`,
2. a commented default in `.env.example`,
3. a row in the README config table.

Leaving either of the latter two stale is a bug, and nothing else in the repo
catches it — `ruff`, `ty` and the full suite are all green against a stale
README. `test_every_setting_is_documented` is what catches it.

There is no fourth site. `config.ENV_VARS` derives every variable name from the
dataclass fields, and `tests/test_config.py` clears *that* rather than a
hand-kept list. This matters because `config.py` loads `.env` at import time
(see Gotchas): a name missing from a hand-kept list would be answered by the
developer's own `.env`, so its default would silently stop being tested. Derived,
that drift is not merely detected — it is inexpressible.

### Why the store factories live in `ingest.py`

`build_embeddings()` and `open_store()` are defined in `ingest.py` and imported
*by* `pipeline.py`, not the reverse. This is deliberate: vectors from different
embedding models are not comparable, and the store's identity is (connection URI,
database, collection, vector index name, embedding function). Indexing and
querying must therefore go through one factory each. **Never construct
`MongoDBAtlasVectorSearch(...)`, `MongoClient(...)` or `VoyageAIEmbeddings(...)`
inline** — route through these factories. `open_store()` returns the vectorstore;
its `.collection` is the raw pymongo handle the incremental bookkeeping needs, and
the `MongoClient` behind it is memoized once per process (see below).

The reranker is the deliberate exception: `build_reranker()` lives in
`pipeline.py`, not here. Reranking is query-only — it has no ingest-side
counterpart, so the "same model must serve both phases" reason that pins the
embedding/store factories here simply does not apply. It sits beside
`build_chat_model()`, the other query-time model factory. This is enforced by an
ordinary behavioral test (the socket block + the injection seam), not a text
invariant, because the risk it guards — offline testability — is one a behavioral
test already covers.

### The process-wide MongoClient (the cache hazard inverts)

`ingest.py` memoizes one `MongoClient` per process (`_client`). Unlike Chroma's
per-directory client — which cached a stale on-disk *snapshot*, so a re-open
after a rebuild read the old index — a `MongoClient` is a connection pool that
always reads current server state. So the hazard flips: the client must be
created *once* and reused, not cleared. `reset_store_cache()` still exists and
callers are unchanged, but its body now *closes and drops* the client (connection
hygiene + fresh-process emulation) rather than clearing a snapshot. Callers
(`grep reset_store_cache`):

- `app.py` calls it before rebuilding the cached pipeline.
- `tests/conftest.py` calls it autouse at every test boundary.
- `tests/test_ingest.py::test_ingest_is_idempotent` calls it directly between
  ingests to emulate a fresh CLI process.
- `tests/test_pipeline.py` does the same around the collection-mismatch tests.

The Streamlit cache key is `index_version()` (a `str`) — a SHA-256 digest over
the corpus fingerprints that `ingest()` stamps into a reserved document
(`_VERSION_ID`, no `content_hash`/`embedding`, so invisible to the incremental
scans and to `$vectorSearch`). It changes on exactly the events an edit, add, or
removal changes and no others, so the running app picks up a `rag ingest` without
a restart and an unchanged re-ingest does not needlessly bust the cache.

### Ingest is incremental, and scoped — never a wholesale wipe

`ingest()` leaves the collection holding exactly the chunks for whatever is in
`data_dir` right now. It gets there by re-embedding only what changed: each
document carries a `content_hash` in its metadata (langchain-mongodb flattens
metadata to top-level fields, so it is queryable directly), and a source whose
hash still matches keeps the vectors it has. Chunks are keyed by a deterministic
`_id` (`source:index:content_hash`), so re-adding is an idempotent upsert-replace,
not a duplicating append.

**The state after the run is the contract, not the work skipped.** Every caller
depends on it — the app rebuilds after an upload and expects the rest of the
corpus to still be answerable, and a file edited by hand between runs is picked
up without being announced. Consequences that are easy to get wrong:

- Deletions are computed over the *indexed* sources, not the fresh ones. A file
  that is gone from `data_dir` has no fresh chunk to compare against, so an
  add-only pass would leave its vectors retrievable forever.
- `_fingerprint()` covers `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `CHUNK_SIZE`
  and `CHUNK_OVERLAP` as well as the text, because all four change what the stored
  vectors *are* — and the Atlas index fixes `numDimensions` at create time, so a
  different-width model produces vectors the existing index cannot serve. Dropping
  the model or dimensions is the dangerous case: the chunks still look current, so
  the skip is silent.
- Every read and delete is **scoped to `{"content_hash": {"$exists": True}}`**, so
  a collection shared with unrelated documents is never read, counted, or deleted
  from — the document-level form of "never a wipe". `_await_searchable` after an
  add, and `_ensure_vector_index` before the query path, exist because Atlas
  indexes asynchronously: an added chunk is in the collection before it is
  searchable, and a query against a not-yet-ready index returns zero rows with no
  error. Ordering is forced: insert (creates the collection), *then* create the
  index (`create_search_index` on a missing namespace fails).

`test_ingest_preserves_foreign_documents_in_a_shared_collection` guards the
scoping (a foreign doc survives a rebuild that deletes). Re-ingest is idempotent:
same input → same chunk count, no duplicate append, and no embedding calls at all.

`ingest()` returns the number of chunks the index *holds*, not the number
re-embedded. That is what keeps re-ingesting the same corpus reporting the same
number, and what both frontends' "Indexed N chunks" means.

### Dependency injection is the test seam

`ingest()`, `open_store()`, and `RAGPipeline.__init__` all accept optional
`embeddings` / `llm` — and `RAGPipeline.__init__` also `reranker`. **Production
always passes `None`**; the parameters exist so tests can inject
`DeterministicFakeEmbedding`, `FakeListChatModel`, and a fake
`BaseDocumentCompressor`. This is why the suite needs no network and no API key:
the real `build_embeddings()`/`build_reranker()` make Voyage AI HTTP calls and the
real LLM calls Claude, but none runs under test. Any new code path touching an
embedding model, the reranker, or the LLM should thread these through rather than
constructing them unconditionally.

Injection is a convention, so `conftest.py` backs it with an autouse fixture. The
store is real now (a local `mongodb-atlas-local` container, started once per
session via testcontainers), so the fixture is a **loopback allowlist** rather
than a blanket block: it patches `socket.socket.connect`/`connect_ex` and
`socket.create_connection` to permit `127.0.0.1` (the container) and reject every
other host. A test that forgets `embeddings=` names no banned symbol and no grep
would find it — but the real Voyage AI path connects to `api.voyageai.com`, a
public host, and that fails. That is the whole guarantee, and it is why the store
being real does not weaken it: only the store is on loopback; the paid APIs are
not. Needs Docker; the suite makes no cloud call and needs no secret.

`app.py` takes no such parameters — it is a script, not a function — so
`test_app.py` reaches the same seam through the factories instead, patching
`ingest.build_embeddings`, `pipeline.build_chat_model`, and
`pipeline.build_reranker` on their modules. That
works only because all are looked up as module globals at call time, which is a
second reason the never-construct-inline rule above is load-bearing: inline a
`VoyageAIEmbeddings(...)` anywhere and the frontend stops being testable
offline, not just inconsistent. `st.cache_resource` is cleared per test, since
its key deliberately ignores `_settings` and would otherwise serve one test's
pipeline to the next.

What `test_app.py` is *for* is the set of guarantees no lower-level test can see,
because they are only observable at the frontend:

- A chat turn is stored as a user/assistant **pair** whatever happens to it —
  success, a failed generation, or the run being torn down mid-answer — so a
  question can never be left in the history with nothing under it. This is what
  the `finally` in `app.py` buys, and the reason a failed turn is stored with an
  `error` flag rather than as ordinary text.
- An uploaded file is *answerable* on the same run, and the uploader is reachable
  when no index exists — the state it is most needed in, and the one a test that
  starts from a built index would never enter.
- A browser-supplied name cannot escape `data_dir` through the widget that
  delivers it.
- A later rerun does not silently re-index. This one is **counted, not
  displayed**: `st.file_uploader` re-reports its files on every rerun, so
  re-indexing on sight would rebuild the whole corpus once per chat message —
  correct output at absurd cost, and invisible to any assertion about what the
  app renders.

## Gotchas

- `config.py` calls `load_dotenv(override=False)` at **import time**. A real
  environment variable wins over `.env`, but a developer's local `.env` will leak
  into test runs — config tests must `monkeypatch.setenv`/`delenv` explicitly.
  This is why `test_config.py` clears `config.ENV_VARS` rather than a
  hand-written list — see the Settings rule above.
- `cli.py` imports `ingest`/`pipeline` lazily inside the command functions. This
  is load-bearing: importing them pulls in the pymongo/langchain stack, so
  `rag --help` and a usage error stay cheap. Keep those imports local.
- Atlas indexes asynchronously and validates a vector's width at *query* time,
  not insert — a wrong-width vector inserts fine and fails at `$vectorSearch`, and
  a query against a missing/not-yet-built index returns zero rows with no error.
  This is why ingest polls (`_await_queryable`/`_await_searchable`) and validates
  `EMBEDDING_DIMENSIONS` against the live model with a probe before writing.
- A free Atlas cluster auto-pauses after 30 idle days and must be resumed
  manually; a paused cluster surfaces as a `ServerSelectionTimeoutError` →
  RuntimeError. `MONGODB_TIMEOUT_MS` is generous so a cold resume is not misread
  as unreachable.

## Enforcing the invariants

The text-level rules live in `tests/invariants.py` as data, and
`tests/test_invariants.py` enforces them across every tracked `.py` file.
**That test is the enforcement** — it runs in CI, for every contributor and
every PR from a fork, whoever wrote the code and whatever editor they used.
There is no second layer, and nothing here depends on which editor you use.

Adding a rule means adding a `Rule` to `RULES`, a case in each direction in
`test_invariants.py`, and a row in the README rule table —
`test_every_rule_is_documented` is what catches the last one, and it exists
because the README's prose had already fallen two rules behind `RULES` with the
whole suite green. Two properties are load-bearing and easy to break:

- Rules match a **masked** copy of the text: string literals are blanked for
  every rule, comments too for all but the suppression rule. Without that, a
  comment describing a rule is blocked by the rule it describes.
- The masking alternation must stay **linear**. An earlier form let two branches
  both match a backslash, and an unterminated quote took 6.5s at 8 lines and
  never finished at 12 — the sweep hanging rather than failing.
  `test_masking_is_linear_on_pathological_input` is the guard.

**Prefer a behavioral test to a rule.** A rule matches spellings; a test
observes the property, so it covers routes nobody thought to enumerate. Reach
for `RULES` only when there is nothing to observe — which is the case exactly
when the point is that some call never happens (`store-factory`,
`embeddings-factory`, `no-suppressions`). Everything else is asserted where the
behavior is:

| Invariant | Enforced by |
| --------- | ----------- |
| the exception union, the empty-collection guard, `source` metadata on loaders | `test_pipeline.py`, `test_ingest.py` |
| `cli.py`'s imports stay cheap | `test_importing_cli_does_not_load_the_heavy_stack` — subprocess-imports the module, asserts pymongo/langchain_mongodb are absent from `sys.modules` |
| `build_chat_model` sets no sampling params | `test_build_chat_model_sets_no_sampling_params` — reads them back off the constructed model |
| `ingest()` never deletes documents it did not write | `test_ingest_preserves_foreign_documents_in_a_shared_collection` — a foreign doc survives a rebuild that deletes |

The last three replaced text rules (`lazy-cli-imports`, `no-sampling-params`,
`no-rmtree`) and are each strictly stronger than the regex they retired.

## Conventions

- New failure modes must fit `FileNotFoundError | RuntimeError | ValueError` —
  the union both frontends catch. `cli.py` catches it in one place (`main()`);
  `app.py` splits it across two, because the sidebar has to render in between:
  `ValueError` from `Settings.from_env()` stops the script above the sidebar,
  and `FileNotFoundError | RuntimeError` from the pipeline load is caught below
  it, so the uploader stays reachable when there is no index. Grep
  `except (FileNotFoundError` rather than trusting a line number. Don't add a
  fourth type — `_add_documents()` catching `OSError` is not one: it is the
  filesystem's own error on a write, and `FileNotFoundError` is already a
  subclass of it.
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
- `RAGPipeline.__init__` has two guards, checked in order. First: the vector
  index exists and is `queryable` — a full collection with no index answers every
  question "I don't know" off zero rows, which the empty-collection guard cannot
  see. Second: the namespace holds at least one of this pipeline's chunks (Mongo
  creates a namespace implicitly on first write and returns zero documents, with
  no error, for one never ingested into — and a wrong `MONGODB_DB`/
  `COLLECTION_NAME`/`VECTOR_INDEX_NAME` is a *different* namespace). Keep both.
- Retrieval-side provider failures are translated in `provider_errors_as_runtime`
  (`ingest.py`), the parallel to `_generate()`'s `anthropic.APIError` handling:
  it wraps every Voyage call and every MongoDB store op, mapping
  `voyageai.error.VoyageError`, `pymongo.errors.PyMongoError` *and*
  `bson.errors.BSONError` (which is outside `PyMongoError`) to **RuntimeError,
  never ValueError** — a bad `MONGODB_URI` must land below `app.py`'s sidebar so
  the uploader stays reachable, not in the `ValueError` branch that stops above it.
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
