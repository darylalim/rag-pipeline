# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv; MLX only on macOS)
uvx --from huggingface_hub hf download <model id>   # once per model (README Setup lists the three); loading never downloads
uv run rag ingest                    # embed data/ into the Chroma collection under chroma_db/
uv run rag query "your question"     # ask from the terminal (loads all three models first)
uv run streamlit run app.py          # chat UI over the same pipeline
uv run pytest                        # full suite (fakes + in-process Chroma; no models, network, Docker or secrets)
uv run pytest -m models              # live tests against the real models (Apple Silicon + models downloaded; ~1 min)
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
here. This venv is ~135 packages and ~940 MB on a Mac (MLX included; Linux gets
~720 MB without it), and several uv commands rebuild it without asking.

Add dependencies with `uv add` / `uv add --dev` rather than hand-editing
`pyproject.toml`, so constraints and `uv.lock` stay derived rather than invented.
`uv add` silently no-ops if the current constraint already allows the resolved
version — pass the locked version as an explicit floor
(`uv add --dev "ruff>=<locked version>"`) to tighten one.
A macOS-only dependency takes the marker the MLX ones carry:
`uv add --marker "sys_platform == 'darwin'" <pkg>`.

The lint select list is broad and the tree is clean against it. **Fix findings
rather than suppressing them** — no `# noqa`, `# ty: ignore`, `# type: ignore`,
or any other form ruff or ty honours in source. The README's rule table lists
them all, and `no-suppressions` rejects them. Prefer `uv run ruff`/`uv run ty`
over `uvx`, so versions match the lock. Ruff's line length and ty's target
version are both inherited (from the default and from `requires-python`) — don't
re-pin them in `pyproject.toml`.

`README.md` covers setup, configuration variables, usage, performance, and what
CI runs; `ci.yml`'s own comments cover why its steps are ordered as they are.
Consult both rather than duplicating that material here. Every CI job must stay
green.

## Architecture

Two phases with a hard boundary between them, one shared config object, and the
local models behind LangChain's interfaces:

```
ingest  (rag_pipeline/ingest.py)      load → split → embed → store (Chroma, under PERSIST_DIR)
query   (rag_pipeline/pipeline.py)    embed question → search → rerank → stuff prompt → local LLM
models  (rag_pipeline/mlx_models.py)  QwenVLEmbeddings · QwenVLReranker · MLXChatModel, over MLX
tracing (rag_pipeline/tracing.py)     optional: each question as one trace, to a self-hosted Phoenix
```

`Settings` (`config.py`) is a frozen dataclass built via `Settings.from_env()`.
Both frontends — `rag_pipeline/cli.py` and `app.py` — construct it the same way,
which is what keeps them agreeing on the persist directory and collection, the
models, and chunking. There are no secrets: every setting is a field with a
literal default.

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

The adapters' private constants in `mlx_models.py` — the embedding and
reranking instructions and prompt formats, the batch sizes, the 8192-token
prompt limit — are deliberately *not* settings. They are each model family's
recipe (and measured optima), not user tunables: the prompts are what the live
tests' model-card scores check, and changing the embedding instruction would
also invalidate every stored vector without changing the fingerprint.

### Why the store factories live in `ingest.py`

`build_embeddings()` and `open_store()` are defined in `ingest.py` and imported
*by* `pipeline.py`, not the reverse. This is deliberate: vectors from different
embedding models are not comparable, and the store's identity is (persist
directory, collection name, embedding function). Indexing and querying must
therefore go through one factory each. **Never construct `Chroma(...)`,
`chromadb.PersistentClient(...)` or `QwenVLEmbeddings(...)` inline** — route
through these factories. `open_store()` returns the langchain `Chroma`;
bookkeeping that needs no model goes through `_collection()`, the raw collection
with no embedding function (chromadb's default is an ONNX model it downloads on
first use). `_client()` is the one `PersistentClient` construction, with a
*fresh* `ChromaSettings(anonymized_telemetry=False)` each call: `PersistentClient`
mutates the settings object it is handed, and two clients on one directory with
unequal settings are a builtins `ValueError`.

The reranker is the deliberate exception: `build_reranker()` lives in
`pipeline.py`, not here. Reranking is query-only — it has no ingest-side
counterpart, so the "same model must serve both phases" reason that pins the
embedding/store factories here simply does not apply. It sits beside
`build_chat_model()`, the other query-time model factory. This is enforced by an
ordinary behavioral test (the MLX block + the injection seam), not a text
invariant, because the risk it guards — offline testability — is one a behavioral
test already covers.

### Models load once per process (`mlx_models.py`)

The three factories construct adapters (`QwenVLEmbeddings`, `QwenVLReranker`,
`MLXChatModel`), and every adapter gets its weights from `load_mlx_model()` —
the only place weights load. It is memoized by resolved snapshot path under a
double-checked lock, so concurrent first calls from Streamlit sessions load once.
That memo is what makes the factories cheap to call again: the app rebuilds its
pipeline after every ingest, and `ingest()` builds its own embedder, and both
wrap the weights already in memory rather than loading second copies of ~22 GB.
Never call `mlx_lm.load` anywhere else.

`load_mlx_model()`'s ordering is load-bearing:

1. It imports `mlx_lm` *first*, so a machine without MLX gets the same
   `RuntimeError` whether or not the model is cached — and the test suite's MLX
   block catches every route to a real model, memoized or not.
2. `resolve_model_path()` finds the model without the network:
   `snapshot_download(local_files_only=True, allow_patterns=<mlx-lm's>)`, then
   checks every shard the weight index lists. Not cached, or incomplete →
   `FileNotFoundError` naming `uvx --from huggingface_hub hf download <id>`. An
   existing directory is used as-is.
3. `mlx_lm.load()` gets that resolved path, never the repo id — given an id,
   mlx-lm tries the network first even when the model is cached.

Concurrency: one lock per *loaded model* (keyed by the model, not the adapter,
so an outgoing and an incoming pipeline share it) serializes forward passes; one
module-level `_GENERATION_LOCK` is held for a whole generation, because mlx-lm
sets and restores the process-wide Metal wired limit around each one and
overlapping calls race on it. mlx-lm's stream is closed while that lock is still
held, and the lock is released however the stream ends — exhausted, failed, or
closed half-way. **A stream that is dropped rather than closed keeps the lock
until the garbage collector finalizes it**, which for one a Streamlit script
holds as a module global can be never — every later question, from every
session, then hangs on the lock. So everything between the model and a frontend
closes rather than drops: `app.py` wraps generation in `closing(chunks)` (the
Stop button raises inside `st.write_stream` with the stream suspended),
`stream_answer`'s tracing wrapper (`_traced`) closes `_generate`'s stream with
its own, and `RAGPipeline._generate` closes the chain's stream with its own. That chain is
`_PROMPT | llm`, with **no `StrOutputParser`**: closing a parser's stream does
not stop the model — langchain-core catches the `GeneratorExit` and drains the
parser's input, i.e. generates on to `MAX_TOKENS` under the lock.
`test_a_real_stop_releases_the_model_and_keeps_the_turn` delivers Stop the way
the button does, with the garbage collector off, over the real `MLXChatModel`.

MLX itself is reached only in a few thin methods (`_pool`, `_forward`, the
generation loop, `_release_buffers`), so prompts, truncation, batching, ordering,
locking and error translation are all unit-tested in CI with a fake `mlx_lm`
(`test_mlx_models.py`). Whether the recipe is *right* is another matter: a
subtly wrong prompt or pooling step still yields plausible vectors and
sensible-looking rankings. Only `tests/test_models_live.py` (`-m models`) notices,
by reproducing the model cards' published scores — run it after touching an
adapter.

### Tracing is the API in the pipeline, the SDK in the frontends

Off unless `PHOENIX_COLLECTOR_ENDPOINT` is set (empty default; `_env_url` refuses
a malformed one as `ValueError`). OpenTelemetry's own split: `pipeline.py`
imports only `opentelemetry-api` and OpenInference's attribute names, which are
a no-op until a provider exists, and `tracing.setup_tracing()` installs one.
`cli.py` calls it in `cmd_query` only (ingest emits no spans, so it would only
load the tracing stack and start an idle exporter thread), and `app.py` on every
rerun, inside the pipeline-load `try`; it is once per process, guarded by the
instrumentor's own state under a lock (`test_concurrent_first_setups_install_once`).
Like the adapters, it raises only `RuntimeError`, because of where the app calls
it: the SDK logs most malformed `OTEL_*` variables and falls back to a default,
but refuses a malformed span limit or an out-of-range batch setting with a
builtins `ValueError` (before OpenTelemetry 1.45, an unknown compression or an
out-of-range sampler ratio too), and the exporter refuses a credential provider
that is not installed with a `RuntimeError` that names no variable.
`setup_tracing` catches either and raises, in its place, a `RuntimeError` that
points at the `OTEL_*` variables. It installs nothing until everything is built,
and shuts down a provider already built when the failure comes — left
registered, its exit hook would keep it alive until the process ends, one more
per failed rerun (`test_a_malformed_otel_variable_is_a_runtime_error_that_installs_nothing`
watches `atexit` for it). Its imports are lazy, so tracing off loads none of the
instrumentation or exporter (`test_tracing_off_loads_none_of_the_tracing_stack`,
which takes the frontends' path: import, then `setup_tracing(Settings())`).

`setup_tracing` assembles the provider from OpenTelemetry's parts, **not
`phoenix.otel.register()`**, whose shortcuts are traps here. Given a base URL,
`register()` posts to it as-is, Phoenix answers 405, and `force_flush()` still
returns True. So `traces_url()` always appends `/v1/traces`. Left to infer a
protocol, `register()` picks gRPC, which also bypasses `_offline` (grpc's C
core). It cannot set the exporter timeout. And it reads `PHOENIX_*` variables
and `.env.phoenix` files that `Settings` does not know about.

Each choice in `setup_tracing` is measured:

- `BatchSpanProcessor`: a simple processor exports inside `span.end()`, so with
  Phoenix down each span waits out the exporter's retries: about 7 s at the
  default timeout, still about 1 s at 2 s.
- `_EXPORT_TIMEOUT_S = 2`: that timeout is all that bounds the exit flush (about
  7 s at the default 10 s, about 1 s at 2 s). `force_flush(timeout)` and
  `OTEL_BSP_EXPORT_TIMEOUT` are ignored.
- OpenInference's `TracerProvider`: the SDK's keeps only 128 attributes per span.
  A reranker span carries three per candidate and four per kept passage (80 at
  the defaults), so from a `FETCH_K` of about 37 the SDK's would silently cut
  it.

A question is one trace, and every path out of it ends the root span:

- **The root span.** `stream_answer` opens it (`start_span`, never
  `start_as_current_span`) and makes it current only around retrieval, a
  synchronous stretch. The LangChain retriever run and the manual reranker span
  nest under it there. A compressor is not a Runnable, so without the manual
  span the rerank would not be traced at all.
- **Generation is lazy**, so it outlives that call. `_traced` re-attaches the
  root's context around each `next()` of `_generate` and never across a
  `yield`. One held across a yield leaks into the consumer between pieces, and
  logs "Failed to detach context" when the stream is closed from another context
  or thread.
- **Primed.** `_traced` yields `""` once, and `stream_answer` consumes it before
  returning. A generator's `finally` exists only once its body has started, so a
  stream closed before its first piece would otherwise end nothing.
- **Status (`_finish`).** An `Exception` is ERROR with the exception recorded. A
  `BaseException` (GeneratorExit, KeyboardInterrupt, Streamlit's StopException)
  leaves the status unset, adds a `stopped` event, and keeps the partial answer
  as the output.

The model's own span still shows ERROR on a Stop: LangChain reports a closed
stream to its tracer as an error. `_tracer()` is looked up per question, not
held at import, because a tracer keeps the provider it first resolved and tests
reset it. `MLXChatModel._get_ls_params` sets `ls_model_name`, since LangChain
fills it only from a `model`/`model_name` field; without it the LLM span names no
model. `tests/test_tracing.py` asserts the trace in-process and the wire format
in a subprocess, against a stand-in collector that runs inside the subprocess.

### Chroma's per-process System (the stale-view hazard)

chromadb shares one System per persist directory per process, and that System's
*vector search* does not see writes another process made after it was opened:
after a terminal `rag ingest` it keeps returning chunks that were deleted, or
raises `InternalError: Error finding id` under a where-filter. Counts, gets and
collection metadata *are* current, so every check but an actual search looks
fine. `reset_store_cache()` is `SharedSystemClient.clear_system_cache()` — it
*drops* the cached System, never closes it: closing stops the System under every
client still holding it (an outgoing pipeline answering in another session),
whose next call is then a builtins `AttributeError` outside the caught union. A
dropped System merely keeps its old view, and what that raises is a
`ChromaError`, which `store_errors_as_runtime` turns into a RuntimeError.
Callers (`grep reset_store_cache`):

- `app.py`'s `load_pipeline` calls it before every build, or the fresh pipeline
  would answer from the old index while its cache key already names the new one.
  `max_entries=1` for the same reason: a second slot would keep a pipeline opened
  before the last reset, and a corpus that changes and changes back mints its
  old key again.
- `ingest()` calls it under its lock, before opening the store. A stale System
  does not only *read* the old view: written through, it saves its vector index
  back over the other process's, keeping the chunks that process deleted for
  good — short searches, and past a few thousand chunks "Error finding id" on
  every filtered one. (The app's upload ingest runs in the sidebar, above the
  reset in `load_pipeline`, so this is the one that covers it.)
- `tests/conftest.py` calls it autouse at every test boundary.
- `test_ingest.py`, `test_pipeline.py` and `test_cli.py` call it directly
  between ingests to emulate a fresh CLI process.

chromadb's System cache is a class-level dict with no lock of its own, and a
client's construction inserts a System, starts it, then reads it back. So every
construction (`_client()`) and every clear (`reset_store_cache()`) shares one
module lock, `_system_cache_lock`: without it one Streamlit session's rebuild,
landing inside another session's client open, surfaced there as a builtins
`KeyError`/`AttributeError`. And `_client()` empties the cache when an open
fails — chromadb caches a System *before* starting it, so one whose start failed
(a corrupt `chroma.sqlite3`) would otherwise be handed, half-built, to the next
open, whose cleanup raises a builtins `AttributeError`.

The Streamlit cache key is `index_version()` (a `str`) — a SHA-256 digest over
the corpus fingerprints that `ingest()` stamps into the *collection metadata*
(`rag_index_version`), because Chroma cannot store a record without an
embedding, and one with a made-up vector could be retrieved. Metadata reads are
fresh across processes even through a cached System, which is what lets the app
notice a `rag ingest` at all. It changes on exactly the events an edit, add, or
removal changes and no others, so the running app picks up a `rag ingest`
without a restart and an unchanged re-ingest does not needlessly bust the cache.
`ingest()` reconciles it on *every* run — compared, and written only when it
differs — because a run that died after its writes but before the stamp leaves
every source looking current, and a stamp written only "when something changed"
would then never be repaired. It returns `""` when nothing has been ingested,
and creates nothing: every read path checks `_has_store()` — for
`chroma.sqlite3`, not merely the directory — before opening a client, which
would create a database in whatever directory it is given. The app calls it on
every rerun of a fresh checkout.

### Ingest is incremental, and scoped — never a wholesale wipe

`ingest()` leaves the collection holding exactly the chunks for whatever is in
`data_dir` right now. It gets there by re-embedding only what changed: each
chunk carries its source's `content_hash` in its metadata, and a source whose
hash still matches keeps the vectors it has. Chunks are keyed by a deterministic
id (`source:index:content_hash`) and added through langchain-chroma's
`add_documents(ids=)`, which upserts — a raw `collection.add` would silently keep
the *old* document under an existing id — so re-adding is an idempotent replace,
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
  vectors *are*. Dropping the model or dimensions is the dangerous case: the
  chunks still look current, so the skip is silent.
- A Chroma collection fixes its width at the first insert and keeps it after
  every row is deleted, so a width change would fail at the add — *after* the
  delete had removed the chunks it was replacing. Hence two checks before any
  write, only when there is something to embed, each a `ValueError` naming its
  own fix. First, the live model's probe width against `EMBEDDING_DIMENSIONS`:
  set it to the model's width. Then that width against the width this
  pipeline's chunks already hold: a new `COLLECTION_NAME`, never a wiped persist
  directory (it may hold other collections). The second reads only `OWN_CHUNKS`,
  so a collection whose width was set by foreign records, or whose own chunks
  are all gone, gives it nothing to compare: there the add itself fails, as
  `store_errors_as_runtime`'s `RuntimeError` with the same `COLLECTION_NAME`
  hint — harmlessly, since with none of our chunks indexed nothing was deleted
  first.
- Every read, delete **and search** is scoped to `OWN_CHUNKS`
  (`{"ingested_by": "rag-pipeline"}`, a marker every chunk carries), so a
  collection shared with unrelated records is never read, counted, deleted from
  or cited — the document-level form of "never a wipe". A dedicated key matched
  by equality, because Chroma has no `$exists`, and the obvious stand-in,
  `{"content_hash": {"$ne": ""}}`, also matches records that *lack* the key: a
  delete built on it removed a foreign document. Deletes are
  `{"$and": [OWN_CHUNKS, {"source": {"$in": superseded}}]}`, issued only when
  `superseded` is non-empty.
- The whole read → delete → add → stamp sequence runs under a `FileLock` on
  `persist_dir/.ingest.lock` — and so does the read of `data_dir` that decides
  it. Two writers on one persist directory at once — a terminal `rag ingest`
  overlapping an upload in the app — corrupt it permanently, and neither writer
  sees an error; only every later query does. And a run that read `data_dir`
  before waiting would apply that older snapshot after the writer it waited on,
  deleting what that writer had just indexed. The embedder is built *before* the
  lock (a multi-second load no other writer should wait on). `_writer_lock()`
  turns the filesystem's own errors creating the directory and the lock file
  (`FileExistsError`, `PermissionError`, …) into `RuntimeError` — around those
  two steps only, so an `OSError` from inside keeps its type. Readers need no
  lock.
- New chunks are added in slices of `get_max_batch_size()` (5461 locally).
  langchain's `add_documents` upserts everything in one call, which chromadb
  refuses above that — a few MB of text — *after* the delete had run. Sliced,
  an interrupted run keeps its progress, and the chunk-count check re-embeds a
  source a failure cut in half. `changed` is sorted, so the same corpus goes
  through the embedder in the same batches — and to the same bits — every run.

`test_ingest_preserves_foreign_documents_in_a_shared_collection` guards the
scoping on the write side (a foreign doc survives a rebuild that deletes), and
`test_retrieve_never_returns_a_foreign_document` on the read side. Re-ingest is
idempotent: same input → same chunk count, no duplicate append, and no embedding
calls at all.

`ingest()` returns the number of chunks the index *holds*, not the number
re-embedded. That is what keeps re-ingesting the same corpus reporting the same
number, and what both frontends' "Indexed N chunks" means.

### Dependency injection is the test seam

`ingest()`, `open_store()`, and `RAGPipeline.__init__` all accept optional
`embeddings` / `llm` — and `RAGPipeline.__init__` also `reranker`. **Production
always passes `None`**; the parameters exist so tests can inject
`DeterministicFakeEmbedding`, `FakeListChatModel`, and a fake
`BaseDocumentCompressor`. This is why the suite needs no model and no Apple
Silicon: the real `build_embeddings()`/`build_reranker()`/`build_chat_model()`
load multi-gigabyte MLX checkpoints from the Hugging Face cache, but none runs
under test. Any new code path touching an embedding model, the reranker, or the
LLM should thread these through rather than constructing them unconditionally.

Injection is a convention, so `conftest.py` backs it with autouse guards:

- `_no_real_models` puts `None` in `sys.modules` for `mlx` and `mlx_lm`, so
  `import mlx_lm` raises and `load_mlx_model()` reports a RuntimeError before it
  looks in the cache. **This is the one that catches a forgotten injection.** A
  test that forgets `embeddings=` names no banned symbol, and with the models
  cached it opens no socket either — a network block alone would let it load
  gigabytes of weights. It behaves identically on a Mac with MLX installed and on
  the Linux CI legs without it.
- `_offline` blocks every socket, to any host: Chroma is in-process and the
  models are local files, so a connection means something is downloading (a
  model, Chroma's default ONNX embedder) or phoning home.
- `_no_tracing` (session-scoped) deletes `PHOENIX_COLLECTOR_ENDPOINT` and forces
  LangSmith's switches to `false` (the pipeline no longer uses LangSmith, but
  langsmith ships inside langchain-core and still acts on them). `.env` is loaded
  at import time, and either would otherwise send test traces from a background
  flush that can land after `_offline` is undone. **`_offline` cannot catch an
  exporter**: the socket block's `RuntimeError` is caught inside the tracing
  stack and only logged — by the exporter's own HTTP transport from
  OpenTelemetry 1.45, by the batch processor before — so the test passes.
- `_no_tracer_left_on` is what makes that failure loud: after every test it
  fails one that left a global tracer provider installed or LangChain
  instrumented. A test that runs `setup_tracing` for real takes `undo_tracing`;
  one that asserts on spans takes `spans`, which records them in memory
  (OpenInference's provider, a synchronous processor, LangChain instrumented)
  and undoes all of it — OpenTelemetry allows one global provider per process,
  and `opentelemetry-test-utils`' `reset_trace_globals()` is what undoes it.
  The guard switches a leak off before failing, through the same
  `_switch_tracing_off()` as `undo_tracing`, so the failure stays with the test
  that leaked. Left on, it would fail the tests after it too: each would trip
  the check again until one undid tracing, and one that sets tracing up would
  find it done and return early, installing and raising nothing.
  `test_a_leak_fails_only_the_test_that_left_tracing_on` runs such a session
  in a pytest subprocess.

`tests/test_offline_guard.py` trips every route to a real model on purpose —
each factory, an ingest and a pipeline left without a fake — and checks the
socket block and the `models` exemption (through the `hide_mlx` fixture), so a
guard that loosens reads as a failure rather than as green.

Tests marked `@pytest.mark.models` are the deliberate exception: they load the
real models, are deselected by pyproject's `addopts = ["-m", "not models"]`, and
run only with `uv run pytest -m models` (a later `-m` replaces the default). They
skip rather than fail when MLX or a model is missing. CI never runs them.

`app.py` takes no such parameters — it is a script, not a function — so
`test_app.py` reaches the same seam through the factories instead, patching
`ingest.build_embeddings`, `pipeline.build_chat_model`, and
`pipeline.build_reranker` on their modules. That
works only because all are looked up as module globals at call time, which is a
second reason the never-construct-inline rule above is load-bearing: inline a
`QwenVLEmbeddings(...)` anywhere and the frontend can no longer be driven with
fakes at all, not just inconsistently. `st.cache_resource` is cleared per test,
since its key deliberately ignores `_settings` and would otherwise serve one
test's pipeline to the next.

What `test_app.py` is *for* is the set of guarantees no lower-level test can see,
because they are only observable at the frontend:

- A chat turn is stored as a user/assistant **pair** whatever happens to it —
  success, a failed generation, or the run being torn down mid-answer — so a
  question can never be left in the history with nothing under it. This is what
  the `finally` in `app.py` buys, and the reason a failed turn is stored with an
  `error` flag rather than as ordinary text. The `finally` writes through a
  `history` bound before the turn, never through `st.session_state`: after a
  real Stop the runner *stays* stopped, and every `st.session_state` access is
  a yield point that raises `StopException` again. (`fail_mid_stream` raises
  from inside generation and cannot show that; the real-Stop test calls
  `script_requests.request_stop()`, as the button does.)
- A stopped answer releases the model: see the concurrency paragraph above.
- A Stop during *retrieval* still closes the stream. It is raised at the
  retrieval spinner's exit (a Streamlit call), after `stream_answer` returned
  and before generation starts, so `app.py` registers `closing(chunks)` inside
  the spinner through an `ExitStack`. No lock is held then, but the stream holds
  the question's root span, and a dropped stream means a trace never sent.
- An upload is reported as added only if it reached the index
  (`indexed_sources()`): ingest skips a textless file — a scanned PDF — without
  failing.
- An uploaded file is *answerable* on the same run, and the uploader is reachable
  when no index exists — the state it is most needed in, and the one a test that
  starts from a built index would never enter.
- A browser-supplied name cannot escape `data_dir` through the widget that
  delivers it.
- Every pipeline build starts from a fresh store client, so the app never
  answers from a stale view of an index another process rebuilt.
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
  hand-written list — see the Settings rule above. At runtime it overrides the
  defaults too: a stale model id left in `.env` fails as "not in the Hugging
  Face cache".
- `cli.py` imports `ingest`/`pipeline` lazily inside the command functions. This
  is load-bearing: importing them pulls in chromadb (a multi-second cold import)
  and the langchain stack, so `rag --help` and a usage error stay cheap. Keep
  those imports local.
- MLX is **macOS-only**: `mlx` and `mlx-lm` are declared
  `; sys_platform == 'darwin'`, so the Linux CI legs never install them. Every
  MLX import is therefore lazy, inside `mlx_models.py`'s functions — a
  module-level one breaks collection on Linux. ty types them as `Any` on every
  platform (`replace-imports-with-any` in `pyproject.toml`), so a Mac and CI give
  the same answer; the price is untyped MLX call sites, which is why they are
  kept thin. It covers `mlx.**`/`mlx_lm.**` only — not a place to park other
  unresolved imports.
- MLX keeps freed buffers for reuse, and with varying batch shapes that cache
  grew to ~7 GB beside a 3.4 GB model. `_release_buffers()` (`mx.clear_cache()`)
  runs after every embed, rerank and generation; a new MLX call path must do the
  same, or three models that fit in 32 GB stop fitting.
- The Qwen3.8 chat template treats an unset `enable_thinking` as *on*: it adds a
  reasoning instruction and the model streams raw reasoning into the answer with
  no `<think>` tag to strip. `MLXChatModel` passes `enable_thinking=False`;
  filtering afterwards cannot work. mlx-lm's `stream_generate` also defaults to
  `max_tokens=256`, so it is passed explicitly.
- Chroma creates `persist_dir`, and `chroma.sqlite3` in it, when a client opens,
  so every read path checks `_has_store()` first; `get_collection` (never
  `get_or_create`) on reads, so a query cannot conjure an empty collection.
  chromadb raises *builtins* `ValueError`/`TypeError` from its own argument
  checks — an empty `$in`, a one-clause `$and`, `hnsw:space` passed to
  `modify`, a `None` metadata value, a search for fewer than one result (hence
  `RAGPipeline`'s `FETCH_K` check) — which the code avoids by construction
  rather than catching (catching `ValueError` would swallow ingest's own width
  errors). `get(include=["embeddings"])` returns a numpy array: use `len()`,
  never truthiness. `modify(metadata=)` *replaces* the dict, so
  `_write_index_version` merges the existing keys, minus `hnsw:*`.
- The stale-view and concurrent-writer hazards above are Chroma's two sharp
  edges; both are silent until a later query, so don't remove
  `reset_store_cache()` from `load_pipeline` or from `ingest()`, or the ingest
  `FileLock`.
- On a Mac, LangChain imports transformers (it arrives with mlx-lm, without
  torch) as soon as the pipeline is imported — langchain-text-splitters, through
  `ingest.py`, and before 1.6 langchain-core too — and transformers then prints
  `PyTorch was not found. Models won't be available` — false here. mlx-lm
  silences that for itself, but too late, so `rag_pipeline/__init__.py` sets
  `TRANSFORMERS_NO_ADVISORY_WARNINGS` first. It must stay in the package
  `__init__`, the earliest import on every entry point;
  `test_importing_the_pipeline_prints_no_pytorch_warning` checks it.
- Phoenix's own clients read `PHOENIX_COLLECTOR_ENDPOINT` too, with different
  semantics: unset means `localhost:6006` to them and *off* here, and given no
  protocol they infer gRPC. The name is shared so Phoenix's docs on it apply;
  the behavior is this repo's, from `Settings`, over HTTP. Phoenix's other
  client variables (`PHOENIX_API_KEY`, its headers variable, `PHOENIX_GRPC_PORT`)
  are not read, so the pipeline sends no credentials and needs a Phoenix with
  authentication off — its default; the README keeps it private with
  `PHOENIX_HOST=127.0.0.1` instead.
- langsmith still ships inside langchain-core and acts on `LANGSMITH_TRACING`
  by itself, uploading every run to LangSmith's cloud. The pipeline does not
  use it, and `_no_tracing` switches it off for tests only; the README and
  `.env.example` tell users with an old `.env` to delete it.
- `.streamlit/config.toml` turns off Streamlit's usage statistics (its front end
  would report to Streamlit) and its file watcher (which walks every loaded
  module on every run and logs a traceback for each of transformers' lazy ones —
  over a hundred a turn), and sets `server.address` to `127.0.0.1`. Unset,
  Streamlit listens on every interface, putting the uploader — which writes into
  `data/` — on the local network with no login; and a headless start then asks
  checkip.amazonaws.com for the machine's external IP, to print it.
  `test_the_app_config_keeps_streamlit_local_and_quiet` pins all three.

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
  comment describing a rule is blocked by the rule it describes. Strings and
  comments are found in one pass, so a quote inside a comment opens no string:
  masking strings first let an apostrophe and a later quote in one comment
  hide the text between them, suppressions included.
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
| the exception union, the empty-collection guards, `source` metadata on loaders | `test_pipeline.py`, `test_ingest.py`, `test_mlx_models.py` |
| `cli.py`'s imports stay cheap | `test_importing_cli_does_not_load_the_heavy_stack` — subprocess-imports the module, asserts chromadb/langchain_chroma/mlx/mlx_lm are absent from `sys.modules`, with the same probe required to see the store stack once the pipeline is imported, so it cannot pass vacuously |
| the chat model decodes greedily, with thinking off | `test_generation_is_greedy_with_thinking_off_and_explicit_max_tokens` — asserts the exact arguments generation is called with, so no sampler reaches mlx-lm by any route |
| `ingest()` never deletes documents it did not write | `test_ingest_preserves_foreign_documents_in_a_shared_collection` — a foreign doc survives a rebuild that deletes |
| a stopped answer releases the generation lock, and its turn is still stored | `test_a_real_stop_releases_the_model_and_keeps_the_turn` — Stop as Streamlit delivers it, garbage collector off, real `MLXChatModel` over a fake MLX |
| no test loads a real model | conftest's `_no_real_models`, pinned by `tests/test_offline_guard.py` |
| a question is one trace, ended however the question ends (answered, failed, stopped, closed unread) | `tests/test_tracing.py`, plus `test_a_stop_during_retrieval_still_sends_the_questions_trace` in `test_app.py` |
| no test leaves tracing on | conftest's `_no_tracer_left_on`, after every test |
| the adapters implement their models' official recipes | `tests/test_models_live.py` (`-m models`, by hand on a Mac) — reproduces the model cards' published scores |

The cheap-imports, greedy-decoding and foreign-document rows replaced text rules
(`lazy-cli-imports`, `no-sampling-params`, `no-rmtree`) and are each strictly
stronger than the regex they retired.

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
  subclass of it. Nor is `cli.py`'s `KeyboardInterrupt` arm ("Interrupted.",
  exit 130): Ctrl-C is how a slow local answer is abandoned, not a failure.
- Nothing on the pipeline-load path may raise `ValueError`, or it lands above
  the sidebar as a configuration error. That is why every adapter's
  *construction* raises only `FileNotFoundError` (model not cached, naming the
  `hf download` command) or `RuntimeError` (MLX missing, a failed load, a model
  of the wrong family, an out-of-range `EMBEDDING_DIMENSIONS`/`MAX_TOKENS`/
  `top_n`) — mlx-lm's own `ValueError` for an unsupported model type and
  huggingface_hub's `HFValidationError` are translated. The pydantic adapters load
  in a `model_validator(mode="after")` (langchain reserves `model_post_init`),
  and raise `RuntimeError` there deliberately: pydantic would wrap a
  `ValueError` in a `ValidationError`.
- Model failures are translated where the model runs — in the adapters in
  `mlx_models.py`. Embedding and reranking failures become `RuntimeError`.
  `MLXChatModel._stream` passes `RuntimeError`/`ValueError` through, wraps
  anything else (a jinja `TemplateError`, say) in `RuntimeError`, and lets
  `BaseException` — Streamlit's stop signal, `GeneratorExit` — through
  untouched. A message type it cannot map, `stop=`, or any generation kwarg is a
  `ValueError` at *call* time, refused rather than silently ignored.
- Generation-level checks live in `_generate()` and nowhere else — today the
  empty-answer guard that stops a frontend presenting no content under a full
  citation list, and the note appended to an answer the model cut off at
  `MAX_TOKENS` (its `finish_reason` is `"length"`), which would otherwise read
  as complete. `stream_answer()` wraps `_generate()` and `answer()` joins
  over that, so every shape inherits it. A new check belongs here rather than in
  a frontend: the one that goes in `app.py` is the one `cli.py` silently doesn't
  get. Note failures surface during *iteration*, not at the `.stream()` call:
  the chain is lazy, so a `try` around the call alone would catch nothing.
- `stream_answer()` returns `(docs, chunks)` because every frontend needs both,
  and handing back the docs the answer was actually generated from is what stops
  displayed citations from drifting via a second search. Its two halves settle at
  different times — retrieval has run when it returns, generation has not — which
  is what lets a caller wrap a spinner around just the call. The app adds a
  second spinner until the *first* chunk arrives, since the model reads the
  whole prompt (several seconds; ~15 s on the first answer after loading)
  before it writes anything. The stream is a `Generator` because a caller that
  stops early must `close()` it (see the concurrency paragraph).
- `RAGPipeline.__init__` checks `FETCH_K >= 1` (a `RuntimeError`), then calls
  `require_index()` before building any model, so a fresh checkout is told to
  run `rag ingest` without first loading ~22 GB. Three cases, each a
  `FileNotFoundError` naming the fix, checked in order: no database in the
  persist directory (first, because opening a client would create one); no
  collection of that name (a wrong `COLLECTION_NAME` is a *different*
  collection, and one that silently searched empty would answer every question
  "I don't know"); and a collection holding none of this pipeline's chunks
  (scoped by `OWN_CHUNKS`, so foreign records do not pass for an index). Only
  then `open_store(create=False)`, the reranker, and the LLM. Keep that order.
- Store failures are translated in `store_errors_as_runtime` (`ingest.py`): it
  wraps every store op — opening the collection, ingest's reads, deletes and
  adds, and the query-time search — mapping `chromadb.errors.ChromaError` to
  **RuntimeError, never ValueError**, so a store failure while the app loads
  its pipeline lands below the sidebar. A message mentioning a dimension gets a
  hint: new `COLLECTION_NAME` (or delete that collection), then `rag ingest` —
  never delete the persist directory. `ChromaError` only; see Gotchas for
  chromadb's builtins errors. `open_store()` also maps `NotFoundError` to
  `FileNotFoundError`.
- `MLXChatModel` has no `temperature`/`top_p`/`top_k` fields and passes no
  sampler: decoding is greedy, so the same question over the same retrieved
  context gets the same answer — grounding comes from the context, and an answer
  that changes from run to run is harder to check against it. Thinking stays off
  (`enable_thinking=False`) and `max_tokens` stays explicit. Don't add sampling
  params.
- Env-var helpers in `config.py` treat set-but-empty (`CHAT_MODEL=`) as unset and
  fall back to the default, and report a malformed value as `ValueError` — for a
  path too, where pathlib's own signal (a `~user` with no home directory) is a
  `RuntimeError` that would slip past `app.py`'s guard above the sidebar. Match
  that behavior for new settings.
- `load_documents()` warns on stderr for unreadable files and *silently* skips
  whitespace-only ones, rather than aborting the ingest. Preserve that resilience.
- Document `source` metadata (path relative to `data_dir`, POSIX-style) is what
  citations key off. Any new loader must set it. Chunk metadata is exactly
  `source`, `content_hash` and `ingested_by`, with `str`/`int`/`float`/`bool`
  values only — Chroma rejects anything else.
- Module and function docstrings explain *why*, not what. Match that register.
