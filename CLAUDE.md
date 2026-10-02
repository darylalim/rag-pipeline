# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                              # install deps (creates .venv; MLX only on macOS)
uvx --from huggingface_hub hf download <model id>   # once per model (README Setup lists the three); loading never downloads
uv run rag ingest                    # embed data/ into the Atlas collection (needs MONGODB_URI)
uv run rag query "your question"     # ask from the terminal (loads the chat model first)
uv run rag eval                      # score the pipeline on evals/questions.json (needs LANGSMITH_API_KEY + ANTHROPIC_API_KEY; indexes evals/corpus itself; ~20 min, ~$1-2)
uv run rag eval --save-baseline      # ...and save the scores as evals/baseline.json
uv run streamlit run streamlit_app.py # chat UI over the same pipeline
uv run streamlit run streamlit_app.py --server.fileWatcherType auto   # while editing streamlit_app.py (config.toml turns the watcher off)
uv run pytest                        # full suite (fakes + atlas-local in Docker; no models, network or secrets; ~4 min)
uv run pytest -m models              # live tests against the real models (Apple Silicon + models downloaded; ~1 min)
uv run pytest tests/test_config.py::test_defaults   # single test
uv run pytest -k idempotent -v                      # by keyword
uv run pytest --cov=rag_pipeline --cov=streamlit_app --cov-report=term-missing   # coverage, on demand
uv run ruff check --fix . && uv run ruff format .   # lint, then format (order matters)
uv run ty check                      # type check
uv sync --locked && uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest   # every check CI runs
actionlint .github/workflows/ci.yml  # after editing ci.yml (Homebrew's, with shellcheck on PATH for the run: scripts)
uv version --bump minor              # release: commit pyproject.toml + uv.lock, push to main; CI tags it and makes a GitHub release (nothing goes to PyPI)
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

Lint runs ruff's own default rule set plus the families pyproject's
`extend-select` adds (a `select` would replace that default rather than extend
it), and the tree is clean against it. **Fix findings rather than suppressing
them** — no `# noqa`, `# ty: ignore`, `# type: ignore`, or any other form ruff
or ty honours in source. The README's rule table lists them all, and
`no-suppressions` rejects them. A rule that does not fit a whole scope (pytest's
`assert` in `tests/`, bandit's password-name heuristics where a token is the
tokenizer's) is switched off in pyproject's `ignore` or `per-file-ignores`,
with its reason beside it — never at the line. Prefer `uv run ruff`/`uv run ty` over `uvx`, so
versions match the lock. Ruff's line length and ty's target version are both
inherited (from the default and from `requires-python`) — don't re-pin them in
`pyproject.toml`.

`README.md` covers setup, configuration variables, usage, performance, and what
CI runs; `ci.yml`'s own comments cover why it is configured as it is. Consult
both rather than duplicating that material here. Every CI job must stay green.
CI does not lint its own workflow, and the release job runs only on a push to
main, so a broken step in it is first seen there — hence `actionlint` above.

## Architecture

Two phases with a hard boundary between them, one shared config object, and the
local models behind LangChain's interfaces:

```
ingest  (rag_pipeline/ingest.py)      load → split → embed → store (MongoDB Atlas + vector index)
query   (rag_pipeline/pipeline.py)    embed question → search → rerank → stuff prompt → local LLM
models  embeddings + rerank: Voyage AI's API (factories in ingest.py / pipeline.py) · chat: MLXChatModel, over MLX (mlx_models.py)
tracing (rag_pipeline/tracing.py)     optional: each question as one trace, to a self-hosted Phoenix
eval    (rag_pipeline/evaluation.py)  rag eval: evals/questions.json as a LangSmith dataset, judged by Claude
```

`Settings` (`config.py`) is a frozen dataclass built via `Settings.from_env()`.
Both frontends — `rag_pipeline/cli.py` and `streamlit_app.py` — construct it the same way,
which is what keeps them agreeing on the database, collection and vector
index, the models, and chunking. Every setting is a field with a literal default.

Credentials are not settings. A key has no literal default to document, and a
field is a value the sidebar displays and a traceback prints, so no key is ever
a `Settings` field (`test_a_credential_never_reaches_settings`). The stage that
needs a key reads it where it is used, through `config.require_env_key()`: a
`RuntimeError` naming the variable when it is unset or empty, which lands in
the pipeline-load guard. A key is documented in `.env.example` and the README in
the same change as the code that first reads it, not before — until then the
docs would ask for a key nothing uses.

All tunables live here — never inline a literal at a call site. Adding one is a
**three-file change**:

1. the field plus its `_env_*` line in `config.py`,
2. a commented default in `.env.example`,
3. a row in the README config table.

Leaving either of the latter two stale is a bug that only
`test_every_setting_is_documented` catches (documented defaults included):
`ruff`, `ty` and the rest of the suite stay green against a stale
`.env.example` or README.

There is no fourth site. `config.ENV_VARS` derives every variable name from the
dataclass fields, and `tests/test_config.py` clears *that* rather than a
hand-kept list. This matters because `config.py` loads `.env` at import time
(see Gotchas): a name missing from a hand-kept list would be answered by the
developer's own `.env`, so its default would silently stop being tested. Derived,
that drift is not merely detected — it is inexpressible.

The Voyage clients' retry budget and timeout (`_VOYAGE_ATTEMPTS`,
`_VOYAGE_TIMEOUT_S` in `ingest.py`) and the store's internal waits are
deliberately *not* settings: they bound how the pipeline copes with a service,
not what it computes.

### Evaluation (`rag eval`)

`evaluation.py` runs the real pipeline over `evals/questions.json` through
`langsmith.evaluate` and scores each answer `retrieval_hit` and
`retrieval_rank` (computed from the chunks in the prompt; the rank, a reciprocal
rank, because the local stack scored a 100% hit rate and a hit rate cannot see a
right file slipping from first place), `correct` and `grounded` (judged by
Claude). It exists so the migration to
hosted models can be measured: each phase is compared with `evals/baseline.json`,
scored on the local stack.

The questions are about `evals/corpus/`, a fictional company's handbook, which
`run()` ingests through `eval_settings()` — the caller's models and chunking,
`EVAL_CORPUS` and its own `EVAL_COLLECTION` — so the user's index is never read
or written. Fictional, so an answer cannot come from the model's own knowledge;
confusable (services documented alike, a deprecated guide contradicting the
current one); and kept at least twice `FETCH_K` in chunks
(`test_the_eval_corpus_outnumbers_what_vector_search_fetches`), or every chunk
reaches the reranker and the embedder goes untested. The sample `data/` made 9
chunks and scored 100% on everything. Editing the corpus or the questions means
a new baseline.

Two things make scores comparable, and both are deliberate:

- **The dataset name is a digest of the questions** (`dataset_name()`), so an
  edited question set is a new dataset, never a dataset edited in place under
  old experiments. `format_report` refuses to compare across datasets.
- **The judge is a constant, `JUDGE_MODEL`, not a setting**, and `ClaudeJudge`
  passes no fallback model: a judge that changed mid-run, or per run, would make
  scores incomparable. A refused or unparseable grade is a `RuntimeError` that
  LangSmith records against the example as unscored, and `--save-baseline`
  refuses any run with an unscored example.

Its two keys are read through `require_env_key` inside `run()`, which checks the
question set, both keys, the judge and the dataset before any model loads.
`anthropic` and `langsmith` are imported only there (cli imports
`evaluation` inside `cmd_eval`; `anthropic` is in `test_cli.py`'s `HEAVY`). The
tests stand in for LangSmith through the `DatasetStore` protocol and for Claude
through the `Judge` callable; nothing in the suite calls either service.

### Why the store factories live in `ingest.py`

`build_embeddings()` and `open_store()` are defined in `ingest.py`; `pipeline.py`
imports `open_store()`, never the reverse, and reaches `build_embeddings()` only
through it. This is deliberate: vectors from different
embedding models are not comparable, and the store's identity is (`MONGODB_URI`,
database, collection, vector index, embedding function). Indexing and querying
must therefore go through one factory each. **Never construct
`MongoDBAtlasVectorSearch(...)`, `MongoClient(...)` or `VoyageAIEmbeddings(...)`
inline** — route through these factories (`tests/conftest.py`'s administration
of the test container is the one exemption). `open_store()` returns the langchain
`MongoDBAtlasVectorSearch`, constructed with `auto_create_index=False` so the
query path creates nothing; bookkeeping that needs no model goes through
`_collection()` and `_meta()`, the raw pymongo handles. Both translate pymongo's
`InvalidName` — raised for a refused database or collection name when the
*handle* is made, before any operation — so they wrap their own construction in
`provider_errors_as_runtime`.

The reranker is the deliberate exception: `build_reranker()` lives in
`pipeline.py`, not here. Reranking is query-only — it has no ingest-side
counterpart, so the "same model must serve both phases" reason that pins the
embedding/store factories here simply does not apply. It sits beside
`build_chat_model()`, the other query-time model factory. This is enforced by an
ordinary behavioral test (the MLX block + the injection seam), not a text
invariant, because the risk it guards — offline testability — is one a behavioral
test already covers.

### Voyage AI: the embedder and the reranker

`build_embeddings()` returns langchain-voyageai's `VoyageAIEmbeddings` at
`output_dimension=EMBEDDING_DIMENSIONS` (256, 512, 1024 or 2048 — anything else
is a `RuntimeError` before any call), and `build_reranker()` its
`VoyageAIRerank` with `top_k=RETRIEVAL_K` (below 1 is a `RuntimeError`). Both
read `VOYAGE_API_KEY` through `require_env_key`, so a missing key is the
`RuntimeError` the pipeline-load guard catches, and both get their clients from
`ingest.voyage_clients()`: langchain-voyageai builds its own with voyageai's
defaults — one attempt, no timeout — and offers no setting for either, so the
factories replace them (`test_the_voyage_clients_retry_and_time_out`). voyageai
retries rate limits, unavailability and timeouts with exponential backoff, and
re-raises the last `VoyageError` as itself, which `provider_errors_as_runtime`
turns into a `RuntimeError`. Every Voyage call happens inside that context
manager: ingest's adds and width probe, the question's embedding in the search,
and the rerank in `retrieve()`. The reranker is `pipeline._VoyageRerank`, a subclass
whose `compress_documents` returns the candidates themselves with only
`relevance_score` added: langchain-voyageai's rebuilds each document without its
id (`source:index:content_hash`, which the reranker trace span records) and adds
a `total_tokens` key to its metadata. It calls the library's private `_rerank`,
so `test_the_reranker_returns_the_candidates_themselves_in_voyages_order` pins
it offline and the live suite checks it against the real API.

A Voyage account with no payment method is held to 3 requests and 10,000 tokens
a minute — less than one ingest slice — and fails with `RateLimitError` even
after the retries. Its free tokens apply with a payment method added.

### The chat model loads once per process (`mlx_models.py`)

`build_chat_model()` constructs `MLXChatModel`, which gets its weights from
`load_mlx_model()` — the only place weights load. It is memoized by resolved
snapshot path under a double-checked lock, so concurrent first calls from
Streamlit sessions load once. That memo is what makes the factory cheap to call
again: the app rebuilds its pipeline after every ingest and wraps the weights
already in memory rather than loading a second copy of ~15 GB. Never call
`mlx_lm.load` anywhere else.

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

Concurrency: one module-level `_GENERATION_LOCK` is held for a whole
generation, because mlx-lm
sets and restores the process-wide Metal wired limit around each one and
overlapping calls race on it. mlx-lm's stream is closed while that lock is still
held, and the lock is released however the stream ends — exhausted, failed, or
closed half-way. **A stream that is dropped rather than closed keeps the lock
until the garbage collector finalizes it**, which for one a Streamlit script
holds as a module global can be never — every later question, from every
session, then hangs on the lock. So everything between the model and a frontend
closes rather than drops: `streamlit_app.py` wraps generation in `closing(chunks)` (the
Stop button raises inside `st.write_stream` with the stream suspended),
`stream_answer`'s tracing wrapper (`_traced`) closes `_generate`'s stream with
its own, and `RAGPipeline._generate` closes the chain's stream with its own. That chain is
`_PROMPT | llm`, with **no `StrOutputParser`**: closing a parser's stream does
not stop the model — langchain-core catches the `GeneratorExit` and drains the
parser's input, i.e. generates on to `MAX_TOKENS` under the lock.
`test_a_real_stop_releases_the_model_and_keeps_the_turn` delivers Stop the way
the button does, with the garbage collector off, over the real `MLXChatModel`.

MLX itself is reached only in the generation loop and `_release_buffers`, so
the prompt, locking, stopping and error translation are all unit-tested in CI
(`test_mlx_models.py`, over the shared fake MLX stack in `tests/fake_mlx.py`).
What the fakes cannot check — that the real chat model streams a grounded
answer with thinking off, and that the Voyage factories embed documents and
questions the right way round at the configured width — is
`tests/test_models_live.py` (`-m models`): run it after touching
`mlx_models.py` or a factory, after a `uv.lock` change that moves `mlx`,
`mlx-lm`, `mlx-metal`, `transformers`, `tokenizers`, `huggingface-hub`,
`voyageai` or `langchain-voyageai`, and after re-downloading the model: the
fakes stay put while all of those move. It calls Voyage's real API, so conftest
exempts the `models` mark from `_offline` and lets it keep `VOYAGE_API_KEY`.
Skipped is not passed.

### Tracing is the API in the pipeline, the SDK in the frontends

Off unless `PHOENIX_COLLECTOR_ENDPOINT` is set (empty default; `_env_url` refuses
a malformed one as `ValueError`). OpenTelemetry's own split: `pipeline.py`
imports only `opentelemetry-api` and OpenInference's attribute names, which are
a no-op until a provider exists, and `tracing.setup_tracing()` installs one.
`cli.py` calls it in `cmd_query` only (ingest emits no spans, so it would only
load the tracing stack and start an idle exporter thread), and `streamlit_app.py` on every
rerun, inside the pipeline-load `try`; it is once per process, guarded by the
instrumentor's own state under a lock (`test_concurrent_first_setups_install_once`).
Like the adapters, it raises only `RuntimeError`, because the app calls it on
the pipeline-load path: the builtins `ValueError` a malformed `OTEL_*` variable
can cause, and the exporter's `RuntimeError` for a credential provider that is
not installed, become one `RuntimeError` pointing at the `OTEL_*` variables. It
installs nothing until everything is built, and shuts down a provider already
built when the failure comes, or each failed rerun would leave one more exit
hook (`test_a_malformed_otel_variable_is_a_runtime_error_that_installs_nothing`
watches `atexit`). Its imports are lazy, so tracing off loads none of the stack
(`test_tracing_off_loads_none_of_the_tracing_stack`, which takes the frontends'
path: import, then `setup_tracing(Settings())`).

`setup_tracing` assembles the provider from OpenTelemetry's parts, **not
`phoenix.otel.register()`**, whose shortcuts are traps here — `tracing.py`'s
module docstring lists them (a base URL posted to as-is, whose 405
`force_flush()` reports as success; gRPC, whose sockets `_offline` cannot see;
no exporter timeout; `PHOENIX_*`/`.env.phoenix` that `Settings` does not know).
So `traces_url()` always builds the collector URL: it appends `/v1/traces` to a
base URL, keeping a path prefix for a reverse proxy, and leaves an endpoint that
already ends in it as is. The `BatchSpanProcessor`, `_EXPORT_TIMEOUT_S = 2` and
OpenInference's `TracerProvider` (the SDK's caps a span at 128 attributes, which
a reranker span passes from a `FETCH_K` of about 37) are each justified in the
comments beside them, and
`test_with_phoenix_down_questions_do_not_wait_and_exit_waits_briefly` measures
what batching and the timeout buy. The exporter timeout is the only bound on the exit flush:
`force_flush(timeout)` and `OTEL_BSP_EXPORT_TIMEOUT` are ignored.

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

### One MongoDB client per process

`_client()` is the one `MongoClient` construction, kept in `_clients` keyed by
(`MONGODB_URI`, `MONGODB_TIMEOUT_MS`) under `_clients_lock`, so concurrent first
opens from several Streamlit sessions build one client
(`test_concurrent_first_opens_share_one_client`). It reads `MONGODB_URI`
through `require_env_key` (a credential, not a setting) and pings on creation,
because pymongo connects lazily: a paused cluster or an IP missing from the
access list would otherwise surface deep inside an ingest or a question. A
client whose ping failed is closed and not kept, so every later attempt reports
the same failure (`test_an_unusable_cluster_is_a_runtime_error_each_time`).

A client is a connection pool that always reads the server's current state, so
**nothing in production resets it**. That inverts the Chroma era's rule, whose
cached on-disk view had to be dropped before every pipeline build:
`reset_store_cache()` now *closes* the clients, and closing one breaks every
pipeline still holding it — an outgoing pipeline answering in another session.
So `load_pipeline` and `ingest()` never call it
(`test_no_pipeline_build_closes_the_shared_store_client`,
`test_an_ingest_leaves_a_held_store_searching`), and a held store sees another
process's ingest with no reset at all
(`test_a_store_held_by_this_process_sees_another_processs_ingest`). Only the
tests call it, at every boundary (conftest's `_reset_store_client`), so each
starts as a fresh process would.

The Streamlit cache key is `index_version()` (a `str`) — a SHA-256 digest over
the corpus fingerprints that `ingest()` stamps into a document of the
`rag_pipeline_meta` collection (`_META_COLLECTION`, keyed `version:<collection>`),
beside the chunks rather than among them, so nothing that reads, counts, deletes
or searches chunks can meet it. It changes on exactly the events an edit, add,
or removal changes and no others, so the running app picks up a `rag ingest`
without a restart and an unchanged re-ingest does not needlessly bust the cache.
`ingest()` reconciles it on *every* run — compared, and written only when it
differs — because a run that died after its writes but before the stamp leaves
every source looking current, and a stamp written only "when something changed"
would then never be repaired. It returns `""` when nothing has been ingested,
and creates nothing (finds and `distinct` over a missing collection are simply
empty), which matters because the app calls it on every rerun.

### Ingest is incremental, and scoped — never a wholesale wipe

`ingest()` leaves the collection holding exactly the chunks for whatever is in
`data_dir` right now. It gets there by re-embedding only what changed: each
chunk carries its source's `content_hash` in its metadata, and a source whose
hash still matches keeps the vectors it has. Chunks are keyed by a deterministic
id (`source:index:content_hash`) and added through langchain-mongodb's
`add_documents(ids=)`, which writes `ReplaceOne(..., upsert=True)` — a raw
`insert_many` would refuse an existing id — so re-adding is an idempotent
replace, not a duplicating append. Its metadata is written as top-level fields
(`source`, `content_hash`, `ingested_by`) beside `text` and `embedding`.

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
- A vector index serves one width (`numDimensions`), and MongoDB itself stores
  a vector of any width, so a mismatch would surface only at search time —
  after a re-ingest had deleted the chunks it was replacing. Hence three checks
  before any write, each a `ValueError` naming its own fix. When there is
  something to embed: the live model's probe width against
  `EMBEDDING_DIMENSIONS` (set it to the model's width), then that width against
  what this pipeline's chunks already hold (`_stored_width`, scoped to
  `OWN_CHUNKS`: a new `COLLECTION_NAME`). On every run: the existing index's own
  definition (`_check_vector_index`) — its width, and that it declares
  `ingested_by` and `source` as filter fields, since `$vectorSearch` refuses a
  pre-filter on an undeclared field. An index this pipeline did not make is
  refused, never rebuilt in place: it may be another tool's. A search at the
  wrong width is Atlas's own `OperationFailure`, which `provider_errors_as_runtime`
  turns into a `RuntimeError` with the `COLLECTION_NAME` hint.
- `_ensure_vector_index` runs on every run, after the deletes and before the
  adds: it creates the collection (Atlas refuses a search index on a collection
  that does not exist) and the index if either is missing — so an index dropped
  by hand is rebuilt — then waits until it is queryable. After the adds,
  `_await_searchable` waits until a written chunk is returned by an exact
  `$vectorSearch` within its own source: Atlas indexes writes asynchronously,
  and the app answers about an upload on the same run that ingested it. Both
  waits poll every `_INDEX_POLL_INTERVAL_S` (0.25 s); it sets the suite's run
  time, since every store test builds an index.
- Every read, delete **and search** is scoped to `OWN_CHUNKS`
  (`{"ingested_by": "rag-pipeline"}`, a marker every chunk carries), so a
  collection shared with unrelated records is never read, counted, deleted from
  or cited — the document-level form of "never a wipe". A dedicated key matched
  by equality rather than a guess at what only our chunks have: the obvious one,
  `{"content_hash": {"$ne": ""}}`, also matches records that *lack* the key: a
  delete built on it removed a foreign document. `OWN_CHUNKS` is spelled
  `{"ingested_by": {"$eq": "rag-pipeline"}}` because that form serves both a
  find and `$vectorSearch`'s `pre_filter`. Deletes are
  `{"$and": [OWN_CHUNKS, {"source": {"$in": superseded}}]}`, issued only when
  `superseded` is non-empty.
- The whole read → delete → add → stamp sequence runs under `_WriterLock` — and
  so does the read of `data_dir` that decides it. Two writers on one collection
  at once — a terminal `rag ingest` overlapping an upload in the app, or two
  machines — would each apply its own reading of the index, one deleting what
  the other just added. And a run that read `data_dir` before waiting would
  apply that older snapshot after the writer it waited on. The lock is a lease
  document in `rag_pipeline_meta` (`ingest-lock:<collection>`), so it binds
  every process and machine; its expiry is computed by the server (`$$NOW`),
  so machines whose clocks differ agree on it. Acquiring is one
  `find_one_and_update(upsert=True)` matched by `_id` alone, whose update
  pipeline takes the lock only if it is missing or expired — MongoDB refuses
  `$expr` in an upsert's query, so the "is it free" test cannot go in the
  filter — and the owner it returns says who won. The holder renews the lease
  (`_LOCK_LEASE_S`, 300 s) after every slice of adds; a renewal that finds the
  lock taken over (it stalled past the lease) raises before writing again, and
  a writer that dies frees the lock when its lease runs out. The embedder is
  built *before* the lock (a multi-second load no other writer should wait on).
  Readers need no lock.
- New chunks are added in slices of `_ADD_SLICE` (256), each embedded and written
  before the next, renewing the lock between them: no slice outlasts the lease,
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
`BaseDocumentCompressor`. This is why the suite needs no model, no API key and
no Apple Silicon: the real `build_embeddings()`/`build_reranker()` call Voyage's
paid API and `build_chat_model()` loads a 15 GB MLX checkpoint, but none runs
under test. Any new code path touching an embedding model, the reranker, or the
LLM should thread these through rather than constructing them unconditionally.

Injection is a convention, so `conftest.py` backs it with autouse guards, each
pinned by `tests/test_offline_guard.py` so one that loosens reads as a failure: `_no_real_models` makes MLX unimportable (**the one that catches a
forgotten chat-model injection** — with the model cached, a network block alone
would let it load), `_no_real_store` removes the developer's real `MONGODB_URI`
and API keys (so a forgotten embedder or reranker injection stops at the
missing `VOYAGE_API_KEY`),
`_offline` blocks every socket to a host other than this machine (the atlas-local
container is on loopback), `_no_tracing` keeps Phoenix and
LangSmith off whatever `.env` says, and `_no_tracer_left_on` fails a test that
left tracing on (which `_offline` cannot see: the tracing stack swallows the
socket error). How each works, and which fixture a new test takes, is in
`tests/CLAUDE.md`.

Tests marked `@pytest.mark.models` are the deliberate exception: they load the
real models, are deselected by pyproject's `addopts = ["-m", "not models"]`, and
run only with `uv run pytest -m models` (a later `-m` replaces the default). They
skip rather than fail when MLX or a model is missing. CI never runs them.

Neither frontend takes such parameters — `streamlit_app.py` is a script, and `cli.py`
builds its own `Settings.from_env()` — so conftest's `wired_env` is the seam for
both: it exports the fixture settings through `ENV_VARS` and patches
`ingest.build_embeddings`, `pipeline.build_chat_model`, and
`pipeline.build_reranker` on their modules. A new frontend test takes it rather
than copying it. That
works only because all are looked up as module globals at call time, which is a
second reason the never-construct-inline rule above is load-bearing: inline a
`VoyageAIEmbeddings(...)` anywhere and the frontend can no longer be driven with
fakes at all, not just inconsistently. `st.cache_resource` is cleared per test,
since its key deliberately ignores `_settings` and would otherwise serve one
test's pipeline to the next.

What `test_streamlit_app.py` is *for* is the set of guarantees no lower-level test can see,
because they are only observable at the frontend:

- A chat turn is stored as a user/assistant **pair** whatever happens to it —
  success, a failed generation, or the run being torn down mid-answer — so a
  question can never be left in the history with nothing under it. This is what
  the `finally` in `streamlit_app.py` buys, and the reason a failed turn is stored with an
  `error` flag rather than as ordinary text. The `finally` writes through a
  `history` bound before the turn, never through `st.session_state`: after a
  real Stop the runner *stays* stopped, and every `st.session_state` access is
  a yield point that raises `StopException` again. (`fail_mid_stream` raises
  from inside generation and cannot show that; the real-Stop test calls
  `script_requests.request_stop()`, as the button does.)
- A stopped answer releases the model: see the concurrency paragraph above.
- A Stop during *retrieval* still closes the stream. It is raised at the
  retrieval spinner's exit (a Streamlit call), after `stream_answer` returned
  and before generation starts, so `streamlit_app.py` registers `closing(chunks)` inside
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
- No pipeline build, upload or rerun closes the shared store client, which
  another session's pipeline may still be searching through.
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
  is load-bearing: importing them pulls in pymongo, langchain-mongodb and the
  langchain stack, so `rag --help` and a usage error stay cheap. Keep
  those imports local — which also keeps them inside `main()`'s `try`, the one
  place a library's import-time `ValueError` (next) is reported.
- Some libraries read settings from the environment once, as they are first
  imported, and refuse a malformed one with a `ValueError` — tracing on or off.
  langsmith, inside langchain-core, imports the OpenTelemetry SDK, whose
  `opentelemetry.sdk.trace` validates `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`; numpy,
  langsmith and huggingface_hub (through transformers, on a Mac) each `int()` a
  variable or two of theirs (`HF_HUB_ETAG_TIMEOUT`, say). So `streamlit_app.py` imports the
  pipeline inside its `Settings` guard, not at the top of the file, where any of
  these was a traceback in place of the whole page. It does so once per process,
  through the cached `_import_failure()`, which keeps a failure as surely as a
  success and logs its traceback: a failed import is not safely repeatable —
  numpy's second attempt is a `RecursionError` — and a library's message need
  not name its variable. Only a fresh interpreter shows any of this — the suite
  imports all of it before its first test — so
  `test_a_variable_refused_at_import_*`, in `test_streamlit_app.py` and `test_cli.py`, run
  the frontends through conftest's `fresh_interpreter`.
- MLX is **macOS-only**: `mlx` and `mlx-lm` are declared
  `; sys_platform == 'darwin'`, so the Linux CI legs never install them. Every
  MLX import is therefore lazy, inside `mlx_models.py`'s functions — a
  module-level one breaks collection on Linux. ty types them as `Any` on every
  platform (`replace-imports-with-any` in `pyproject.toml`), so a Mac and CI give
  the same answer; the price is untyped MLX call sites, which is why they are
  kept thin. It covers `mlx.**`/`mlx_lm.**` only — not a place to park other
  unresolved imports.
- MLX keeps freed buffers for reuse, and with varying shapes that cache grew to
  several GB beside a model. `_release_buffers()` (`mx.clear_cache()`) runs
  after every generation; a new MLX call path must do the same.
- The Qwen3.8 chat template treats an unset `enable_thinking` as *on*: it adds a
  reasoning instruction and the model streams raw reasoning into the answer with
  no `<think>` tag to strip. `MLXChatModel` passes `enable_thinking=False`;
  filtering afterwards cannot work. mlx-lm's `stream_generate` also defaults to
  `max_tokens=256`, so it is passed explicitly.
- Atlas Vector Search is **asynchronous** twice over, and each is silent: an index
  is queryable some time after it is created, and a write is searchable some
  time after it lands — until then a search returns nothing, with no error.
  `ingest()` waits out both (see the ingest section). A test that writes a
  document directly and then asserts a search does *not* return it must first
  see an unscoped search return it, or it passes only because the index has not
  caught up (`test_retrieve_never_returns_a_foreign_document` does).
- MongoDB refuses `$expr` in an upsert's query (code 224): a conditional upsert
  decides inside its update pipeline instead, as `_WriterLock.try_acquire` does.
  And atlas-local's `update_search_index` treats a definition as a full-text one
  ("mappings is required"), so a test that needs a different vector index drops
  the old one (asynchronously — wait for it to go) and creates the new one.
- Reads create nothing: a find, `distinct` or `list_collection_names` on a
  missing collection is simply empty, and MongoDB creates a collection only on
  its first write — which only `ingest()` makes. `require_index()` and
  `index_version()` rely on that, and the tests check it
  (`test_index_version_is_empty_before_any_ingest_and_creates_nothing`).
- Don't call `reset_store_cache()` outside the tests (see "One MongoDB client per
  process"), and don't remove the ingest's `_WriterLock`: two writers at once are
  silent until a later query finds the index missing what one of them added.
- `rag_pipeline/__init__.py` sets `TRANSFORMERS_NO_ADVISORY_WARNINGS` before
  anything imports LangChain, which on a Mac imports transformers (with mlx-lm,
  without torch) and prints a false "PyTorch was not found". It must stay in the
  package `__init__`, the earliest import on every entry point;
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
- `.streamlit/config.toml` keeps Streamlit local and quiet: usage statistics,
  the first-run email prompt and the file watcher off, and `server.address =
  127.0.0.1` (unset, the uploader — which writes into `data/` — is on the local
  network with no login). Each setting's reason is beside it, and
  `test_the_app_config_keeps_streamlit_local_and_quiet` pins all four. With the
  watcher off an edit to `streamlit_app.py` is not picked up at all — Rerun reuses the
  compiled script until every tab has closed or the server restarts; see
  Commands.

## Enforcing the invariants

The text-level rules live in `tests/invariants.py` as data, and
`tests/test_invariants.py` enforces them across every `.py` file git does not
ignore, whether or not it has been added yet. **That test is the only
enforcement**: it runs locally and in CI for every contributor, and there is no
hook or editor layer. How to add a rule, and the two properties of its masking
that are easy to break, are in `tests/CLAUDE.md`.

**Prefer a behavioral test to a rule.** A rule matches spellings; a test
observes the property, so it covers routes nobody thought to enumerate. Reach
for `RULES` only when there is nothing to observe — which is the case exactly
when the point is that some call never happens (`store-factory`,
`embeddings-factory`, `no-suppressions`). Everything else is asserted where the
behavior is:

| Invariant | Enforced by |
| --------- | ----------- |
| the exception union, the empty-collection guards, `source` metadata on loaders | `test_pipeline.py`, `test_ingest.py`, `test_mlx_models.py` |
| `cli.py`'s imports stay cheap | `test_importing_cli_does_not_load_the_heavy_stack` — subprocess-imports the module, asserts pymongo/langchain_mongodb/mlx/mlx_lm/anthropic are absent from `sys.modules`, with the same probe required to see the store stack once the pipeline is imported, so it cannot pass vacuously |
| the chat model decodes greedily, with thinking off | `test_generation_is_greedy_with_thinking_off_and_explicit_max_tokens` — asserts the exact arguments generation is called with, so no sampler reaches mlx-lm by any route |
| `ingest()` never deletes documents it did not write | `test_ingest_preserves_foreign_documents_in_a_shared_collection` — a foreign doc survives a rebuild that deletes |
| a stopped answer releases the generation lock, and its turn is still stored | `test_a_real_stop_releases_the_model_and_keeps_the_turn` — Stop as Streamlit delivers it, garbage collector off, real `MLXChatModel` over a fake MLX |
| no test loads a real model | conftest's `_no_real_models`, pinned by `tests/test_offline_guard.py` |
| a question is one trace, ended however the question ends (answered, failed, stopped, closed unread) | `tests/test_tracing.py`, plus `test_a_stop_during_retrieval_still_sends_the_questions_trace` in `test_streamlit_app.py` |
| no test leaves tracing on | conftest's `_no_tracer_left_on`, after every test |
| secrets (`.env`, `.env.*`, `.streamlit/secrets.toml`), the user's documents in `data/`, coverage's parallel data files and Claude Code worktrees stay out of git; `.env.example` and the three samples stay addable | `test_gitignore_keeps_secrets_and_your_documents_out_of_git` — the repo's `.gitignore` in a scratch repository made with no template, global excludes off |
| the real chat model and Voyage's API behave as the fakes assume | `tests/test_models_live.py` (`-m models`, by hand on a Mac with `VOYAGE_API_KEY`) |

The cheap-imports, greedy-decoding and foreign-document rows each replaced a
text rule and are strictly stronger than it: don't reintroduce one.

## Conventions

- New failure modes must fit `FileNotFoundError | RuntimeError | ValueError` —
  the union both frontends catch. `cli.py` catches it in one place (`main()`);
  `streamlit_app.py` splits it across two, because the sidebar has to render in between:
  `ValueError` from `Settings.from_env()`, or from the pipeline's imports (a
  malformed variable a library reads as it is imported; see Gotchas), stops the
  script above the sidebar, and `FileNotFoundError | RuntimeError` from the
  pipeline load is caught below it, so the uploader stays reachable when there
  is no index. A chat turn catches the union too, then anything else — a bug,
  not a failure mode — which it shows the same way but also logs with its
  traceback, since once caught it never reaches Streamlit's own log. Grep
  `except (FileNotFoundError` rather than trusting a line number. Don't add a
  fourth type — `_add_documents()` catching `OSError` is not one: it is the
  filesystem's own error on a write, and `FileNotFoundError` is already a
  subclass of it. Nor is `cli.py`'s `KeyboardInterrupt` arm ("Interrupted.",
  exit 130): Ctrl-C is how a slow local answer is abandoned, not a failure.
- Nothing on the pipeline-load path may raise `ValueError`: its guard catches
  only the other two, so one escapes as a traceback under the sidebar, every
  rerun. That is why every adapter's *construction* raises only
  `FileNotFoundError` (model not cached, naming the `hf download` command) or
  `RuntimeError` (MLX missing, a failed load, a model of the wrong family, an
  out-of-range `EMBEDDING_DIMENSIONS`/`MAX_TOKENS`/`top_n`) — mlx-lm's own
  `ValueError` for an unsupported model type and huggingface_hub's
  `HFValidationError` are translated. The pydantic adapters load in a
  `model_validator(mode="after")` (langchain reserves `model_post_init`), and
  raise `RuntimeError` there deliberately: pydantic would wrap a `ValueError` in
  a `ValidationError`.
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
  a frontend: the one that goes in `streamlit_app.py` is the one `cli.py` silently doesn't
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
- `RAGPipeline.__init__` checks `1 <= FETCH_K <= 1000` (a `RuntimeError`;
  `$vectorSearch` caps candidates at 10,000 and langchain-mongodb asks for ten
  per result), then calls `require_index()` before building any model, so a
  fresh setup is told to run `rag ingest` without first loading ~15 GB. Three
  cases, each a `FileNotFoundError` naming the fix, checked in order: no
  collection of that name (a wrong `COLLECTION_NAME` is a *different*
  collection, and one that silently searched empty would answer every question
  "I don't know"); a collection holding none of this pipeline's chunks (scoped
  by `OWN_CHUNKS`, so foreign records do not pass for an index); and no vector
  index named `VECTOR_INDEX_NAME`. An unreachable cluster or a missing
  `MONGODB_URI` is `_client()`'s `RuntimeError`. Only then `open_store()`, the
  reranker, and the LLM. Keep that order.
- Store failures are translated in `provider_errors_as_runtime` (`ingest.py`): it
  wraps every store op — connecting, getting a handle, ingest's reads, deletes,
  adds and index management, the writer lock, and the query-time search —
  mapping `pymongo.errors.PyMongoError` and `bson.errors.BSONError` (not a
  `PyMongoError`) to **RuntimeError, never ValueError**, so a store failure
  while the app loads its pipeline lands below the sidebar; a malformed
  `MONGODB_URI` (pymongo's `ConfigurationError`) lands there too. A message
  mentioning a dimension gets a hint: a new `COLLECTION_NAME` (or drop that
  collection's vector index), then `rag ingest`.
- `MLXChatModel` has no `temperature`/`top_p`/`top_k` fields and passes no
  sampler: decoding is greedy, so the same question over the same retrieved
  context gets the same answer — grounding comes from the context, and an answer
  that changes from run to run is harder to check against it. Thinking stays off
  (`enable_thinking=False`) and `max_tokens` stays explicit. Don't add sampling
  params.
- Env-var helpers in `config.py` treat set-but-empty (`CHAT_MODEL=`) as unset and
  fall back to the default, and report a malformed value as `ValueError` — for a
  path too, where pathlib's own signal (a `~user` with no home directory) is a
  `RuntimeError` that would slip past `streamlit_app.py`'s guard above the sidebar. Match
  that behavior for new settings.
- `load_documents()` warns on stderr for unreadable files and *silently* skips
  whitespace-only ones, rather than aborting the ingest. Preserve that resilience.
  It catches only `OSError | ValueError`: `_read_pdf` translates whatever pypdf
  raises on a malformed file — builtins errors included — into `ValueError`
  where pypdf runs, the adapters' pattern for their models. A new loader does
  the same for its parser. A new file type is its suffix in `SUPPORTED_SUFFIXES`
  (the uploader's accepted types derive from it) plus a branch in
  `load_documents()`; README's two hand-written `.md`/`.txt`/`.pdf` lists (under
  *Build the index* and *Add your own documents*) and the `data_dir` comment in
  `config.py` need it too, and no test checks them.
- Document `source` metadata (path relative to `data_dir`, POSIX-style) is what
  citations key off. Any new loader must set it. Chunk metadata is exactly
  `source`, `content_hash` and `ingested_by`, with `str`/`int`/`float`/`bool`
  values only. That limit is ours to keep, not MongoDB's: it stores lists and
  `null` too, and a chunk whose `ingested_by` came out `null` would fall outside
  `OWN_CHUNKS` for good.
- Module and function docstrings explain *why*, not what. Match that register.
