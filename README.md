# RAG Pipeline

A small, readable **Retrieval-Augmented Generation** pipeline built with
[LangChain](https://docs.langchain.com). Documents are embedded with **Voyage
AI**, stored and searched in **MongoDB Atlas Vector Search**, and answers are
generated with **Claude**. It ships with a reusable core library, a CLI, and a
Streamlit chat app — all sharing the same code.

```
Ingest (once):   data/ ──load──▶ split ──embed──▶ store (MongoDB Atlas Vector Search)
Query (per Q):   question ──embed──▶ search ──rerank──▶ [top-k chunks + question] ──▶ Claude ──▶ grounded answer + sources
```

Every step of a query is a network call: embedding the question and reranking the
candidates (both Voyage AI), the `$vectorSearch` itself (MongoDB Atlas), and
generation (Claude).

**Contents** — [Prerequisites](#prerequisites) · [Setup](#setup) ·
[Deploy a free Atlas cluster](#deploy-a-free-atlas-cluster) · [Usage](#usage) ·
[Add your own documents](#add-your-own-documents) ·
[Configuration](#configuration) · [Development](#development) ·
[Project structure](#project-structure) · [How it works](#how-it-works) ·
[Invariants](#invariants)

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.11+
- An **Anthropic API key** (generation) and a **Voyage AI API key** (embedding
  and reranking). Ingest embeds too, so it needs the Voyage AI key as well.
- A **MongoDB Atlas cluster** and its connection string (the vector store). The
  free tier is enough — see [below](#deploy-a-free-atlas-cluster).
- **Docker** — only to run the test suite, which uses a local Atlas container.
  Not needed to run the app itself.

## Setup

```bash
uv sync                      # create the venv and install dependencies
cp .env.example .env         # then add ANTHROPIC_API_KEY, VOYAGE_API_KEY, MONGODB_URI
```

Embedding calls the Voyage AI API — there is no local embedding model. The
`voyageai` SDK does fetch a small tokenizer from the Hugging Face Hub on first
use (for client-side token counting), cached under `~/.cache/huggingface`; the
*"unauthenticated requests to the HF Hub"* notice it prints is harmless.

## Deploy a free Atlas cluster

The vectors live in MongoDB Atlas. A free (M0) cluster is enough for a demo:

1. In the [Atlas console](https://cloud.mongodb.com), **Create** a cluster and
   pick the **Free** tier (still labelled `M0` in places), a provider, and a
   region near you.
2. In the **Security Quickstart** that appears, create a **database user**
   (save the password) and, under **Network Access**, **Add My Current IP
   Address** (or `0.0.0.0/0` for a laptop on changing networks — that leaves the
   password as the only barrier, fine for a demo, not for real data).
3. **Connect → Drivers → Python**, copy the `mongodb+srv://…` string, put your
   database-user password in it (percent-encode any special characters), and set
   it as `MONGODB_URI` in `.env`. Plain `pymongo` resolves `mongodb+srv://` — no
   `[srv]` extra needed.

`uv run rag ingest` creates the vector search index itself, so nothing else to
set up. Free-tier limits worth knowing: at most **3** search + vector indexes,
**0.5 GB** storage, and a cluster **auto-pauses after 30 idle days** and must be
resumed manually — a paused cluster surfaces as a connection error, so if queries
start hanging or erroring after a break, resume it in the console.

## Usage

### 1. Build the index

```bash
uv run rag ingest
```

Loads the `.md`/`.txt`/`.pdf` files under `data/` — recursively, matching the
extension case-insensitively — splits them into overlapping chunks, embeds them
with Voyage AI, and stores them in the MongoDB Atlas collection, creating the
vector search index if it does not exist yet.

A file it cannot use is skipped rather than aborting the run: an unreadable one
(bad encoding, corrupt PDF, permissions) warns on stderr, and one that yields no
text is dropped silently. The silent case is the one to know about — it includes
a scanned, image-only PDF, whose text extraction returns empty without failing.

Re-run it whenever the documents change. It is incremental: each document is
fingerprinted, and only new, edited, or removed ones are re-embedded, so adding
one file to a large corpus costs one file rather than the corpus — embedding is
a billed API call. Afterwards the collection holds exactly what is in `data/`, so
deletions and edits are picked up too, there are never duplicates, and any
unrelated documents sharing the collection are left alone. Re-running with
nothing changed makes no embedding calls at all.

Changing `EMBEDDING_MODEL`, `CHUNK_SIZE` or `CHUNK_OVERLAP` re-embeds
everything, since all three change what the stored vectors represent.

### 2. Ask questions from the terminal

```bash
uv run rag query "What is chunking and why do we overlap chunks?"
```

Streams the grounded answer as Claude produces it, then prints the source files
it drew from.

### 3. Or use the chat app

```bash
uv run streamlit run app.py
```

A browser chat UI over the same pipeline, streaming each answer token by token,
with a sidebar showing the active configuration and a per-answer Sources panel
holding the retrieved passages themselves — so a claim can be checked against
the text it was generated from, not just against a filename.

The sidebar also takes uploads, so the whole loop — add a document, index it,
ask about it — can happen in the browser. See below.

## Add your own documents

Two routes, same result:

- **From the app** — drag `.md`/`.txt`/`.pdf` files onto **Add documents** in the
  sidebar and click **Add to index**. They are written into `data/` and the index
  is refreshed on the spot — incrementally, as above — so the answer to your next
  question already includes them, with no reload and no terminal. This also works
  before any index exists, which is how a fresh checkout can be brought up
  entirely from the browser. A file whose name already exists in `data/` replaces
  it, so re-uploading a corrected document updates it rather than leaving both
  versions retrievable.
- **From the filesystem** — drop files into `data/` (the three sample files are
  just a starter corpus — delete them if you like), then re-run
  `uv run rag ingest`.

Either way the CLI and the app immediately answer against the new content: they
read the same Atlas collection, and the app reloads its pipeline when the corpus
fingerprint `rag ingest` stamps changes.

Uploaded filenames are treated as untrusted input: `save_upload()` reduces a name
to its final path component and rejects unsupported suffixes before writing, so
an upload cannot choose its own directory. Its docstring covers the details,
including where that boundary deliberately stops.

## Configuration

Everything is set in `.env` (see `.env.example`). `ANTHROPIC_API_KEY`,
`VOYAGE_API_KEY` and `MONGODB_URI` are required; the rest have sensible defaults:

| Variable            | Default            | Purpose |
| ------------------- | ------------------ | ------- |
| `ANTHROPIC_API_KEY` | —                  | Anthropic key (generation) |
| `VOYAGE_API_KEY`    | —                  | Voyage AI key — embedding (ingest + query) and reranking (query) |
| `MONGODB_URI`       | —                  | MongoDB Atlas connection string (the vector store); carries credentials, so it is a key, not a tunable |
| `CHAT_MODEL`        | `claude-haiku-4-5` | Generation model (e.g. `claude-opus-4-8` for higher-quality answers) |
| `MAX_TOKENS`        | `1024`             | Maximum length of a generated answer |
| `EMBEDDING_MODEL`   | `voyage-4-lite`    | Voyage AI embedding model |
| `EMBEDDING_DIMENSIONS` | `1024`          | Width of the embedding vectors; pins the Atlas index's `numDimensions` |
| `RETRIEVAL_K`       | `4`                | Chunks kept after reranking |
| `FETCH_K`           | `20`               | Candidates retrieved before reranking |
| `RERANK_MODEL`      | `rerank-2.5-lite`  | Voyage AI reranker (cross-encoder; e.g. `rerank-2.5` for higher quality) |
| `CHUNK_SIZE`        | `1000`             | Characters per chunk |
| `CHUNK_OVERLAP`     | `200`              | Overlap between adjacent chunks |
| `DATA_DIR`          | `./data`           | Source documents |
| `MONGODB_DB`        | `rag_db`           | Atlas database holding the collection |
| `COLLECTION_NAME`   | `rag_docs`         | Atlas collection holding the vectors — must match between ingest and query |
| `VECTOR_INDEX_NAME` | `vector_index`     | Atlas Vector Search index over the collection — must match between ingest and query |
| `ATLAS_SIMILARITY`  | `cosine`           | Vector similarity metric (`cosine`, `euclidean`, or `dotProduct`) |
| `MONGODB_TIMEOUT_MS` | `10000`           | `serverSelectionTimeoutMS` for the Mongo client |

Optional LangSmith tracing (`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`, with
`LANGSMITH_PROJECT` naming the project) is picked up automatically if set — see
`.env.example`.

## Development

[Ruff](https://docs.astral.sh/ruff/) (lint + format) and
[ty](https://docs.astral.sh/ty/) (type check) are pinned in the dev dependency
group and configured in `pyproject.toml`. The order below is the order CI runs
them in.

### Linting and type checking

```bash
uv run ruff check --fix .    # lint, applying safe fixes
uv run ruff format .         # format
uv run ty check              # type check
```

Run `ruff check` before `ruff format` — lint fixes can reorder code that
formatting then tidies.

### Tests

```bash
uv run pytest
```

The suite needs **no API keys and no cloud Atlas**: it injects a deterministic
fake embedding model and a fake chat model, and reaches only a local
`mongodb-atlas-local` container (started by testcontainers) over loopback. A
relaxed socket guard still blocks every third-party host, so a test that forgets
to inject a fake fails instead of quietly calling the Voyage AI API. It does need
**Docker** for that container — the first run pulls the image, then it is cached.

It covers the configuration, the loader and splitter, ingest idempotency, upload
handling, an ingest→retrieve→generate round trip, the CLI as a terminal program,
and the Streamlit app driven headlessly — so the frontend is covered by CI rather
than by hand. `CLAUDE.md` has the design behind the injection seam and what the
app-level tests are there to guarantee.

Coverage is measured on demand rather than in CI, and carries no threshold — a
number to keep green invites tests that execute code without asserting anything:

```bash
uv run pytest --cov=rag_pipeline --cov=app --cov-report=term-missing
```

The uncovered lines are meant to be uncovered: `cli.py`'s
`if __name__ == "__main__"` guard, and the two real model constructions in
`build_embeddings()` / `build_reranker()` — the lines whose injected fakes stand
in for the paid Voyage AI calls, so the suite never makes them.

### Continuous integration

`.github/workflows/ci.yml` runs on every push — any branch — and on every pull
request, as two jobs:

| Job    | Status check name                    | Runs |
| ------ | ------------------------------------ | ---- |
| `lint` | `ruff + ty`                          | `ruff check`, `ruff format --check`, `ty check` |
| `test` | `pytest (py3.11)`, `pytest (py3.13)` | the pytest suite on both ends of `requires-python` |

Both install with `uv sync --locked`, which fails if `uv.lock` has drifted from
`pyproject.toml` — so a dependency added by hand without re-locking is caught
rather than silently skipped. The `lint` job installs only the dev group before
running ruff, and the full environment only for `ty check`.

Tests need no secrets and no cloud Atlas — they run against a local container
over loopback — but they do need Docker, which the GitHub runners provide.

Every branch push gets CI immediately, so a branch that has been broken for
several commits is visible before review rather than after. A same-repo pull
request then reuses that run; a fork's pull request produces no push event here,
so its jobs run for real.

Nothing gates `main` — it accepts direct pushes, and CI reports on the result
rather than blocking it. To gate merges instead, add a repository ruleset
requiring the three status check names in the table above. To run the same
checks locally beforehand:

```bash
uv sync --locked && uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

## Project structure

```
rag_pipeline/
  config.py     Settings, loaded from environment variables
  ingest.py     load → split → embed → store (build_embeddings lives here)
  pipeline.py   RAGPipeline: load index + Claude, stream_answer(...) / answer(...)
cli.py          entrypoint → rag_pipeline/cli.py  (rag ingest | rag query "...")
app.py          Streamlit chat UI
data/           sample documents (swap in your own)
```

## How it works

- **Voyage AI embeddings** (`langchain-voyageai`) embed documents at ingest and
  questions at query; the *same* model must embed both for their vectors to
  compare, so a single factory (`build_embeddings()`) is shared by ingest and
  query.
- **MongoDB Atlas Vector Search** (`langchain-mongodb`) stores the vectors and
  serves `$vectorSearch`, so querying searches the collection instead of
  re-embedding. `ingest()` creates the index; the factory that opens the store
  (`open_store()`) lives in `ingest.py` and is imported by the query side, so
  both open it the same way.
- **Voyage AI reranking** (`langchain-voyageai`) sharpens retrieval: vector search
  casts a wide net (`FETCH_K` candidates), then a cross-encoder reranker scores
  each candidate against the question *jointly* — which embedding cosine
  similarity only approximates — and keeps the top `RETRIEVAL_K`. It defaults to
  `rerank-2.5-lite`, the cost/latency tier matching the `voyage-4-lite` /
  `claude-haiku-4-5` defaults; set `RERANK_MODEL=rerank-2.5` for higher-quality
  ranking. This is the single query-time factory that lives in `pipeline.py`
  rather than `ingest.py`, because reranking has no ingest-side counterpart.
- **Claude generation** (`langchain-anthropic`) is prompted to answer only from
  the retrieved context and to cite its sources, which is what turns a general
  chat model into a document-grounded question-answerer. Both frontends stream
  it: `stream_answer(question)` hands back the retrieved sources and a lazy
  stream of the answer together, and `answer()` — for library callers who just
  want the finished string — is a join over the same path.

Because these are LangChain integrations, swapping the embedding model or the
chat model is a one-line change in `.env`. Swapping the vector store is a code
change rather than a config one: `open_store()` in `ingest.py` is the single
place the Mongo client and `MongoDBAtlasVectorSearch` are constructed, so it is
the main place to edit — though the incremental bookkeeping in `ingest()` also
speaks pymongo directly (its reads and deletes).

## Invariants

A few of this project's rules are properties of the source *text* rather than of
its behavior — they say some call never happens, so there is nothing to observe.
Those live as data in `tests/invariants.py`, and `tests/test_invariants.py`
enforces them across every tracked `.py` file:

| Rule                 | Forbids                                                        | Why |
| -------------------- | -------------------------------------------------------------- | --- |
| `store-factory`      | constructing `MongoDBAtlasVectorSearch(...)` or `MongoClient(...)` outside `ingest.py` — `tests/` exempt | a collection's identity is (connection URI, database, collection, vector index name, embedding function); ingest and query must open it the same way |
| `embeddings-factory` | constructing an embedding model (`VoyageAIEmbeddings(...)`) outside `ingest.py`, `tests/` included | the same model must embed documents and questions; in tests, inject a fake instead |
| `no-suppressions`    | lint/type suppression comments                                   | fix the finding instead |

Two documentation rules ride along: every `Settings` field must appear in both
`.env.example` and the configuration table above, and every rule in the table
above must have its row — `test_every_setting_is_documented` and
`test_every_rule_is_documented` are what catch either falling behind, since
`ruff`, `ty` and the rest of the suite stay green against stale docs.

Everything else is asserted where the behavior is, because a test observes the
property while a rule only matches spellings. See CLAUDE.md's *Enforcing the
invariants* for those and for the reasoning behind the split.
