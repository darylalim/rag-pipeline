# rag-pipeline

A small, readable **Retrieval-Augmented Generation** pipeline built with
[LangChain](https://docs.langchain.com). Documents are embedded **locally** with
a `sentence-transformers` model (no embedding API key needed) and answers are
generated with **Claude**. It ships with a reusable core library, a CLI, and a
Streamlit chat app — all sharing the same code.

```
Ingest (once):   data/ ──load──▶ split ──embed──▶ store (Chroma, on disk)
Query (per Q):   question ──embed──▶ search ──▶ [top-k chunks + question] ──▶ Claude ──▶ grounded answer + sources
```

The only network call at query time is generation; embedding and retrieval run
entirely on your machine.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.11+
- An **Anthropic API key** — needed only for the query/generation step. Ingest
  runs fully locally.

## Setup

```bash
uv sync                      # create the venv and install dependencies
cp .env.example .env         # then add your ANTHROPIC_API_KEY
```

The first ingest downloads the embedding model (`all-MiniLM-L6-v2`, ~90 MB) once.

## Usage

### 1. Build the index

```bash
uv run rag ingest
```

Loads every `.md`/`.txt`/`.pdf` in `data/`, splits them into overlapping chunks,
embeds them locally, and persists a Chroma index to `chroma_db/`. Re-run this
whenever the documents change — it clears this collection's existing vectors and
re-adds fresh chunks, so no duplicates, and nothing else in `chroma_db/` is
touched.

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
with per-answer source citations and a sidebar showing the active configuration.

## Add your own documents

Drop `.md`, `.txt`, or `.pdf` files into `data/` (the three sample files are just
a starter corpus — delete them if you like), then re-run `uv run rag ingest`.
That's it; the CLI and app immediately answer against the new content.

## Configuration

Everything is set in `.env` (see `.env.example`). Only `ANTHROPIC_API_KEY` is
required; the rest have sensible defaults:

| Variable          | Default                                    | Purpose                                  |
| ----------------- | ------------------------------------------ | ---------------------------------------- |
| `ANTHROPIC_API_KEY` | —                                        | Claude key (query step only)             |
| `CHAT_MODEL`      | `claude-haiku-4-5`                         | Generation model (e.g. `claude-opus-4-8` for higher-quality answers) |
| `MAX_TOKENS`      | `1024`                                     | Maximum length of a generated answer     |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2`   | Local embedding model                    |
| `RETRIEVAL_K`     | `4`                                        | Chunks retrieved per question            |
| `CHUNK_SIZE`      | `1000`                                     | Characters per chunk                     |
| `CHUNK_OVERLAP`   | `200`                                      | Overlap between adjacent chunks          |
| `DATA_DIR`        | `./data`                                   | Source documents                         |
| `PERSIST_DIR`     | `./chroma_db`                              | Where the index is stored                |
| `COLLECTION_NAME` | `rag_docs`                                 | Chroma collection holding the vectors — must match between ingest and query |

Optional LangSmith tracing (`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`) is
picked up automatically if set — see `.env.example`.

## Tests

```bash
uv run pytest
```

The suite runs fully offline — it injects a deterministic fake embedding model
(no model download, no network) and a fake chat model in place of Claude, so no
API key is needed. An autouse fixture blocks network sockets, so a test that
forgets to inject a fake fails instead of quietly downloading a model. Most of
its ~9s warm wall time is process startup and the transitive `torch` import; the
tests themselves take ~3s. It covers configuration, the loader/splitter, ingest
idempotency, the source helpers, the setup guards, an ingest→retrieve
round-trip, and the generation path end-to-end (answer text plus source
citations, and that retrieved context is injected into the prompt).

`tests/test_app.py` drives the Streamlit app itself headlessly, through the same
fakes, so the frontend is covered by CI rather than by hand. Its main job is the
one guarantee no lower-level test can see: that a chat turn is stored as a
user/assistant *pair* whatever happens to it — success, a failed generation, or
the run being torn down mid-answer — so a question can never be left in the
history with nothing under it.

Two files enforce the project's own invariants rather than its behavior.
`tests/test_invariants.py` checks every tracked `.py` file against the rules in
`tests/invariants.py` — no inline store/embedding construction outside the
factories, lazy CLI imports, no lint suppressions, and that every setting is
documented here and in `.env.example`. `tests/test_hooks.py` covers the two
optional Claude Code hooks in `.claude/` that report the same problems earlier;
they are a convenience for one editor, and deleting them changes nothing about
what CI enforces.

## Linting and type checking

[Ruff](https://docs.astral.sh/ruff/) (lint + format) and
[ty](https://docs.astral.sh/ty/) (type check) are pinned in the dev dependency
group and configured in `pyproject.toml`:

```bash
uv run ruff check --fix .    # lint, applying safe fixes
uv run ruff format .         # format
uv run ty check              # type check
```

Run `ruff check` before `ruff format` — lint fixes can reorder code that
formatting then tidies.

## Continuous integration

`.github/workflows/ci.yml` runs on every push — any branch — and on every pull
request, as two jobs:

| Job    | Status check name                     | Runs                                             |
| ------ | ------------------------------------- | ------------------------------------------------ |
| `lint` | `ruff + ty`                           | `ruff check`, `ruff format --check`, `ty check`   |
| `test` | `pytest (py3.11)`, `pytest (py3.13)`  | the pytest suite on both ends of `requires-python` |

Both install with `uv sync --locked`, which fails if `uv.lock` has drifted from
`pyproject.toml` — so a dependency added by hand without re-locking is caught
rather than silently skipped. The `lint` job installs only the dev group before
running ruff, and the full environment only for `ty check`.

Tests need no secrets: the suite is fully offline.

Every branch push gets CI immediately, so a branch that has been broken for
several commits is visible before review rather than after. A same-repo pull
request then skips both jobs, because its own branch push already ran them on
the identical commit; a pull request from a fork produces no push event here, so
its jobs run for real.

Nothing gates `main` — it accepts direct pushes, and CI reports on the result
rather than blocking it. To gate merges instead, add a repository ruleset
requiring the three status check names in the table above. To run the same
checks locally beforehand, see the commands above.

## Project structure

```
rag_pipeline/
  config.py     Settings, loaded from environment variables
  ingest.py     load -> split -> embed -> store (build_embeddings lives here)
  pipeline.py   RAGPipeline: load index + Claude, stream_answer(...) / answer(...)
cli.py entrypoint ->  rag_pipeline/cli.py   (rag ingest | rag query "...")
app.py                Streamlit chat UI
data/                 sample documents (swap in your own)
```

## How it works

- **Local embeddings** (`langchain-huggingface`) keep ingest free and offline;
  the *same* model must embed both documents and questions, so a single factory
  (`build_embeddings`) is shared by ingest and query.
- **Persistent Chroma** writes vectors to disk once at ingest, so querying just
  reloads the index instead of re-embedding.
- **Claude generation** (`langchain-anthropic`) is prompted to answer only from
  the retrieved context and to cite its sources, which is what turns a general
  chat model into a document-grounded question-answerer. Both frontends stream
  it: `stream_answer(question)` hands back the retrieved sources and a lazy
  stream of the answer together, and `answer()` — for library callers who just
  want the finished string — is a join over the same path.

Because these are LangChain integrations, swapping any piece — a different
embedding model, vector store, or chat model — is a one-line change in
`config.py` or `.env`.
