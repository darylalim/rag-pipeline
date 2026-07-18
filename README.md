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
whenever the documents change — it rebuilds from scratch, so no duplicates.

### 2. Ask questions from the terminal

```bash
uv run rag query "What is chunking and why do we overlap chunks?"
```

Prints a grounded answer followed by the source files it drew from.

### 3. Or use the chat app

```bash
uv run streamlit run app.py
```

A browser chat UI over the same pipeline, with per-answer source citations and a
sidebar showing the active configuration.

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
API key is needed. Most of its ~7s wall time is the transitive `torch` import;
the tests themselves run in well under a second. It covers configuration, the
loader/splitter, ingest idempotency, the source helpers, the setup guards, an
ingest→retrieve round-trip, and the generation path end-to-end (answer text plus
source citations, and that retrieved context is injected into the prompt).

A GitHub Actions workflow (`.github/workflows/ci.yml`) runs this suite on every
push, on Python 3.11 and 3.13.

## Project structure

```
rag_pipeline/
  config.py     Settings, loaded from environment variables
  ingest.py     load -> split -> embed -> store (build_embeddings lives here)
  pipeline.py   RAGPipeline: load index + Claude, answer(question) -> (text, sources)
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
  chat model into a document-grounded question-answerer.

Because these are LangChain integrations, swapping any piece — a different
embedding model, vector store, or chat model — is a one-line change in
`config.py` or `.env`.
