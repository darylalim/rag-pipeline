# RAG Pipeline

A small, readable **Retrieval-Augmented Generation** pipeline built with
[LangChain](https://docs.langchain.com). Documents are embedded with **Voyage
AI** and answers are generated with **Claude**. It ships with a reusable core
library, a CLI, and a Streamlit chat app — all sharing the same code.

```
Ingest (once):   data/ ──load──▶ split ──embed──▶ store (Chroma, on disk)
Query (per Q):   question ──embed──▶ search ──rerank──▶ [top-k chunks + question] ──▶ Claude ──▶ grounded answer + sources
```

At query time the vector search runs locally against the on-disk index; the
network calls are embedding the question and reranking the candidates (both
Voyage AI), then generation (Claude).

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and Python 3.11+
- An **Anthropic API key** (generation) and a **Voyage AI API key** (embedding
  and reranking). Both ingest and query embed text, and query reranks on top of
  that, so both steps need the Voyage key.

## Setup

```bash
uv sync                      # create the venv and install dependencies
cp .env.example .env         # then add your ANTHROPIC_API_KEY and VOYAGE_API_KEY
```

Embedding calls the Voyage AI API — there is no local embedding model. The
`voyageai` SDK does fetch a small tokenizer from the Hugging Face Hub on first
use (for client-side token counting), cached under `~/.cache/huggingface`; the
*"unauthenticated requests to the HF Hub"* notice it prints is harmless.

## Usage

### 1. Build the index

```bash
uv run rag ingest
```

Loads the `.md`/`.txt`/`.pdf` files under `data/` — recursively, matching the
extension case-insensitively — splits them into overlapping chunks, embeds them
with Voyage AI, and persists a Chroma index to `chroma_db/`.

A file it cannot use is skipped rather than aborting the run: an unreadable one
(bad encoding, corrupt PDF, permissions) warns on stderr, and one that yields no
text is dropped silently. The silent case is the one to know about — it includes
a scanned, image-only PDF, whose text extraction returns empty without failing.

Re-run it whenever the documents change. It is incremental: each document is
fingerprinted, and only new, edited, or removed ones are re-embedded, so adding
one file to a large corpus costs one file rather than the corpus — embedding is
a billed API call. Afterwards the index holds exactly what is in `data/`, so
deletions and edits are picked up too, there are never duplicates, and nothing
else in `chroma_db/` is touched. Re-running with nothing changed makes no
embedding calls at all.

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
  is refreshed on the spot — only the new files are embedded, the rest of the
  corpus keeps the vectors it has — and the answer to your next question already
  includes them, with no reload and no terminal. This also works before any index
  exists, which is how a fresh checkout can be brought up entirely from the
  browser. A file whose name already exists in `data/` replaces it, so
  re-uploading a corrected document updates it rather than leaving both versions
  retrievable.
- **From the filesystem** — drop files into `data/` (the three sample files are
  just a starter corpus — delete them if you like), then re-run
  `uv run rag ingest`.

Either way the CLI and the app immediately answer against the new content: they
read the same index, and the app reloads it when its `chroma.sqlite3` mtime
changes.

Uploaded filenames are treated as untrusted input. Streamlit validates the
extension server-side but not the path, so `save_upload()` reduces a name to its
final component before writing: an upload cannot choose its own directory, and
`../../.ssh/authorized_keys` lands as `authorized_keys` in `data/` or is rejected
for its suffix. It does write through a symlink already sitting in `data/`, as
any program would — putting one there needs the access it would grant, so that
is not a boundary this is trying to hold.

## Configuration

Everything is set in `.env` (see `.env.example`). `ANTHROPIC_API_KEY` and
`VOYAGE_API_KEY` are required; the rest have sensible defaults:

| Variable          | Default                                    | Purpose                                  |
| ----------------- | ------------------------------------------ | ---------------------------------------- |
| `ANTHROPIC_API_KEY` | —                                        | Claude key (generation)                  |
| `VOYAGE_API_KEY`  | —                                          | Voyage AI key — embedding (ingest + query) and reranking (query) |
| `CHAT_MODEL`      | `claude-haiku-4-5`                         | Generation model (e.g. `claude-opus-4-8` for higher-quality answers) |
| `MAX_TOKENS`      | `1024`                                     | Maximum length of a generated answer     |
| `EMBEDDING_MODEL` | `voyage-4-lite`                            | Voyage AI embedding model                |
| `RETRIEVAL_K`     | `4`                                        | Chunks kept after reranking              |
| `FETCH_K`         | `20`                                       | Candidates retrieved before reranking    |
| `RERANK_MODEL`    | `rerank-2.5-lite`                          | Voyage AI reranker (cross-encoder; e.g. `rerank-2.5` for higher quality) |
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
forgets to inject a fake fails instead of quietly calling the Voyage API. Most of
its ~6s warm wall time is process startup and importing the chromadb/langchain
stack; the tests themselves take ~3s. It covers configuration, the loader/splitter
(including PDF extraction and the files it must skip), ingest idempotency, the
source helpers, upload handling (that a saved file comes back out of the loader,
that a name cannot escape `data_dir`, and that a same-named file is replaced
rather than duplicated), the setup guards, an ingest→retrieve round-trip (including that reranking reorders
the candidates and caps them at `RETRIEVAL_K`), the generation path end-to-end
(answer text plus source citations, and that retrieved context is injected into
the prompt), and the CLI as a terminal program — its
output, its deduplicated sources block, and the exit code each member of the
caught exception union produces.

Coverage is measured on demand rather than in CI, and carries no threshold — a
number to keep green invites tests that execute code without asserting anything:

```bash
uv run pytest --cov=rag_pipeline --cov=app --cov-report=term-missing
```

It currently reports 99%, and the uncovered lines are meant to be uncovered:
`cli.py`'s `if __name__ == "__main__"` guard, and the `VoyageAIEmbeddings(...)` /
`VoyageAIRerank(...)` constructions in `build_embeddings()` / `build_reranker()` —
the lines the offline suite exists to never execute.

`tests/test_app.py` drives the Streamlit app itself headlessly, through the same
fakes, so the frontend is covered by CI rather than by hand. Its main job is the
one guarantee no lower-level test can see: that a chat turn is stored as a
user/assistant *pair* whatever happens to it — success, a failed generation, or
the run being torn down mid-answer — so a question can never be left in the
history with nothing under it.

The upload path is tested there for the same reason: what it must get right is
only observable at the frontend. That an uploaded file becomes *answerable* on
the same run; that the uploader is reachable when no index exists, which is the
state it is most needed in; that a browser-supplied name cannot escape `data_dir`
through the widget that delivers it; and that a later rerun does not silently
re-index. That last one is counted rather than displayed: `st.file_uploader`
re-reports its files on every rerun, so re-indexing on sight would rebuild the
whole corpus once per chat message — correct output at absurd cost, and
invisible in any assertion about what the app renders.

Two files enforce the project's own invariants rather than its behavior.
`tests/test_invariants.py` checks every tracked `.py` file against the rules
declared in `tests/invariants.py`:

| Rule                  | Forbids                                                        | Why |
| --------------------- | -------------------------------------------------------------- | --- |
| `chroma-factory`      | constructing `Chroma(...)` outside `ingest.py` — `tests/` exempt   | a collection's identity is (persist dir, name, embedding function); ingest and query must open it the same way |
| `embeddings-factory`  | constructing an embedding model (`VoyageAIEmbeddings(...)`) outside `ingest.py`, `tests/` included | the same model must embed documents and questions; in tests, inject a fake instead |
| `no-suppressions`     | lint/type suppression comments                                   | fix the finding instead |

Those are the rules about how source is *written*. An invariant that can be
observed instead is asserted where the behavior is, because that catches every
route to the mistake rather than the spellings someone enumerated:

| Invariant | Enforced by |
| --------- | ----------- |
| `cli.py` imports stay cheap | `test_importing_cli_does_not_load_the_heavy_stack` — imports the module in a subprocess, asserts chromadb/langchain never loaded |
| `build_chat_model` sets no sampling params | `test_build_chat_model_sets_no_sampling_params` — reads them back off the constructed model |
| ingest never wipes the persist dir | `test_ingest_preserves_unrelated_files_in_persist_dir` — a neighbouring file survives a rebuild |

Plus the settings triad: every `Settings` field must be documented here and in
`.env.example`. The rule table is itself checked — `test_every_rule_is_documented`
fails if a rule is added without a row, the same way `test_every_setting_is_documented`
guards the config table above.

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

- **Voyage AI embeddings** (`langchain-voyageai`) embed documents at ingest and
  questions at query; the *same* model must embed both for their vectors to
  compare, so a single factory (`build_embeddings`) is shared by ingest and query.
- **Persistent Chroma** writes vectors to disk once at ingest, so querying just
  reloads the index instead of re-embedding.
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

Because these are LangChain integrations, swapping any piece — a different
embedding model, vector store, or chat model — is a one-line change in
`config.py` or `.env`.
