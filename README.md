# RAG Pipeline

A small, readable **Retrieval-Augmented Generation** pipeline built with
[LangChain](https://docs.langchain.com) that runs entirely on an Apple Silicon
Mac. Documents are embedded with **Qwen3-VL-Embedding**, stored and searched in
**Chroma** on disk, reranked with **Qwen3-VL-Reranker**, and answered by
**Qwen3.8-27B** — all three models running locally with
[MLX](https://github.com/ml-explore/mlx). It ships with a reusable core library,
a CLI, and a Streamlit chat app — all sharing the same code.

```
Ingest (once):   data/ ──load──▶ split ──embed──▶ store (Chroma, on disk)
Query (per Q):   question ──embed──▶ search ──rerank──▶ [top-k chunks + question] ──▶ local LLM ──▶ grounded answer + sources
```

No step of either phase touches the network: the models load from the local
Hugging Face cache, and Chroma runs in-process against a directory on disk. The
one network step is downloading the models, once, during setup. There are no API
keys and no accounts. The chat app keeps to that too: Streamlit's own usage
statistics, which its browser front end would otherwise send to Streamlit, are
switched off in `.streamlit/config.toml`.

**Contents** — [Prerequisites](#prerequisites) · [Setup](#setup) ·
[Usage](#usage) · [Add your own documents](#add-your-own-documents) ·
[Configuration](#configuration) · [Development](#development) ·
[Project structure](#project-structure) · [How it works](#how-it-works) ·
[Invariants](#invariants)

## Prerequisites

- An **Apple Silicon Mac** (M1 or later) on **macOS 14** or newer. The project
  targets MLX's Metal backend and declares MLX for macOS only, so the models run
  only there. Linux can install the project and run its test suite — that is
  what CI does — but not the models.
- About **32 GB of unified memory**. The three models together hold about 22 GB
  and peak near 24 GB while answering, against the roughly 26.8 GB macOS lets
  the GPU use on a 32 GB machine — it fits, with little to spare. A 16 GB Mac
  cannot hold the default chat model, which is about 15 GB on its own.
- About **24 GB of disk** for the models (15 GB for the chat model, about
  4.3 GB each for the embedder and the reranker), in the Hugging Face cache.
- [uv](https://docs.astral.sh/uv/) and Python 3.11+.

Nothing else: no API keys, no cloud account, no Docker.

## Setup

```bash
uv sync                      # create the venv and install dependencies
```

Then download the three models, once:

```bash
uvx --from huggingface_hub hf download mlx-community/Qwen3-VL-Embedding-2B-bf16
uvx --from huggingface_hub hf download mlx-community/Qwen3-VL-Reranker-2B-bf16
uvx --from huggingface_hub hf download mlx-community/Qwen3.8-27B-4bit
```

They land in the Hugging Face cache (`~/.cache/huggingface/hub`, or wherever
`HF_HOME`/`HF_HUB_CACHE` points — set it the same way for the download and for
the app). Loading is **cache-only**: the pipeline never downloads a model. One
that is missing, or whose download stopped partway, fails with an error naming
the exact `hf download` command, rather than starting a 15 GB download in the
middle of a question. A model setting can also be a path to a model directory,
which is used as-is.

A `.env` file is optional — `cp .env.example .env` only to override a default
(see [Configuration](#configuration)).

## Usage

### 1. Build the index

```bash
uv run rag ingest
```

Loads the `.md`/`.txt`/`.pdf` files under `data/` — recursively, matching the
extension case-insensitively — splits them into overlapping chunks, embeds them
with the local embedding model, and stores them in a Chroma collection under
`chroma_db/`. Ingest loads only the embedding model (about 3.4 GB), never the
reranker or the chat model.

A file it cannot use is skipped rather than aborting the run: an unreadable one
(bad encoding, corrupt PDF, permissions) warns on stderr, and one that yields no
text is dropped silently. The silent case is the one to know about — it includes
a scanned, image-only PDF, whose text extraction returns empty without failing.

Re-run it whenever the documents change. It is incremental: each document is
fingerprinted, and only new, edited, or removed ones are re-embedded, so adding
one file to a large corpus costs one file rather than the corpus — embedding
runs at roughly nine 1000-character chunks a second. Afterwards the collection
holds exactly what is in `data/`, so deletions and edits are picked up too,
there are never duplicates, and any unrelated documents sharing the collection
are left alone. Re-running with nothing changed makes no embedding calls at all.

Changing `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `CHUNK_SIZE` or
`CHUNK_OVERLAP` re-embeds everything, since all four change what the stored
vectors represent. A change to the vectors' *width* (`EMBEDDING_DIMENSIONS`) also
needs a new `COLLECTION_NAME`: a Chroma collection keeps the width it was
created with, even once emptied. Ingest checks this before it deletes anything
and stops with an error saying so.

One ingest writes at a time. A second one — a terminal `rag ingest` while the
app is indexing an upload, say — waits on a lock file in `chroma_db/` rather
than writing alongside the first, because two writers at once corrupt a Chroma
directory for good.

### 2. Ask questions from the terminal

```bash
uv run rag query "What is chunking and why do we overlap chunks?"
```

Streams the grounded answer as the local model produces it, then prints the
source files it drew from. Each `rag query` is a fresh process, so it loads all
three models before it answers; for more than a question or two, the app, which
keeps them loaded, is the faster route. Ctrl-C abandons an answer (exit status
130) once the model finishes the step it is on — a few seconds at most while
it is still reading the prompt.

### 3. Or use the chat app

```bash
uv run streamlit run app.py
```

A browser chat UI over the same pipeline, streaming each answer token by token,
with a sidebar showing the active configuration and a per-answer panel of the
retrieved passages themselves — so a claim can be checked against the text it
was generated from, not just against a filename.

Opening the app loads the three models, behind a spinner; after that they stay
in memory for the life of the server, including across the index rebuilds an
upload or a `rag ingest` triggers. Answers are generated one at a time: a
question asked from a second tab while one is being answered waits for it to
finish. The toolbar's **Stop** ends an answer where it is — the model stops
generating there and then — and the turn stays in the history, marked as
interrupted.

Editing `app.py` does not rerun it: `.streamlit/config.toml` turns Streamlit's
file watcher off, because on every run it walks every loaded module and logs a
traceback for each of transformers' lazily loaded ones — over a hundred a turn.
While working on the app itself, run it with
`uv run streamlit run app.py --server.fileWatcherType auto`.

The sidebar also takes uploads, so the whole loop — add a document, index it,
ask about it — can happen in the browser. See below.

### What to expect

Measured on an M2 Max with 32 GB, using the default models and settings:

| Step | Time |
| ---- | ---- |
| Loading the three models | about 10 s once the weights are in macOS's file cache; longer on a first load after a reboot, which reads them from disk |
| Embedding documents (ingest) | about 9 chunks per second |
| Embedding a question | about 30 ms |
| Reranking the 20 candidates | about 2.5 s |
| First token of the answer | about 6–8 s with the default four 1000-character chunks (about 1k tokens of prompt) — the model reads the whole prompt, at 100–150 tokens per second, before it writes anything; about 15 s for the first answer after the models load, while MLX warms up |
| The rest of the answer | about 17–21 tokens per second |

So a question takes roughly 10–15 seconds, most of it before the first word
appears (the app shows a spinner for that wait), and the first one after the
models load nearer 20. Each `rag query` is a fresh process, so it pays for the
load and that slow first answer every time. The first-token wait scales with
the retrieved context: each extra chunk (`RETRIEVAL_K`) or longer one
(`CHUNK_SIZE`) adds a couple of seconds.

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
  just a starter corpus — delete them if you like; the one live test that asks
  about them skips without them), then re-run `uv run rag ingest`.

Either way the CLI and the app immediately answer against the new content: they
read the same Chroma collection, and the app reloads its pipeline when the corpus
fingerprint `rag ingest` stamps into the collection changes.

Uploaded filenames are treated as untrusted input: `save_upload()` reduces a name
to its final path component and rejects unsupported suffixes before writing, so
an upload cannot choose its own directory. Its docstring covers the details,
including where that boundary deliberately stops.

## Configuration

Nothing is required. Every setting has a default and can be overridden in `.env`
(see `.env.example`) or the environment:

| Variable            | Default            | Purpose |
| ------------------- | ------------------ | ------- |
| `CHAT_MODEL`        | `mlx-community/Qwen3.8-27B-4bit` | Generation model, run with mlx-lm (thinking off, greedy decoding); any mlx-lm chat checkpoint whose chat template accepts a system turn works (Gemma 2's, for one, rejects it, and every question then fails) |
| `MAX_TOKENS`        | `1024`             | Maximum length of a generated answer, in tokens; an answer cut off there ends with a note saying so |
| `EMBEDDING_MODEL`   | `mlx-community/Qwen3-VL-Embedding-2B-bf16` | Embedding model (ingest + query); must be a Qwen3-VL-Embedding checkpoint, since the adapter implements that family's prompt format and pooling |
| `EMBEDDING_DIMENSIONS` | `2048`          | Width of the stored vectors, up to the model's native 2048: a narrower one is the full vector's re-normalized prefix (Matryoshka); a change needs a new `COLLECTION_NAME` |
| `RETRIEVAL_K`       | `4`                | Chunks kept after reranking and put in the prompt; each adds a couple of seconds before the first token |
| `FETCH_K`           | `20`               | Candidates retrieved before reranking |
| `RERANK_MODEL`      | `mlx-community/Qwen3-VL-Reranker-2B-bf16` | Reranker; must be a Qwen3-VL-Reranker checkpoint with tied embeddings — the 2B, bf16 or quantized — since the adapter scores off the tied embedding matrix |
| `CHUNK_SIZE`        | `1000`             | Characters per chunk |
| `CHUNK_OVERLAP`     | `200`              | Overlap between adjacent chunks |
| `DATA_DIR`          | `./data`           | Source documents |
| `PERSIST_DIR`       | `./chroma_db`      | Where Chroma keeps the index on disk |
| `COLLECTION_NAME`   | `rag_docs`         | Chroma collection holding the vectors — must match between ingest and query |

Each model setting is a Hugging Face repo id, resolved from the local cache, or a
path to a model directory.

Optional LangSmith tracing (`LANGSMITH_TRACING=true` + `LANGSMITH_API_KEY`, with
`LANGSMITH_PROJECT` naming the project) is picked up automatically if set — see
`.env.example`. It is the one setting that sends anything off the machine: when
it is on, every traced run — the prompt, the retrieved passages and the answer —
is uploaded to LangSmith's cloud. It is off unless you set it. (Streamlit run
headless — `--server.headless true` — also looks up the machine's external IP
address, to print it; setting `server.address` in `.streamlit/config.toml`
skips that.)

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

ty treats the MLX modules as `Any` on every platform (`replace-imports-with-any`
in `pyproject.toml`): they are installed only on macOS, so this is what keeps a
Mac and CI giving the same answer.

### Tests

```bash
uv run pytest
```

The suite needs **no models, no MLX, no network and no Docker**. It injects a
deterministic fake embedding model, a fake reranker and a fake chat model, and
runs a real Chroma store in-process under a per-test temp directory. Three
guards keep it that way: MLX is made unimportable, so a test that forgets to
inject a fake fails instead of quietly loading gigabytes of weights from the
Hugging Face cache; every socket is blocked; and LangSmith tracing is forced off,
whatever `.env` says. It runs the same on a Mac with MLX installed as on Linux
without it.

It covers the configuration, the loader and splitter, ingest idempotency and
scoping, upload handling, an ingest→retrieve→generate round trip, the CLI as a
terminal program, and the Streamlit app driven headlessly — so the frontend is
covered by CI rather than by hand. The model adapters in `mlx_models.py` are
unit-tested the same way, against a fake MLX and a fake tokenizer: their prompt
formats, truncation, batching, ordering, locking and error translation are all
checked where MLX does not exist. `CLAUDE.md` has the design behind the
injection seam and what the app-level tests are there to guarantee.

What the fakes cannot check is whether the models themselves are driven
correctly, so that has a suite of its own:

```bash
uv run pytest -m models
```

These tests load the real checkpoints (an Apple Silicon Mac with the three
models downloaded and the memory to hold them; about a minute on an M2 Max). They reproduce the embedding and
reranker model cards' published scores — a subtly wrong prompt format or pooling
step still produces plausible vectors and sensible-looking rankings, so this is
the only check that notices — confirm that batching does not change a result,
and run an ingest-and-answer pass over `data/`. They skip, rather than fail,
when MLX or a model is missing — and the ingest-and-answer pass skips when the
sample document its question is about is no longer in `data/`. A plain
`uv run pytest` deselects them
(`addopts = ["-m", "not models"]`), so it never loads a model; run this suite by
hand after changing anything in `mlx_models.py`, since CI cannot.

Coverage is measured on demand rather than in CI, and carries no threshold — a
number to keep green invites tests that execute code without asserting anything:

```bash
uv run pytest --cov=rag_pipeline --cov=app --cov-report=term-missing
```

The lines it reports uncovered should be the ones only a real terminal or a real
model reaches: `cli.py`'s `if __name__ == "__main__"` guard, and the MLX calls in
`mlx_models.py` that the fakes stand in for — which is what `-m models` is for.

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

Both run on Linux, where the project does not install MLX. That costs nothing:
MLX is declared macOS-only in `pyproject.toml` (`; sys_platform == 'darwin'`), so
the runners never install it, and the suite never imports it anyway. Tests need no secrets, no models and no Docker.
The `models` tests are the part CI never runs.

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
  config.py      Settings, loaded from environment variables
  ingest.py      load → split → embed → store (build_embeddings and open_store live here)
  pipeline.py    RAGPipeline: load index + local models, stream_answer(...) / answer(...)
  mlx_models.py  the three local models behind LangChain's interfaces, loaded once per process
  cli.py         rag ingest | rag query "..."
app.py           Streamlit chat UI
.streamlit/      config.toml: usage statistics and the file watcher off
data/            sample documents (swap in your own)
chroma_db/       the index, created by rag ingest (git-ignored)
```

## How it works

- **Local embeddings** (Qwen3-VL-Embedding-2B) embed documents at ingest and
  questions at query; the *same* model must embed both for their vectors to
  compare, so a single factory (`build_embeddings()`) is shared by ingest and
  query. The adapter follows the model's official recipe: an instruction as the
  system turn, the text as the user turn, then one appended `<|endoftext|>`
  token whose final hidden state, normalized, is the vector. Documents and
  questions get different instructions, as the model's own retrieval examples
  do.
- **Chroma** (`langchain-chroma`) stores the vectors under `chroma_db/` and runs
  in-process — no server. Every chunk this pipeline writes carries a marker, and
  every read, delete and search is filtered on it, so a collection shared with
  other data is never read, deleted from, or cited. The factory that opens the
  store (`open_store()`) lives in `ingest.py` and is imported by the query side,
  so both open it the same way.
- **Local reranking** (Qwen3-VL-Reranker-2B) sharpens retrieval: vector search
  casts a wide net (`FETCH_K` candidates), then the reranker scores each
  candidate against the question *jointly* — as the model's probability of
  answering "yes" to whether the passage meets the query — which embedding
  similarity only approximates, and keeps the top `RETRIEVAL_K`. This is the
  single query-time factory that lives in `pipeline.py` rather than `ingest.py`,
  because reranking has no ingest-side counterpart.
- **Local generation** (Qwen3.8-27B, 4-bit, via `mlx-lm`) is prompted to answer
  only from the retrieved context and to cite its sources, which is what turns a
  general chat model into a document-grounded question-answerer. Thinking is
  switched off — left on, the model streams its reasoning straight into the
  answer — and decoding is greedy, so the same question over the same context
  gets the same answer. Both frontends stream it: `stream_answer(question)` hands
  back the retrieved sources and a lazy stream of the answer together, and
  `answer()` — for library callers who just want the finished string — is a join
  over the same path.

Each model's weights load once per process, however many times the pipeline is
rebuilt: the app rebuilds after every ingest, and ingest builds its own embedder,
and both reuse what is already in memory rather than loading second copies.

Swapping a model is a one-line change in `.env` only within its family — any
mlx-lm chat checkpoint whose template takes a system turn for `CHAT_MODEL` (the
grounding prompt is one), another Qwen3-VL-Embedding checkpoint for
`EMBEDDING_MODEL`, and another tied-embedding Qwen3-VL-Reranker checkpoint (the
2B, in any quantization) for `RERANK_MODEL`. A different embedding or
reranking family is a code change in `mlx_models.py`, because each adapter
implements its family's own prompt format and scoring. Swapping the vector store
is a code change too: `open_store()` in `ingest.py` is the single place
`Chroma` is constructed, so it is the main place to edit — though the
incremental bookkeeping in `ingest()` also speaks Chroma's filters directly (its
reads and deletes).

## Invariants

A few of this project's rules are properties of the source *text* rather than of
its behavior — they say some call never happens, so there is nothing to observe.
Those live as data in `tests/invariants.py`, and `tests/test_invariants.py`
enforces them across every tracked `.py` file:

| Rule                 | Forbids                                                        | Why |
| -------------------- | -------------------------------------------------------------- | --- |
| `store-factory`      | constructing the vector store (`Chroma(...)`, `Chroma.from_*(...)`, `chromadb.PersistentClient(...)` or `chromadb.Client(...)`) outside `ingest.py`, `tests/` included | a collection's identity is (persist dir, collection name, embedding function); ingest and query must open it the same way, and chromadb rejects a second client on the same directory with different settings |
| `embeddings-factory` | constructing an embedding model (`QwenVLEmbeddings(...)` or `HuggingFaceEmbeddings(...)`) outside `ingest.py`, `tests/` included — the class definition itself excepted | the same model must embed documents and questions; in tests, inject a fake instead |
| `no-suppressions`    | lint/type suppression comments                                   | fix the finding instead |

Two documentation rules ride along: every `Settings` field must appear in both
`.env.example` and the configuration table above, and every rule in the table
above must have its row — `test_every_setting_is_documented` and
`test_every_rule_is_documented` are what catch either falling behind, since
`ruff`, `ty` and the rest of the suite stay green against stale docs.

Everything else is asserted where the behavior is, because a test observes the
property while a rule only matches spellings. See CLAUDE.md's *Enforcing the
invariants* for those and for the reasoning behind the split.
