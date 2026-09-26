"""Tests for the local-model adapters, with no MLX and no model.

The project installs MLX only on Apple Silicon macOS, and the real checkpoints
are about 22 GB, so everything here runs against a fake ``mlx_lm`` module
injected into ``sys.modules`` (over the ``None`` that conftest's
``_no_real_models`` puts there) and a character-level fake tokenizer --
conftest's ``fake_mlx``, from ``fake_mlx.py``. That leaves the MLX calls
themselves -- the forward passes and ``stream_generate`` -- to
``test_models_live.py``, and covers everything around them, which is where the
contract lives: which weights load and how often, which errors come out, what
text reaches the model, where it is cut, how results are ordered, that forward
passes over one model never overlap, and that the generation lock is released
however a stream ends -- with mlx-lm's own stream finished first.

The embedder is built through ``ingest.build_embeddings`` rather than by name:
that factory is its one sanctioned constructor (the ``embeddings-factory``
rule), and going through it also covers the wiring production uses.
"""

from __future__ import annotations

import gc
import math
import subprocess
import sys
import threading
import time
import types
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
from huggingface_hub import constants as hf_constants
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline.config import Settings
from rag_pipeline.mlx_models import (
    MLXChatModel,
    QwenVLReranker,
    load_mlx_model,
    resolve_model_path,
)
from tests.fake_mlx import HIDDEN, FakeModel, decode, write_model

_EMBED_TAIL = "<|im_end|>\n<|im_start|>assistant\n<|endoftext|>"
_RERANK_TAIL = "<|im_end|>\n<|im_start|>assistant\n"


@pytest.fixture
def no_hub(monkeypatch):
    """Fail the test if the Hugging Face cache is consulted at all."""

    def refuse(*_args, **_kwargs):
        pytest.fail("the Hugging Face cache was consulted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)


def _embedder(model_id: str, dimensions: int = HIDDEN) -> Embeddings:
    return ingest_mod.build_embeddings(
        Settings(embedding_model=model_id, embedding_dimensions=dimensions)
    )


def _reranker(model_id: str, top_n: int = 3) -> QwenVLReranker:
    return QwenVLReranker(model_id=model_id, top_n=top_n)


def _chat(model_id: str, max_tokens: int = 7) -> MLXChatModel:
    return MLXChatModel(model_id=model_id, max_tokens=max_tokens)


# --- resolving a model --------------------------------------------------------


def test_a_model_directory_is_used_as_is(model_dir, no_hub):
    assert resolve_model_path(model_dir) == Path(model_dir)


def test_a_directory_missing_a_shard_is_incomplete(tmp_path, no_hub):
    root = write_model(
        tmp_path / "m", shards=["model-1.safetensors", "model-2.safetensors"]
    )
    (root / "model-2.safetensors").unlink()

    with pytest.raises(FileNotFoundError, match=r"model-2\.safetensors"):
        resolve_model_path(str(root))


@pytest.mark.parametrize(
    "index",
    [
        '{"weight_map": ',
        "[]",
        "{}",
        '{"weight_map": []}',
        '{"weight_map": {"w": null}}',
    ],
    ids=[
        "truncated",
        "not-an-object",
        "no-map",
        "map-not-an-object",
        "non-string-shard",
    ],
)
def test_an_unreadable_weight_index_is_a_runtime_error(tmp_path, no_hub, index):
    """An index cut off by an interrupted convert or copy must stay a RuntimeError.

    Its JSONDecodeError is a ValueError, which app.py's pipeline-load guard does
    not catch: a traceback under the sidebar in place of the error. Every
    malformed shape is covered, including a shard name that is not a string,
    which would otherwise fail outside the translation.
    """
    root = write_model(tmp_path / "m", shards=["model-1.safetensors"])
    (root / "model.safetensors.index.json").write_text(index)

    with pytest.raises(RuntimeError, match="Unreadable weight index"):
        resolve_model_path(str(root))


def test_an_uncached_model_names_the_download_command(tmp_path, monkeypatch):
    """The real lookup, against an empty cache: never downloaded.

    Also shows the lookup stays offline -- conftest blocks every socket, so a
    lookup that tried the network would fail with that error instead.
    """
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    with pytest.raises(FileNotFoundError) as info:
        resolve_model_path("some-org/some-model")

    assert "uvx --from huggingface_hub hf download some-org/some-model" in str(
        info.value
    )


def test_the_cache_lookup_is_local_only_and_filtered_like_mlx_lm(tmp_path, monkeypatch):
    """Without mlx-lm's own file patterns, a snapshot that mlx-lm downloaded
    looks incomplete to huggingface_hub (it has no README.md), so a model that
    is present would be reported missing."""
    seen: dict[str, Any] = {}

    def lookup(repo_id, **kwargs):
        seen.update(kwargs, repo_id=repo_id)
        return str(write_model(tmp_path / "snap"))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lookup)

    assert resolve_model_path("org/model") == tmp_path / "snap"
    assert seen["local_files_only"] is True
    assert "model*.safetensors" in seen["allow_patterns"]


def test_a_cached_snapshot_missing_a_shard_is_incomplete(tmp_path, monkeypatch):
    """An interrupted download: the snapshot resolves, one shard never landed."""
    snap = write_model(tmp_path / "snap", shards=["a.safetensors", "b.safetensors"])
    (snap / "b.safetensors").unlink()
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *_a, **_k: str(snap)
    )

    with pytest.raises(FileNotFoundError, match="hf download org/model") as info:
        resolve_model_path("org/model")

    assert "b.safetensors" in str(info.value)


def test_an_id_that_is_neither_a_directory_nor_a_repo_is_a_runtime_error(tmp_path):
    """huggingface_hub rejects it with a ValueError, which must not escape:
    app.py's pipeline-load guard does not catch it, so it would be a traceback
    under the sidebar in place of the error."""
    with pytest.raises(RuntimeError, match="neither a model directory"):
        resolve_model_path(str(tmp_path / "no" / "such" / "model"))


# --- loading ------------------------------------------------------------------


def test_missing_mlx_is_a_runtime_error_before_any_cache_lookup(monkeypatch):
    """conftest has made mlx_lm unimportable, as it is on Linux."""

    def refuse(_model_id):
        pytest.fail("the model was looked up before MLX was imported")

    monkeypatch.setattr(mlx_models, "resolve_model_path", refuse)

    with pytest.raises(RuntimeError, match="Apple Silicon"):
        load_mlx_model("org/model")


def test_missing_mlx_wins_even_over_a_memoized_model(fake_mlx, model_dir, monkeypatch):
    """The suite's MLX block must catch every route to a real model, including
    one some earlier code already loaded."""
    load_mlx_model(model_dir)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)

    with pytest.raises(RuntimeError, match="MLX"):
        load_mlx_model(model_dir)


def test_weights_load_once_across_every_adapter(fake_mlx, model_dir):
    """What keeps an app rebuild (after every ingest) from reloading 22 GB."""
    first = load_mlx_model(model_dir)
    for _ in range(2):
        _embedder(model_dir)
        _reranker(model_dir)
        _chat(model_dir)

    assert load_mlx_model(model_dir) is first
    assert len(fake_mlx.loads) == 1


def test_a_repo_id_loads_its_cached_snapshot_path_once(
    fake_mlx, model_dir, monkeypatch
):
    """Production names every model by repo id, so this is the load that matters.

    mlx-lm is handed the cached snapshot's path, never the id: given an id it
    tries the network first even when the model is cached. And the id and that
    path share one copy of the weights -- the three models are about 22 GB.
    """
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *_a, **_k: model_dir
    )

    first = load_mlx_model("org/model")

    assert load_mlx_model(model_dir) is first
    assert load_mlx_model("org/model") is first
    assert fake_mlx.loads == [str(Path(model_dir).resolve())]


def test_each_model_loads_separately(fake_mlx, tmp_path):
    a = write_model(tmp_path / "a")
    b = write_model(tmp_path / "b")

    load_mlx_model(str(a))
    load_mlx_model(str(b))
    load_mlx_model(str(a))

    assert len(fake_mlx.loads) == 2


def test_concurrent_first_loads_load_once(fake_mlx, model_dir):
    """Streamlit sessions can ask for the model at the same moment."""
    fake_mlx.load_delay = 0.05
    start = threading.Barrier(6)
    results: list[Any] = []

    def worker():
        start.wait()
        results.append(load_mlx_model(model_dir))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(fake_mlx.loads) == 1
    assert all(r is results[0] for r in results)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Model type not_a_real_arch not supported."),
        RuntimeError("[load_safetensors] Invalid json header length"),
        KeyError("language_model"),
        OSError("disk read failed"),
    ],
    ids=["unsupported-type", "corrupt-weights", "key-error", "os-error"],
)
def test_a_failed_load_is_a_runtime_error(fake_mlx, model_dir, error):
    fake_mlx.load_error = error

    with pytest.raises(RuntimeError, match="Could not load model") as info:
        load_mlx_model(model_dir)

    assert info.value.__cause__ is error


def test_a_load_that_finds_files_missing_is_file_not_found(fake_mlx, model_dir):
    fake_mlx.load_error = FileNotFoundError("No safetensors found")

    with pytest.raises(FileNotFoundError, match="No safetensors found"):
        load_mlx_model(model_dir)


def test_a_failed_load_is_not_memoized(fake_mlx, model_dir):
    fake_mlx.load_error = RuntimeError("transient")
    with pytest.raises(RuntimeError):
        load_mlx_model(model_dir)

    fake_mlx.load_error = None
    load_mlx_model(model_dir)

    assert len(fake_mlx.loads) == 2


_ADAPTERS = {
    "embeddings": _embedder,
    "reranker": _reranker,
    "chat": _chat,
}


@pytest.mark.parametrize("build", _ADAPTERS.values(), ids=_ADAPTERS.keys())
def test_no_adapter_construction_raises_value_error(fake_mlx, model_dir, build):
    """mlx-lm reports an unsupported model as ValueError, which app.py's
    pipeline-load guard does not catch: a traceback in place of its error."""
    fake_mlx.load_error = ValueError("Model type not supported.")

    with pytest.raises(RuntimeError):
        build(model_dir)


@pytest.mark.parametrize("build", _ADAPTERS.values(), ids=_ADAPTERS.keys())
def test_an_uncached_model_fails_every_adapter_as_file_not_found(
    fake_mlx, tmp_path, monkeypatch, build
):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    with pytest.raises(FileNotFoundError, match="hf download"):
        build("some-org/not-downloaded")
    assert fake_mlx.loads == []


@pytest.mark.parametrize(
    "build", [_embedder, _reranker], ids=["embeddings", "reranker"]
)
def test_a_model_of_the_wrong_family_is_a_runtime_error(fake_mlx, model_dir, build):
    """A plain text model has no ``language_model`` wrapper to read."""
    fake_mlx.load_result = (types.SimpleNamespace(), fake_mlx.tokenizer)

    with pytest.raises(RuntimeError, match="does not look like"):
        build(model_dir)


def test_a_reranker_with_an_untied_output_head_is_refused(fake_mlx, model_dir):
    """The score is read off the embedding matrix, which is the output head only
    when the two are tied. With a separate ``lm_head`` it would still produce a
    ranking -- a plausible, wrong one -- so construction refuses it instead."""
    untied = FakeModel()
    untied.language_model.lm_head = object()
    fake_mlx.load_result = (untied, fake_mlx.tokenizer)

    with pytest.raises(RuntimeError, match="untied output head"):
        _reranker(model_dir)


def test_importing_the_module_does_not_import_mlx():
    """The Linux CI legs import every module with no MLX installed.

    In a subprocess, because this suite has already imported everything; the
    question is what a fresh interpreter loads. The positive control keeps the
    check from passing vacuously on a broken probe.
    """
    probe = (
        "import sys, rag_pipeline.mlx_models; "
        "print(','.join(m for m in ('mlx', 'mlx.core', 'mlx_lm', "
        "'rag_pipeline.mlx_models') if m in sys.modules))"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    ).stdout.strip()

    assert loaded == "rag_pipeline.mlx_models"


# --- embeddings ---------------------------------------------------------------


def test_a_template_without_the_user_turn_is_a_runtime_error(
    fake_mlx, model_dir, monkeypatch
):
    """Without the slot the text would land after the generation prompt and be
    pooled from a position the model was never trained to read -- plausible
    vectors, no error -- so the embedder refuses the template at construction."""
    render = fake_mlx.tokenizer.apply_chat_template

    def drop_user(messages, **kwargs):
        return render([m for m in messages if m["role"] != "user"], **kwargs)

    monkeypatch.setattr(fake_mlx.tokenizer, "apply_chat_template", drop_user)

    with pytest.raises(RuntimeError, match="does not include the user's text"):
        _embedder(model_dir)


@pytest.fixture
def pooled(monkeypatch) -> list[list[list[int]]]:
    """Replace the forward pass: record each batch, return a vector per prompt.

    Each vector is ``[len(prompt)]``, so a test can tell which text a result
    came from and whether the order survived batching.
    """
    batches: list[list[list[int]]] = []

    def pool(_self, batch):
        batches.append(batch)
        return [[float(len(ids))] for ids in batch]

    monkeypatch.setattr(mlx_models.QwenVLEmbeddings, "_pool", pool)
    return batches


def test_documents_and_queries_use_the_official_prompts(fake_mlx, model_dir, pooled):
    """System turn = instruction, user turn = text, the generation prompt, then
    one appended <|endoftext|> -- the token whose state is the vector. Documents
    and questions take different instructions, as the model was trained."""
    embedder = _embedder(model_dir)

    embedder.embed_documents(["Alpha chunk."])
    embedder.embed_query("What is alpha?")

    doc, query = (decode(batch[0]) for batch in pooled)
    assert doc == (
        "<|im_start|>system\nRepresent the user's input.<|im_end|>\n"
        "<|im_start|>user\nAlpha chunk.<|im_end|>\n"
        "<|im_start|>assistant\n<|endoftext|>"
    )
    assert query == (
        "<|im_start|>system\nRetrieve passages that answer this question.<|im_end|>\n"
        "<|im_start|>user\nWhat is alpha?<|im_end|>\n"
        "<|im_start|>assistant\n<|endoftext|>"
    )


def test_an_overlong_text_is_cut_but_the_prompt_tail_survives(
    fake_mlx, model_dir, pooled, monkeypatch
):
    """Cutting the prompt from the right, as the official script does, would
    drop the pooled token and read the vector from the middle of the text."""
    monkeypatch.setattr(mlx_models, "_MAX_PROMPT_TOKENS", 150)
    embedder = _embedder(model_dir)

    embedder.embed_documents(["~" * 500])  # "~" appears nowhere in the template

    (ids,) = pooled[0]
    text = decode(ids)
    kept = text.count("~")
    assert len(ids) == 150
    assert 0 < kept < 500
    assert text.endswith("<|im_start|>user\n" + "~" * kept + _EMBED_TAIL)


def test_embeddings_are_batched_by_eight_and_keep_their_order(
    fake_mlx, model_dir, pooled
):
    texts: list[str] = ["t" * n for n in range(1, 21)]

    vectors = _embedder(model_dir).embed_documents(texts)

    assert [len(batch) for batch in pooled] == [8, 8, 4]
    overhead = len(pooled[0][0]) - 1
    assert vectors == [[float(overhead + n)] for n in range(1, 21)]


def test_embedding_nothing_runs_no_forward_pass(fake_mlx, model_dir, pooled):
    assert _embedder(model_dir).embed_documents([]) == []
    assert pooled == []


def test_every_embedding_call_empties_the_mlx_buffer_cache(fake_mlx, model_dir, pooled):
    """Left alone, MLX's buffer cache grew to 7 GB beside a 3.4 GB model."""
    embedder = _embedder(model_dir)

    embedder.embed_documents(["a", "b"])
    embedder.embed_query("c")

    assert fake_mlx.cache_clears == 2


@pytest.mark.parametrize(
    "error",
    [ValueError("[reshape] bad"), KeyError("x"), RuntimeError("[metal::malloc]")],
)
def test_a_failed_embedding_is_a_runtime_error(fake_mlx, model_dir, monkeypatch, error):
    def pool(_self, _batch):
        raise error

    monkeypatch.setattr(mlx_models.QwenVLEmbeddings, "_pool", pool)
    embedder = _embedder(model_dir)

    with pytest.raises(RuntimeError, match="Embedding with") as info:
        embedder.embed_query("q")

    assert info.value.__cause__ is error
    assert fake_mlx.cache_clears == 1  # released on failure too


@pytest.mark.parametrize("dimensions", [0, -1, HIDDEN + 1])
def test_dimensions_outside_the_model_width_are_a_runtime_error(
    fake_mlx, model_dir, dimensions
):
    with pytest.raises(RuntimeError, match="EMBEDDING_DIMENSIONS"):
        _embedder(model_dir, dimensions)


@pytest.mark.parametrize("dimensions", [1, HIDDEN])
def test_dimensions_at_the_ends_of_the_range_are_accepted(
    fake_mlx, model_dir, dimensions
):
    _embedder(model_dir, dimensions)


# --- reranking ----------------------------------------------------------------

# Logits the fake forward pass gives each document, keyed by its text.
_LOGITS = {"alpha": 2.0, "beta": -1.0, "gamma": 0.5, "delta": 3.5, "epsilon": 0.5}


def _document_text(ids: list[int]) -> str:
    text = decode(ids)
    return text[text.index("<Document>:") + len("<Document>:") : -len(_RERANK_TAIL)]


@pytest.fixture
def scored(monkeypatch) -> list[list[list[int]]]:
    """Replace the forward pass with a lookup in ``_LOGITS``; record batches."""
    batches: list[list[list[int]]] = []

    def forward(_self, batch):
        batches.append(batch)
        return [_LOGITS.get(_document_text(ids), 0.0) for ids in batch]

    monkeypatch.setattr(QwenVLReranker, "_forward", forward)
    return batches


def _docs(*texts: str) -> list[Document]:
    return [
        Document(page_content=t, metadata={"source": f"{t}.md"}, id=f"id-{t}")
        for t in texts
    ]


def test_the_reranker_prompt_is_the_official_one(fake_mlx, model_dir, scored):
    """Byte for byte: spacing included, and no <think> block -- the text-only
    Qwen3-Reranker's format moves the card example's logit from 1.77 to 1.28."""
    _reranker(model_dir).compress_documents(_docs("alpha"), "Why overlap?")

    assert decode(scored[0][0]) == (
        "<|im_start|>system\nJudge whether the Document meets the requirements "
        "based on the Query and the Instruct provided. Note that the answer can "
        'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        "<Instruct>: Given a search query, retrieve relevant candidates that "
        "answer the query.<Query>:Why overlap?\n<Document>:alpha<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def test_reranking_returns_top_n_best_first_with_scores(fake_mlx, model_dir, scored):
    docs = _docs("alpha", "beta", "gamma", "delta", "epsilon")

    ranked = _reranker(model_dir, top_n=3).compress_documents(docs, "q")

    assert [d.page_content for d in ranked] == ["delta", "alpha", "gamma"]
    for d in ranked:
        logit = _LOGITS[d.page_content]
        assert d.metadata["relevance_score"] == pytest.approx(
            1 / (1 + math.exp(-logit))
        )
        assert d.metadata["source"] == f"{d.page_content}.md"
        assert d.id == f"id-{d.page_content}"
    # Copies: the retriever's documents are not annotated behind its back.
    assert all("relevance_score" not in d.metadata for d in docs)


def test_tied_scores_keep_the_retrievers_order(fake_mlx, model_dir, scored):
    ranked = _reranker(model_dir, top_n=2).compress_documents(
        _docs("epsilon", "gamma", "beta"), "q"
    )

    assert [d.page_content for d in ranked] == ["epsilon", "gamma"]


def test_fewer_documents_than_top_n_returns_them_all(fake_mlx, model_dir, scored):
    ranked = _reranker(model_dir, top_n=4).compress_documents(
        _docs("beta", "alpha"), "q"
    )

    assert [d.page_content for d in ranked] == ["alpha", "beta"]


def test_reranking_nothing_runs_no_forward_pass(fake_mlx, model_dir, scored):
    assert _reranker(model_dir).compress_documents([], "q") == []
    assert scored == []
    assert fake_mlx.cache_clears == 0


def test_reranking_batches_by_eight_in_length_order(fake_mlx, model_dir, monkeypatch):
    """Similar lengths share a batch, so little padding is computed; every
    score still lands on its own document.

    Twenty candidates, as FETCH_K's default hands the reranker, so three
    batches: a score written back to the wrong slot shows up only from the
    second batch on, and only when every document scores differently.
    """
    texts = [f"doc{'.' * ((i * 7) % 20)}{i}" for i in range(20)]
    own = {text: float(i) - 9.5 for i, text in enumerate(texts)}  # none is 0.0
    scored: list[list[list[int]]] = []

    def forward(_self, batch):
        scored.append(batch)
        return [own[_document_text(ids)] for ids in batch]

    monkeypatch.setattr(QwenVLReranker, "_forward", forward)

    ranked = _reranker(model_dir, top_n=20).compress_documents(_docs(*texts), "q")

    assert [len(batch) for batch in scored] == [8, 8, 4]
    flat = [len(ids) for batch in scored for ids in batch]
    assert flat == sorted(flat)
    assert [d.page_content for d in ranked] == sorted(
        texts, key=own.__getitem__, reverse=True
    )
    for d in ranked:
        assert d.metadata["relevance_score"] == pytest.approx(
            1 / (1 + math.exp(-own[d.page_content]))
        )
    assert fake_mlx.cache_clears == 1


def test_an_overlong_document_is_cut_but_the_prompt_tail_survives(
    fake_mlx, model_dir, scored, monkeypatch
):
    """The answer position is the prompt's last token; losing the tail would
    score the pair from the middle of the document."""
    monkeypatch.setattr(mlx_models, "_MAX_PROMPT_TOKENS", 400)

    _reranker(model_dir).compress_documents(_docs("~" * 1000), "q")

    (ids,) = scored[0]
    text = decode(ids)
    kept = text.count("~")
    assert len(ids) == 400
    assert 0 < kept < 1000
    assert text.endswith("<Document>:" + "~" * kept + _RERANK_TAIL)


def test_a_query_too_long_to_leave_room_for_a_document_is_a_runtime_error(
    fake_mlx, model_dir, scored, monkeypatch
):
    monkeypatch.setattr(mlx_models, "_MAX_PROMPT_TOKENS", 400)

    with pytest.raises(RuntimeError, match="token limit"):
        _reranker(model_dir).compress_documents(_docs("alpha"), "q" * 1000)
    assert scored == []


def test_a_failed_rerank_is_a_runtime_error(fake_mlx, model_dir, monkeypatch):
    error = ValueError("[broadcast_shapes] bad")

    def forward(_self, _batch):
        raise error

    monkeypatch.setattr(QwenVLReranker, "_forward", forward)

    with pytest.raises(RuntimeError, match="Reranking with") as info:
        _reranker(model_dir).compress_documents(_docs("alpha"), "q")

    assert info.value.__cause__ is error
    assert fake_mlx.cache_clears == 1


def test_a_top_n_below_one_is_a_runtime_error(fake_mlx, model_dir):
    with pytest.raises(RuntimeError, match="top_n"):
        _reranker(model_dir, top_n=0)
    assert fake_mlx.loads == []


def test_forward_passes_over_one_model_are_serialized(fake_mlx, model_dir, monkeypatch):
    """Every adapter over one set of weights shares one forward lock.

    After an upload, ingest's new embedder starts while another session's
    pipeline may still be embedding a question over the same weights, and MLX
    documents no thread safety. Overlap is counted, not timed, so the result is
    deterministic: any unguarded route -- a lock dropped, or one per adapter
    rather than per model -- shows as a peak above one.
    """
    active = 0
    peak = 0
    guard = threading.Lock()

    def enter() -> None:
        nonlocal active, peak
        with guard:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)  # a window wide enough for an unguarded pass to enter
        with guard:
            active -= 1

    def pool(_self, batch):
        enter()
        return [[1.0] for _ in batch]

    def forward(_self, batch):
        enter()
        return [0.0 for _ in batch]

    monkeypatch.setattr(mlx_models.QwenVLEmbeddings, "_pool", pool)
    monkeypatch.setattr(QwenVLReranker, "_forward", forward)
    embedders = [_embedder(model_dir), _embedder(model_dir)]
    rerankers = [_reranker(model_dir), _reranker(model_dir)]
    calls = [lambda e=e: e.embed_query("q") for e in embedders] + [
        lambda r=r: r.compress_documents(_docs("alpha"), "q") for r in rerankers
    ]
    start = threading.Barrier(len(calls))

    def run(call) -> None:
        start.wait()
        call()

    # A thread each, or the barrier waits for good; result() raises here what
    # any of them failed with, traceback and all. Not `with`: its exit joins the
    # workers unbounded, so a pass stuck on the lock would hang here, not fail.
    executor = ThreadPoolExecutor(max_workers=len(calls))
    try:
        passes = [executor.submit(run, call) for call in calls]
        for future in passes:
            future.result(timeout=10)
    finally:
        executor.shutdown(wait=False)

    assert peak == 1


# --- generation ---------------------------------------------------------------

_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Answer from the context."), ("human", "Q: {question}")]
)


def test_generation_is_greedy_with_thinking_off_and_explicit_max_tokens(
    fake_mlx, model_dir
):
    """No sampler means mlx-lm's argmax. Thinking left on would stream the
    model's reasoning into the answer; max_tokens left out would be mlx-lm's
    silent 256."""
    (_PROMPT | _chat(model_dir, max_tokens=7) | StrOutputParser()).invoke(
        {"question": "why?"}
    )

    ((prompt, kwargs),) = fake_mlx.generate_calls
    assert kwargs == {"max_tokens": 7}
    template = fake_mlx.tokenizer.template_kwargs[-1]
    assert template["enable_thinking"] is False
    assert template["add_generation_prompt"] is True
    assert decode(prompt) == (
        "<|im_start|>system\nAnswer from the context.<|im_end|>\n"
        "<|im_start|>user\nQ: why?<|im_end|>\n<|im_start|>assistant\n"
    )


def test_the_chat_model_has_no_sampling_parameters():
    fields = set(MLXChatModel.model_fields)
    assert not fields & {"temperature", "top_p", "top_k", "min_p", "sampler"}


def test_generation_options_are_refused_not_ignored(fake_mlx, model_dir):
    with pytest.raises(ValueError, match="greedily"):
        _chat(model_dir).invoke("hi", temperature=0.7)


def test_stop_sequences_are_refused_not_ignored(fake_mlx, model_dir):
    """Greedy decoding here has no stop-string support; ignoring one would
    generate past the point the caller asked the answer to end."""
    with pytest.raises(ValueError, match="stop sequences"):
        _chat(model_dir).invoke("hi", stop=["\n"])
    assert fake_mlx.generate_calls == []


def test_the_answer_streams_in_pieces_through_a_prompt_chain(fake_mlx, model_dir):
    fake_mlx.pieces = ["Chunks ", "overlap ", "to keep context."]
    chain = _PROMPT | _chat(model_dir) | StrOutputParser()

    pieces = [p for p in chain.stream({"question": "why?"}) if p]

    assert pieces == ["Chunks ", "overlap ", "to keep context."]


def test_invoke_joins_the_stream_and_reports_why_it_stopped(fake_mlx, model_dir):
    message = _chat(model_dir).invoke("why?")

    assert message.content == "Chunks overlap."
    assert message.response_metadata["finish_reason"] == "stop"
    assert message.usage_metadata is not None
    assert message.usage_metadata["output_tokens"] == 2


def test_message_roles_map_to_the_chat_template(fake_mlx, model_dir):
    _chat(model_dir).invoke(
        [SystemMessage("s"), HumanMessage("h"), AIMessage("a"), HumanMessage("h2")]
    )

    ((prompt, _),) = fake_mlx.generate_calls
    assert decode(prompt) == (
        "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nh<|im_end|>\n"
        "<|im_start|>assistant\na<|im_end|>\n<|im_start|>user\nh2<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


@pytest.mark.parametrize(
    "message",
    [
        ChatMessage(content="x", role="critic"),
        ToolMessage(content="x", tool_call_id="1"),
    ],
    ids=["chat-message", "tool-message"],
)
def test_an_unsupported_message_is_a_value_error_at_call_time(
    fake_mlx, model_dir, message
):
    model = _chat(model_dir)  # constructing is fine; only the call is refused

    with pytest.raises(ValueError, match="cannot take"):
        model.invoke([HumanMessage("h"), message])
    assert fake_mlx.generate_calls == []


def test_the_lock_is_held_while_a_stream_is_open(fake_mlx, model_dir):
    stream = _chat(model_dir).stream("why?")
    assert isinstance(stream, Generator)

    next(stream)
    assert mlx_models._GENERATION_LOCK.locked()

    stream.close()
    assert not mlx_models._GENERATION_LOCK.locked()


def test_closing_a_prompt_chain_stops_the_model(fake_mlx, model_dir):
    """The pipeline's chain shape, prompt then model, closed half-way -- as the
    app closes it when Streamlit's Stop interrupts an answer.

    The model's own stream must be closed there and then, not run to the end:
    everything it generates holds the process-wide lock, and every other
    question waits on it. A `StrOutputParser` on the end would do exactly that
    (langchain-core drains a transform's input when it is closed), which is why
    the pipeline has none -- and why this counts what was generated.
    """
    fake_mlx.pieces = ["a", "b", "c", "d"]
    stream = (_PROMPT | _chat(model_dir)).stream({"question": "q"})
    assert isinstance(stream, Generator)

    next(stream)
    stream.close()

    assert fake_mlx.pieces_generated == 1, "the model ran on after the close"
    assert fake_mlx.lock_held_at_close == [True]
    assert not mlx_models._GENERATION_LOCK.locked()
    assert fake_mlx.cache_clears == 1


@pytest.mark.parametrize("end", ["closed", "abandoned"])
def test_the_mlx_stream_ends_while_the_lock_is_held(fake_mlx, model_dir, end):
    """mlx-lm restores the process-wide wired limit as its generator exits, so
    that must happen before the lock is released -- after, it races the next
    generation's own setting of the limit, the race the lock exists for."""
    fake_mlx.pieces = ["a", "b", "c", "d"]
    stream = _chat(model_dir).stream("why?")
    assert isinstance(stream, Generator)
    next(stream)

    if end == "closed":
        stream.close()
    else:
        del stream
        gc.collect()

    assert fake_mlx.lock_held_at_close == [True]
    assert not mlx_models._GENERATION_LOCK.locked()


def test_the_lock_is_released_when_a_stream_is_abandoned(fake_mlx, model_dir):
    fake_mlx.pieces = ["a", "b", "c", "d"]
    stream = _chat(model_dir).stream("why?")
    next(stream)

    del stream
    gc.collect()

    assert not mlx_models._GENERATION_LOCK.locked()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("[metal::malloc] too big"), RuntimeError),
        (ValueError("empty prompt"), ValueError),
        (KeyError("token"), RuntimeError),
        (TypeError("bad"), RuntimeError),
    ],
    ids=["runtime-error", "value-error", "key-error", "type-error"],
)
def test_a_failed_generation_stays_in_the_union(fake_mlx, model_dir, error, expected):
    """Mid-stream, as a real failure lands: the frontend is already rendering."""
    fake_mlx.pieces = ["partial "]
    fake_mlx.stream_error = error
    stream = _chat(model_dir).stream("why?")

    assert next(stream).content == "partial "
    with pytest.raises(expected) as info:
        next(stream)

    assert info.value is error or info.value.__cause__ is error
    assert not mlx_models._GENERATION_LOCK.locked()


def test_a_template_error_is_a_runtime_error(fake_mlx, model_dir, monkeypatch):
    """jinja2's TemplateError is outside the union both frontends catch."""

    class TemplateError(Exception):
        pass

    def reject(*_args, **_kwargs):
        raise TemplateError("System message must be at the beginning.")

    monkeypatch.setattr(fake_mlx.tokenizer, "apply_chat_template", reject)

    with pytest.raises(RuntimeError, match="System message"):
        _chat(model_dir).invoke("why?")
    assert not mlx_models._GENERATION_LOCK.locked()


def test_base_exceptions_pass_through_untouched(fake_mlx, model_dir):
    """Streamlit stops a script with a BaseException; translating it into a
    RuntimeError would render the Stop button as an error."""

    class StopScript(BaseException):
        pass

    stop = StopScript()
    fake_mlx.stream_error = stop

    with pytest.raises(StopScript) as info:
        _chat(model_dir).invoke("why?")

    assert info.value is stop
    assert not mlx_models._GENERATION_LOCK.locked()


def test_a_max_tokens_below_one_is_a_runtime_error(fake_mlx, model_dir):
    """mlx-lm reads a negative max_tokens as "no limit"."""
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        _chat(model_dir, max_tokens=0)
    assert fake_mlx.loads == []
