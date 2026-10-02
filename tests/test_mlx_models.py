"""Tests for the local chat-model adapter, with no MLX and no model.

The project installs MLX only on Apple Silicon macOS, and the real checkpoints
are about 22 GB, so everything here runs against a fake ``mlx_lm`` module
injected into ``sys.modules`` (over the ``None`` that conftest's
``_no_real_models`` puts there) and a character-level fake tokenizer --
conftest's ``fake_mlx``, from ``fake_mlx.py``. That leaves the MLX calls
themselves -- ``stream_generate`` -- to ``test_models_live.py``, and covers
everything around them, which is where the contract lives: which weights load
and how often, which errors come out, what text reaches the model, and that the
generation lock is released however a stream ends -- with mlx-lm's own stream
finished first.
"""

from __future__ import annotations

import gc
import subprocess
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
from huggingface_hub import constants as hf_constants
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline import mlx_models
from rag_pipeline.mlx_models import (
    MLXChatModel,
    load_mlx_model,
    resolve_model_path,
)
from tests.fake_mlx import decode, write_model


@pytest.fixture
def no_hub(monkeypatch):
    """Fail the test if the Hugging Face cache is consulted at all."""

    def refuse(*_args, **_kwargs):
        pytest.fail("the Hugging Face cache was consulted")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", refuse)


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

    Its JSONDecodeError is a ValueError, which streamlit_app.py's pipeline-load
    guard does not catch: a traceback under the sidebar in place of the error. Every
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
    streamlit_app.py's pipeline-load guard does not catch it, so it would be a traceback
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


def test_weights_load_once_however_often_the_model_is_built(fake_mlx, model_dir):
    """What keeps an app rebuild (after every ingest) from reloading 15 GB."""
    first = load_mlx_model(model_dir)
    for _ in range(2):
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


def test_construction_never_raises_value_error(fake_mlx, model_dir):
    """mlx-lm reports an unsupported model as ValueError, which streamlit_app.py's
    pipeline-load guard does not catch: a traceback in place of its error."""
    fake_mlx.load_error = ValueError("Model type not supported.")

    with pytest.raises(RuntimeError):
        _chat(model_dir)


def test_an_uncached_model_is_file_not_found(fake_mlx, tmp_path, monkeypatch):
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))

    with pytest.raises(FileNotFoundError, match="hf download"):
        _chat("some-org/not-downloaded")
    assert fake_mlx.loads == []


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


@pytest.mark.parametrize(
    ("finish", "flush", "content", "tokens"),
    [
        # The usual end: the EOS arrives after the text, in a response of its
        # own with none, and is counted.
        pytest.param("stop", "", "Chunks overlap.", 3, id="stop"),
        # A trailing space is held back until the detokenizer sees what follows,
        # so the stop's own response carries it.
        pytest.param("stop", " ", "Chunks overlap. ", 3, id="stop-with-flush"),
        # Cut off: the last token is decoded, then the loop ends on the count.
        pytest.param("length", "", "Chunks overlap.", 2, id="length"),
    ],
)
def test_invoke_joins_the_stream_and_reports_why_it_stopped(
    fake_mlx, model_dir, finish, flush, content, tokens
):
    """Why generation ended, and what it cost, come from mlx-lm's final response.

    That response usually has no text, so an adapter that looked only at the
    responses that do would report every normal answer as ended for no reason,
    with its EOS uncounted.
    """
    fake_mlx.finish_reason = finish
    fake_mlx.stop_flush = flush

    message = _chat(model_dir).invoke("why?")

    assert message.content == content
    assert message.response_metadata["finish_reason"] == finish
    assert message.usage_metadata is not None
    assert message.usage_metadata["output_tokens"] == tokens


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
