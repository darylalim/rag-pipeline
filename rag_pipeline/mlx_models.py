"""The local chat model, behind LangChain's interface.

``MLXChatModel`` -- a langchain ``BaseChatModel`` over mlx-lm generation -- is
what ``pipeline.build_chat_model`` constructs. (Embedding and reranking moved
to Voyage AI's API; their local adapters are gone.)

Guarantees it keeps:

- Weights are loaded only through ``load_mlx_model``, once per process per
  model, so rebuilding a pipeline (the app does after every ingest) never
  reloads them.
- ``mlx``/``mlx_lm`` are imported lazily, inside functions: the project
  installs MLX only on macOS (a ``sys_platform == 'darwin'`` marker), and the
  Linux CI legs import this module without it.
  ``load_mlx_model`` imports ``mlx_lm`` *before* touching the Hugging Face
  cache, so a missing MLX fails the same way on every machine.
- Loading never reaches the network: a model id resolves to its local cached
  snapshot (``local_files_only=True``) or is a path to a model directory.
- Failures stay inside ``FileNotFoundError | RuntimeError | ValueError`` --
  and construction never raises ``ValueError`` (streamlit_app.py catches only
  ``FileNotFoundError | RuntimeError`` around the pipeline load):
  model not cached / incomplete -> ``FileNotFoundError`` naming the download
  command; MLX unavailable, a failed load or generation -> ``RuntimeError``.

The MLX calls themselves are confined to the generation loop, so everything
around them -- the prompt, locking, stopping, error translation -- is tested in
CI, where MLX does not exist.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any, Self

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.language_models.chat_models import generate_from_stream
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import PrivateAttr, model_validator

# --- loading -----------------------------------------------------------------

# The file patterns mlx-lm itself downloads a model with. A snapshot fetched that
# way has no README.md or .gitattributes, and huggingface_hub checks a
# local_files_only lookup against the repo's full cached file listing: without
# the same filter it raises IncompleteSnapshotError for a model that is present.
_ALLOW_PATTERNS = [
    "*.json",
    "model*.safetensors",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
]

# Keyed by resolved snapshot path, so a repo id and a path to the same snapshot
# share one copy of the weights -- about 15 GB for the default chat model.
_LOADED: dict[str, tuple[Any, Any]] = {}
_LOAD_LOCK = threading.Lock()


def _not_cached(model_id: str, detail: str = "") -> FileNotFoundError:
    return FileNotFoundError(
        f"Model {model_id!r} is not in the Hugging Face cache (or incomplete"
        f"{detail}). Download it with: uvx --from huggingface_hub hf download "
        f"{model_id}"
    )


def _missing_files(path: Path) -> list[str]:
    """The files a load needs that ``path`` lacks.

    Checked up front because nothing else catches an interrupted download in
    time: ``snapshot_download(local_files_only=True)`` returns a snapshot with
    no weights in it without complaint, and a sharded model missing one shard
    fails only deep inside the load, as an error that names no remedy.
    """
    index = path / "model.safetensors.index.json"
    if index.is_file():
        # Every way a cut-off or malformed index can fail is a RuntimeError --
        # a JSONDecodeError is a ValueError, which streamlit_app.py's
        # pipeline-load guard does not catch: a traceback under its sidebar. A
        # shard name that is not a string is checked here too, or it would fail
        # below, outside the try.
        try:
            shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            if not all(isinstance(name, str) for name in shards):
                raise TypeError("a shard name is not a string")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError(f"Unreadable weight index {index}: {exc}") from exc
    else:
        shards = ["model.safetensors"]
    return [name for name in ["config.json", *shards] if not (path / name).is_file()]


def resolve_model_path(model_id: str) -> Path:
    """The local directory holding ``model_id``'s weights, without the network.

    A path to an existing directory is used as-is; anything else is a Hugging
    Face repo id resolved from the local cache. ``FileNotFoundError`` (with the
    ``hf download`` command) when it is not cached, or when the snapshot is
    incomplete -- every weight shard its index lists must be present. Anything
    else that stops the lookup (an id that is neither a directory nor a valid
    repo id) is a ``RuntimeError``, never ``ValueError``.
    """
    local = Path(model_id).expanduser()
    if local.is_dir():
        missing = _missing_files(local)
        if missing:
            raise FileNotFoundError(
                f"Model directory {local} is incomplete; missing {', '.join(missing)}."
            )
        return local

    from huggingface_hub import snapshot_download

    try:
        path = Path(
            snapshot_download(
                model_id, local_files_only=True, allow_patterns=_ALLOW_PATTERNS
            )
        )
    except FileNotFoundError as exc:
        # LocalEntryNotFoundError (never downloaded) and IncompleteSnapshotError
        # (files missing from the snapshot) both subclass it.
        raise _not_cached(model_id) from exc
    except Exception as exc:
        # HFValidationError, for an id that is not a repo id, is a ValueError --
        # which streamlit_app.py's pipeline-load guard does not catch: a traceback.
        raise RuntimeError(
            f"Model {model_id!r} is neither a model directory nor a cached "
            f"Hugging Face repo: {exc}"
        ) from exc
    missing = _missing_files(path)
    if missing:
        raise _not_cached(model_id, f"; missing {', '.join(missing)}")
    return path


def load_mlx_model(model_id: str) -> tuple[Any, Any]:
    """``(model, tokenizer)`` for ``model_id``, loaded once per process.

    The only place weights are loaded. Memoized per resolved model and guarded
    by a lock, so concurrent first calls (Streamlit sessions) load it once.

    ``mlx_lm`` is imported first, before any cache lookup, so a machine without
    MLX gets the same RuntimeError whether or not the model is cached -- and
    the test suite's MLX block catches every route to a real model, memoized or
    not. The load is given the resolved local path, never the repo id: mlx-lm
    tries the network for an id even when the model is cached.
    """
    try:
        import mlx_lm
    except ImportError as exc:
        raise RuntimeError(
            f"Cannot load {model_id!r}: the local models run on MLX, which needs "
            f"Apple Silicon macOS ({exc})."
        ) from exc

    path = resolve_model_path(model_id)
    key = str(path.resolve())
    loaded = _LOADED.get(key)
    if loaded is not None:
        return loaded
    with _LOAD_LOCK:
        loaded = _LOADED.get(key)
        if loaded is None:
            try:
                model, tokenizer = mlx_lm.load(key)
            except FileNotFoundError as exc:
                raise _not_cached(model_id, f": {exc}") from exc
            except Exception as exc:
                # Includes mlx-lm's ValueError for an unsupported model type and
                # the RuntimeError of a corrupt safetensors header.
                raise RuntimeError(
                    f"Could not load model {model_id!r} from {path}: {exc}"
                ) from exc
            loaded = _LOADED[key] = (model, tokenizer)
    return loaded


def _release_buffers() -> None:
    """Empty MLX's buffer cache.

    MLX keeps freed buffers for reuse, and with varying batch shapes that cache
    grew to 7 GB beside a 3.4 GB model in probing -- beside a 15 GB generator,
    enough to exhaust a 32 GB machine. Emptying it after every call cost no
    measurable throughput. It is process-wide, so it also trims what the other
    models left behind.
    """
    import mlx.core as mx

    mx.clear_cache()


# --- generation --------------------------------------------------------------

# One generation at a time per process, across every chat model: mlx-lm sets
# and restores the process-wide Metal wired limit around each generation, so
# overlapping calls race on it (probing left it stuck raised), and one 27B model
# saturates the GPU anyway -- two at once finished no sooner than in turn.
_GENERATION_LOCK = threading.Lock()

_ROLES = ((SystemMessage, "system"), (HumanMessage, "user"), (AIMessage, "assistant"))


def _chat_turn(message: BaseMessage) -> dict[str, str]:
    for cls, role in _ROLES:
        if isinstance(message, cls):
            return {"role": role, "content": str(message.text)}
    raise ValueError(f"The local chat model cannot take a {message.type!r} message.")


class MLXChatModel(BaseChatModel):
    """A local mlx-lm chat model as a langchain ``BaseChatModel``.

    Streams token pieces; thinking disabled; greedy decoding (no sampling
    parameters). Loads the model eagerly (at construction).

    Thinking must be switched off in the template, not filtered afterwards:
    left on, the template adds a reasoning instruction and the model streams
    its reasoning straight into the answer, with no tag to strip. Greedy
    decoding is the deliberate absence of sampling parameters -- answers are
    grounded in the retrieved context, and the same question over the same
    context should get the same answer.
    """

    model_id: str
    max_tokens: int

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _load_chat_model(self) -> Self:
        # mlx-lm reads a negative max_tokens as "no limit".
        if self.max_tokens < 1:
            raise RuntimeError(f"MAX_TOKENS must be at least 1, not {self.max_tokens}.")
        self._model, self._tokenizer = load_mlx_model(self.model_id)
        return self

    @property
    def _llm_type(self) -> str:
        return "mlx-lm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_id": self.model_id, "max_tokens": self.max_tokens}

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        # What a tracer names the model by. LangChain fills it from a field
        # called `model` or `model_name`, so without this a trace's model span
        # names no model at all, and the provider is the lower-cased class name.
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "mlx"
        params["ls_model_name"] = self.model_id
        return params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the answer, holding the generation lock until the stream ends.

        The lock is released however the stream ends -- exhausted, failed, or
        closed half-way. Dropped half-way, it is released only when the garbage
        collector finalizes the stream, which for a stream something still
        refers to can be never, so a consumer that stops early must close it:
        streamlit_app.py does when Streamlit's Stop interrupts an answer, and the
        pipeline closes this stream when its own is closed. Errors keep to the union:
        RuntimeError and ValueError pass through and anything else (a template
        error, say) becomes RuntimeError, while BaseException -- Streamlit's own
        stop signal, GeneratorExit -- passes untouched.
        """
        if stop:
            raise ValueError("The local chat model does not support stop sequences.")
        if kwargs:
            raise ValueError(
                f"The local chat model takes no generation options ({sorted(kwargs)}); "
                "it decodes greedily by design."
            )
        turns = [_chat_turn(message) for message in messages]
        last = None
        with _GENERATION_LOCK:
            try:
                import mlx_lm

                prompt = self._tokenizer.apply_chat_template(
                    turns, add_generation_prompt=True, enable_thinking=False
                )
                # No sampler is greedy decoding. max_tokens is explicit because
                # mlx-lm's default (256) would cut answers short silently. The
                # stream is closed while the lock is still held, because its
                # exit is what restores the wired limit the lock protects.
                with closing(
                    mlx_lm.stream_generate(
                        self._model, self._tokenizer, prompt, max_tokens=self.max_tokens
                    )
                ) as stream:
                    for last in stream:
                        if last.text:
                            chunk = ChatGenerationChunk(
                                message=AIMessageChunk(content=last.text)
                            )
                            if run_manager is not None:
                                run_manager.on_llm_new_token(last.text, chunk=chunk)
                            yield chunk
            except (RuntimeError, ValueError):
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"Generation with {self.model_id!r} failed: {exc}"
                ) from exc
            finally:
                _release_buffers()
        if last is not None:
            # "length" here means the answer was cut off at MAX_TOKENS.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    usage_metadata={
                        "input_tokens": last.prompt_tokens,
                        "output_tokens": last.generation_tokens,
                        "total_tokens": last.prompt_tokens + last.generation_tokens,
                    },
                    response_metadata={
                        "finish_reason": last.finish_reason,
                        "model_name": self.model_id,
                    },
                )
            )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # One code path for invoke and stream, so they cannot disagree.
        return generate_from_stream(
            self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        )
