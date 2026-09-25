"""Local MLX models behind LangChain's interfaces.

The three adapters the factories construct (``ingest.build_embeddings``,
``pipeline.build_reranker``, ``pipeline.build_chat_model``):

- ``QwenVLEmbeddings``  -- langchain ``Embeddings`` over Qwen3-VL-Embedding.
- ``QwenVLReranker``    -- langchain ``BaseDocumentCompressor`` over
  Qwen3-VL-Reranker.
- ``MLXChatModel``      -- langchain ``BaseChatModel`` over mlx-lm generation.

Guarantees every adapter keeps:

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
  and construction never raises ``ValueError`` (app.py catches only
  ``FileNotFoundError | RuntimeError`` around the pipeline load):
  model not cached / incomplete -> ``FileNotFoundError`` naming the download
  command; MLX unavailable, a failed load, a failed forward pass or generation
  -> ``RuntimeError``.

The embedder and reranker implement their model family's official recipe
(prompt format, pooled token, score head) by hand over mlx-lm's text-only model,
because the MLX packages that ship one were unusable here: mlx-embeddings fails
from its second call on, and mlx-vlm loads the vision tower, pulls in a server
stack and rounds scores to bf16. A recipe that is subtly wrong still
yields plausible vectors and sensible-looking rankings, so the only check that
notices is ``tests/test_models_live.py`` reproducing the model cards' published
scores. The MLX calls themselves are confined to a few thin methods (``_pool``,
``_forward``, the generation loop), so everything around them -- prompts,
truncation, batching, ordering, locking, error translation -- is tested in CI,
where MLX does not exist.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

from langchain_core.callbacks import CallbackManagerForLLMRun, Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
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
# share one copy of the weights -- the three models together are about 22 GB.
_LOADED: dict[str, tuple[Any, Any]] = {}
_LOAD_LOCK = threading.Lock()
# One lock per loaded model, shared by every adapter over it: after an app
# rebuild the old pipeline's embedder may still be running in another session
# while the new one starts, and both drive the same weights.
_FORWARD_LOCKS: dict[int, threading.Lock] = {}


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
        # a JSONDecodeError is a ValueError, which app.py would read as a bad
        # setting and stop on above its sidebar. A shard name that is not a
        # string is checked here too, or it would fail below, outside the try.
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
        # which app.py would take for a configuration error above its sidebar.
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


def _forward_lock(model: Any) -> threading.Lock:
    """The lock serializing forward passes through ``model``.

    MLX documents no thread-safety guarantee, and Streamlit sessions plus
    langchain's executor-based ``aembed_*`` call from many threads. The GPU runs
    one pass at a time anyway, so serializing costs no throughput (probing
    measured it faster) and bounds peak memory to one batch.
    """
    with _LOAD_LOCK:
        return _FORWARD_LOCKS.setdefault(id(model), threading.Lock())


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


def _fit_ids(
    encode: Callable[[str], list[int]], head: str, body: str, tail: str, limit: int
) -> list[int]:
    """Token ids for ``head + body + tail`` in at most ``limit``, cutting only ``body``.

    Both models read their output at the prompt's tail -- the embedder pools
    its appended token, the reranker scores the answer position -- so a
    prompt truncated from the right, as the official embedding script does, is
    read at the wrong place. The body is trimmed instead, as the official
    reranker does. The common case is tokenized whole, exactly as the model was
    trained; only an overlong prompt is spliced from separately encoded parts.
    """
    ids = encode(head + body + tail)
    if len(ids) <= limit:
        return ids
    head_ids, tail_ids = encode(head), encode(tail)
    budget = limit - len(head_ids) - len(tail_ids)
    if budget <= 0:
        raise RuntimeError(
            f"The prompt is over the model's {limit}-token limit before any "
            "document text is added."
        )
    return head_ids + encode(body)[:budget] + tail_ids


# The official scripts' MAX_LENGTH, for both the embedder and the reranker.
_MAX_PROMPT_TOKENS = 8192


# --- embeddings --------------------------------------------------------------

# The official RAG asymmetry: documents take the model's default instruction,
# questions a retrieval one, which in probing widened the gap between relevant
# and irrelevant passages (0.445 vs 0.318 at the narrowest).
_EMBED_DOCUMENT_INSTRUCTION = "Represent the user's input."
_EMBED_QUERY_INSTRUCTION = "Retrieve passages that answer this question."
# The token whose final hidden state is the embedding. It is not part of the
# chat template: the official pipeline gets it from the tokenizer's
# post-processor, which only some tokenize calls run, so it is appended by id.
_EMBED_TOKEN = "<|endoftext|>"
# Throughput plateaus at 8 texts per pass for 1000-char chunks.
_EMBED_BATCH = 8
# Marks where the text goes in a rendered template: a private-use character,
# which no chat template emits on its own.
_SLOT = "\ue000"


def _split_template(tokenizer: Any, instruction: str) -> tuple[str, str]:
    """The rendered chat template around the user text, as ``(head, tail)``.

    Rendered once rather than per text, and split so ``_fit_ids`` can trim the
    text alone.
    """
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": _SLOT},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, slot, tail = rendered.partition(_SLOT)
    if not slot:
        raise RuntimeError("the chat template does not include the user's text")
    return head, tail


class QwenVLEmbeddings(Embeddings):
    """Qwen3-VL-Embedding as a langchain ``Embeddings``.

    ``dimensions`` truncates each vector to its first ``dimensions`` values and
    re-normalizes (Matryoshka); outside ``[1, native width]`` is a
    ``RuntimeError`` at construction. Loads the model eagerly.

    The recipe is the official one: instruction as the system turn, text as the
    user turn, the generation prompt, then one appended ``<|endoftext|>`` whose
    last hidden state is the vector. Leaving that token off moves the model
    card's scores by up to 0.18 while every vector still looks reasonable.
    """

    def __init__(self, model_id: str, dimensions: int) -> None:
        model, tokenizer = load_mlx_model(model_id)
        try:
            backbone = model.language_model.model
            native = int(model.language_model.args.hidden_size)
            pooled_id = int(tokenizer.get_vocab()[_EMBED_TOKEN])
            document_template = _split_template(tokenizer, _EMBED_DOCUMENT_INSTRUCTION)
            query_template = _split_template(tokenizer, _EMBED_QUERY_INSTRUCTION)
        except Exception as exc:
            raise RuntimeError(
                f"{model_id!r} does not look like a Qwen3-VL-Embedding model: {exc}"
            ) from exc
        if not 1 <= dimensions <= native:
            raise RuntimeError(
                f"EMBEDDING_DIMENSIONS={dimensions} is outside 1..{native}, the "
                f"range {model_id!r} can produce."
            )
        self._model_id = model_id
        self._tokenizer = tokenizer
        self._backbone = backbone
        self._pooled_id = pooled_id
        self._dimensions = dimensions
        self._document_template = document_template
        self._query_template = query_template
        self._lock = _forward_lock(model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self._document_template)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self._query_template)[0]

    def _embed(self, texts: list[str], template: tuple[str, str]) -> list[list[float]]:
        if not texts:
            return []
        head, tail = template
        vectors: list[list[float]] = []
        try:
            prompts = [self._prompt_ids(head, text, tail) for text in texts]
            with self._lock:
                try:
                    for start in range(0, len(prompts), _EMBED_BATCH):
                        vectors.extend(
                            self._pool(prompts[start : start + _EMBED_BATCH])
                        )
                finally:
                    _release_buffers()
        except Exception as exc:
            raise RuntimeError(
                f"Embedding with {self._model_id!r} failed: {exc}"
            ) from exc
        return vectors

    def _prompt_ids(self, head: str, text: str, tail: str) -> list[int]:
        # One token of the limit is reserved for the appended pooled token.
        body = _fit_ids(self._encode, head, text, tail, _MAX_PROMPT_TOKENS - 1)
        return [*body, self._pooled_id]

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _pool(self, batch: list[list[int]]) -> list[list[float]]:
        """One forward pass: each prompt's vector, in order."""
        import mlx.core as mx

        width = max(len(ids) for ids in batch)
        # Right padding needs no mask: under the causal mask no real token
        # attends to a pad after it, so the pad id is irrelevant (it happens to
        # equal the pooled token's). That is also why each row's pooled position
        # comes from its length -- searching for the token id would find pads.
        tokens = mx.array(
            [ids + [self._pooled_id] * (width - len(ids)) for ids in batch]
        )
        last = mx.array([len(ids) - 1 for ids in batch])
        hidden = self._backbone(tokens)
        pooled = hidden[mx.arange(len(batch)), last][:, : self._dimensions]
        pooled = pooled.astype(mx.float32)
        pooled = pooled / mx.linalg.norm(pooled, axis=-1, keepdims=True)
        return pooled.tolist()


# --- reranking ---------------------------------------------------------------

# The official prompt, byte for byte. The spacing is part of it ("<Instruct>: "
# has a space, "<Query>:" and "<Document>:" do not), and so is the absence of a
# <think> block: the text-only Qwen3-Reranker's format moves the card example's
# logit from 1.77 to 1.28.
_RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
_RERANK_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)
_RERANK_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
_RERANK_BATCH = 8


def _rerank_head(query: str) -> str:
    """The prompt up to the document text.

    Reads ``_RERANK_INSTRUCTION`` at call time, which is how the live tests
    reproduce the model card's example under the card's own instruction.
    """
    return (
        f"<|im_start|>system\n{_RERANK_SYSTEM}<|im_end|>\n<|im_start|>user\n"
        f"<Instruct>: {_RERANK_INSTRUCTION}<Query>:{query}\n<Document>:"
    )


def _yes_minus_no(embed: Any, yes_no: tuple[int, int]) -> Any:
    """The score head: the tied embedding's "yes" row minus its "no" row, in fp32.

    Read by calling the layer on the two ids, never by indexing its ``.weight``:
    a quantized checkpoint's ``embed_tokens`` is a ``QuantizedEmbedding``, whose
    ``.weight`` holds packed uint32 words -- a vector of the wrong width, and
    every rerank would fail at the matmul. The call dequantizes. For the bf16
    default it returns the same rows bit for bit, and the subtraction stays in
    the weights' own dtype before widening, so its scores are the model card's.
    """
    import mlx.core as mx

    rows = embed(mx.array(list(yes_no)))
    return (rows[0] - rows[1]).astype(mx.float32)


def _sigmoid(logit: float) -> float:
    # The tanh form cannot overflow, unlike 1 / (1 + exp(-logit)).
    return 0.5 * (1.0 + math.tanh(logit / 2.0))


def _scored(doc: Document, score: float) -> Document:
    # A copy, so the score never leaks into the retriever's own documents.
    metadata = deepcopy(doc.metadata)
    metadata["relevance_score"] = score
    return Document(page_content=doc.page_content, metadata=metadata, id=doc.id)


class QwenVLReranker(BaseDocumentCompressor):
    """Qwen3-VL-Reranker as a langchain ``BaseDocumentCompressor``.

    ``compress_documents`` returns the ``top_n`` highest-scoring documents in
    descending score order, as copies carrying ``metadata["relevance_score"]``.
    Loads the model eagerly (at construction).

    The checkpoint has no score head: a pair's score is the model's probability
    of answering "yes" rather than "no", read off the tied embedding matrix --
    so only a checkpoint whose output head *is* that matrix can be scored. The
    2B ties them (in any quantization); a checkpoint with a separate
    ``lm_head`` is refused at construction rather than scored off the wrong
    weights with no error.
    Activations run in fp32 over the bf16 weights; pure bf16 made a chunk's
    score depend on which other chunks shared its padded batch, enough to change
    the top four.
    """

    model_id: str
    top_n: int

    _backbone: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _yes_no: tuple[int, int] = PrivateAttr(default=(0, 0))
    _pad_id: int = PrivateAttr(default=0)
    _score_vector: Any = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default=None)

    # An "after" validator rather than model_post_init, which langchain reserves;
    # it raises RuntimeError deliberately, since pydantic would turn a
    # ValueError into a ValidationError.
    @model_validator(mode="after")
    def _load_reranker(self) -> Self:
        if self.top_n < 1:
            raise RuntimeError(
                f"The reranker's top_n must be at least 1, not {self.top_n}."
            )
        model, tokenizer = load_mlx_model(self.model_id)
        try:
            self._backbone = model.language_model.model
            vocab = tokenizer.get_vocab()
            self._yes_no = (int(vocab["yes"]), int(vocab["no"]))
            self._pad_id = int(tokenizer.pad_token_id)
        except Exception as exc:
            raise RuntimeError(
                f"{self.model_id!r} does not look like a Qwen3-VL-Reranker model: {exc}"
            ) from exc
        if hasattr(model.language_model, "lm_head"):
            raise RuntimeError(
                f"{self.model_id!r} has an untied output head; the reranker scores "
                "off the tied embedding matrix, so it takes only a tied-embedding "
                "Qwen3-VL-Reranker checkpoint (the 2B, in any quantization)."
            )
        self._tokenizer = tokenizer
        self._lock = _forward_lock(model)
        return self

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        logits = self._logits(query, [doc.page_content for doc in documents])
        # Stable, so tied scores keep the retriever's order.
        ranked = sorted(range(len(documents)), key=logits.__getitem__, reverse=True)
        return [
            _scored(documents[i], _sigmoid(logits[i])) for i in ranked[: self.top_n]
        ]

    def _logits(self, query: str, texts: list[str]) -> list[float]:
        logits = [0.0] * len(texts)
        try:
            head = _rerank_head(query)
            seqs = [
                _fit_ids(self._encode, head, text, _RERANK_TAIL, _MAX_PROMPT_TOKENS)
                for text in texts
            ]
            # Batching similar lengths together keeps the padding, which is
            # computed and then thrown away, small.
            order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
            with self._lock:
                try:
                    for start in range(0, len(order), _RERANK_BATCH):
                        batch = order[start : start + _RERANK_BATCH]
                        scores = self._forward([seqs[i] for i in batch])
                        for i, logit in zip(batch, scores, strict=True):
                            logits[i] = logit
                finally:
                    _release_buffers()
        except Exception as exc:
            raise RuntimeError(
                f"Reranking with {self.model_id!r} failed: {exc}"
            ) from exc
        return logits

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _forward(self, batch: list[list[int]]) -> list[float]:
        """One forward pass: each prompt's yes-minus-no logit, in order."""
        import mlx.core as mx
        from mlx_lm.models.cache import BatchKVCache

        embed = self._backbone.embed_tokens
        if self._score_vector is None:
            self._score_vector = _yes_minus_no(embed, self._yes_no)
        # Left padding keeps every row's last real token at position -1, where
        # the score is read. BatchKVCache masks the pads and gives each row its
        # own RoPE offset, which is what makes a padded batch score the same as
        # one pair at a time.
        width = max(len(ids) for ids in batch)
        pads = [width - len(ids) for ids in batch]
        tokens = mx.array(
            [[self._pad_id] * pad + ids for pad, ids in zip(pads, batch, strict=True)]
        )
        cache = [BatchKVCache(pads) for _ in self._backbone.layers]
        hidden = self._backbone(tokens, cache, embed(tokens).astype(mx.float32))
        return (hidden[:, -1, :].astype(mx.float32) @ self._score_vector).tolist()


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
        app.py does when Streamlit's Stop interrupts an answer, and the pipeline
        closes this stream when its own is closed. Errors keep to the union:
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
