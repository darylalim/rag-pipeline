"""A fake MLX stack, for driving the real model adapters with no MLX and no model.

Shared rather than private to ``test_mlx_models.py`` because the adapters'
contract does not end at the adapter: whether Streamlit's Stop releases the
generation lock depends on how the pipeline and app.py close the stream, so the
frontend and pipeline tests drive the real ``MLXChatModel`` over this fake too.
conftest's ``fake_mlx`` fixture installs it into ``sys.modules`` for one test.

The tokenizer is character-level, so the ids a fake forward pass or generation
receives decode back to the exact prompt text, and a test can assert on what
reached the model -- and where it was cut -- as a string.
"""

from __future__ import annotations

import json
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_pipeline import mlx_models

# The fake model's native embedding width.
HIDDEN = 16

# The fake vocabulary's special tokens: the ones the adapters look up by name,
# given ids above the Unicode range so no character can collide with them.
SPECIAL_IDS = {"<|endoftext|>": 0x110000, "yes": 0x110001, "no": 0x110002}


class CharTokenizer:
    """One token per character, so ids decode back to the exact prompt text.

    Stands in for mlx-lm's TokenizerWrapper. What the tests check is which text
    reaches the model and where it is cut, and a character-level vocabulary
    makes that readable as a string assertion.
    """

    pad_token_id = SPECIAL_IDS["<|endoftext|>"]

    def __init__(self) -> None:
        self.template_kwargs: list[dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        # The adapters append the pooled token themselves; relying on a
        # tokenizer's post-processor to add it is the bug this guards against.
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        names = {v: k for k, v in SPECIAL_IDS.items()}
        return "".join(names.get(i) or chr(i) for i in ids)

    def get_vocab(self) -> dict[str, int]:
        return dict(SPECIAL_IDS)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.template_kwargs.append(
            {"add_generation_prompt": add_generation_prompt, **kwargs}
        )
        text = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return [ord(c) for c in text] if tokenize else text


def decode(ids: list[int]) -> str:
    return CharTokenizer().decode(ids)


class FakeModel:
    """The attributes the adapters read off a loaded Qwen3-VL model.

    No ``lm_head`` on the language model: the checkpoints the adapters take tie
    their output head to the embedding matrix, and mlx-lm leaves the attribute
    off exactly then.
    """

    def __init__(self) -> None:
        backbone = types.SimpleNamespace(embed_tokens=None, layers=[])
        self.language_model = types.SimpleNamespace(
            model=backbone, args=types.SimpleNamespace(hidden_size=HIDDEN)
        )


@dataclass
class Response:
    """The fields of mlx-lm's GenerationResponse the chat model reads."""

    text: str
    finish_reason: str | None
    prompt_tokens: int
    generation_tokens: int


@dataclass
class FakeMLX:
    """What the fake ``mlx_lm``/``mlx.core`` modules do, and what they saw.

    ``on_piece`` runs before each piece of a generation is produced, in the
    thread consuming the stream -- which is how a test delivers a Streamlit
    Stop at a chosen point, the way the toolbar button does. ``pieces_generated``
    counts what was produced, so a stream that was drained rather than closed
    shows up; ``lock_held_at_close`` records the generation lock's state as each
    generation's stream ends, since mlx-lm restores the process-wide wired limit
    at exactly that point.
    """

    loads: list[str] = field(default_factory=list)
    load_error: BaseException | None = None
    load_result: Any = None
    load_delay: float = 0.0
    pieces: list[str] = field(default_factory=lambda: ["Chunks ", "overlap."])
    finish_reason: str = "stop"
    stream_error: BaseException | None = None
    on_piece: Callable[[int], None] | None = None
    generate_calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)
    pieces_generated: int = 0
    streams_closed: int = 0
    lock_held_at_close: list[bool] = field(default_factory=list)
    cache_clears: int = 0
    tokenizer: CharTokenizer = field(default_factory=CharTokenizer)

    def load(self, path: str) -> Any:
        self.loads.append(path)
        time.sleep(self.load_delay)
        if self.load_error is not None:
            raise self.load_error
        return self.load_result or (FakeModel(), self.tokenizer)

    def stream_generate(self, _model: Any, _tokenizer: Any, prompt: Any, **kwargs: Any):
        self.generate_calls.append((prompt, kwargs))
        try:
            for i, text in enumerate(self.pieces):
                if self.on_piece is not None:
                    self.on_piece(i)
                done = i == len(self.pieces) - 1 and self.stream_error is None
                self.pieces_generated += 1
                yield Response(
                    text, self.finish_reason if done else None, len(prompt), i + 1
                )
            if self.stream_error is not None:
                raise self.stream_error
        finally:
            self.streams_closed += 1
            self.lock_held_at_close.append(mlx_models._GENERATION_LOCK.locked())

    def clear_cache(self) -> None:
        self.cache_clears += 1


class _FakeMlxLm(types.ModuleType):
    def __init__(self, fake: FakeMLX) -> None:
        super().__init__("mlx_lm")
        self.load = fake.load
        self.stream_generate = fake.stream_generate


class _FakeMlxCore(types.ModuleType):
    def __init__(self, fake: FakeMLX) -> None:
        super().__init__("mlx.core")
        self.clear_cache = fake.clear_cache


class _FakeMlx(types.ModuleType):
    def __init__(self, core: types.ModuleType) -> None:
        super().__init__("mlx")
        self.core = core


def modules(fake: FakeMLX) -> dict[str, types.ModuleType]:
    """The ``sys.modules`` entries that stand in for MLX, driven by ``fake``."""
    core = _FakeMlxCore(fake)
    return {"mlx_lm": _FakeMlxLm(fake), "mlx": _FakeMlx(core), "mlx.core": core}


def write_model(root: Path, shards: list[str] | None = None) -> Path:
    """A model directory as a load needs it: config plus every weight file."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}")
    if shards is None:
        (root / "model.safetensors").write_bytes(b"")
    else:
        weight_map = {f"w{i}": name for i, name in enumerate(shards)}
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
        for name in shards:
            (root / name).write_bytes(b"")
    return root
