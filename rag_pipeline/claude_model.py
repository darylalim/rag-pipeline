"""The chat model, Claude, behind LangChain's interface.

``ClaudeChatModel`` is what ``pipeline.build_chat_model`` constructs: a
langchain ``BaseChatModel`` over the Anthropic SDK's streaming helper.

An adapter of its own rather than langchain-anthropic's ``ChatAnthropic``,
because that one iterates the SDK's HTTP stream without closing it: when its
consumer stops early -- the app's Stop button, a closed ``rag query`` -- the
response is abandoned rather than closed, and Claude can keep generating, and
billing, to MAX_TOKENS. Here the stream is a context manager, so however the
answer ends -- exhausted, failed, or closed half-way -- the request ends with
it. That property is what ``tests/test_claude_model.py`` drives through a real
client over a mock transport.

Guarantees it keeps:

- The request is this pipeline's, set here and nowhere else: thinking at its
  lowest, a fixed effort, no sampling parameters, server-side fallback on a
  refusal another model may answer. ``stop`` and generation options are refused,
  not ignored.
- Failures stay inside ``FileNotFoundError | RuntimeError | ValueError``, and
  construction raises only ``RuntimeError`` (streamlit_app.py catches
  ``FileNotFoundError | RuntimeError`` around the pipeline load): a missing
  ``ANTHROPIC_API_KEY`` or a ``MAX_TOKENS`` below one. An API failure, a refusal
  and anything unexpected while streaming become ``RuntimeError``; a message
  type it cannot send is a ``ValueError``; ``BaseException`` -- Streamlit's stop
  signal, ``GeneratorExit`` -- passes untouched.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

import anthropic
from anthropic.types.beta import BetaMessageParam, BetaThinkingConfigParam
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
from pydantic import model_validator

from rag_pipeline.config import require_env_key

# Thinking off. `between_tools` is Claude Sonnet 5.5's lowest thinking setting:
# with no tools in the request, the model does no extended thinking at all. An
# answer grounded in a few retrieved passages needs none, and thinking would add
# to the wait before the first word. (`{"type": "disabled"}` is a 400 on this
# model, and `between_tools` is refused by every other model.)
_THINKING: BetaThinkingConfigParam = {"type": "between_tools"}
# Anthropic's starting point for question answering; `between_tools` allows
# `high` or below.
_EFFORT = "low"
# A request Claude Sonnet 5.5's safety classifiers decline in a category another
# model may answer is re-run on it server-side, on the same stream; a decline
# that falls back nowhere still ends with `stop_reason: "refusal"`.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
# The SDK retries rate limits, overloads and connection errors with backoff;
# two (its default) is thin for a question a user is waiting on.
_MAX_RETRIES = 4

_ROLES = ((HumanMessage, "user"), (AIMessage, "assistant"))


def _split(messages: list[BaseMessage]) -> tuple[str, list[BetaMessageParam]]:
    """The system prompt, and the turns after it, as the Messages API takes them."""
    system: list[str] = []
    turns: list[BetaMessageParam] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            system.append(str(message.text))
            continue
        for cls, role in _ROLES:
            if isinstance(message, cls):
                turns.append({"role": role, "content": str(message.text)})
                break
        else:
            raise ValueError(f"The chat model cannot take a {message.type!r} message.")
    return "\n\n".join(system), turns


class ClaudeChatModel(BaseChatModel):
    """Claude as a langchain ``BaseChatModel``, streaming the answer's text."""

    model: str
    max_tokens: int
    # Injectable for tests (a real client over a mock transport); production
    # leaves it None and gets one built from ANTHROPIC_API_KEY.
    client: Any = None

    @model_validator(mode="after")
    def _connect(self) -> Self:
        # RuntimeError, not ValueError: pydantic would wrap a ValueError in a
        # ValidationError, and this runs on the app's pipeline-load path.
        if self.max_tokens < 1:
            raise RuntimeError(f"MAX_TOKENS must be at least 1, not {self.max_tokens}.")
        if self.client is None:
            key = require_env_key("ANTHROPIC_API_KEY", "Answers come from Claude")
            self.client = anthropic.Anthropic(api_key=key, max_retries=_MAX_RETRIES)
        return self

    @property
    def _llm_type(self) -> str:
        return "anthropic"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model, "max_tokens": self.max_tokens}

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        # What a tracer names the model by.
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "anthropic"
        params["ls_model_name"] = self.model
        return params

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the answer's text, then one chunk saying why it stopped.

        The SDK's stream is open only inside its ``with`` block, so closing this
        generator half-way -- at a ``yield`` -- closes the HTTP response then
        and there. A stop reason of ``max_tokens`` is reported as LangChain's
        ``finish_reason: "length"``, which ``RAGPipeline._generate`` turns into
        a note on the answer; ``refusal`` is a RuntimeError, so a declined
        question never reads as an empty or partial answer.
        """
        if stop:
            raise ValueError("The chat model does not support stop sequences.")
        if kwargs:
            raise ValueError(
                f"The chat model takes no generation options ({sorted(kwargs)}); "
                "its request is fixed in claude_model.py."
            )
        system, turns = _split(messages)
        try:
            with self.client.beta.messages.stream(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=turns,
                thinking=_THINKING,
                output_config={"effort": _EFFORT},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        chunk = ChatGenerationChunk(
                            message=AIMessageChunk(content=text)
                        )
                        if run_manager is not None:
                            run_manager.on_llm_new_token(text, chunk=chunk)
                        yield chunk
                final = stream.get_final_message()
        except (RuntimeError, ValueError):
            raise
        except anthropic.APIError as exc:
            raise RuntimeError(f"Claude ({self.model}) request failed: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Generation with {self.model!r} failed: {exc}") from exc

        if final.stop_reason == "refusal":
            category = getattr(final.stop_details, "category", None)
            raise RuntimeError(
                "Claude declined to answer this question"
                + (f" (a {category} refusal)." if category else ".")
            )
        usage = final.usage
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                usage_metadata={
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.input_tokens + usage.output_tokens,
                },
                response_metadata={
                    # "length" means cut off at MAX_TOKENS, as LangChain says it.
                    "finish_reason": "length"
                    if final.stop_reason == "max_tokens"
                    else final.stop_reason,
                    "model_name": final.model,
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
