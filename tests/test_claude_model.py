"""Tests for the chat-model adapter, over a real Anthropic client and no network.

conftest's ``fake_claude`` is a real SDK client whose server is a mock
transport (``tests/fake_claude.py``), so what is tested is the whole path a
request takes -- the body the SDK builds from what the adapter passes, the
SDK's own stream parsing, and above all whether closing the answer closes the
HTTP response -- with only the server replaced.
"""

from __future__ import annotations

import httpx2
import pytest
from langchain_core.messages import (
    AIMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate

from rag_pipeline.claude_model import ClaudeChatModel
from tests.fake_claude import MODEL

_CHAIN_PROMPT = ChatPromptTemplate.from_messages(
    [("system", "Answer from the context."), ("human", "{question}")]
)


# --- the request ---------------------------------------------------------------


def test_the_request_is_the_pipelines_with_no_sampling_parameters(fake_claude):
    """Thinking at its lowest, a fixed effort, fallback on a refusal, and no
    sampler -- Claude Sonnet 5.5 rejects one -- by any route."""
    list(
        (_CHAIN_PROMPT | fake_claude.chat(max_tokens=321)).stream({"question": "Why?"})
    )

    sent = fake_claude.sent
    assert sent["model"] == MODEL
    assert sent["max_tokens"] == 321
    assert sent["thinking"] == {"type": "between_tools"}
    assert sent["output_config"] == {"effort": "low"}
    assert sent["fallbacks"] == "default"
    assert sent["system"] == "Answer from the context."
    assert sent["messages"] == [{"role": "user", "content": "Why?"}]
    assert not {"temperature", "top_p", "top_k"} & set(sent)
    assert (
        "server-side-fallback-2026-07-01"
        in fake_claude.requests[-1].headers["anthropic-beta"]
    )


def test_the_chat_model_has_no_sampling_parameters():
    sampling = {"temperature", "top_p", "top_k", "sampler"}
    assert not sampling & set(ClaudeChatModel.model_fields)


def test_generation_options_are_refused_not_ignored(fake_claude):
    with pytest.raises(ValueError, match="no generation options"):
        list(fake_claude.chat().stream("hi", temperature=0.2))


def test_stop_sequences_are_refused_not_ignored(fake_claude):
    with pytest.raises(ValueError, match="stop sequences"):
        list(fake_claude.chat().stream("hi", stop=["\n"]))


@pytest.mark.parametrize(
    "message",
    [ToolMessage(content="x", tool_call_id="1"), ChatMessage(content="x", role="tool")],
    ids=["tool", "chat"],
)
def test_an_unsupported_message_is_a_value_error_at_call_time(fake_claude, message):
    with pytest.raises(ValueError, match="cannot take"):
        list(fake_claude.chat().stream([HumanMessage(content="hi"), message]))


def test_message_roles_map_to_the_messages_api(fake_claude):
    list(
        fake_claude.chat().stream(
            [
                SystemMessage(content="Rules."),
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2"),
            ]
        )
    )

    assert fake_claude.sent["system"] == "Rules."
    assert fake_claude.sent["messages"] == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]


# --- the answer ------------------------------------------------------------------


def test_the_answer_streams_in_pieces_through_a_prompt_chain(fake_claude):
    pieces = [
        chunk.text
        for chunk in (_CHAIN_PROMPT | fake_claude.chat()).stream({"question": "Hi?"})
        if chunk.text
    ]

    assert pieces == ["Hello", " there."]


def test_invoke_joins_the_stream_and_reports_why_it_stopped(fake_claude):
    message = fake_claude.chat().invoke("Hi?")

    assert message.text == "Hello there."
    assert message.response_metadata["finish_reason"] == "end_turn"
    assert message.usage_metadata is not None
    assert message.usage_metadata["output_tokens"] == len(fake_claude.pieces)


def test_an_answer_cut_off_at_max_tokens_reports_length(fake_claude):
    """LangChain's word for it, which RAGPipeline._generate turns into a note."""
    fake_claude.pieces, fake_claude.stop_reason = ["Half"], "max_tokens"

    message = fake_claude.chat().invoke("Hi?")

    assert message.response_metadata["finish_reason"] == "length"


def test_a_refusal_is_a_runtime_error_not_an_answer(fake_claude):
    """A decline is HTTP 200 with a stop reason, so it has to be read: taken as
    an answer, it would be an empty one under a full citation list."""
    fake_claude.pieces, fake_claude.stop_reason = [], "refusal"
    fake_claude.stop_details = {
        "type": "refusal",
        "category": "cyber",
        "explanation": None,
    }

    with pytest.raises(RuntimeError, match="declined to answer"):
        fake_claude.chat().invoke("Hi?")


# --- ending a stream ---------------------------------------------------------------


def test_closing_a_prompt_chain_closes_the_http_response(fake_claude):
    """The reason this adapter exists: a Stop must end the request, not abandon
    it. Closed after its first piece, the answer's HTTP response is closed
    then, with most of what the server would send never read -- where an
    abandoned one would keep Claude generating, and billing, to MAX_TOKENS."""
    fake_claude.pieces = [f"w{i} " for i in range(200)]
    stream = (_CHAIN_PROMPT | fake_claude.chat()).stream({"question": "Hi?"})

    first = next(chunk for chunk in stream if chunk.text)
    stream.close()

    (body,) = fake_claude.bodies
    assert first.text == "w0 "
    assert body.closed, "closing the answer left the HTTP response open"
    assert body.pieces_sent < len(fake_claude.pieces) / 2


def test_a_finished_answer_closes_its_response_too(fake_claude):
    list(fake_claude.chat().stream("Hi?"))

    (body,) = fake_claude.bodies
    assert body.closed


# --- failures -------------------------------------------------------------------


@pytest.mark.parametrize("status", [400, 429, 500, 529])
def test_an_api_error_is_a_runtime_error(fake_claude, status):
    fake_claude.respond = lambda _r: httpx2.Response(
        status,
        json={"type": "error", "error": {"type": "api_error", "message": "boom"}},
    )

    with pytest.raises(RuntimeError, match="request failed") as excinfo:
        fake_claude.chat().invoke("Hi?")

    assert excinfo.type is RuntimeError


def test_a_connection_failure_is_a_runtime_error(fake_claude):
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    fake_claude.respond = refuse

    with pytest.raises(RuntimeError, match="request failed"):
        fake_claude.chat().invoke("Hi?")


def test_base_exceptions_pass_through_untouched(fake_claude):
    """Ctrl-C, or Streamlit's stop signal, is not a failure to translate."""

    def interrupt(_request: httpx2.Request) -> httpx2.Response:
        raise KeyboardInterrupt

    fake_claude.respond = interrupt

    with pytest.raises(KeyboardInterrupt):
        fake_claude.chat().invoke("Hi?")


# --- construction ------------------------------------------------------------------


def test_a_missing_key_is_a_runtime_error_at_construction(monkeypatch):
    """RuntimeError, on the app's pipeline-load path; conftest has removed the
    developer's key."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match=r"^ANTHROPIC_API_KEY is not set"):
        ClaudeChatModel(model=MODEL, max_tokens=64)


def test_a_max_tokens_below_one_is_a_runtime_error(fake_claude):
    with pytest.raises(RuntimeError, match="MAX_TOKENS"):
        fake_claude.chat(max_tokens=0)


def test_a_trace_names_the_model(fake_claude):
    params = fake_claude.chat()._get_ls_params()

    assert params["ls_model_name"] == MODEL
    assert params["ls_provider"] == "anthropic"
