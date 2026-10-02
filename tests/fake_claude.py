"""A stand-in for the Anthropic API: a real client, answered by a mock transport.

The client is the real SDK's, so what the tests drive is the whole path a
request takes -- the body the SDK builds, its parsing of the event stream, and
whether closing an answer closes the HTTP response. Only the server is
replaced: ``FakeClaude.handle`` answers each request with a Messages API event
stream, as server-sent events, and records what it was sent and how much of
each answer was read before the client stopped.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from typing import Any

import anthropic
import httpx2

from rag_pipeline.claude_model import ClaudeChatModel

MODEL = "claude-sonnet-5-5"


def _event(name: str, data: dict[str, Any]) -> bytes:
    return f"event: {name}\ndata: {json.dumps({'type': name, **data})}\n\n".encode()


class _Body(httpx2.SyncByteStream):
    """One answer's response body: records how many pieces were read, and
    whether the client closed it -- its side of a request ending."""

    def __init__(self, fake: FakeClaude) -> None:
        self.fake = fake
        self.pieces_sent = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        fake = self.fake
        yield _event(
            "message_start",
            {
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": MODEL,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 1},
                }
            },
        )
        yield _event(
            "content_block_start",
            {"index": 0, "content_block": {"type": "text", "text": ""}},
        )
        for i, piece in enumerate(fake.pieces):
            if fake.on_piece is not None:
                fake.on_piece(i)
            self.pieces_sent += 1
            yield _event(
                "content_block_delta",
                {"index": 0, "delta": {"type": "text_delta", "text": piece}},
            )
        yield _event("content_block_stop", {"index": 0})
        delta: dict[str, Any] = {"stop_reason": fake.stop_reason, "stop_sequence": None}
        if fake.stop_details is not None:
            delta["stop_details"] = fake.stop_details
        yield _event(
            "message_delta",
            {"delta": delta, "usage": {"output_tokens": len(fake.pieces)}},
        )
        yield _event("message_stop", {})

    def close(self) -> None:
        self.closed = True


class FakeClaude:
    """What the stand-in server answers with, and what it saw."""

    def __init__(self) -> None:
        self.pieces: list[str] = ["Hello", " there."]
        self.stop_reason = "end_turn"
        self.stop_details: dict[str, Any] | None = None
        # Called before each piece is sent -- where a test stops the app's run.
        self.on_piece: Callable[[int], None] | None = None
        # Answers in place of the event stream: an error status, a raise.
        self.respond: Callable[[httpx2.Request], httpx2.Response] | None = None
        self.requests: list[httpx2.Request] = []
        self.bodies: list[_Body] = []

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.respond is not None:
            return self.respond(request)
        body = _Body(self)
        self.bodies.append(body)
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=body
        )

    def chat(self, max_tokens: int = 64) -> ClaudeChatModel:
        """The real adapter, over a real client whose server is this one."""
        client = anthropic.Anthropic(
            api_key="sk-ant-test",
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(
                transport=httpx2.MockTransport(self.handle)
            ),
        )
        return ClaudeChatModel(model=MODEL, max_tokens=max_tokens, client=client)

    @property
    def sent(self) -> dict[str, Any]:
        """The last request's JSON body."""
        return json.loads(self.requests[-1].content)
