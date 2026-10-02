"""Tests for tracing: the trace a question becomes, and what it costs.

Split the way the code is. ``pipeline.py`` builds each question's runs, so
the shape of a trace -- one root, every step nested under it, the right ending
however the question ends -- is asserted in-process, against conftest's
``traces``: the real LangSmith client, recording what it would send.
``tracing.py`` builds the client that sends them in production, and what
matters about it -- that it batches, and what an unreachable LangSmith costs
-- is only visible from a whole process, so it is asserted in a subprocess,
against a stand-in LangSmith running inside that subprocess: this process
still opens no socket.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import sys
import threading

import httpx2
import pytest
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.language_models import FakeListChatModel
from langsmith import trace, tracing_context

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import tracing as tracing_mod
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline

_QUESTION = "Why do chunks overlap?"


def _on(settings: Settings) -> Settings:
    return dataclasses.replace(settings, langsmith_tracing=True)


def _pipeline(settings, fake_embeddings, llm, reranker) -> RAGPipeline:
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    return RAGPipeline(settings, embeddings=fake_embeddings, llm=llm, reranker=reranker)


class _ScoringReranker(BaseDocumentCompressor):
    """Keeps the last ``top_n`` candidates, reversed, scored as a reranker does."""

    top_n: int

    def compress_documents(self, documents, query, callbacks=None):
        kept = list(reversed(documents))[: self.top_n]
        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": 0.9 - rank / 10},
                id=doc.id,
            )
            for rank, doc in enumerate(kept)
        ]


# --- the trace a question becomes --------------------------------------------


def test_a_question_is_one_trace_with_every_step_under_it(
    settings, fake_embeddings, fake_reranker, traces, canned_answer
):
    """Retrieval and generation run at different times -- the stream is lazy --
    but they are one question, so they must be one trace.

    Without the root run they were two unrelated ones (the search, then the
    generation), and the rerank between them was in neither.
    """
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    answer = pipeline.answer(_QUESTION)

    assert traces.parents() == {
        "RAGPipeline": None,
        "VectorStoreRetriever": "RAGPipeline",
        "_SliceReranker": "RAGPipeline",
        "generate": "RAGPipeline",
        "ChatPromptTemplate": "generate",
        "FakeListChatModel": "generate",
    }
    runs = traces.runs
    root = traces.named("RAGPipeline")
    assert {run["trace_id"] for run in runs} == {root["id"]}
    assert {run["session_name"] for run in runs if "session_name" in run} == {
        settings.langsmith_project
    }
    kinds = {run["name"]: run["run_type"] for run in runs}
    assert kinds["RAGPipeline"] == "chain"
    assert kinds["VectorStoreRetriever"] == "retriever"
    assert kinds["_SliceReranker"] == "retriever"
    assert kinds["FakeListChatModel"] == "llm"

    assert root["inputs"] == {"question": _QUESTION}
    assert root["outputs"] == {"answer": answer.text} == {"answer": canned_answer}
    # Every step of an answered question ended, and none in error.
    assert all(run.get("end_time") for run in runs)
    assert not any(run.get("error") for run in runs)


def test_the_rerank_run_shows_what_went_in_and_what_was_kept(
    settings, fake_embeddings, traces, canned_answer
):
    """The step LangChain does not trace, and the one that decides what the
    model is shown: every candidate in, the kept ones out, with their scores."""
    reranker = _ScoringReranker(top_n=settings.retrieval_k)
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        reranker,
    )

    docs, chunks = pipeline.stream_answer(_QUESTION)
    "".join(chunks)

    rerank = traces.named("_ScoringReranker")
    assert rerank["inputs"]["query"] == _QUESTION
    metadata = rerank["extra"]["metadata"]
    assert (metadata["model"], metadata["top_k"]) == (
        settings.rerank_model,
        settings.retrieval_k,
    )
    retrieved = traces.named("VectorStoreRetriever")["outputs"]["documents"]
    candidates = rerank["inputs"]["documents"]
    kept = rerank["outputs"]["documents"]
    # In: everything the search returned, in its order, each with its chunk id.
    assert [c["page_content"] for c in candidates] == [
        r["page_content"] for r in retrieved
    ]
    assert all(c["id"].count(":") == 2 for c in candidates)  # source:index:hash
    # Out: exactly what the model was given, with the scores it was ranked by.
    assert [k["page_content"] for k in kept] == [doc.page_content for doc in docs]
    assert [k["id"] for k in kept] == [doc.id for doc in docs]
    assert [k["metadata"] for k in kept] == [doc.metadata for doc in docs]


def test_the_model_run_names_the_model(
    settings, fake_embeddings, fake_reranker, fake_claude, traces
):
    """The adapter tells LangSmith which model answered and whose it is --
    LangChain would otherwise name the provider after the class."""
    pipeline = _pipeline(
        _on(settings), fake_embeddings, fake_claude.chat(), fake_reranker
    )

    pipeline.answer(_QUESTION)

    metadata = traces.named("ClaudeChatModel")["extra"]["metadata"]
    assert metadata["ls_model_name"] == "claude-sonnet-5-5"
    assert metadata["ls_provider"] == "anthropic"


def test_a_stopped_answer_ends_its_trace_as_stopped_not_failed(
    settings, fake_embeddings, fake_reranker, fake_claude, traces
):
    """A Stop is the reader's choice, not a failure of the pipeline.

    Recorded -- a tag, and the partial answer as the output -- with no error,
    so LangSmith does not count every Stop as a failed question. And tracing
    must not cost the property the Stop exists for: closing the stream still
    ends the model's request there.
    """
    fake_claude.pieces = [f"word{i} " for i in range(200)]
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        fake_claude.chat(max_tokens=500),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer(_QUESTION)
    assert next(chunks) == "word0 "
    chunks.close()

    (body,) = fake_claude.bodies
    assert body.closed, "the model's request was left open after the close"
    root = traces.named("RAGPipeline")
    assert root.get("error") is None
    assert root["end_time"]
    assert "stopped" in root["tags"]
    assert root["extra"]["metadata"]["stopped_by"] == "GeneratorExit"
    assert root["outputs"] == {"answer": "word0 "}


def test_a_stream_closed_before_its_first_piece_still_ends_the_trace(
    settings, fake_embeddings, fake_reranker, canned_answer, traces
):
    """The case a generator's own `finally` cannot see: one never started.

    Closing an unstarted generator runs none of its code, so a root run ended
    only there would stay open for good. This is the app's Stop landing
    between retrieval and generation.
    """
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer(_QUESTION)
    chunks.close()

    root = traces.named("RAGPipeline")
    assert root["end_time"]
    assert "stopped" in root["tags"]
    assert "generate" not in {run["name"] for run in traces.runs}, "generation ran"


@pytest.mark.parametrize("empty_answer", [False, True], ids=["model-error", "empty"])
def test_a_failed_answer_ends_its_trace_in_error(
    settings, fake_embeddings, fake_reranker, fake_claude, traces, empty_answer
):
    """A failure while generating -- the model's own, or the pipeline's
    empty-answer check -- marks the question failed, with the error recorded."""
    if empty_answer:
        llm = FakeListChatModel(responses=["   "])  # whitespace: nothing to show
    else:
        fake_claude.respond = lambda _r: httpx2.Response(
            529,
            json={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "busy"},
            },
        )
        llm = fake_claude.chat()
    pipeline = _pipeline(_on(settings), fake_embeddings, llm, fake_reranker)

    with pytest.raises(RuntimeError):
        pipeline.answer(_QUESTION)

    root = traces.named("RAGPipeline")
    assert root["error"].startswith("RuntimeError: ")
    assert "stopped" not in root.get("tags", [])


def _raising_reranker(error: BaseException) -> BaseDocumentCompressor:
    class _RaisingReranker(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            raise error

    return _RaisingReranker()


@pytest.mark.parametrize(
    ("error", "failed"),
    [
        pytest.param(RuntimeError("reranking failed"), True, id="failed"),
        pytest.param(KeyboardInterrupt(), False, id="ctrl-c"),
    ],
)
def test_a_question_ended_during_retrieval_still_ends_its_trace(
    settings, fake_embeddings, canned_answer, traces, error, failed
):
    """Retrieval ends before there is a stream to end the root run, so
    stream_answer must end it itself -- for a failure, and for Ctrl-C at the
    terminal while the question is embedded or reranked, which the CLI treats as
    an ordinary way out. The rerank's own run ends in error either way: it was
    cut short."""
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        _raising_reranker(error),
    )

    with pytest.raises(type(error)):
        pipeline.stream_answer(_QUESTION)

    root = traces.named("RAGPipeline")
    assert root["end_time"]
    assert bool(root.get("error")) is failed
    assert ("stopped" in root.get("tags", [])) is not failed
    assert traces.named("_RaisingReranker")["error"]


def test_a_failed_answer_keeps_what_was_generated(
    settings,
    fake_embeddings,
    fake_reranker,
    canned_answer,
    traces,
    fail_mid_stream,
    partial_answer,
):
    """The text before a failure is the most useful part of a failed trace."""
    fail_mid_stream(RuntimeError("the model died"))
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    with pytest.raises(RuntimeError, match="the model died"):
        pipeline.answer(_QUESTION)

    root = traces.named("RAGPipeline")
    assert root["error"] == "RuntimeError: the model died"
    assert root["outputs"] == {"answer": partial_answer}


def test_the_question_is_never_current_between_pieces(
    settings, fake_embeddings, fake_reranker, canned_answer, traces, tracing_left_on
):
    """Its context is entered around each step of generation, never across a
    yield.

    Held across one, the question's run would be current in the consumer's code
    too -- the app's rendering, the CLI's printing -- and anything traced there
    would be filed under the question.
    """
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer(_QUESTION)
    left = [tracing_left_on()]
    for _piece in chunks:
        left.append(tracing_left_on())

    assert not any(left), left
    assert len(left) > 2  # it did stream, so the check ran between pieces


def test_a_stream_closed_from_another_thread_ends_cleanly(
    settings, fake_embeddings, fake_reranker, canned_answer, traces, caplog
):
    """How a Stop can arrive: the stream closed from outside the frame that
    read it. A context held across a yield cannot be reset there, and says so."""
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )
    _docs, chunks = pipeline.stream_answer(_QUESTION)
    next(chunks)

    with caplog.at_level(logging.WARNING):
        closer = threading.Thread(target=chunks.close)
        closer.start()
        closer.join()

    assert "Context" not in caplog.text, caplog.text
    assert "stopped" in traces.named("RAGPipeline")["tags"]


# --- off, and inside another trace -------------------------------------------


def test_off_sends_nothing_even_inside_a_trace_that_is_on(
    settings, fake_embeddings, fake_reranker, canned_answer, traces
):
    """Off is the pipeline's decision, not the environment's.

    LangSmith switches itself on from its own variables (an old
    ``LANGCHAIN_TRACING_V2=true``) and inside any run that is current. Either
    is simulated here by an enclosing context that traces to the recorder --
    and with LANGSMITH_TRACING off, the question still sends nothing.
    """
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    with tracing_context(enabled=True, client=traces.client, project_name="outer"):
        pipeline.answer(_QUESTION)

    assert traces.runs == []


def test_inside_a_run_already_current_the_question_nests_under_it(
    settings, fake_embeddings, fake_reranker, canned_answer, traces
):
    """`rag eval` traces each question it asks, as a run of its experiment; the
    question's own trace belongs under that run, in that project, not in a
    trace of its own beside it."""
    pipeline = _pipeline(
        _on(settings),
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    # Enabled explicitly, as `rag eval`'s evaluate() does: conftest switches
    # LangSmith's own variables off.
    with (
        tracing_context(enabled=True),
        trace("experiment-row", client=traces.client, project_name="experiment"),
    ):
        pipeline.answer(_QUESTION)

    parents = traces.parents()
    assert parents["RAGPipeline"] == "experiment-row"
    assert parents["generate"] == "RAGPipeline"
    runs = traces.runs
    assert len({run["trace_id"] for run in runs}) == 1
    assert {run["session_name"] for run in runs if "session_name" in run} == {
        "experiment"
    }


# --- the client ---------------------------------------------------------------


def test_tracing_on_without_a_key_stops_the_load(
    settings, fake_embeddings, fake_reranker, canned_answer
):
    """Reported where the app reports a setup failure -- a RuntimeError while
    the pipeline loads -- not as every trace silently lost afterwards. The real
    factory: conftest has removed the developer's key."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)

    with pytest.raises(RuntimeError, match=r"^LANGSMITH_API_KEY is not set"):
        RAGPipeline(
            _on(settings),
            embeddings=fake_embeddings,
            llm=FakeListChatModel(responses=[canned_answer]),
            reranker=fake_reranker,
        )


def test_the_client_is_shared_and_capped(settings, monkeypatch):
    """One client per process, however often the app rebuilds its pipeline --
    each owns a sending thread -- with its retries off and short timeouts, which
    are what bound an exit when LangSmith is unreachable (the subprocess test
    below measures it). Built through a stand-in, since a real one starts a
    thread that would reach for LangSmith."""
    built: list[dict] = []

    class _Client:
        def __init__(self, **kwargs) -> None:
            built.append(kwargs)

    monkeypatch.setattr(tracing_mod, "Client", _Client)
    monkeypatch.setattr(tracing_mod, "_clients", {})
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test")

    assert tracing_mod.tracing_client(settings) is None
    first = tracing_mod.tracing_client(_on(settings))
    assert tracing_mod.tracing_client(_on(settings)) is first

    (kwargs,) = built
    assert kwargs["api_key"] == "lsv2_test"
    assert kwargs["retry_config"].total == 0
    assert kwargs["timeout_ms"] == tracing_mod._TIMEOUT_MS


# --- a whole process ------------------------------------------------------------

# Runs inside a subprocess: a stand-in LangSmith on a free local port, or a port
# nothing listens on, then a question's worth of runs through production's own
# client. The child prints what the stand-in received and how long sending
# took; the parent times the exit, which is where an unreachable LangSmith
# costs.
_PROBE = """
import json, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

received = []


class LangSmith(BaseHTTPRequestHandler):
    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        received.append({"method": self.command, "path": self.path, "body": body})
        payload = b"{}"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = do_PATCH = _answer

    def log_message(self, *args):
        pass


if sys.argv[1] == "up":
    server = HTTPServer(("127.0.0.1", 0), LangSmith)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
else:
    with socket.socket() as probe:  # bound, then closed: nothing listens there
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

import os
os.environ["LANGSMITH_ENDPOINT"] = f"http://127.0.0.1:{port}"

from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context
from langsmith.run_trees import RunTree

from rag_pipeline.config import Settings
from rag_pipeline.tracing import tracing_client

settings = Settings(langsmith_tracing=True, langsmith_project="probe-project")
client = tracing_client(settings)
emitting = time.monotonic()
root = RunTree(
    name="RAGPipeline", run_type="chain", inputs={"question": "ping"},
    ls_client=client, project_name=settings.langsmith_project,
)
root.post()
with tracing_context(enabled=True, parent=root):
    RunnableLambda(lambda x: x).invoke("ping")
root.end(outputs={"answer": "pong"})
root.patch()
emit_s = time.monotonic() - emitting
client.flush(timeout=10)
runs = b"".join(r["body"] for r in received if r["path"].startswith("/runs"))
print(json.dumps({
    "emit_s": emit_s,
    "paths": sorted({r["path"] for r in received}),
    "has_root": b"RAGPipeline" in runs,
    "has_child": b"RunnableLambda" in runs,
    "has_project": b"probe-project" in runs,
}), flush=True)
"""


def _run_probe(langsmith: str) -> tuple[dict, float, str]:
    # The developer's own settings must not reach the child: .env would bring
    # their key and endpoint straight back (config.py loads it), and a proxy
    # would carry a post to 127.0.0.1 somewhere else.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LANGSMITH_", "LANGCHAIN_"))
    }
    env |= {
        "PYTHON_DOTENV_DISABLED": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "LANGSMITH_API_KEY": "lsv2_probe",
        "LANGSMITH_TRACING": "false",  # only the pipeline's own context traces
    }
    started = time_monotonic()
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, langsmith],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env=env,
    )
    took = time_monotonic() - started
    return json.loads(result.stdout.splitlines()[-1]), took, result.stderr


def time_monotonic() -> float:
    import time

    return time.monotonic()


def test_runs_reach_langsmith_batched_with_the_project():
    """What LangSmith needs to file a trace, checked where it counts: on the
    wire. The root and LangChain's run under it both arrive, in the configured
    project, sent in a batch from the background -- the path that keeps a
    question from waiting on LangSmith."""
    report, _took, stderr = _run_probe("up")

    assert report["has_root"], (report, stderr)
    assert report["has_child"], report
    assert report["has_project"], report
    assert "/runs/multipart" in report["paths"], report


def test_with_langsmith_down_questions_do_not_wait_and_exit_waits_briefly():
    """What tracing costs when LangSmith cannot be reached.

    Nothing while the question runs: runs are only queued, and sent from a
    background thread. And little at exit, where the queue is drained: the
    SDK's defaults -- three retries on top of its own three attempts, a 10 s
    connect -- held this exit for about 12 s; capped, it is a fraction of a
    second. (A host that never answers costs about 10 s however the client is
    set: three attempts at a fixed 3 s connect, which the SDK offers no
    setting for -- and which an offline test cannot reach.)
    """
    report, took, _stderr = _run_probe("down")

    assert report["emit_s"] < 0.5, (
        f"sending a question's runs took {report['emit_s']:.1f}s"
    )
    # The child's whole life, interpreter start and imports included.
    assert took < 10, f"the process took {took:.1f}s to finish"
