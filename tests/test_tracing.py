"""Tests for tracing: the trace a question becomes, and how it reaches Phoenix.

Split the way the code is. ``pipeline.py`` opens spans through the
OpenTelemetry API, so the shape of a question's trace -- one root, every step
nested under it, the right status however the question ends -- is asserted
in-process, with conftest's ``spans`` recording them in memory. ``tracing.py``
installs the provider that sends them, and what matters about that is only
visible on the wire -- the URL, the protocol, the project, the flush at exit --
so it is asserted in a subprocess, against a stand-in collector running inside
that subprocess: this process still opens no socket.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import logging
import os
import subprocess
import sys
import threading

import pytest
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.language_models import FakeListChatModel
from openinference.instrumentation import TracerProvider
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider
from opentelemetry.trace import StatusCode

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import mlx_models
from rag_pipeline.mlx_models import MLXChatModel
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.tracing import setup_tracing, traces_url

_QUESTION = "Why do chunks overlap?"


def _pipeline(settings, fake_embeddings, llm, reranker) -> RAGPipeline:
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    return RAGPipeline(settings, embeddings=fake_embeddings, llm=llm, reranker=reranker)


def _named(spans: tuple[ReadableSpan, ...], name: str) -> ReadableSpan:
    (span,) = [span for span in spans if span.name == name]
    return span


def _attributes(span: ReadableSpan) -> dict:
    assert span.attributes is not None
    return dict(span.attributes)


def _documents(attributes: dict, key: str) -> list[dict]:
    """A flattened OpenInference document list, put back together."""
    documents: dict[int, dict] = {}
    for name, value in attributes.items():
        if name.startswith(f"{key}."):
            index, _, field = name[len(key) + 1 :].partition(".document.")
            documents.setdefault(int(index), {})[field] = value
    return [documents[i] for i in sorted(documents)]


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
    settings, fake_embeddings, fake_reranker, spans, canned_answer
):
    """Retrieval and generation run at different times -- the stream is lazy --
    but they are one question, so they must be one trace.

    Without the root span they were two unrelated ones (the search, then the
    generation), and the rerank between them was in neither.
    """
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    answer = pipeline.answer(_QUESTION)

    finished = spans.get_finished_spans()
    root = _named(finished, "RAGPipeline")
    by_id = {span.context.span_id: span.name for span in finished}
    parents = {
        span.name: by_id[span.parent.span_id] for span in finished if span.parent
    }
    assert root.parent is None
    assert parents == {
        "VectorStoreRetriever": "RAGPipeline",
        "_SliceReranker": "RAGPipeline",
        "generate": "RAGPipeline",
        "ChatPromptTemplate": "generate",
        "FakeListChatModel": "generate",
    }
    assert {span.context.trace_id for span in finished} == {root.context.trace_id}

    kinds = {
        span.name: _attributes(span)["openinference.span.kind"] for span in finished
    }
    assert kinds["RAGPipeline"] == "CHAIN"
    assert kinds["VectorStoreRetriever"] == "RETRIEVER"
    assert kinds["_SliceReranker"] == "RERANKER"
    assert kinds["FakeListChatModel"] == "LLM"

    attributes = _attributes(root)
    assert attributes["input.value"] == _QUESTION
    assert attributes["output.value"] == answer.text == canned_answer
    # Every step of an answered question reads as a success -- the manual
    # rerank span included, which OpenTelemetry would otherwise leave unset.
    assert {span.name: span.status.status_code for span in finished} == dict.fromkeys(
        kinds, StatusCode.OK
    )


def test_the_rerank_span_shows_what_went_in_and_what_was_kept(
    settings, fake_embeddings, spans, canned_answer
):
    """The step no instrumentation sees, and the one that decides what the
    model is shown: every candidate in, the kept ones out, with their scores."""
    reranker = _ScoringReranker(top_n=settings.retrieval_k)
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        reranker,
    )

    docs, chunks = pipeline.stream_answer(_QUESTION)
    "".join(chunks)

    finished = spans.get_finished_spans()
    attributes = _attributes(_named(finished, "_ScoringReranker"))
    assert attributes["reranker.query"] == _QUESTION
    assert attributes["reranker.model_name"] == settings.rerank_model
    assert attributes["reranker.top_k"] == settings.retrieval_k

    retrieved = _documents(
        _attributes(_named(finished, "VectorStoreRetriever")), "retrieval.documents"
    )
    candidates = _documents(attributes, "reranker.input_documents")
    kept = _documents(attributes, "reranker.output_documents")
    # In: everything the search returned, in its order, each with its chunk id.
    assert [c["content"] for c in candidates] == [r["content"] for r in retrieved]
    assert all(c["id"].count(":") == 2 for c in candidates)  # source:index:hash
    # Out: exactly what the model was given, with the scores it was ranked by.
    assert [k["content"] for k in kept] == [doc.page_content for doc in docs]
    assert [k["id"] for k in kept] == [doc.id for doc in docs]
    assert [k["score"] for k in kept] == [
        doc.metadata["relevance_score"] for doc in docs
    ]
    assert json.loads(kept[0]["metadata"])["source"] == docs[0].metadata["source"]


def test_the_model_span_names_the_local_model(
    settings, fake_embeddings, fake_reranker, fake_mlx, model_dir, spans
):
    """LangChain names a model from a `model` or `model_name` field, and the
    adapter's is `model_id`: without its override, no trace says which model
    answered."""
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        MLXChatModel(model_id=model_dir, max_tokens=50),
        fake_reranker,
    )

    pipeline.answer(_QUESTION)

    attributes = _attributes(_named(spans.get_finished_spans(), "MLXChatModel"))
    assert attributes["llm.model_name"] == model_dir
    assert attributes["llm.provider"] == "mlx"


def test_a_stopped_answer_ends_its_trace_as_stopped_not_failed(
    settings, fake_embeddings, fake_reranker, fake_mlx, model_dir, spans
):
    """A Stop is the reader's choice, not a failure of the pipeline.

    Recorded -- an event, and the partial answer as the output -- with the
    status left unset, so Phoenix does not count every Stop as a failed
    question. And tracing must not cost the property the Stop exists for:
    closing the stream still stops the model there, lock released.
    """
    fake_mlx.pieces = [f"word{i} " for i in range(20)]
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        MLXChatModel(model_id=model_dir, max_tokens=50),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer(_QUESTION)
    assert next(chunks) == "word0 "
    chunks.close()

    assert fake_mlx.pieces_generated == 1, "the model ran on after the close"
    assert not mlx_models._GENERATION_LOCK.locked()
    root = _named(spans.get_finished_spans(), "RAGPipeline")
    assert root.status.status_code is StatusCode.UNSET
    assert [(e.name, dict(e.attributes or {})) for e in root.events] == [
        ("stopped", {"reason": "GeneratorExit"})
    ]
    assert _attributes(root)["output.value"] == "word0 "


def test_a_stream_closed_before_its_first_piece_still_ends_the_trace(
    settings, fake_embeddings, fake_reranker, canned_answer, spans
):
    """The case a generator's own `finally` cannot see: one never started.

    Closing an unstarted generator runs none of its code, so a root span ended
    only there would stay open for good -- and an open span is never exported.
    This is the app's Stop landing between retrieval and generation.
    """
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer(_QUESTION)
    chunks.close()

    finished = spans.get_finished_spans()
    root = _named(finished, "RAGPipeline")
    assert [e.name for e in root.events] == ["stopped"]
    assert "generate" not in {span.name for span in finished}, "generation started"


@pytest.mark.parametrize("empty_answer", [False, True], ids=["model-error", "empty"])
def test_a_failed_answer_ends_its_trace_in_error(
    settings, fake_embeddings, fake_reranker, fake_mlx, model_dir, spans, empty_answer
):
    """A failure while generating -- the model's own, or the pipeline's
    empty-answer check -- marks the question failed, with the error recorded."""
    if empty_answer:
        llm = FakeListChatModel(responses=["   "])  # whitespace: nothing to show
    else:
        fake_mlx.stream_error = OSError("metal ran out of memory")
        llm = MLXChatModel(model_id=model_dir, max_tokens=50)
    pipeline = _pipeline(settings, fake_embeddings, llm, fake_reranker)

    with pytest.raises(RuntimeError):
        pipeline.answer(_QUESTION)

    root = _named(spans.get_finished_spans(), "RAGPipeline")
    assert root.status.status_code is StatusCode.ERROR
    assert (root.status.description or "").startswith("RuntimeError: ")
    assert [e.name for e in root.events] == ["exception"]


def _raising_reranker(error: BaseException) -> BaseDocumentCompressor:
    class _RaisingReranker(BaseDocumentCompressor):
        def compress_documents(self, documents, query, callbacks=None):
            raise error

    return _RaisingReranker()


@pytest.mark.parametrize(
    ("error", "status", "events"),
    [
        pytest.param(
            RuntimeError("reranking failed"),
            StatusCode.ERROR,
            ["exception"],
            id="failed",
        ),
        pytest.param(KeyboardInterrupt(), StatusCode.UNSET, ["stopped"], id="ctrl-c"),
    ],
)
def test_a_question_ended_during_retrieval_still_ends_its_trace(
    settings, fake_embeddings, canned_answer, spans, error, status, events
):
    """Retrieval ends before there is a stream to end the span, so
    stream_answer must end it itself -- for a failure, and for Ctrl-C at the
    terminal while the question is embedded or reranked, which the CLI treats as
    an ordinary way out. The exception is recorded once, not by both the
    retrieval's scope and the span's end."""
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        _raising_reranker(error),
    )

    with pytest.raises(type(error)):
        pipeline.stream_answer(_QUESTION)

    root = _named(spans.get_finished_spans(), "RAGPipeline")
    assert root.status.status_code is status
    assert [e.name for e in root.events] == events


def test_a_failed_answer_keeps_what_was_generated(
    settings,
    fake_embeddings,
    fake_reranker,
    canned_answer,
    spans,
    fail_mid_stream,
    partial_answer,
):
    """The text before a failure is the most useful part of a failed trace."""
    fail_mid_stream(RuntimeError("the model died"))
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    with pytest.raises(RuntimeError, match="the model died"):
        pipeline.answer(_QUESTION)

    root = _named(spans.get_finished_spans(), "RAGPipeline")
    assert root.status.status_code is StatusCode.ERROR
    assert _attributes(root)["output.value"] == partial_answer


def test_the_question_span_is_never_current_between_pieces(
    settings, fake_embeddings, fake_reranker, canned_answer, spans
):
    """Made current around each step of generation, never across a yield.

    Held across one, it would be the current span in the consumer's code too --
    the app's rendering, the CLI's printing -- and anything traced there would
    be filed under the question.
    """
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )

    _docs, chunks = pipeline.stream_answer(_QUESTION)
    current = [trace.get_current_span().get_span_context().is_valid]
    for _piece in chunks:
        current.append(trace.get_current_span().get_span_context().is_valid)

    assert not any(current)


def test_a_stream_closed_from_another_thread_ends_cleanly(
    settings, fake_embeddings, fake_reranker, canned_answer, spans, caplog
):
    """How a Stop can arrive: the stream closed from outside the frame that
    read it. A span held across a yield fails to detach there, and logs it."""
    pipeline = _pipeline(
        settings,
        fake_embeddings,
        FakeListChatModel(responses=[canned_answer]),
        fake_reranker,
    )
    _docs, chunks = pipeline.stream_answer(_QUESTION)
    next(chunks)

    with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
        closer = threading.Thread(target=chunks.close)
        closer.start()
        closer.join()

    assert "Failed to detach context" not in caplog.text
    root = _named(spans.get_finished_spans(), "RAGPipeline")
    assert [e.name for e in root.events] == ["stopped"]


# --- installing the provider -------------------------------------------------


@pytest.fixture
def provider_exit_hooks(monkeypatch) -> list[object]:
    """The exit hooks of tracer providers built during the test, while still
    registered.

    A provider registers one -- its own shutdown, the exit-time flush -- as it is
    built, and unregisters it when shut down. One left registered keeps its
    provider alive, to be flushed at exit, however long ago it was abandoned.
    `atexit` cannot be asked what it holds, so the calls are watched instead.
    """
    hooks: list[object] = []
    register, unregister = atexit.register, atexit.unregister

    def watched_register(func, /, *args, **kwargs):
        if isinstance(getattr(func, "__self__", None), SDKTracerProvider):
            hooks.append(func)
        return register(func, *args, **kwargs)

    def watched_unregister(func, /) -> None:
        if func in hooks:
            hooks.remove(func)
        unregister(func)

    monkeypatch.setattr(atexit, "register", watched_register)
    monkeypatch.setattr(atexit, "unregister", watched_unregister)
    return hooks


@pytest.mark.parametrize(
    ("endpoint", "url"),
    [
        ("http://localhost:6006", "http://localhost:6006/v1/traces"),
        ("http://localhost:6006/", "http://localhost:6006/v1/traces"),
        ("https://proxy.example/phoenix", "https://proxy.example/phoenix/v1/traces"),
        ("http://localhost:6006/v1/traces", "http://localhost:6006/v1/traces"),
        ("http://localhost:6006/v1/traces/", "http://localhost:6006/v1/traces"),
    ],
)
def test_the_collector_url_is_the_endpoint_plus_v1_traces(endpoint, url):
    """Phoenix's collector is `/v1/traces` under its base URL; posted to the
    base URL itself, every span is refused (405) and the exporter logs only
    that the batch failed."""
    assert traces_url(endpoint) == url


def test_with_no_endpoint_tracing_stays_off(settings, tracing_left_on):
    setup_tracing(settings)

    assert tracing_left_on() == []


def test_setup_installs_one_provider_for_the_process(
    settings, undo_tracing, caplog, provider_exit_hooks
):
    """The app calls setup on every rerun, from every session: the first call
    installs, and every later one is a no-op -- not a second provider, which
    OpenTelemetry would refuse with a warning and leave running beside it."""
    traced = dataclasses.replace(
        settings,
        phoenix_collector_endpoint="http://127.0.0.1:9",
        phoenix_project="probe-project",
    )

    with caplog.at_level(logging.WARNING):
        setup_tracing(traced)
        provider = trace.get_tracer_provider()
        setup_tracing(traced)

    assert trace.get_tracer_provider() is provider
    # OpenInference's provider: the SDK's would cut a reranker span over a
    # large FETCH_K to 128 attributes.
    assert isinstance(provider, TracerProvider)
    # Its exit flush, and only its -- which also shows `provider_exit_hooks`
    # seeing the SDK register one, so the empty list a failed setup must leave,
    # below, cannot come from a watch that sees nothing.
    assert provider_exit_hooks == [provider.shutdown]
    assert provider.resource.attributes["openinference.project.name"] == (
        "probe-project"
    )
    assert LangChainInstrumentor().is_instrumented_by_opentelemetry
    assert not caplog.records, [r.getMessage() for r in caplog.records]


# Values the SDK refuses rather than logging and falling back to a default: one
# while the provider is built, one after. An unknown compression was one until
# OpenTelemetry 1.45, which logs it and sends uncompressed. Not
# OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT: the SDK also reads that one as it is imported,
# which chromadb does, so in the app a malformed one fails before tracing is
# reached. And the exporter's own refusal, of a credential provider that is not
# installed: a RuntimeError already, but one that comes after the provider is
# built, and names no variable.
@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("OTEL_ATTRIBUTE_COUNT_LIMIT", "abc"),
        ("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "4096"),
        ("OTEL_PYTHON_EXPORTER_OTLP_HTTP_CREDENTIAL_PROVIDER", "no-such-provider"),
    ],
)
def test_a_malformed_otel_variable_is_a_runtime_error_that_installs_nothing(
    settings, monkeypatch, tracing_left_on, provider_exit_hooks, variable, value
):
    """The SDK refuses some malformed OTEL_* variables itself, with a builtins
    ValueError -- which, on the app's pipeline-load path, would escape its guard
    as a crash page on every rerun. Translated, and with nothing installed, so
    tracing stays off and the next rerun says the same thing. Nothing left
    registered to run at exit either, or every failed rerun would strand one
    more provider, kept alive until the process ends."""
    monkeypatch.setenv(variable, value)
    threads = sum(
        t.name == "OtelBatchSpanRecordProcessor" for t in threading.enumerate()
    )
    traced = dataclasses.replace(
        settings, phoenix_collector_endpoint="http://127.0.0.1:9"
    )

    with pytest.raises(RuntimeError, match="OTEL_"):
        setup_tracing(traced)

    assert tracing_left_on() == []
    assert (
        sum(t.name == "OtelBatchSpanRecordProcessor" for t in threading.enumerate())
        == threads
    ), "a failed setup left an exporter thread running"
    assert provider_exit_hooks == [], (
        "a failed setup left a provider's exit hook registered"
    )


def test_concurrent_first_setups_install_once(settings, undo_tracing, caplog):
    """Streamlit sessions can rerun at the same moment -- after a server
    restart, every open tab does -- and each calls setup. One provider must
    come of it, not one per session, each with its own exporter thread and exit
    flush, with LangChain's spans going to whichever instrumented first."""
    traced = dataclasses.replace(
        settings, phoenix_collector_endpoint="http://127.0.0.1:9"
    )
    before = sum(
        t.name == "OtelBatchSpanRecordProcessor" for t in threading.enumerate()
    )
    barrier = threading.Barrier(6)

    def first_run() -> None:
        barrier.wait()
        setup_tracing(traced)

    with caplog.at_level(logging.WARNING):
        sessions = [threading.Thread(target=first_run) for _ in range(6)]
        for session in sessions:
            session.start()
        for session in sessions:
            session.join()

    after = sum(t.name == "OtelBatchSpanRecordProcessor" for t in threading.enumerate())
    assert after - before == 1
    assert not caplog.records, [r.getMessage() for r in caplog.records]


# --- on the wire -------------------------------------------------------------

# Installs tracing for real in a fresh interpreter, against a stand-in collector
# that decodes what it is sent -- or, with "down", against a port nothing
# listens on -- then runs the flush a process makes at exit, timed.
_WIRE_PROBE = r"""
import json, socket, sys, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)

received = []


class Collector(BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        for resource_spans in ExportTraceServiceRequest.FromString(body).resource_spans:
            resource = {
                a.key: a.value.string_value for a in resource_spans.resource.attributes
            }
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    received.append({
                        "path": self.path,
                        "content_type": self.headers["Content-Type"],
                        "project": resource.get("openinference.project.name"),
                        "name": span.name,
                        "attributes": len(span.attributes),
                    })
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


if sys.argv[1] == "up":
    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
else:
    with socket.socket() as probe:  # bound, then closed: nothing listens there
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

from langchain_core.runnables import RunnableLambda
from opentelemetry import trace

from rag_pipeline.config import Settings
from rag_pipeline.tracing import setup_tracing

settings = Settings(
    phoenix_collector_endpoint=f"http://127.0.0.1:{port}",
    phoenix_project="probe-project",
)
setup_tracing(settings)
setup_tracing(settings)
emitting = time.monotonic()
RunnableLambda(lambda x: x).invoke("ping")
trace.get_tracer("probe").start_span(
    "wide", attributes={f"passage.{i}": i for i in range(200)}
).end()

start = time.monotonic()
trace.get_tracer_provider().shutdown()
print(json.dumps({
    "received": received,
    "emit_s": start - emitting,
    "flush_s": time.monotonic() - start,
}))
"""


def _run_wire_probe(collector: str) -> tuple[dict, str, str]:
    # The developer's own settings must not reach the child: their exporter
    # variables would change what is sent, and a proxy would carry the post to
    # 127.0.0.1 somewhere else (up to OpenTelemetry 1.44, whose exporter posted
    # through requests; 1.45's urllib3 transport ignores proxy variables).
    # Scrubbing the environment is not enough on its own -- config.py's
    # load_dotenv() would read .env straight back in -- so .env is switched off
    # too. (LangSmith's switches arrive as "false", from conftest's
    # _no_tracing.)
    env = {k: v for k, v in os.environ.items() if not k.startswith("OTEL_")}
    env["PYTHON_DOTENV_DISABLED"] = "1"
    env["NO_PROXY"] = env["no_proxy"] = "*"
    result = subprocess.run(
        [sys.executable, "-c", _WIRE_PROBE, collector],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
        env=env,
    )
    return json.loads(result.stdout.splitlines()[-1]), result.stdout, result.stderr


def test_traces_reach_the_collector_as_phoenix_expects_them():
    """What Phoenix needs to file a trace, checked where it counts: on the wire.

    OTLP over HTTP, posted to /v1/traces (not the base URL, which Phoenix
    refuses), under the configured project, which is how Phoenix picks the
    project to file it in. LangChain's runs arrive as well as spans opened
    through the API -- so the instrumentation is on -- each exactly once,
    though setup ran twice. A wide span arrives whole. And nothing is printed:
    `rag query`'s stdout is its answer.
    """
    report, stdout, stderr = _run_wire_probe("up")

    received = report["received"]
    assert sorted(span["name"] for span in received) == ["RunnableLambda", "wide"]
    assert {span["path"] for span in received} == {"/v1/traces"}
    assert {span["content_type"] for span in received} == {"application/x-protobuf"}
    assert {span["project"] for span in received} == {"probe-project"}
    wide = next(span for span in received if span["name"] == "wide")
    assert wide["attributes"] == 200
    assert stdout.count("\n") == 1, stdout
    assert "Overriding" not in stderr
    assert "already instrumented" not in stderr


def test_with_phoenix_down_questions_do_not_wait_and_exit_waits_briefly():
    """What tracing costs when Phoenix is not running.

    Nothing while the question runs: spans are only queued, and exported from
    a background thread -- a processor that exported inside span.end() would
    hold each of a question's spans for the whole retry budget. And little at
    exit: the exporter retries until its timeout, which at the default (10 s)
    holds a finished `rag query` for about 7 s, and at the one set here for
    about 1. The failure is still said, on stderr: the batch's final failure,
    not only the retry warnings before it.
    """
    report, _stdout, stderr = _run_wire_probe("down")

    assert report["received"] == []
    assert report["emit_s"] < 0.5, f"emitting two spans took {report['emit_s']:.1f}s"
    assert report["flush_s"] < 4, f"the exit flush took {report['flush_s']:.1f}s"
    # "span batch" up to OpenTelemetry 1.44, "spans batch" from 1.45; a retry
    # warning never says "Failed to export".
    assert "Failed to export span" in stderr, stderr
