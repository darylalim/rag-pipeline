"""conftest's guards are the offline guarantee -- assert they actually hold.

A test that forgets to inject a fake must *fail*, not quietly run a real model.
With the models local, a network block alone would never notice: weights
already in the Hugging Face cache load without opening a socket. So
``_no_real_models`` makes MLX unimportable, and ``_offline`` separately blocks
every socket, for whatever still reaches for the network -- a download,
telemetry, Chroma's default embedder. Either one silently loosening reads as
green everywhere else, so each route to a real model is tripped here on purpose
-- every factory, and both entry points that build one when a fake is left out
-- and the socket block is checked on its own. So is the tracing guard: an
exporter is the one route out the socket block cannot stop, since the tracing
SDK catches its error.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import socket
import sys
from types import ModuleType
from typing import Any

import pytest
from langchain_core.language_models import FakeListChatModel

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline
from rag_pipeline.tracing import setup_tracing


def _assert_stopped_by_the_mlx_guard(
    excinfo: pytest.ExceptionInfo[RuntimeError], model_id: str
) -> None:
    # Exact type: the loader's RuntimeError, not a FileNotFoundError from a
    # cache lookup it should never have reached. And the model id, so the
    # message says which fake was forgotten.
    assert excinfo.type is RuntimeError
    assert "MLX" in str(excinfo.value)
    assert model_id in str(excinfo.value)


# --- the model guard ---------------------------------------------------------


@pytest.mark.parametrize("module", ["mlx_lm", "mlx.core"])
def test_mlx_cannot_be_imported(module):
    """What every other check here rests on: MLX is absent, as on the Linux legs."""
    with pytest.raises(ImportError):
        importlib.import_module(module)


@pytest.mark.parametrize(
    ("module", "factory", "setting"),
    [
        pytest.param(ingest_mod, "build_embeddings", "embedding_model", id="embedder"),
        pytest.param(pipeline_mod, "build_reranker", "rerank_model", id="reranker"),
        pytest.param(pipeline_mod, "build_chat_model", "chat_model", id="chat-model"),
    ],
)
def test_every_model_factory_is_stopped_by_the_guard(
    settings, module: ModuleType, factory: str, setting: str
):
    """Each factory reaches the loader, and the loader stops at the MLX import.

    Run against the real default model ids, which may well be fully cached on
    the machine running this: the guard has to fire before the cache is looked
    at, or it protects only the machines that never downloaded anything.
    """
    with pytest.raises(RuntimeError) as excinfo:
        getattr(module, factory)(settings)

    _assert_stopped_by_the_mlx_guard(excinfo, getattr(settings, setting))


def test_an_ingest_without_injected_embeddings_is_stopped(settings):
    with pytest.raises(RuntimeError) as excinfo:
        ingest_mod.ingest(settings)

    _assert_stopped_by_the_mlx_guard(excinfo, settings.embedding_model)


@pytest.mark.parametrize(
    ("forgotten", "setting"),
    [
        ("embeddings", "embedding_model"),
        ("reranker", "rerank_model"),
        ("llm", "chat_model"),
    ],
)
def test_a_pipeline_missing_any_one_fake_is_stopped(
    settings, fake_embeddings, fake_reranker, forgotten: str, setting: str
):
    """Leaving out any one of the three is enough to reach a real model.

    Against a real index, so the pipeline's own guards pass and construction
    gets as far as the model factories -- with no index it would stop earlier,
    at the missing-index check, and prove nothing about them.
    """
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    fakes: dict[str, Any] = {
        "embeddings": fake_embeddings,
        "reranker": fake_reranker,
        "llm": FakeListChatModel(responses=["unused"]),
    }
    del fakes[forgotten]

    with pytest.raises(RuntimeError) as excinfo:
        RAGPipeline(settings, **fakes)

    _assert_stopped_by_the_mlx_guard(excinfo, getattr(settings, setting))


@pytest.mark.parametrize(("marked", "hidden"), [(False, True), (True, False)])
def test_only_a_models_marked_test_may_import_mlx(request, hide_mlx, marked, hidden):
    """The exemption is exactly the ``models`` marker, in both directions.

    The live tests load the real checkpoints and would all fail under the guard;
    every other test must stay under it. Checked through the guard's own logic on
    this test's node, because a test that is really marked is deselected here.
    """
    names = ("mlx", "mlx_lm")
    if marked:
        request.node.add_marker(pytest.mark.models)
    with pytest.MonkeyPatch.context() as mp:
        # Start from a clean slate: the autouse guard has already run for this
        # (unmarked, at the time) test and left None entries behind.
        for name in names:
            mp.delitem(sys.modules, name, raising=False)

        hide_mlx(request.node, mp)

        blocked = [name in sys.modules and sys.modules[name] is None for name in names]
    assert blocked == [hidden] * len(names)


# --- the socket block --------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        pytest.param(("huggingface.co", 443), id="model-download"),
        pytest.param(("127.0.0.1", 9), id="loopback"),
    ],
)
def test_create_connection_is_blocked(address):
    """The route every HTTP client takes, a Hugging Face download included.

    Loopback included: nothing in the suite needs a socket any more, so the
    block is blanket rather than an allowlist with holes to keep correct. It
    raises RuntimeError rather than OSError on purpose -- httpcore turns an
    OSError into a ConnectError, which huggingface_hub catches and answers by
    falling back to the local cache, so an OSError would turn a blocked
    download into a silent load of whatever is cached.
    """
    with pytest.raises(RuntimeError, match="network socket"):
        socket.create_connection(address, timeout=1)


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_a_bare_socket_cannot_connect(method):
    """The other route: a socket opened directly, patched at the method level."""
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="network socket"):
            getattr(sock, method)(("127.0.0.1", 9))
    finally:
        sock.close()


# --- the tracing guard -------------------------------------------------------


def test_tracing_is_off_whatever_the_env_file_says():
    """`_no_tracing`, seen from inside a test.

    The endpoint both frontends read is gone, so neither installs an exporter,
    and LangSmith -- which langchain-core still carries, and which still acts
    on its own switch -- is off.
    """
    assert "PHOENIX_COLLECTOR_ENDPOINT" not in os.environ
    assert Settings.from_env().phoenix_collector_endpoint == ""
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_every_test_is_checked_for_tracing_left_on(request):
    """The tripwire must run after every test, not only where requested: a
    test that forgets to undo tracing does not know it."""
    assert "_no_tracer_left_on" in request.fixturenames


def test_the_tripwire_sees_what_a_real_setup_leaves(
    settings, undo_tracing, tracing_left_on
):
    """Tripped for real: tracing set up with an endpoint leaves a provider and
    LangChain's hook installed process-wide, and the check reports both. (No
    span is emitted, so the exporter never reaches for its socket.)"""
    setup_tracing(
        dataclasses.replace(settings, phoenix_collector_endpoint="http://127.0.0.1:9")
    )

    assert tracing_left_on() == [
        "a tracer provider is installed",
        "LangChain is instrumented",
    ]
