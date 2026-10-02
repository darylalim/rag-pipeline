"""conftest's guards are the offline guarantee -- assert they actually hold.

A test that forgets to inject a fake must *fail*, not quietly call a real model.
Every model is an API -- the embedder and the reranker Voyage AI's, the chat
model Anthropic's -- so two guards cover them all: ``_no_real_store`` removes
the API keys, so each factory stops before it builds a client, and ``_offline``
blocks the socket besides. Either silently loosening reads as green everywhere
else, so each route to a real model is tripped here on purpose
-- every factory, and both entry points that build one when a fake is left out
-- and the socket block is checked on its own. So is the tracing guard: a
client sending from a background thread is the one route out the socket block
cannot stop, since that thread logs the error and carries on. And a tracing
context a test leaves behind must fail only that test, not every test after.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.language_models import FakeListChatModel
from langsmith import tracing_context
from langsmith.run_trees import RunTree

from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline.config import Settings
from rag_pipeline.pipeline import RAGPipeline
from tests.conftest import _LIVE_CREDENTIALS, is_live

# The key each factory needs, and so the one its refusal must name.
_KEY_OF = {
    "embedding_model": "VOYAGE_API_KEY",
    "rerank_model": "VOYAGE_API_KEY",
    "chat_model": "ANTHROPIC_API_KEY",
}


def _assert_stopped_by_the_key_guard(
    excinfo: pytest.ExceptionInfo[RuntimeError], setting: str
) -> None:
    # Exact type, and the right key: the factory refused before building a
    # client, so no request was made -- and the message says which fake was
    # forgotten.
    assert excinfo.type is RuntimeError
    assert str(excinfo.value).startswith(f"{_KEY_OF[setting]} is not set")


# --- the key guard -----------------------------------------------------------


def test_no_api_key_reaches_a_test():
    """What every other check here rests on: config.py loaded the developer's
    .env at import, and `_no_real_store` has taken the keys back out."""
    assert not {"VOYAGE_API_KEY", "ANTHROPIC_API_KEY"} & set(os.environ)


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
    """Each factory stops at its missing key, before any client exists."""
    with pytest.raises(RuntimeError) as excinfo:
        getattr(module, factory)(settings)

    _assert_stopped_by_the_key_guard(excinfo, setting)


def test_an_ingest_without_injected_embeddings_is_stopped(settings):
    with pytest.raises(RuntimeError) as excinfo:
        ingest_mod.ingest(settings)

    _assert_stopped_by_the_key_guard(excinfo, "embedding_model")


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

    _assert_stopped_by_the_key_guard(excinfo, setting)


@pytest.mark.parametrize("marked", [False, True])
def test_only_a_live_marked_test_is_exempt(request, marked):
    """The exemption is exactly the ``live`` marker, in both directions.

    The live tests call the real APIs and would all fail under the guards;
    every other test must stay under them. Checked through the guards' own
    decision on this test's node, because a test that is really marked is
    deselected here.
    """
    if marked:
        request.node.add_marker(pytest.mark.live)

    assert is_live(request.node) is marked


def test_no_test_keeps_the_developers_cluster():
    """Even the exemption keeps only the API keys: a live test still writes to
    the test container, never to the cluster in the developer's .env."""
    assert "MONGODB_URI" not in _LIVE_CREDENTIALS


# --- the socket block --------------------------------------------------------


@pytest.mark.parametrize(
    "address",
    [
        pytest.param(("api.anthropic.com", 443), id="the-anthropic-api"),
        pytest.param(("api.voyageai.com", 443), id="the-voyage-api"),
        pytest.param(("cluster0.example.mongodb.net", 27017), id="an-atlas-cluster"),
        pytest.param(("10.0.0.1", 27017), id="the-local-network"),
    ],
)
def test_create_connection_is_blocked(address):
    """The route every HTTP client takes, the model APIs' included.

    Refused before the name is resolved, so not even a DNS lookup leaves. It
    raises RuntimeError rather than OSError on purpose -- httpcore turns an
    OSError into a ConnectError, which the SDKs retry with backoff and then
    report as an ordinary outage, so a blocked call would read as a slow,
    flaky network rather than as a test that reached for one.
    """
    with pytest.raises(RuntimeError, match="network socket"):
        socket.create_connection(address, timeout=1)


@pytest.mark.parametrize("method", ["connect", "connect_ex"])
def test_a_bare_socket_cannot_connect(method):
    """The other route: a socket opened directly, patched at the method level."""
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="network socket"):
            getattr(sock, method)(("10.0.0.1", 9))
    finally:
        sock.close()


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_this_machine_is_reachable(host):
    """The one opening: the store is the atlas-local container, on loopback.

    Nothing listens on port 9, so the connection is refused -- by the operating
    system, which is the proof that the guard let it through.
    """
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((host, 9), timeout=1)


def test_the_developers_cluster_is_out_of_reach():
    """config.py loads .env at import, so a developer's real MONGODB_URI is in
    the environment before any test runs; `_no_real_store` takes it out, so a
    test that forgets the `atlas` fixture cannot write into a real cluster."""
    assert "MONGODB_URI" not in os.environ

    with pytest.raises(RuntimeError, match="MONGODB_URI is not set"):
        ingest_mod.index_version(Settings())


# --- the tracing guard -------------------------------------------------------


def test_tracing_is_off_whatever_the_env_file_says():
    """`_no_tracing` and `_no_real_store`, seen from inside a test.

    The switch both frontends read is off, so neither builds a client; every
    other spelling LangSmith acts on by itself is off too; and the key that
    would let anything through is gone.
    """
    assert Settings.from_env().langsmith_tracing is False
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ


def test_every_test_is_checked_for_tracing_left_on(request):
    """The tripwire must run after every test, not only where requested: a
    test that leaves a tracing context behind does not know it."""
    assert "_no_tracer_left_on" in request.fixturenames


def test_the_tripwire_sees_a_context_left_open(tracing_left_on):
    """Tripped for real: a tracing context entered and not yet left -- what a
    context held across a yield amounts to -- is a current run and tracing
    switched on, and the check reports both. Left properly afterwards, so this
    test does not trip the guard itself. (Nothing is posted: the run is
    never sent.)"""
    leaked = tracing_context(enabled=True, parent=RunTree(name="leaked"))
    leaked.__enter__()
    try:
        assert tracing_left_on() == [
            "run 'leaked' is current",
            "tracing is set to True",
        ]
    finally:
        leaked.__exit__(None, None, None)
    assert tracing_left_on() == []


# A session whose first test leaks a tracing context. Left in place, the leak
# would fail both tests after it: the one that ignores tracing, by tripping the
# same teardown check, and the one that looks, by finding a run still current.
_LEAK_THEN_CARRY_ON = """
from langsmith import tracing_context
from langsmith.run_helpers import get_current_run_tree
from langsmith.run_trees import RunTree


# Held: a context manager nothing refers to is collected, and collecting it
# runs its exit -- which would undo the leak before the guard could see it.
_held = []


def test_leaks():
    leaked = tracing_context(enabled=True, parent=RunTree(name="leaked"))
    leaked.__enter__()
    _held.append(leaked)


def test_ignores_tracing():
    pass


def test_finds_no_run_current():
    assert get_current_run_tree() is None
"""


def test_a_leak_fails_only_the_test_that_left_tracing_on(pytester, monkeypatch):
    """The tripwire clears a leak before failing the test that left it, so the
    tests after it run as if it had never happened.

    A pytest process of its own, with this conftest as its plugin: a context
    left set in this process would leak into this run's tests instead. This checkout goes on the path ahead of the one the
    environment has installed, so the session loads the conftest under test.
    """
    monkeypatch.setenv(
        "PYTHONPATH", str(Path(__file__).resolve().parents[1]), prepend=os.pathsep
    )
    pytester.makepyfile(_LEAK_THEN_CARRY_ON)
    pytester.plugins.append("tests.conftest")

    result = pytester.runpytest_subprocess("-p", "no:cacheprovider", timeout=60)

    # The guard fails from its teardown, which pytest reports as an error of the
    # leaking test, whose body passed; the other two pass whole.
    result.assert_outcomes(passed=3, errors=1)
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_leaks*",
            "E *AssertionError: the test left tracing on: *",
        ]
    )
