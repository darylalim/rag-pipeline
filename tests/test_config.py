"""Tests for Settings defaults and environment-variable overrides."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from rag_pipeline.config import ENV_VARS, Settings

# The checkout this test file sits in. The path defaults are anchored to the
# repository, not the working directory, so `rag` behaves the same whichever
# directory it is run from.
_ROOT = Path(__file__).resolve().parents[1]

# Every field, so a setting added without a default here fails `test_defaults`
# rather than going untested.
_DEFAULTS = {
    "data_dir": _ROOT / "data",
    "persist_dir": _ROOT / "chroma_db",
    "collection_name": "rag_docs",
    "embedding_model": "mlx-community/Qwen3-VL-Embedding-2B-bf16",
    "embedding_dimensions": 2048,
    "chat_model": "mlx-community/Qwen3.8-27B-4bit",
    "max_tokens": 1024,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retrieval_k": 4,
    "fetch_k": 20,
    "rerank_model": "mlx-community/Qwen3-VL-Reranker-2B-bf16",
    # Empty: tracing is off unless asked for, so a fresh checkout sends nothing.
    "phoenix_collector_endpoint": "",
    "phoenix_project": "rag-pipeline",
}


def test_defaults():
    assert dataclasses.asdict(Settings()) == _DEFAULTS


def test_from_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAT_MODEL", "mlx-community/Qwen3-8B-4bit")
    monkeypatch.setenv("RERANK_MODEL", str(tmp_path / "reranker"))
    monkeypatch.setenv("RETRIEVAL_K", "7")
    monkeypatch.setenv("FETCH_K", "30")
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "256")
    monkeypatch.setenv("COLLECTION_NAME", "custom")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.internal:6006")
    monkeypatch.setenv("PHOENIX_PROJECT", "docs-qa")
    # Relative, so the resolution is observable: a path setting is fixed to an
    # absolute path when read, not reinterpreted against whatever directory a
    # later call happens to run in.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PERSIST_DIR", "index")

    s = Settings.from_env()

    assert s.chat_model == "mlx-community/Qwen3-8B-4bit"
    # A model setting is a string passed through as-is: a local directory is as
    # valid as a repo id, and resolving either is the loader's job.
    assert s.rerank_model == str(tmp_path / "reranker")
    assert s.retrieval_k == 7
    assert isinstance(s.retrieval_k, int)
    assert s.fetch_k == 30
    assert s.chunk_size == 512
    assert s.embedding_dimensions == 256
    assert s.collection_name == "custom"
    assert s.data_dir == tmp_path.resolve()
    assert s.persist_dir == (tmp_path / "index").resolve()
    # Kept as given: the collector path is appended where spans are sent, so a
    # base URL and one naming a proxy prefix both mean what they say.
    assert s.phoenix_collector_endpoint == "http://phoenix.internal:6006"
    assert s.phoenix_project == "docs-qa"


@pytest.mark.parametrize("var", ["DATA_DIR", "PERSIST_DIR"])
def test_an_unusable_path_setting_is_a_value_error_naming_it(monkeypatch, var):
    """pathlib signals a `~user` with no home directory as a RuntimeError.

    Left as that, it would slip past the ValueError guard app.py puts around
    Settings -- the one that stops above the sidebar with "Fix it" -- and reach
    the user as a crash page instead. A malformed setting is a ValueError,
    whichever reader found it.
    """
    monkeypatch.setenv(var, "~no_such_user_xyz/somewhere")

    with pytest.raises(ValueError, match=var):
        Settings.from_env()


@pytest.mark.parametrize(
    "endpoint",
    [
        "localhost:6006",  # no scheme: urlsplit reads "localhost" as one
        "grpc://localhost:4317",
        "http://",
        "http://localhost:not-a-port",
        "http://localhost:0",
    ],
)
def test_an_unusable_phoenix_endpoint_is_a_value_error_naming_it(monkeypatch, endpoint):
    """Refused where it is read, not where spans are sent.

    The exporter takes any string, and one it cannot post to fails only on
    export, as a log line on a background thread -- every trace lost, and
    nothing on screen to say why. A ValueError here is what app.py stops on
    above its sidebar and what the CLI prints as its one-line error.
    """
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)

    with pytest.raises(ValueError, match="PHOENIX_COLLECTOR_ENDPOINT"):
        Settings.from_env()


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://localhost:6006",
        "https://phoenix.example.com",
        "http://127.0.0.1:6006/phoenix/",  # behind a reverse proxy
        "http://[::1]:6006",
    ],
)
def test_a_usable_phoenix_endpoint_is_accepted(monkeypatch, endpoint):
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)

    assert Settings.from_env().phoenix_collector_endpoint == endpoint


def test_from_env_uses_defaults_when_unset(monkeypatch):
    # ENV_VARS, not a hand-kept list: config.py loads .env at import time, so a
    # name missing here would be answered by the developer's own .env and this
    # test would keep passing while no longer covering that default.
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    assert dataclasses.asdict(Settings.from_env()) == _DEFAULTS


def test_from_env_empty_string_falls_back_to_default(monkeypatch):
    # A set-but-empty var should fall back to the default, not pass "" through --
    # for every setting, so each of the str/int/path readers is covered.
    for var in ENV_VARS:
        monkeypatch.setenv(var, "")

    assert dataclasses.asdict(Settings.from_env()) == _DEFAULTS


def test_env_vars_are_exactly_what_from_env_reads(monkeypatch):
    """ENV_VARS is derived from the fields, so it can only be as right as the
    reads in `from_env` are.

    Recorded rather than compared by eye: a read of a variable that is no
    longer a field -- a leftover from a removed store or provider -- would be
    configuration nothing documents and no test clears, and a field read under
    a misspelled name would never be overridable at all.
    """
    read: list[str] = []
    getenv = os.getenv

    def recording_getenv(name, default=None):
        read.append(name)
        return getenv(name, default)

    monkeypatch.setattr(os, "getenv", recording_getenv)

    Settings.from_env()

    assert sorted(read) == sorted(ENV_VARS)
