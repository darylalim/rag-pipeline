"""Tests for Settings defaults and environment-variable overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rag_pipeline.config import ENV_VARS, Settings, require_env_key

# The checkout this test file sits in. The path defaults are anchored to the
# repository, not the working directory, so `rag` behaves the same whichever
# directory it is run from.
_ROOT = Path(__file__).resolve().parents[1]


def test_defaults(fresh_interpreter):
    """The defaults that are properties rather than choices.

    Each default is declared once, in config.py, and
    `test_every_setting_is_documented` holds the README and .env.example to it,
    so a default changed without its documentation already fails. A copy of
    every value here was a fourth place to change one, which the three-file
    recipe for a setting does not name. What stays pinned is what the documents
    cannot show: that the paths are anchored to this checkout at run time (they
    document `./data`), and that tracing starts off.

    The paths are read in an interpreter started in another directory: this one
    runs from the checkout, where paths anchored to the working directory would
    come out the same.
    """
    result = fresh_interpreter(
        "import json\n"
        "from rag_pipeline.config import Settings\n"
        "print(json.dumps([str(Settings().data_dir), str(Settings().persist_dir)]))\n"
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [
        str(_ROOT / "data"),
        str(_ROOT / "chroma_db"),
    ]
    # Empty: tracing is off unless asked for, so a fresh checkout sends nothing.
    assert Settings().phoenix_collector_endpoint == ""


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

    Left as that, it would slip past the ValueError guard streamlit_app.py puts around
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
    nothing on screen to say why. A ValueError here is what streamlit_app.py stops on
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

    assert Settings.from_env() == Settings()


def test_from_env_empty_string_falls_back_to_default(monkeypatch):
    # A set-but-empty var should fall back to the default, not pass "" through --
    # for every setting, so each of the str/int/path readers is covered.
    for var in ENV_VARS:
        monkeypatch.setenv(var, "")

    assert Settings.from_env() == Settings()


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


# The credentials the cloud migration reads. Exemplary rather than exhaustive:
# what is tested is how `require_env_key` treats any name, and these are the
# ones a developer's .env is most likely to hold -- which is why each test
# deletes or sets them explicitly rather than trusting the environment.
_CREDENTIALS = (
    "ANTHROPIC_API_KEY",
    "VOYAGE_API_KEY",
    "MONGODB_URI",
    "LANGSMITH_API_KEY",
)


@pytest.mark.parametrize("name", _CREDENTIALS)
@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_a_missing_credential_is_a_runtime_error_naming_it(monkeypatch, name, value):
    # RuntimeError, not ValueError: a key is first needed on the pipeline-load
    # path, where the app's guard catches only FileNotFoundError | RuntimeError.
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=rf"^{name} is not set\. ") as excinfo:
        require_env_key(name, "Answers come from Claude")

    message = str(excinfo.value)
    assert "Answers come from Claude; set it" in message
    assert ".env.example" in message


def test_a_set_credential_is_returned(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "mongodb+srv://user:pw@cluster.example.net")

    assert (
        require_env_key("MONGODB_URI", "The index is stored in Atlas")
        == "mongodb+srv://user:pw@cluster.example.net"
    )


def test_a_credential_never_reaches_settings(monkeypatch):
    """No key is a Settings field, so none is displayed or printed with them.

    The sidebar renders Settings values, and a traceback through any function
    holding a Settings prints its repr. A key read into a field -- under its
    own name or any other -- would carry the sentinel into that repr.
    """
    sentinel = "sk-sentinel-must-not-appear"
    for name in _CREDENTIALS:
        monkeypatch.setenv(name, sentinel)

    assert sentinel not in repr(Settings.from_env())
