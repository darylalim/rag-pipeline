"""Tests for the terminal frontend.

`cli.py` is the entry point with no return value to assert on: what it produces
is stdout, stderr, and an exit code, and those *are* its contract -- a script
that pipes `rag query` into something reads the sources block, and a shell that
branches on failure reads the status. Both are invisible to every other test in
the suite, which asserts on what functions return.

The mapping at `cli.py`'s handler is the reason this file is mostly about
failure. `FileNotFoundError | RuntimeError | ValueError` is the union both
frontends catch, and here is where it becomes an exit code and a one-line
message instead of a traceback -- so each member is exercised through the real
command rather than trusted to stay caught.

Wired through the same seam as `test_app.py`: `cli.py` builds its own Settings
from the environment and reaches models only through the two factories, so
setting the environment and patching the factories keeps these tests inside the
suite's offline guarantee.
"""

from __future__ import annotations

import pytest

from rag_pipeline import cli
from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod


@pytest.fixture
def indexed(wired_env, fake_embeddings):
    """A `wired_env` whose index already exists, as `rag query` requires."""
    ingest_mod.ingest(wired_env, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()
    return wired_env


# --- the happy paths ---------------------------------------------------------


def test_ingest_reports_where_it_wrote_and_how_much(wired_env, capsys):
    assert cli.main(["ingest"]) == 0

    out = capsys.readouterr().out
    assert "Indexed" in out
    assert "chunks" in out
    # The path is the actionable half: an ingest that silently wrote somewhere
    # else is the failure a user cannot otherwise see.
    assert str(wired_env.persist_dir) in out


def test_query_prints_the_answer_then_its_sources(indexed, capsys, canned_answer):
    assert cli.main(["query", "why overlap chunks?"]) == 0

    out = capsys.readouterr().out
    assert canned_answer in out
    # Sources come after the answer, so a reader meets the claim before its
    # provenance and a pipe can split on the header.
    assert out.index("Sources:") > out.index(canned_answer)
    assert "- a.md" in out


def test_the_sources_block_lists_each_file_once(
    tmp_path, wired_env, capsys, monkeypatch
):
    """Two chunks from one file must cite it once, not twice.

    `retrieval_k` counts chunks while the block lists files, so any corpus whose
    chunks outnumber its files exercises the difference -- and a citation list
    that repeats a filename reads as two independent sources for one claim.
    """
    root = tmp_path / "solo"
    root.mkdir()
    (root / "only.md").write_text("overlap. " * 200, encoding="utf-8")
    monkeypatch.setenv("DATA_DIR", str(root))

    assert cli.main(["ingest"]) == 0
    ingest_mod.reset_store_cache()
    capsys.readouterr()

    assert cli.main(["query", "overlap?"]) == 0

    sources = capsys.readouterr().out.split("Sources:")[1]
    assert sources.count("only.md") == 1


# --- the exception union, as exit codes --------------------------------------


def test_a_missing_index_is_an_error_not_a_traceback(wired_env, capsys):
    """FileNotFoundError: the query ran before any ingest."""
    assert cli.main(["query", "anything"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "Traceback" not in err
    # The message has to name the fix; this is the first thing a new user hits.
    assert "rag ingest" in err


def test_a_missing_api_key_is_an_error_not_a_traceback(indexed, capsys, monkeypatch):
    """RuntimeError: generation needs a key, and the guard fires before load."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Unpatch the chat factory: the guard only fires when the real client would
    # be built, which is exactly the production path this test is about.
    monkeypatch.setattr(pipeline_mod, "build_chat_model", pipeline_mod.build_chat_model)

    assert cli.main(["query", "anything"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "ANTHROPIC_API_KEY" in err


def test_a_malformed_numeric_setting_is_an_error_not_a_traceback(
    wired_env, capsys, monkeypatch
):
    """ValueError: raised by int() inside `from_env`, before any command runs.

    Inside the try for this reason -- `Settings.from_env()` is the first thing
    main() does, and a typo'd CHUNK_SIZE would otherwise be a raw traceback
    before the user has typed anything wrong about the command itself.
    """
    monkeypatch.setenv("CHUNK_SIZE", "not-a-number")

    assert cli.main(["ingest"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "Traceback" not in err


def test_a_failure_partway_through_the_stream_terminates_the_line(
    indexed, capsys, fail_mid_stream, partial_answer
):
    """The partial answer must not share a line with the error.

    Streaming means a provider failure lands with text already on screen, and
    `cmd_query` prints the closing newline from a `finally` for exactly that
    case. Without it main()'s "Error: ..." collides with the partial answer.
    """
    fail_mid_stream(RuntimeError("Claude API request failed: boom"))

    assert cli.main(["query", "anything"]) == 1

    captured = capsys.readouterr()
    assert captured.out.endswith("\n")
    assert partial_answer in captured.out
    assert captured.err.startswith("Error: ")


# --- argparse itself ---------------------------------------------------------


@pytest.mark.parametrize(
    "argv", [[], ["query"]], ids=["no-subcommand", "query-without-a-question"]
)
def test_an_incomplete_invocation_exits_with_usage(argv: list[str]) -> None:
    """Neither reaches a command: the subparser and `question` are both required.

    SystemExit(2) rather than a return value: argparse exits during parsing, so
    this is the one failure main()'s handler never sees.
    """
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2


def test_settings_come_from_the_environment_not_a_literal(
    wired_env, monkeypatch, capsys
):
    """The CLI must honour an override, or `.env` silently means nothing here.

    Asserted through `rag ingest`'s own output rather than by reading Settings,
    so it covers the wiring from environment to command and not just `from_env`.
    """
    elsewhere = wired_env.persist_dir.parent / "moved"
    monkeypatch.setenv("PERSIST_DIR", str(elsewhere))

    assert cli.main(["ingest"]) == 0

    # Compared against the path this test chose, not against a second
    # `from_env()` call -- that would derive both sides from one source and pass
    # even if the command ignored the environment entirely.
    assert str(elsewhere) in capsys.readouterr().out
    assert elsewhere.exists()
