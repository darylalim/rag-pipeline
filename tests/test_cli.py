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
from the environment and reaches models only through the three factories, so
setting the environment and patching the factories keeps these tests inside the
suite's no-real-model guarantee.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
import types

import huggingface_hub.constants as hf_constants
import pytest

from rag_pipeline import cli
from rag_pipeline import ingest as ingest_mod
from rag_pipeline import pipeline as pipeline_mod
from rag_pipeline import tracing as tracing_mod

# The real factories, bound before `wired_env` swaps fakes in on the modules:
# the tests below that are about the production path put these back.
from rag_pipeline.ingest import build_embeddings
from rag_pipeline.pipeline import build_chat_model


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
    chunks = ingest_mod.split_documents(
        ingest_mod.load_documents(wired_env.data_dir), wired_env
    )
    assert f"Indexed {len(chunks)} chunks" in out
    # The location is the actionable half: an ingest that silently wrote to a
    # different directory or collection is the failure a user cannot otherwise
    # see.
    assert f"'{wired_env.collection_name}'" in out
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


def test_a_machine_without_mlx_is_an_error_not_a_traceback(
    indexed, capsys, monkeypatch
):
    """RuntimeError: the local models need MLX, which only Apple Silicon has.

    conftest already makes MLX unimportable, which is exactly the state of a
    machine without it, so all this needs is the real chat factory back -- the
    production path this test is about. The message has to say why, or a Linux
    user is left with an import error from a package they never asked for.
    """
    monkeypatch.setattr(pipeline_mod, "build_chat_model", build_chat_model)

    assert cli.main(["query", "anything"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "Traceback" not in err
    assert "Apple Silicon" in err


def test_a_model_missing_from_the_cache_is_an_error_not_a_traceback(
    wired_env, capsys, monkeypatch, tmp_path
):
    """FileNotFoundError: a model that was never downloaded names the download.

    Through `rag ingest`, because the embedding model is the first one a new
    user needs. MLX is a stand-in -- the loader only needs its import to succeed
    before it looks in the cache -- and the cache is an empty directory, so the
    developer's own is not what answers. Nothing is fetched: conftest blocks
    every socket, so a lookup that tried the network would fail differently.
    """
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty-hub"))
    monkeypatch.setattr(ingest_mod, "build_embeddings", build_embeddings)
    monkeypatch.setenv("EMBEDDING_MODEL", "some-org/never-downloaded")

    assert cli.main(["ingest"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "Traceback" not in err
    assert "hf download some-org/never-downloaded" in err


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


def test_a_variable_refused_at_import_is_an_error_not_a_traceback(fresh_interpreter):
    """ValueError again, raised by an import rather than by `from_env`.

    The OpenTelemetry SDK refuses a malformed OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT as
    it is imported, which chromadb does whether or not tracing is on -- so the
    command's own imports raise it, and they are inside main()'s try only
    because they are lazy. In a fresh interpreter, because this one has long
    since imported the SDK and would never read the variable again.
    """
    result = fresh_interpreter(
        "from rag_pipeline.cli import main\nraise SystemExit(main())",
        "query",
        "anything",
        OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT="abc",
    )

    assert result.returncode == 1, result.stderr
    assert result.stderr.startswith("Error: ")
    assert "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT" in result.stderr
    assert "Traceback" not in result.stderr


def test_a_failure_partway_through_the_stream_terminates_the_line(
    indexed, capsys, fail_mid_stream, partial_answer
):
    """The partial answer must not share a line with the error.

    Streaming means a model failure lands with text already on screen, and
    `cmd_query` prints the closing newline from a `finally` for exactly that
    case. Without it main()'s "Error: ..." collides with the partial answer.
    """
    fail_mid_stream(RuntimeError("Generation with 'some-org/some-model' failed: boom"))

    assert cli.main(["query", "anything"]) == 1

    captured = capsys.readouterr()
    assert captured.out.endswith("\n")
    assert partial_answer in captured.out
    assert captured.err.startswith("Error: ")


def test_an_unusable_persist_dir_is_an_error_not_a_traceback(
    wired_env, capsys, monkeypatch, tmp_path
):
    """The filesystem's own errors on PERSIST_DIR, as the one-line message.

    A regular file where the directory should be: creating it is a
    FileExistsError, an OSError outside the union main() catches, so it must be
    translated before it gets here -- and name the setting, since the path
    alone does not say which one to fix.
    """
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("a file", encoding="utf-8")
    monkeypatch.setenv("PERSIST_DIR", str(not_a_dir))

    assert cli.main(["ingest"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "Traceback" not in err
    assert "PERSIST_DIR" in err


def test_a_fetch_k_below_one_is_an_error_not_a_traceback(indexed, capsys, monkeypatch):
    """Chroma rejects a search for no results with a builtins TypeError, raised
    only when the first question is asked; the pipeline refuses the setting
    up front instead, inside the union."""
    monkeypatch.setenv("FETCH_K", "0")

    assert cli.main(["query", "anything"]) == 1

    err = capsys.readouterr().err
    assert err.startswith("Error: ")
    assert "FETCH_K" in err


def test_ctrl_c_mid_answer_is_one_line_and_the_sigint_status(
    indexed, capsys, fail_mid_stream, partial_answer
):
    """Ctrl-C is how a slow local answer is abandoned, so it is an ordinary exit.

    One line and the shell's status for SIGINT (130), not a traceback through
    the model's generation loop -- and the partial answer's line still ends.
    """
    fail_mid_stream(KeyboardInterrupt())

    try:
        status = cli.main(["query", "anything"])
    except KeyboardInterrupt:
        pytest.fail("Ctrl-C escaped main() as a traceback")  # not pytest's own

    assert status == 130

    captured = capsys.readouterr()
    assert partial_answer in captured.out
    assert captured.out.endswith("\n")
    assert captured.err == "Interrupted.\n"


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
    monkeypatch.setenv("COLLECTION_NAME", "a_distinctive_collection")

    assert cli.main(["ingest"]) == 0

    # Compared against the values this test chose, not against a second
    # `from_env()` call -- that would derive both sides from one source and pass
    # even if the command ignored the environment entirely.
    out = capsys.readouterr().out
    assert str(elsewhere) in out
    assert "a_distinctive_collection" in out
    # And the write landed there, not merely the message: that collection at
    # that path now carries a version stamp, and the fixture's own location was
    # never touched.
    moved = dataclasses.replace(
        wired_env, persist_dir=elsewhere, collection_name="a_distinctive_collection"
    )
    assert ingest_mod.index_version(moved)
    assert not wired_env.persist_dir.exists()


def test_a_question_sets_up_tracing_and_an_ingest_does_not(indexed, monkeypatch):
    """From the command's own Settings, and only for `rag query`.

    A question is the one thing traced. An ingest emits no spans, so setting
    tracing up there would only load the instrumentation and the exporter and
    start an exporter thread with nothing to send.
    """
    seen: list[str] = []
    monkeypatch.setattr(
        tracing_mod,
        "setup_tracing",
        lambda settings: seen.append(settings.phoenix_collector_endpoint),
    )
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix.test:6006")

    assert cli.main(["ingest"]) == 0
    assert seen == []
    assert cli.main(["query", "Why do chunks overlap?"]) == 0
    assert seen == ["http://phoenix.test:6006"]


# --- the cost of `rag --help` ------------------------------------------------

# The heavy half of the dependency tree: the vector store and the model runtime.
# cli.py reaches all of it, but only from inside a command function: importing
# the module must not pay for a stack the user may never reach, since `rag
# --help` and a usage error load cli.py and then exit.
HEAVY = ("chromadb", "langchain_chroma", "mlx", "mlx_lm")

# What tracing loads once it is on: the LangChain instrumentation and the span
# exporter. Off, the pipeline carries the OpenTelemetry API alone.
TRACING_STACK = (
    "openinference.instrumentation",
    "opentelemetry.exporter.otlp.proto.http",
)

# Records which HEAVY modules are loaded after importing cli.py, then again after
# importing the store and query modules -- one interpreter, so the second
# reading is the control for the first. Whether transformers came along is
# recorded too: the stderr check below means something only when it did.
_IMPORT_PROBE = """
import json, sys
loaded = {{}}
import rag_pipeline.cli
loaded["cli"] = [m for m in {heavy!r} if m in sys.modules]
import rag_pipeline.ingest, rag_pipeline.pipeline
loaded["pipeline"] = [m for m in {heavy!r} if m in sys.modules]
from rag_pipeline.config import Settings
from rag_pipeline.tracing import setup_tracing
setup_tracing(Settings())  # what both frontends do with tracing off
loaded["tracing"] = [m for m in {tracing!r} if m in sys.modules]
loaded["transformers"] = "transformers" in sys.modules
print(json.dumps(loaded))
"""


@pytest.fixture(scope="module")
def heavy_modules_loaded() -> dict:
    """What a fresh interpreter holds of HEAVY, after cli.py and after the rest,
    and what it printed on stderr doing so.

    In a subprocess because this suite has already imported all of it; the
    question is what a fresh interpreter loads, which is the only place the
    difference is observable. Module-scoped because it is the same answer for
    every test that asks, and a fresh interpreter importing chromadb is not free.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _IMPORT_PROBE.format(heavy=HEAVY, tracing=TRACING_STACK),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    return {**loaded, "stderr": result.stderr}


def test_importing_cli_does_not_load_the_heavy_stack(heavy_modules_loaded):
    """`import rag_pipeline.cli` must not drag in chromadb or MLX.

    The behavioral form of what used to be a text rule matching import
    spellings in cli.py. Asserting on `sys.modules` is strictly stronger: it
    fails for *any* route to the heavy stack -- a new module that imports it
    eagerly, a re-export, an `importlib` call -- rather than the handful of
    spellings a regex was able to enumerate.

    The second reading is the control: a list of module names that nothing
    imports passes the first assertion vacuously, and a provider swap is exactly
    when HEAVY drifts out of date. The same probe must see the store stack once
    the pipeline itself is imported.
    """
    assert not heavy_modules_loaded["cli"], (
        f"importing cli.py loaded: {heavy_modules_loaded['cli']}"
    )
    assert {"chromadb", "langchain_chroma"} <= set(heavy_modules_loaded["pipeline"])


def test_importing_the_pipeline_does_not_load_mlx(heavy_modules_loaded):
    """MLX is imported when a model loads, not when the pipeline does.

    The project installs MLX only on macOS, while CI runs on Linux: an eager
    import anywhere in ingest.py or pipeline.py would fail the Linux run at
    collection, and on a Mac it would pass unnoticed until then.
    """
    loaded = set(heavy_modules_loaded["pipeline"])
    assert not {"mlx", "mlx_lm"} & loaded, f"importing the pipeline loaded: {loaded}"


def test_tracing_off_loads_none_of_the_tracing_stack(heavy_modules_loaded):
    """Off is the default, and costs nothing: setup_tracing imports the
    instrumentation and the exporter itself, and only once it has an endpoint.

    The probe takes the frontends' own path -- import everything, then call
    setup_tracing with the default Settings -- so an import hoisted to the top
    of tracing.py, or above its endpoint check, shows up here. The control is
    that every name is a real, installed module: a misspelled one is never
    loaded, and would pass forever.
    """
    assert all(importlib.util.find_spec(name) for name in TRACING_STACK)
    assert not heavy_modules_loaded["tracing"], (
        f"tracing off loaded: {heavy_modules_loaded['tracing']}"
    )


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers arrives only with mlx-lm, which is installed only on macOS",
)
def test_importing_the_pipeline_prints_no_pytorch_warning(heavy_modules_loaded):
    """No "PyTorch was not found" on every command.

    transformers is only mlx-lm's tokenizer backend, and torch is deliberately
    absent -- but LangChain imports transformers as soon as the pipeline is
    imported, before mlx-lm can silence the notice, so without the package's
    own setting every `rag` command and the app would open by announcing that
    models won't be available. The first assertion is the control: transformers
    has to have been imported for the second to mean anything.
    """
    assert heavy_modules_loaded["transformers"]
    assert "PyTorch was not found" not in heavy_modules_loaded["stderr"]
