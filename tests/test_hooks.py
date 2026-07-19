"""Tests for the CLAUDE.md enforcement hooks in `.claude/hooks/`.

The hooks are exercised as subprocesses fed JSON on stdin rather than imported,
because that *is* the contract Claude Code invokes them under: payload in, exit
code out (2 = reject). Importing them would test the regexes while leaving the
stdin/exit-code wiring — the part that silently disables a hook when it breaks —
uncovered.

The regression sweep matters most: a guard with a false-positive rate gets
switched off within a day, and it fails closed, blocking correct work. Every
real source file is fed back in verbatim and must produce no hit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / ".claude" / "hooks" / "invariant-guard.py"
TRIAD = ROOT / ".claude" / "hooks" / "settings-triad.py"

ALLOW = 0
BLOCK = 2  # the exit code Claude Code reads as "reject this edit / turn"

# Every tracked .py file, globbed rather than listed: a hardcoded list fails by
# silently not covering a new file, which is the wrong direction for a guard.
REAL_FILES = sorted(
    subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
)


def run_script(
    argv: list[str],
    stdin: str,
    project_dir: Path = ROOT,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a hook with the environment Claude Code gives it.

    Separate from run_hook so the wiring tests can vary one axis each — the
    interpreter, the cwd, a non-JSON payload — without restating the other
    four arguments, which are the load-bearing part.
    """
    return subprocess.run(
        argv,
        input=stdin,
        capture_output=True,
        text=True,
        cwd=cwd,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)},
        check=False,
    )


def run_hook(
    script: Path,
    payload: dict[str, object],
    project_dir: Path = ROOT,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke a hook the way Claude Code does: JSON on stdin, exit code out."""
    return run_script(
        [sys.executable, str(script)], json.dumps(payload), project_dir, cwd
    )


def edit_payload(relpath: str, text: str, field: str = "content") -> dict[str, object]:
    """A PreToolUse payload for writing `text` to `relpath`."""
    return {"tool_input": {"file_path": str(ROOT / relpath), field: text}}


# --- invariant-guard: violations that must be blocked ------------------------

VIOLATIONS = [
    pytest.param(
        "rag_pipeline/cli.py",
        "from rag_pipeline.ingest import ingest",
        id="cli-import-dotted",
    ),
    pytest.param(
        "rag_pipeline/cli.py",
        "from .pipeline import RAGPipeline",
        id="cli-import-relative",
    ),
    pytest.param(
        "rag_pipeline/cli.py",
        "from rag_pipeline import ingest, pipeline",
        id="cli-import-bare-module",
    ),
    pytest.param("rag_pipeline/cli.py", "from . import ingest", id="cli-import-dot"),
    pytest.param(
        "rag_pipeline/cli.py", "import rag_pipeline.pipeline", id="cli-import-plain"
    ),
    pytest.param("app.py", 'store = Chroma(collection_name="x")', id="inline-chroma"),
    pytest.param(
        "app.py", "e = HuggingFaceEmbeddings(model_name=m)", id="inline-embeddings"
    ),
    pytest.param(
        # tests/ may open Chroma directly, but never build a real embedding
        # model: that downloads ~90MB and puts the suite back on the network.
        "tests/test_pipeline.py",
        'emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")',
        id="embeddings-in-tests-breaks-offline",
    ),
    pytest.param("rag_pipeline/config.py", "import os  # noqa: F401", id="noqa"),
    pytest.param("rag_pipeline/config.py", "x = y  # ty: ignore", id="ty-ignore"),
    pytest.param("rag_pipeline/ingest.py", "rmtree(settings.persist_dir)", id="rmtree"),
    pytest.param(
        "rag_pipeline/pipeline.py",
        "ChatAnthropic(model=m, temperature=0.2)",
        id="temperature",
    ),
]


@pytest.mark.parametrize(("relpath", "text"), VIOLATIONS)
def test_guard_blocks_violations(relpath: str, text: str) -> None:
    result = run_hook(GUARD, edit_payload(relpath, text))
    assert result.returncode == BLOCK
    assert "CLAUDE.md invariant violation" in result.stderr


# --- invariant-guard: code that must NOT be blocked --------------------------

ALLOWED = [
    pytest.param(
        "rag_pipeline/cli.py",
        "from rag_pipeline.config import Settings",
        id="cli-config-import-is-cheap",
    ),
    pytest.param(
        "rag_pipeline/cli.py",
        "from rag_pipeline import config  # pipeline notes",
        id="comment-mentioning-pipeline",
    ),
    pytest.param(
        "rag_pipeline/cli.py",
        "def cmd_ingest(args):\n    from rag_pipeline.ingest import ingest\n",
        id="lazy-import-inside-function",
    ),
    pytest.param(
        "rag_pipeline/ingest.py",
        "return Chroma(collection_name=n)",
        id="ingest-is-the-factory-home",
    ),
    pytest.param(
        "tests/test_ingest.py",
        'store = Chroma(collection_name="x")',
        id="tests-may-open-chroma-directly",
    ),
    pytest.param(
        "rag_pipeline/pipeline.py",
        '"""No temperature/top_p: grounding comes from context."""',
        id="docstring-naming-sampling-params",
    ),
    pytest.param(
        # The guard quotes every banned pattern in its own rule table and
        # messages. Masking is what lets it edit itself; there is no longer a
        # path exemption doing that work.
        ".claude/hooks/invariant-guard.py",
        're.compile(r"Chroma\\(")\nMSG = "rmtree and temperature= are banned"',
        id="guard-may-rewrite-itself",
    ),
    pytest.param(
        "tests/test_hooks.py",
        'BAD = "import os  # noqa: F401"',
        id="hook-test-may-quote-banned-patterns",
    ),
    # Prose describing a rule must not trip it. Without literal/comment masking
    # the author cannot document the invariant being enforced.
    pytest.param(
        "app.py",
        "# Never construct Chroma(...) inline -- use open_store().",
        id="comment-describing-the-chroma-rule",
    ),
    pytest.param(
        "rag_pipeline/pipeline.py",
        '"""build_chat_model sets no temperature= by design."""',
        id="docstring-describing-the-sampling-rule",
    ),
    pytest.param(
        "rag_pipeline/ingest.py",
        '# never rmtree the persist dir\nRULE = "no rmtree here"',
        id="prose-describing-the-rmtree-rule",
    ),
    pytest.param(
        "rag_pipeline/config.py",
        'DOC = "write # noqa and it gets rejected"',
        id="noqa-inside-a-string-literal",
    ),
    pytest.param(
        # Column 0 inside a triple-quoted block: the `^` anchor would match this
        # without masking, so it is the case that proves masking is doing work.
        "rag_pipeline/cli.py",
        'HELP = """\nfrom rag_pipeline.ingest import ingest\n"""',
        id="cli-import-inside-a-docstring",
    ),
]


@pytest.mark.parametrize(("relpath", "text"), ALLOWED)
def test_guard_allows_legitimate_code(relpath: str, text: str) -> None:
    assert run_hook(GUARD, edit_payload(relpath, text)).returncode == ALLOW


@pytest.mark.parametrize("relpath", REAL_FILES)
def test_guard_is_silent_on_every_real_file(relpath: str) -> None:
    """No false positives on the tree as it stands.

    A guard that fires on correct code fails closed and gets disabled, so this
    is the check to re-run after touching any pattern.
    """
    text = (ROOT / relpath).read_text()
    assert run_hook(GUARD, edit_payload(relpath, text)).returncode == ALLOW


# --- invariant-guard: payload shapes -----------------------------------------


def test_guard_reads_the_edit_tool_new_string_field() -> None:
    payload = edit_payload("app.py", 'Chroma(collection_name="x")', field="new_string")
    assert run_hook(GUARD, payload).returncode == BLOCK


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"tool_input": {"command": "uv run pytest"}}, id="bash-no-path"),
        pytest.param(
            {"tool_input": {"file_path": "/tmp/scratch.py", "content": "Chroma()"}},
            id="outside-the-repo",
        ),
        pytest.param(
            # Absolute, so this isolates the empty-text exit rather than also
            # depending on how a relative path resolves.
            {"tool_input": {"file_path": str(ROOT / "app.py"), "content": "   "}},
            id="empty-text",
        ),
    ],
)
def test_guard_ignores_irrelevant_payloads(payload: dict[str, object]) -> None:
    assert run_hook(GUARD, payload).returncode == ALLOW


def test_guard_resolves_a_relative_path_against_the_project(tmp_path: Path) -> None:
    """A relative file_path is project-relative, not cwd-relative.

    Resolving against the hook's own cwd would place it outside the project and
    skip enforcement — a silent fail-open, which is the worst outcome for a
    guard because the green result is indistinguishable from a real pass.
    """
    payload: dict[str, object] = {
        "tool_input": {"file_path": "app.py", "content": 'Chroma(name="x")'}
    }
    # cwd deliberately not the project directory
    assert run_hook(GUARD, payload, cwd=tmp_path).returncode == BLOCK


@pytest.mark.parametrize("script", [GUARD, TRIAD], ids=["guard", "triad"])
def test_hooks_report_rather_than_crash_on_bad_payloads(script: Path) -> None:
    """Malformed stdin must not block, and must not exit silently either.

    Exit 1 is Claude Code's non-blocking-error path: the edit proceeds, but the
    first stderr line surfaces, so a payload-shape change in a future release
    cannot disable enforcement without anyone noticing.
    """
    result = run_script([sys.executable, str(script)], "not json")
    assert result.returncode == 1
    assert "not enforcing" in result.stderr
    assert "Traceback" not in result.stderr


# --- the wiring that invokes the hooks ---------------------------------------


def test_settings_json_points_at_hooks_that_exist() -> None:
    """The hooks can be perfect and still never run.

    Nothing else covers this: every other test invokes the scripts by path, so
    renaming one leaves the suite green while no hook fires in any session.
    """
    settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    configured = settings["hooks"]

    assert set(configured) == {"PreToolUse", "Stop"}
    assert configured["PreToolUse"][0]["matcher"] == "Edit|Write"

    commands = [
        hook["command"]
        for matchers in configured.values()
        for matcher in matchers
        for hook in matcher["hooks"]
    ]
    assert len(commands) == 2
    for command in commands:
        script = Path(command.replace("${CLAUDE_PROJECT_DIR}", str(ROOT)))
        assert script.is_file(), f"{command} does not exist"
        assert os.access(script, os.X_OK), f"{command} is not executable"


@pytest.mark.parametrize("script", [GUARD, TRIAD], ids=["guard", "triad"])
def test_hooks_run_under_their_shebang(script: Path) -> None:
    """settings.json execs the scripts directly, so the shebang is the real
    entry point — not the venv interpreter every other test uses."""
    result = run_script([str(script)], json.dumps({"stop_hook_active": True}))
    assert result.returncode == ALLOW, result.stderr


# --- settings-triad ----------------------------------------------------------


def build_project(
    tmp_path: Path,
    *,
    var: str = "RERANK_TOP_N",
    helper: str = "_env_int",
    in_env_example: bool = False,
    in_readme: bool = False,
    in_delenv: bool = False,
    edit_test_config: Callable[[str], str] = lambda text: text,
) -> Path:
    """A fake project root declaring `var`, documented only where asked.

    `edit_test_config` transforms tests/test_config.py before the `in_delenv`
    insertion, so the two compose instead of one silently replacing the other.
    """
    (tmp_path / "rag_pipeline").mkdir()
    (tmp_path / "tests").mkdir()

    config = (ROOT / "rag_pipeline/config.py").read_text()
    # Append after the anchor rather than rewriting it: the hook scans for
    # `_env_*("NAME"` with double quotes, so re-quoting the anchor would drop
    # RETRIEVAL_K from the fixture's declared set and weaken every case below.
    anchor = 'retrieval_k=_env_int("RETRIEVAL_K", cls.retrieval_k),'
    # Without this, a reformatted config.py makes `replace` a no-op and every
    # BLOCK case fails as `0 == 2`, pointing at the hook instead of the anchor.
    assert anchor in config, f"anchor no longer present in config.py: {anchor!r}"
    (tmp_path / "rag_pipeline/config.py").write_text(
        config.replace(anchor, f'{anchor}\n            extra={helper}("{var}", 5),')
    )

    env_example = (ROOT / ".env.example").read_text()
    if in_env_example:
        env_example += f"\n# {var}=5\n"
    (tmp_path / ".env.example").write_text(env_example)

    readme = (ROOT / "README.md").read_text()
    if in_readme:
        readme = readme.replace(
            "| `COLLECTION_NAME` |",
            f"| `{var}` | `5` | Extra |\n| `COLLECTION_NAME` |",
            1,
        )
    (tmp_path / "README.md").write_text(readme)

    test_config = edit_test_config((ROOT / "tests/test_config.py").read_text())
    if in_delenv:
        delenv_anchor = '        "MAX_TOKENS",'
        assert delenv_anchor in test_config, "delenv tuple no longer ends as expected"
        test_config = test_config.replace(
            delenv_anchor, f'{delenv_anchor}\n        "{var}",', 1
        )
    (tmp_path / "tests/test_config.py").write_text(test_config)

    return tmp_path


def make_git_repo(project: Path) -> None:
    """Commit everything in `project`, so its working tree starts clean.

    Isolated from the developer's global git config: a signing key or hook
    there would otherwise make these tests fail for unrelated reasons.
    """
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.email=t@example.com", "-c", "user.name=T", "commit", "-qm", "x"],
    ):
        subprocess.run(
            ["git", "-C", str(project), *args], capture_output=True, check=True, env=env
        )


def test_triad_passes_on_the_real_repo() -> None:
    """The hook never blocks a turn in this repo.

    Note this passes for either of two reasons — the working-tree gate finding
    no settings site dirty, or the full check running clean. The fixture tests
    below cover the validation itself; this one is the end-to-end smoke test.
    """
    assert run_hook(TRIAD, {}).returncode == ALLOW


def test_triad_honors_stop_hook_active() -> None:
    """Nudge once rather than looping — Claude Code overrides after 8 blocks."""
    assert run_hook(TRIAD, {"stop_hook_active": True}).returncode == ALLOW


@pytest.mark.parametrize(
    ("in_env_example", "in_readme", "in_delenv", "expected_in_message"),
    [
        pytest.param(False, False, False, ".env.example", id="documented-nowhere"),
        pytest.param(
            True, True, False, "tests/test_config.py", id="delenv-missing-latent-hazard"
        ),
        pytest.param(True, False, True, "README.md", id="readme-missing"),
        pytest.param(False, True, True, ".env.example", id="env-example-missing"),
    ],
)
def test_triad_blocks_on_each_missing_site(
    tmp_path: Path,
    in_env_example: bool,
    in_readme: bool,
    in_delenv: bool,
    expected_in_message: str,
) -> None:
    project = build_project(
        tmp_path,
        in_env_example=in_env_example,
        in_readme=in_readme,
        in_delenv=in_delenv,
    )
    result = run_hook(TRIAD, {}, project)
    assert result.returncode == BLOCK
    assert expected_in_message in result.stderr


def test_triad_passes_when_all_four_sites_present(tmp_path: Path) -> None:
    project = build_project(
        tmp_path, in_env_example=True, in_readme=True, in_delenv=True
    )
    assert run_hook(TRIAD, {}, project).returncode == ALLOW


def test_triad_catches_a_future_env_helper(tmp_path: Path) -> None:
    """The scan matches `_env_*`, so a new `_env_bool` is covered on day one."""
    project = build_project(tmp_path, helper="_env_bool", var="STRICT_MODE")
    assert run_hook(TRIAD, {}, project).returncode == BLOCK


def test_triad_survives_renaming_the_defaults_test(tmp_path: Path) -> None:
    """Anchored on `monkeypatch.delenv`, not the test's name."""
    project = build_project(
        tmp_path,
        in_env_example=True,
        in_readme=True,
        in_delenv=True,  # composes with the rename below
        edit_test_config=lambda text: text.replace(
            "def test_from_env_uses_defaults_when_unset", "def test_renamed"
        ),
    )
    assert run_hook(TRIAD, {}, project).returncode == ALLOW


def test_triad_ignores_committed_drift_when_nothing_changed(tmp_path: Path) -> None:
    """An unrelated turn must not be blocked by drift it did not cause.

    Without the working-tree gate, one stale README row would exit 2 at the end
    of every turn until someone fixed it — including turns that never touched
    config.py.
    """
    project = build_project(tmp_path)  # declares a var documented nowhere
    make_git_repo(project)  # ...but the tree is clean, so this turn is unrelated
    assert run_hook(TRIAD, {}, project).returncode == ALLOW


def test_triad_checks_once_a_settings_site_is_dirty(tmp_path: Path) -> None:
    """The gate is about *scope*, not about weakening the check."""
    project = build_project(tmp_path)
    make_git_repo(project)

    config = project / "rag_pipeline" / "config.py"
    config.write_text(
        config.read_text().replace(
            "extra=_env_int(", "extra2=_env_int('X', 1)\n            extra=_env_int("
        )
    )

    result = run_hook(TRIAD, {}, project)
    assert result.returncode == BLOCK
    assert "RERANK_TOP_N" in result.stderr


def test_triad_checks_when_git_cannot_answer(tmp_path: Path) -> None:
    """No repo means the gate must fail toward enforcing, not toward silence."""
    project = build_project(tmp_path)  # no git init
    assert not (project / ".git").exists()
    assert run_hook(TRIAD, {}, project).returncode == BLOCK


def test_triad_reports_once_when_the_delenv_loop_is_gone(tmp_path: Path) -> None:
    """One message, not one per variable — nine would read as a broken hook."""
    project = build_project(
        tmp_path,
        in_env_example=True,
        in_readme=True,
        edit_test_config=lambda text: text.replace(
            "monkeypatch.delenv", "monkeypatch.setenv"
        ),
    )
    result = run_hook(TRIAD, {}, project)
    assert result.returncode == BLOCK
    assert result.stderr.count("could not find") == 1
