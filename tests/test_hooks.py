"""Tests for the wiring that carries the invariants into a Claude Code session.

The rules themselves are covered in-process by `test_invariants.py`; what is
left here is everything that can silently disconnect them — the settings.json
that names the hooks, the shebang they are executed under, the payload shapes
they must survive, and the exit codes Claude Code reads.

That layer is worth its subprocesses precisely because it is invisible when it
breaks: a hook that never fires and a hook that fires and finds nothing look
identical from the outside.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.invariants import RULES

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / ".claude" / "hooks" / "invariant-guard.py"
TRIAD = ROOT / ".claude" / "hooks" / "settings-triad.py"

ALLOW = 0
BLOCK = 2  # the exit code Claude Code reads as "reject this edit / turn"


def run_script(
    argv: list[str],
    stdin: str,
    project_dir: Path = ROOT,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a hook with the environment Claude Code gives it."""
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


# --- the wiring that makes hooks run at all ----------------------------------


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


@pytest.mark.parametrize("script", [GUARD, TRIAD], ids=["guard", "triad"])
def test_hooks_report_when_the_rules_are_unreachable(
    script: Path, tmp_path: Path
) -> None:
    """A project without tests/invariants.py must say so, not fail open quietly.

    The hooks load their rules from the project tree, so a moved or renamed
    rules module would otherwise disable them with no signal.
    """
    payload = json.dumps({"tool_input": {"file_path": "app.py", "content": "Chroma()"}})
    result = run_script([sys.executable, str(script)], payload, project_dir=tmp_path)
    assert result.returncode == 1
    assert "not enforcing" in result.stderr


# --- invariant-guard: payload handling ---------------------------------------


def test_guard_blocks_a_violation_end_to_end() -> None:
    """One case through the real payload path; the rules live elsewhere."""
    result = run_hook(GUARD, edit_payload("app.py", 'Chroma(collection_name="x")'))
    assert result.returncode == BLOCK
    assert "CLAUDE.md invariant violation" in result.stderr


def test_guard_allows_clean_code_end_to_end() -> None:
    result = run_hook(GUARD, edit_payload("app.py", "pipeline = build_pipeline(s)"))
    assert result.returncode == ALLOW


def test_guard_reads_the_edit_tool_new_string_field() -> None:
    payload = edit_payload("app.py", 'Chroma(collection_name="x")', field="new_string")
    assert run_hook(GUARD, payload).returncode == BLOCK


def test_guard_resolves_a_relative_path_against_the_project(tmp_path: Path) -> None:
    """A relative file_path is project-relative, not cwd-relative.

    Resolving against the hook's own cwd would place it outside the project and
    skip enforcement — a silent fail-open, the worst outcome for a guard because
    the green result is indistinguishable from a real pass.
    """
    payload: dict[str, object] = {
        "tool_input": {"file_path": "app.py", "content": 'Chroma(name="x")'}
    }
    assert run_hook(GUARD, payload, cwd=tmp_path).returncode == BLOCK


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"tool_input": {"command": "uv run pytest"}}, id="bash-no-path"),
        pytest.param(
            {"tool_input": {"file_path": "/tmp/scratch.py", "content": "Chroma()"}},
            id="outside-the-repo",
        ),
        pytest.param(
            {"tool_input": {"file_path": str(ROOT / "app.py"), "content": "   "}},
            id="empty-text",
        ),
    ],
)
def test_guard_ignores_irrelevant_payloads(payload: dict[str, object]) -> None:
    assert run_hook(GUARD, payload).returncode == ALLOW


# --- settings-triad: the working-tree gate -----------------------------------


def build_project(
    tmp_path: Path, *, documented: bool, rules_documented: bool = True
) -> Path:
    """A minimal project declaring two settings, one of them undocumented.

    Hand-built rather than copied from the repo: the hook only parses a Settings
    dataclass and two documentation files, so a faithful copy would add coupling
    without adding coverage.

    The rule rows are the exception, and are generated from the real `RULES`
    rather than written out: the hook loads the repo's own `invariants.py`, so
    the rules this project must document are whatever that module declares. A
    hand-written list here would be a third copy of exactly the list the rule
    under test exists to stop anyone from keeping by hand.
    """
    (tmp_path / "rag_pipeline").mkdir()
    (tmp_path / "rag_pipeline" / "config.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Settings:\n"
        '    chat_model: str = "x"\n'
        "    rerank_top_n: int = 5\n"
    )
    env = "# CHAT_MODEL=x\n"
    readme = "| `CHAT_MODEL` | `x` | Model |\n"
    if documented:
        env += "# RERANK_TOP_N=5\n"
        readme += "| `RERANK_TOP_N` | `5` | Rerank |\n"
    if rules_documented:
        readme += "".join(f"| `{rule.name}` | forbids | why |\n" for rule in RULES)
    (tmp_path / ".env.example").write_text(env)
    (tmp_path / "README.md").write_text(readme)

    (tmp_path / "tests").mkdir()
    shutil.copy(ROOT / "tests" / "invariants.py", tmp_path / "tests" / "invariants.py")
    return tmp_path


def make_git_repo(project: Path) -> None:
    """Commit everything, so the project's working tree starts clean."""
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
    """End-to-end smoke test; completeness itself is test_invariants.py's job."""
    assert run_hook(TRIAD, {}).returncode == ALLOW


def test_triad_honors_stop_hook_active() -> None:
    """Nudge once rather than looping — Claude Code overrides after 8 blocks."""
    assert run_hook(TRIAD, {"stop_hook_active": True}).returncode == ALLOW


def test_triad_blocks_an_undocumented_setting(tmp_path: Path) -> None:
    project = build_project(tmp_path, documented=False)
    result = run_hook(TRIAD, {}, project)
    assert result.returncode == BLOCK
    assert "RERANK_TOP_N" in result.stderr


def test_triad_passes_when_documented(tmp_path: Path) -> None:
    project = build_project(tmp_path, documented=True)
    assert run_hook(TRIAD, {}, project).returncode == ALLOW


def test_triad_blocks_an_undocumented_rule(tmp_path: Path) -> None:
    """The rule table is gated exactly like the settings sites.

    Same hook, second derived-documentation check: a rule that reaches `RULES`
    without reaching the README is reported by name, so the omission is fixed
    while invariants.py is still on screen.
    """
    project = build_project(tmp_path, documented=True, rules_documented=False)
    result = run_hook(TRIAD, {}, project)
    assert result.returncode == BLOCK
    assert "no-rmtree" in result.stderr


def test_triad_ignores_committed_drift_when_nothing_changed(tmp_path: Path) -> None:
    """An unrelated turn must not be blocked by drift it did not cause.

    The CI sweep in test_invariants.py is what catches this case; the hook is
    scoped to what the turn touched so it stays out of the way.
    """
    project = build_project(tmp_path, documented=False)
    make_git_repo(project)  # tree now clean, so this turn is unrelated
    assert run_hook(TRIAD, {}, project).returncode == ALLOW


def test_triad_reports_once_a_site_is_dirty(tmp_path: Path) -> None:
    project = build_project(tmp_path, documented=False)
    make_git_repo(project)
    (project / ".env.example").write_text("# CHAT_MODEL=x\n# touched\n")
    assert run_hook(TRIAD, {}, project).returncode == BLOCK


def test_triad_reports_when_git_cannot_answer(tmp_path: Path) -> None:
    """No repo means the gate must fail toward enforcing, not toward silence."""
    project = build_project(tmp_path, documented=False)
    assert not (project / ".git").exists()
    assert run_hook(TRIAD, {}, project).returncode == BLOCK
