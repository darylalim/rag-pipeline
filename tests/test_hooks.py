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
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GUARD = ROOT / ".claude" / "hooks" / "invariant-guard.py"
TRIAD = ROOT / ".claude" / "hooks" / "settings-triad.py"

ALLOW = 0
BLOCK = 2  # the exit code Claude Code reads as "reject this edit / turn"

# Every file the guard must stay silent on. Kept explicit rather than globbed so
# a new source file is a deliberate addition here.
REAL_FILES = [
    "app.py",
    "rag_pipeline/config.py",
    "rag_pipeline/ingest.py",
    "rag_pipeline/pipeline.py",
    "rag_pipeline/cli.py",
    "tests/conftest.py",
    "tests/test_config.py",
    "tests/test_ingest.py",
    "tests/test_pipeline.py",
    ".claude/hooks/invariant-guard.py",
    ".claude/hooks/settings-triad.py",
]


def run_hook(
    script: Path, payload: dict[str, object], project_dir: Path = ROOT
) -> subprocess.CompletedProcess[str]:
    """Invoke a hook the way Claude Code does: JSON on stdin, exit code out."""
    return subprocess.run(
        [sys.executable, str(script)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)},
        check=False,
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
        ".claude/hooks/invariant-guard.py",
        're.compile(r"Chroma\\(")  # noqa: E501\nrmtree\ntemperature=1',
        id="guard-may-rewrite-itself",
    ),
    pytest.param(
        "tests/test_hooks.py",
        'BAD = "import os  # noqa: F401"',
        id="hook-test-may-quote-banned-patterns",
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
            {"tool_input": {"file_path": "app.py", "content": "   "}}, id="empty-text"
        ),
    ],
)
def test_guard_ignores_irrelevant_payloads(payload: dict[str, object]) -> None:
    assert run_hook(GUARD, payload).returncode == ALLOW


# --- settings-triad ----------------------------------------------------------


def build_project(
    tmp_path: Path,
    *,
    var: str = "RERANK_TOP_N",
    helper: str = "_env_int",
    in_env_example: bool = False,
    in_readme: bool = False,
    in_delenv: bool = False,
    test_config_override: str | None = None,
) -> Path:
    """A fake project root declaring `var`, documented only where asked."""
    (tmp_path / "rag_pipeline").mkdir()
    (tmp_path / "tests").mkdir()

    config = (ROOT / "rag_pipeline/config.py").read_text()
    # Append after the anchor rather than rewriting it: the hook scans for
    # `_env_*("NAME"` with double quotes, so re-quoting the anchor would drop
    # RETRIEVAL_K from the fixture's declared set and weaken every case below.
    anchor = 'retrieval_k=_env_int("RETRIEVAL_K", cls.retrieval_k),'
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

    test_config = (ROOT / "tests/test_config.py").read_text()
    if in_delenv:
        test_config = test_config.replace(
            '        "MAX_TOKENS",', f'        "MAX_TOKENS",\n        "{var}",', 1
        )
    if test_config_override is not None:
        test_config = test_config_override
    (tmp_path / "tests/test_config.py").write_text(test_config)

    return tmp_path


def test_triad_passes_on_the_real_repo() -> None:
    """All 9 settings are documented at all four sites today."""
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
    renamed = (
        (ROOT / "tests/test_config.py")
        .read_text()
        .replace("def test_from_env_uses_defaults_when_unset", "def test_renamed")
        .replace(
            '        "MAX_TOKENS",', '        "MAX_TOKENS",\n        "RERANK_TOP_N",', 1
        )
    )
    project = build_project(
        tmp_path, in_env_example=True, in_readme=True, test_config_override=renamed
    )
    assert run_hook(TRIAD, {}, project).returncode == ALLOW


def test_triad_reports_once_when_the_delenv_loop_is_gone(tmp_path: Path) -> None:
    """One message, not one per variable — nine would read as a broken hook."""
    without_loop = (
        (ROOT / "tests/test_config.py")
        .read_text()
        .replace("monkeypatch.delenv", "monkeypatch.setenv")
    )
    project = build_project(
        tmp_path,
        in_env_example=True,
        in_readme=True,
        test_config_override=without_loop,
    )
    result = run_hook(TRIAD, {}, project)
    assert result.returncode == BLOCK
    assert result.stderr.count("could not find") == 1
