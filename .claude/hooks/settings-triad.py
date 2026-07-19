#!/usr/bin/env python3
"""Stop hook: check that a new Settings field is documented before the turn ends.

Thin by design, like its sibling. The rule lives in ``tests/invariants.py`` and
is enforced for everyone by ``tests/test_invariants.py``; this hook only makes
the feedback arrive while config.py is still on screen instead of at the next
pytest run.

Stop rather than PostToolUse: when config.py is written, `.env.example` and the
README legitimately do not have their entry yet, so a per-edit check would fire
on every *correct* change. Stop is the first moment "the change is finished" is
true.

Scope: problems are reported only when the working tree has touched one of the
sites. Validation is cheap enough to run unconditionally, but gating the report
means a turn about something else is never blocked by drift it did not cause.
The tradeoff is that already-committed drift goes unreported until someone next
touches one of those files -- the CI test is what catches that case. If git
cannot answer, the report is not suppressed: the failure direction is "enforce".
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

try:
    data = json.load(sys.stdin)
except ValueError as exc:
    # Exit 1, not 2: unparseable input must not wedge the turn, but it means the
    # hook is not enforcing, so report rather than exit silently.
    print(f"settings-triad: unreadable payload, not enforcing ({exc})", file=sys.stderr)
    sys.exit(1)

if data.get("stop_hook_active"):
    sys.exit(0)  # already nudged once; let the turn end

root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".")


def load_invariants(project_root: Path) -> ModuleType | None:
    """Import tests/invariants.py by path (no package context in a hook)."""
    spec = importlib.util.spec_from_file_location(
        "_invariants", project_root / "tests" / "invariants.py"
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sites_touched(project: Path, sites: tuple[str, ...]) -> bool | None:
    """Has the working tree modified config.py or any documentation site?

    None when git cannot answer, which the caller treats as "report anyway"
    rather than "nothing changed" -- an unavailable git must not silently
    disable enforcement.
    """
    paths = ["rag_pipeline/config.py", *sites]
    try:
        result = subprocess.run(
            ["git", "-C", str(project), "status", "--porcelain", "--", *paths],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


try:
    invariants = load_invariants(root)
except (OSError, SyntaxError) as exc:
    print(f"settings-triad: cannot load rules, not enforcing ({exc})", file=sys.stderr)
    sys.exit(1)

if invariants is None:
    print("settings-triad: cannot load rules, not enforcing", file=sys.stderr)
    sys.exit(1)

problems = invariants.settings_problems(root)

# The git call is the expensive part of this hook (~11.6ms, versus ~0.07ms to
# read and scan the files), so consult it only once there is something to
# report. On a consistent tree -- the steady state -- git is never forked.
if problems and sites_touched(root, invariants.SETTINGS_SITES) is not False:
    print(
        "Settings sites incomplete (CLAUDE.md: adding a Settings field means "
        "config.py + .env.example + the README config table).\n" + "\n".join(problems),
        file=sys.stderr,
    )
    sys.exit(2)
