#!/usr/bin/env python3
"""Stop hook: check the derived-documentation rules before the turn ends.

Two of them, and they are the same rule twice: a Settings field must reach
`.env.example` and the README config table, and an invariant in ``RULES`` must
reach the README rule table. Both exist because the documentation site is the
one place nothing else looks -- ruff, ty and the whole suite stay green against
a stale README -- so both are derived from the declaration rather than kept by
hand.

Thin by design, like its sibling. The rules live in ``tests/invariants.py`` and
are enforced for everyone by ``tests/test_invariants.py``; this hook only makes
the feedback arrive while the declaration is still on screen instead of at the
next pytest run.

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


def paths_touched(project: Path, paths: tuple[str, ...]) -> bool | None:
    """Has the working tree modified a declaration or its documentation site?

    None when git cannot answer, which the caller treats as "report anyway"
    rather than "nothing changed" -- an unavailable git must not silently
    disable enforcement.
    """
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

# Each check pairs its problems with the paths whose modification makes those
# problems this turn's business, and with how to explain them.
checks = (
    (
        invariants.settings_problems(root),
        ("rag_pipeline/config.py", *invariants.SETTINGS_SITES),
        "Settings sites incomplete (CLAUDE.md: adding a Settings field means "
        "config.py + .env.example + the README config table).",
    ),
    (
        invariants.rules_problems(root),
        ("tests/invariants.py", invariants.RULES_SITE),
        "Invariant rules undocumented (CLAUDE.md: adding a rule means a Rule in "
        "RULES + a row in the README rule table).",
    ),
)

# The git call is the expensive part of this hook (~11.6ms, versus ~0.07ms to
# read and scan the files), so consult it only once there is something to
# report. On a consistent tree -- the steady state -- git is never forked.
reports = [
    header + "\n" + "\n".join(problems)
    for problems, paths, header in checks
    if problems and paths_touched(root, paths) is not False
]

if reports:
    print("\n\n".join(reports), file=sys.stderr)
    sys.exit(2)
