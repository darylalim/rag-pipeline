#!/usr/bin/env python3
"""PreToolUse hook: reject edits that violate mechanical CLAUDE.md invariants.

Thin by design. The rules live in ``tests/invariants.py`` and are enforced for
everyone by ``tests/test_invariants.py`` in CI; this hook exists only to give
the same answer earlier, at write time rather than at review time. Delete it and
nothing stops being enforced -- feedback just arrives one pytest run later.

All this file does is turn a hook payload into a call to ``violations()``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

try:
    data = json.load(sys.stdin)
except ValueError as exc:
    # Exit 1, not 2: a payload we cannot parse must not block the edit, but it
    # means the hook is not enforcing, so say so rather than exiting silently.
    # Claude Code shows the first stderr line for a non-blocking error.
    print(
        f"invariant-guard: unreadable payload, not enforcing ({exc})", file=sys.stderr
    )
    sys.exit(1)

tool_input = data.get("tool_input") or {}
raw_path = tool_input.get("file_path") or ""
if not raw_path:
    sys.exit(0)

root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".").resolve()
# A relative file_path is relative to the project, not to wherever this hook
# happens to be running; resolving against cwd would place it outside root and
# skip enforcement entirely.
candidate = Path(raw_path)
if not candidate.is_absolute():
    candidate = root / candidate
try:
    rel = candidate.resolve().relative_to(root).as_posix()
except ValueError:
    sys.exit(0)  # outside the repo

chunks = [tool_input.get("content"), tool_input.get("new_string")]
text = "\n".join(c for c in chunks if isinstance(c, str))
if not text.strip():
    sys.exit(0)


def load_invariants(project_root: Path) -> ModuleType | None:
    """Import tests/invariants.py by path.

    By path rather than by name because a hook runs as a bare script with no
    package context, and because the project root is only known at runtime.
    """
    spec = importlib.util.spec_from_file_location(
        "_invariants", project_root / "tests" / "invariants.py"
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    invariants = load_invariants(root)
except (OSError, SyntaxError) as exc:
    print(f"invariant-guard: cannot load rules, not enforcing ({exc})", file=sys.stderr)
    sys.exit(1)

if invariants is None:
    print("invariant-guard: cannot load rules, not enforcing", file=sys.stderr)
    sys.exit(1)

hits = invariants.violations(rel, text)
if hits:
    print(f"Blocked edit to {rel} - CLAUDE.md invariant violation:", file=sys.stderr)
    for msg in hits:
        print(f"  - {msg}", file=sys.stderr)
    sys.exit(2)
