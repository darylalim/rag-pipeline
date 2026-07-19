#!/usr/bin/env python3
"""Stop hook: enforce the four-file Settings rule from CLAUDE.md.

Runs at the end of a turn rather than on each edit: when ``config.py`` is
written, the other files legitimately do not have their entry yet, so a
PostToolUse check would fire on every correct change. Stop is the first moment
"the change is finished" is true.

The fourth site is the subtle one. ``config.py`` calls ``load_dotenv()`` at
import time, so a variable missing from the delenv tuple in
``test_from_env_uses_defaults_when_unset`` is resolved from the developer's
own environment instead of its default — the test keeps passing while quietly
no longer testing that default.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

data = json.load(sys.stdin)
if data.get("stop_hook_active"):
    sys.exit(0)  # already nudged once; let the turn end

root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".")
config = root / "rag_pipeline" / "config.py"
env_example = root / ".env.example"
readme = root / "README.md"
test_config = root / "tests" / "test_config.py"

if not config.is_file():
    sys.exit(0)

# `_env_\w+` (not an explicit path|int|str list) so a future `_env_bool` helper
# is covered the day it is added rather than silently escaping the check.
declared = re.findall(r'_env_\w+\(\s*"([A-Z0-9_]+)"', config.read_text())
if not declared:
    sys.exit(0)

env_text = env_example.read_text() if env_example.is_file() else ""
readme_text = readme.read_text() if readme.is_file() else ""
test_text = test_config.read_text() if test_config.is_file() else ""

# Anchor on the delenv call rather than the test's name, so renaming the test
# does not silently disable the check. Empty means the loop is gone or was
# restructured -- reported once below rather than once per variable.
delenv_tuple = "".join(
    re.findall(
        r"for\s+\w+\s+in\s+\((.*?)\):\s*\n\s*monkeypatch\.delenv",
        test_text,
        re.DOTALL,
    )
)

problems = []
for name in dict.fromkeys(declared):
    missing = []
    # A commented default line, e.g. "# RETRIEVAL_K=4".
    if not re.search(rf"^#\s*{name}=", env_text, re.MULTILINE):
        missing.append(f".env.example (add a commented default: `# {name}=<default>`)")
    # A row in the README config table, e.g. "| `RETRIEVAL_K` | `4` | ... |".
    if not re.search(rf"^\|\s*`{name}`\s*\|", readme_text, re.MULTILINE):
        missing.append(f"README.md config table (add a row for `{name}`)")
    # The delenv tuple in tests/test_config.py. Without it, config.py's
    # import-time load_dotenv() lets the developer's own .env supply the value
    # and the defaults test silently stops covering this variable.
    if delenv_tuple and f'"{name}"' not in delenv_tuple:
        missing.append(
            "tests/test_config.py (add it to the delenv tuple in "
            "test_from_env_uses_defaults_when_unset)"
        )
    if missing:
        problems.append(f"  {name}: missing from " + "; ".join(missing))

if not delenv_tuple and test_text:
    problems.append(
        "  tests/test_config.py: could not find the `for var in (...)` / "
        "monkeypatch.delenv loop, so the defaults test could not be checked. "
        "Restore it or update this hook."
    )

if problems:
    print(
        "Settings sites incomplete (CLAUDE.md: adding a Settings field is a "
        "four-file change - config.py + .env.example + README config table + "
        "the delenv tuple in tests/test_config.py).\n" + "\n".join(problems),
        file=sys.stderr,
    )
    sys.exit(2)
