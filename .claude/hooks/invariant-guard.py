#!/usr/bin/env python3
"""PreToolUse hook: reject edits that violate mechanical CLAUDE.md invariants.

Only the text being written is inspected, so pre-existing code never trips it.

Patterns are matched against a masked copy of that text rather than the raw
bytes, because otherwise a rule fires on the prose describing it: a comment
reading `# never construct Chroma(...) inline` would be blocked by the very
rule it documents. String literals are masked for every rule; comments are
masked for the code rules but kept for the suppression rule, which exists to
find `# noqa` in a comment and nowhere else.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

try:
    data = json.load(sys.stdin)
except ValueError as exc:
    # Exit 1, not 2: a payload we cannot parse must not block the edit, but it
    # means the hook is not enforcing, so say so rather than exiting silently.
    # Claude Code shows the first stderr line for a non-blocking error.
    print(
        f"invariant-guard: unreadable hook payload, not enforcing ({exc})",
        file=sys.stderr,
    )
    sys.exit(1)

tool_input = data.get("tool_input") or {}

raw_path = tool_input.get("file_path") or ""
if not raw_path:
    sys.exit(0)

root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".").resolve()
# A relative file_path is relative to the project, not to wherever this hook
# happens to be running; resolving it against cwd would place it outside root
# and skip enforcement entirely.
candidate = Path(raw_path)
if not candidate.is_absolute():
    candidate = root / candidate
try:
    rel = candidate.resolve().relative_to(root).as_posix()
except ValueError:
    sys.exit(0)  # outside the repo

# The hook scripts and their test quote the banned patterns; masking (below)
# already covers the quoted forms, but these files exist to describe the rules
# and have no business being policed by them.
if rel.startswith(".claude/") or rel == "tests/test_hooks.py":
    sys.exit(0)

chunks = [tool_input.get("content"), tool_input.get("new_string")]
text = "\n".join(c for c in chunks if isinstance(c, str))
if not text.strip():
    sys.exit(0)


def blank(match: re.Match[str]) -> str:
    """Replace a span with just its newlines, so `^` anchors keep their lines."""
    return "\n" * match.group(0).count("\n")


# Applied in order: triple-quoted strings, then single-line strings, then
# comments. Text may be an Edit fragment rather than a parseable module, so
# this is deliberately regex-based rather than tokenize/ast.
TRIPLE_QUOTED = re.compile(r"(\"\"\"|''')(?:\\.|(?!\1).)*?\1", re.DOTALL)
QUOTED = re.compile(r"(\"|')(?:\\.|(?!\1)[^\n])*?\1")
COMMENT = re.compile(r"#[^\n]*")

with_comments = QUOTED.sub(blank, TRIPLE_QUOTED.sub(blank, text))
code_only = COMMENT.sub(blank, with_comments)

CODE = "code"
COMMENTS = "comments"

RULES = [
    (
        lambda p: (
            p.endswith(".py")
            and not p.startswith("tests/")
            and p != "rag_pipeline/ingest.py"
        ),
        re.compile(r"(?<![\w.])Chroma\s*\("),
        CODE,
        "Constructing Chroma(...) inline. A collection's identity is (persist dir, "
        "collection name, embedding function), so indexing and querying must go "
        "through open_store() in rag_pipeline/ingest.py.",
    ),
    (
        # tests/ is NOT exempt here, unlike Chroma above. test_ingest.py opens a
        # collection directly on purpose, but a test that builds a real embedding
        # model would download ~90MB and put the suite back on the network.
        lambda p: p.endswith(".py") and p != "rag_pipeline/ingest.py",
        re.compile(r"(?<![\w.])HuggingFaceEmbeddings\s*\("),
        CODE,
        "Constructing HuggingFaceEmbeddings(...) inline. Route through "
        "build_embeddings() in rag_pipeline/ingest.py -- vectors from different "
        "models are incomparable, and in tests this breaks the offline guarantee "
        "(inject DeterministicFakeEmbedding instead).",
    ),
    (
        lambda p: p == "rag_pipeline/cli.py",
        re.compile(
            # `from rag_pipeline.ingest import ...` / `from .pipeline import ...`
            r"^(?:from\s+(?:rag_pipeline\.|\.)(?:ingest|pipeline)\b"
            # `from rag_pipeline import ingest, pipeline` / `from . import ingest`
            r"|from\s+(?:rag_pipeline|\.)\s+import\s+[^\n]*\b(?:ingest|pipeline)\b"
            # `import rag_pipeline.ingest`
            r"|import\s+rag_pipeline\.(?:ingest|pipeline)\b)",
            re.MULTILINE,
        ),
        CODE,
        "Top-level import of ingest/pipeline in cli.py. These pull in "
        "sentence-transformers/torch (~4.3s vs ~0.08s for `rag --help`). Keep the "
        "import inside the command function.",
    ),
    (
        lambda p: p.endswith(".py"),
        re.compile(r"#\s*(?:noqa|ty:\s*ignore)"),
        COMMENTS,
        "Adding a lint/type suppression. CLAUDE.md: fix the finding instead.",
    ),
    (
        lambda p: p == "rag_pipeline/ingest.py",
        re.compile(r"\brmtree\b"),
        CODE,
        "rmtree in ingest.py. ingest() is a scoped collection rebuild and must never "
        "wipe the persist directory -- it may hold unrelated data.",
    ),
    (
        lambda p: p == "rag_pipeline/pipeline.py",
        re.compile(r"\b(?:temperature|top_p)\s*="),
        CODE,
        "Setting temperature/top_p. build_chat_model() deliberately sets neither -- "
        "some models (Opus 4.8) reject sampling params outright.",
    ),
]

haystack = {CODE: code_only, COMMENTS: with_comments}
hits = [
    msg
    for applies, pattern, scope, msg in RULES
    if applies(rel) and pattern.search(haystack[scope])
]

if hits:
    print(f"Blocked edit to {rel} - CLAUDE.md invariant violation:", file=sys.stderr)
    for msg in hits:
        print(f"  - {msg}", file=sys.stderr)
    sys.exit(2)
