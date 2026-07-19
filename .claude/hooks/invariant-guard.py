#!/usr/bin/env python3
"""PreToolUse hook: reject edits that violate mechanical CLAUDE.md invariants.

Only the text being written is inspected, so pre-existing code never trips it.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

data = json.load(sys.stdin)
tool_input = data.get("tool_input") or {}

raw_path = tool_input.get("file_path") or ""
if not raw_path:
    sys.exit(0)

root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or ".").resolve()
try:
    rel = Path(raw_path).resolve().relative_to(root).as_posix()
except ValueError:
    sys.exit(0)  # outside the repo

# Files that exist to *describe* the rules necessarily contain the patterns
# they ban: the hooks quote them in their own error messages, and the hook test
# uses them as fixtures. Without this exemption the guard makes both
# permanently uneditable.
if rel.startswith(".claude/") or rel == "tests/test_hooks.py":
    sys.exit(0)

chunks = [tool_input.get("content"), tool_input.get("new_string")]
for edit in tool_input.get("edits") or []:
    chunks.append(edit.get("new_string"))
text = "\n".join(c for c in chunks if isinstance(c, str))
if not text.strip():
    sys.exit(0)

RULES = [
    (
        lambda p: (
            p.endswith(".py")
            and not p.startswith("tests/")
            and p != "rag_pipeline/ingest.py"
        ),
        re.compile(r"(?<![\w.])(?:Chroma|HuggingFaceEmbeddings)\s*\("),
        "Constructing Chroma(...) / HuggingFaceEmbeddings(...) inline. A collection's "
        "identity is (persist dir, collection name, embedding function) and vectors "
        "from different models are incomparable -- route through build_embeddings() / "
        "open_store() in rag_pipeline/ingest.py.",
    ),
    (
        lambda p: p == "rag_pipeline/cli.py",
        re.compile(
            # `from rag_pipeline.ingest import ...` / `from .pipeline import ...`
            r"^(?:from\s+(?:rag_pipeline\.|\.)(?:ingest|pipeline)\b"
            # `from rag_pipeline import ingest, pipeline` / `from . import ingest`.
            # [^#\n]* stops a trailing comment mentioning "pipeline" from matching.
            r"|from\s+(?:rag_pipeline|\.)\s+import\s+[^#\n]*\b(?:ingest|pipeline)\b"
            # `import rag_pipeline.ingest`
            r"|import\s+rag_pipeline\.(?:ingest|pipeline)\b)",
            re.MULTILINE,
        ),
        "Top-level import of ingest/pipeline in cli.py. These pull in "
        "sentence-transformers/torch (~4.3s vs ~0.08s for `rag --help`). Keep the "
        "import inside the command function.",
    ),
    (
        lambda p: p.endswith(".py"),
        re.compile(r"#\s*(?:noqa|ty:\s*ignore)"),
        "Adding a lint/type suppression. CLAUDE.md: fix the finding instead.",
    ),
    (
        lambda p: p == "rag_pipeline/ingest.py",
        re.compile(r"\brmtree\b"),
        "rmtree in ingest.py. ingest() is a scoped collection rebuild and must never "
        "wipe the persist directory -- it may hold unrelated data.",
    ),
    (
        lambda p: p == "rag_pipeline/pipeline.py",
        re.compile(r"\b(?:temperature|top_p)\s*="),
        "Setting temperature/top_p. build_chat_model() deliberately sets neither -- "
        "some models (Opus 4.8) reject sampling params outright.",
    ),
]

hits = [msg for applies, pattern, msg in RULES if applies(rel) and pattern.search(text)]

if hits:
    print(f"Blocked edit to {rel} - CLAUDE.md invariant violation:", file=sys.stderr)
    for msg in hits:
        print(f"  - {msg}", file=sys.stderr)
    sys.exit(2)
