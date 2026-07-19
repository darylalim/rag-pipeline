"""The mechanically checkable invariants from CLAUDE.md.

Owned here rather than in ``.claude/hooks/`` because the test suite is the
primary enforcement layer: it runs in CI, for every contributor and every PR
from a fork, not only inside a Claude Code session. The hooks in ``.claude/``
import this module to give the same answers earlier — at write time rather than
at review time — which makes them a latency optimization over the tests, not a
separate source of truth. Delete the hooks and the invariants still hold.

Two callers, two shapes of input. :func:`violations` takes a whole file when the
test sweeps the tree, and an Edit fragment when a hook checks what is about to
be written; the rules are the same either way.

Not everything in CLAUDE.md belongs here. These are the rules expressible as a
property of source *text*. Behavioral invariants — the exception union, the
empty-collection guard, ``source`` metadata on loaders — are enforced by
ordinary tests in ``test_pipeline.py`` and ``test_ingest.py``, which is a better
layer for them: they assert what the code does rather than how it is spelled.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple


class Rule(NamedTuple):
    """One invariant: where it applies, what violates it, and why it exists."""

    name: str
    applies: Callable[[str], bool]
    pattern: re.Pattern[str]
    # True for rules that must see comments (finding a lint suppression in one
    # is the whole job). False for rules that must not, so that prose
    # describing a rule is not blocked by the rule it describes.
    scan_comments: bool
    message: str


# One alternation, longest opener first. Every branch is unambiguous -- `[^\\]`
# cannot match what `\\.` matches -- which keeps this linear. An earlier
# two-pass form used `(?:\\.|(?!\1).)*?`, whose branches BOTH match a backslash:
# on an unterminated quote followed by backslash-heavy text that backtracks
# exponentially, and an unterminated quote is the *normal* case in an Edit
# fragment. Measured on the old pattern, an unterminated ''' plus 8 lines of
# `re.compile(r"\d+\w*\s")` took 6.5s and 12 lines never finished.
#
# Regex rather than tokenize/ast for the same reason: a fragment is usually not
# a parseable module, so a real parser rejects the common case.
STRING = re.compile(
    r'"""(?:[^\\]|\\.)*?"""'
    r"|'''(?:[^\\]|\\.)*?'''"
    r'|"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'",
    re.DOTALL,
)
COMMENT = re.compile(r"#[^\n]*")


def _blank(match: re.Match[str]) -> str:
    """Replace a span with just its newlines, so `^` anchors keep their lines."""
    return "\n" * match.group(0).count("\n")


RULES = [
    Rule(
        name="chroma-factory",
        applies=lambda p: (
            p.endswith(".py")
            and not p.startswith("tests/")
            and p != "rag_pipeline/ingest.py"
        ),
        pattern=re.compile(r"(?<![\w.])Chroma\s*\("),
        scan_comments=False,
        message=(
            "Constructing Chroma(...) inline. A collection's identity is (persist "
            "dir, collection name, embedding function), so indexing and querying "
            "must go through open_store() in rag_pipeline/ingest.py."
        ),
    ),
    Rule(
        name="embeddings-factory",
        # tests/ is NOT exempt here, unlike Chroma above: test_ingest.py opens a
        # collection directly on purpose, but nothing in tests should build a
        # real embedding model. Note the offline guarantee does not rest on this
        # rule -- conftest's socket-blocking fixture does, because it catches the
        # realistic version of the mistake (a forgotten `embeddings=` argument,
        # which names no banned symbol at all). This rule is authoring-time
        # feedback for the deliberate spelling.
        applies=lambda p: p.endswith(".py") and p != "rag_pipeline/ingest.py",
        pattern=re.compile(r"(?<![\w.])HuggingFaceEmbeddings\s*\("),
        scan_comments=False,
        message=(
            "Constructing HuggingFaceEmbeddings(...) inline. Route through "
            "build_embeddings() in rag_pipeline/ingest.py; in tests, inject "
            "DeterministicFakeEmbedding instead."
        ),
    ),
    Rule(
        name="lazy-cli-imports",
        applies=lambda p: p == "rag_pipeline/cli.py",
        pattern=re.compile(
            # `from rag_pipeline.ingest import ...` / `from .pipeline import ...`
            r"^(?:from\s+(?:rag_pipeline\.|\.)(?:ingest|pipeline)\b"
            # `from rag_pipeline import ingest, pipeline` / `from . import ingest`
            r"|from\s+(?:rag_pipeline|\.)\s+import\s+[^\n]*\b(?:ingest|pipeline)\b"
            # `import rag_pipeline.ingest`
            r"|import\s+rag_pipeline\.(?:ingest|pipeline)\b)",
            re.MULTILINE,
        ),
        scan_comments=False,
        message=(
            "Top-level import of ingest/pipeline in cli.py. These pull in "
            "sentence-transformers/torch (~4.3s vs ~0.08s for `rag --help`). "
            "Keep the import inside the command function."
        ),
    ),
    Rule(
        name="no-suppressions",
        applies=lambda p: p.endswith(".py"),
        # Spelled indirectly so this module does not match itself; the docstring
        # above may spell it, because string literals are masked.
        pattern=re.compile(r"#\s*(?:n[o]qa|ty:\s*ignore)"),
        scan_comments=True,
        message="Adding a lint/type suppression. CLAUDE.md: fix the finding instead.",
    ),
    Rule(
        name="no-rmtree",
        applies=lambda p: p == "rag_pipeline/ingest.py",
        pattern=re.compile(r"\brmtree\b"),
        scan_comments=False,
        message=(
            "rmtree in ingest.py. ingest() is a scoped collection rebuild and must "
            "never wipe the persist directory -- it may hold unrelated data."
        ),
    ),
    Rule(
        name="no-sampling-params",
        applies=lambda p: p == "rag_pipeline/pipeline.py",
        pattern=re.compile(r"\b(?:temperature|top_p)\s*="),
        scan_comments=False,
        message=(
            "Setting temperature/top_p. build_chat_model() deliberately sets "
            "neither -- some models (Opus 4.8) reject sampling params outright."
        ),
    ),
]


def violations(relpath: str, text: str) -> list[str]:
    """Messages for every invariant `text` breaks, as content of `relpath`.

    `relpath` is POSIX-style and relative to the project root. `text` may be a
    whole file or an Edit fragment; only the text given is inspected, so a hook
    never fires on code that was already on disk.
    """
    applicable = [rule for rule in RULES if rule.applies(relpath)]
    if not applicable:
        # Nothing can fire for this path -- skip masking, which for a large
        # non-Python write is the only real work here.
        return []

    with_comments = STRING.sub(_blank, text)
    code_only = COMMENT.sub(_blank, with_comments)

    return [
        rule.message
        for rule in applicable
        if rule.pattern.search(with_comments if rule.scan_comments else code_only)
    ]


# --- the rule documentation rule ---------------------------------------------

# Where a rule must be documented, beyond RULES itself.
RULES_SITE = "README.md"


def rules_problems(root: Path) -> list[str]:
    """Report any rule in :data:`RULES` with no row in the README's rule table.

    A rule is met as a test failure, so an undocumented rule is one the
    contributor was never told about — they read a style preference and get a
    hard CI failure naming a rule no document mentions. Nothing else catches
    that: a rule's existence is invisible to ruff, ty and the rest of the suite,
    which is exactly the hole the settings triad already found the hard way (the
    README's prose fell two rules behind ``RULES`` while everything stayed
    green). Deriving the requirement from ``RULES`` costs one test and makes that
    drift inexpressible rather than merely detectable — the same trade
    ``config.ENV_VARS`` makes for environment variables.
    """
    readme = root / RULES_SITE
    if not readme.is_file():
        return []

    readme_text = readme.read_text(errors="ignore")
    return [
        f"  {rule.name}: missing from {RULES_SITE} (add a row for `{rule.name}`)"
        for rule in RULES
        # A row in the README rule table, e.g. "| `no-rmtree` | ... |".
        if not re.search(rf"^\|\s*`{rule.name}`\s*\|", readme_text, re.MULTILINE)
    ]


# --- the Settings documentation rule -----------------------------------------

# Where a Settings field must appear, beyond config.py itself.
SETTINGS_SITES = (".env.example", "README.md")


def settings_fields(config_source: str) -> list[str]:
    """Environment variable names declared by the Settings dataclass.

    Parsed rather than regexed: this is the one place where missing a field
    means silently not enforcing it, and `ast` cannot be fooled by a field
    mentioned in a docstring. A field's variable is its name uppercased, which
    is also how `config.ENV_VARS` derives them — one convention, not two lists.
    """
    try:
        module = ast.parse(config_source)
    except SyntaxError:
        return []  # mid-edit; the next check sees a parseable file
    return [
        stmt.target.id.upper()
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    ]


def settings_problems(root: Path) -> list[str]:
    """Report any Settings field missing from `.env.example` or the README table.

    Adding a tunable means touching three files; nothing else in the repo
    notices when the latter two go stale, because `ruff`, `ty` and the whole
    suite stay green with a stale README.
    """
    config = root / "rag_pipeline" / "config.py"
    if not config.is_file():
        return []

    declared = settings_fields(config.read_text())
    if not declared:
        return []

    env_text = (root / ".env.example").read_text(errors="ignore")
    readme_text = (root / "README.md").read_text(errors="ignore")

    problems = []
    for name in dict.fromkeys(declared):
        missing = []
        # A commented default line, e.g. "# RETRIEVAL_K=4".
        if not re.search(rf"^#\s*{name}=", env_text, re.MULTILINE):
            missing.append(f".env.example (add `# {name}=<default>`)")
        # A row in the README config table, e.g. "| `RETRIEVAL_K` | `4` | ... |".
        if not re.search(rf"^\|\s*`{name}`\s*\|", readme_text, re.MULTILINE):
            missing.append(f"README.md config table (add a row for `{name}`)")
        if missing:
            problems.append(f"  {name}: missing from " + "; ".join(missing))
    return problems
