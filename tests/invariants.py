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

**This module must stay standard-library only.** The hooks exec on
``#!/usr/bin/env python3`` -- the *system* interpreter, not the project venv --
so importing anything from ``rag_pipeline`` would pull in ``dotenv`` and friends
and crash the hook instead of checking anything. This is why field defaults are
read out of config.py's AST rather than off ``Settings``. Nothing in the suite
catches a violation: pytest runs under uv, where those imports resolve fine.

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
        pattern=re.compile(
            r"(?<![\w.])(?:VoyageAIEmbeddings|HuggingFaceEmbeddings)\s*\("
        ),
        scan_comments=False,
        message=(
            "Constructing an embedding model (VoyageAIEmbeddings/"
            "HuggingFaceEmbeddings) inline. Route through build_embeddings() in "
            "rag_pipeline/ingest.py; in tests, inject DeterministicFakeEmbedding "
            "instead."
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
            "Top-level import of ingest/pipeline in cli.py. These pull in the "
            "chromadb/langchain stack (~0.9s of imports vs ~0.02s to import cli "
            "alone). Keep the import inside the command function."
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


# --- documented-in-the-README, the shape both rules below share ---------------


def has_table_row(readme_text: str, name: str, default: str | None = None) -> bool:
    """Is `name` the first cell of a README table row, optionally stating `default`?

    The one thing the two documentation rules below genuinely share, so it is
    defined once. Both would otherwise carry their own copy of what counts as a
    documented row, and a change to the table style would be applied to one --
    leaving the other quietly matching nothing, which reads as success.

    The backticks around `default` are required rather than tolerated: without
    them `1000 chars` and a bare `1024` both pass as near-misses, and a cell that
    is nearly right is the one a reader trusts.

    Not anchored to a particular table, so setting names, rule names and the CI
    job names share one namespace across the file. No collision is possible with
    the current names, and anchoring costs more regex than the risk earns.
    """
    cell = rf"\s*`{re.escape(default)}`\s*\|" if default is not None else ""
    return bool(re.search(rf"^\|\s*`{name}`\s*\|{cell}", readme_text, re.MULTILINE))


# --- the rule documentation rule ---------------------------------------------

# Where a rule must be documented, beyond RULES itself. A tuple like
# SETTINGS_SITES below, so a second site is an appended string rather than a
# type change here and at every caller.
RULES_SITES = ("README.md",)


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
    (site,) = RULES_SITES
    readme = root / site
    if not readme.is_file():
        return []

    readme_text = readme.read_text(errors="ignore")
    return [
        f"  {rule.name}: missing from {site} (add a row for `{rule.name}`)"
        for rule in RULES
        if not has_table_row(readme_text, rule.name)
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


def _render_default(node: ast.expr) -> str | None:
    """How a field's default should be spelled in the docs, or None if unknowable.

    Read from the AST rather than from `Settings` itself, because this module is
    imported by a hook that runs on the *system* interpreter -- where
    rag_pipeline's dependencies, `dotenv` among them, are not installed.
    Importing config.py there would crash the hook rather than check anything.

    Reading the class default is also the only correct reading: `from_env()`
    would answer to the developer's own `.env`, which is the drift
    `config.ENV_VARS` exists to make inexpressible.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str | int):
        return str(node.value)
    # `_ROOT / "data"` is an absolute path at runtime but documented as `./data`,
    # since an absolute one would be this machine's layout rather than a default.
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Div)
        and isinstance(node.left, ast.Name)
        and node.left.id == "_ROOT"
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, str)
    ):
        return f"./{node.right.value}"
    return None


def settings_defaults(config_source: str) -> dict[str, str | None]:
    """Each Settings field's documented default, keyed by environment variable.

    A field whose default does not render (a call, a computed expression) maps
    to None, and its value simply goes unchecked -- an unrenderable default is
    one no documentation could state literally either.
    """
    try:
        module = ast.parse(config_source)
    except SyntaxError:
        return {}
    return {
        stmt.target.id.upper(): _render_default(stmt.value)
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Settings"
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.value is not None
    }


def settings_problems(root: Path) -> list[str]:
    """Report any Settings field missing from `.env.example` or the README table.

    Adding a tunable means touching three files; nothing else in the repo
    notices when the latter two go stale, because `ruff`, `ty` and the whole
    suite stay green with a stale README.
    """
    config = root / "rag_pipeline" / "config.py"
    if not config.is_file():
        return []

    config_source = config.read_text()
    declared = settings_fields(config_source)
    if not declared:
        return []
    defaults = settings_defaults(config_source)

    env_text = (root / ".env.example").read_text(errors="ignore")
    readme_text = (root / "README.md").read_text(errors="ignore")

    problems = []
    for name in dict.fromkeys(declared):
        default = defaults.get(name)
        shown = f"`{default}`" if default is not None else "<default>"
        missing = []
        # A commented default line, e.g. "# RETRIEVAL_K=4", stating the value the
        # code actually declares -- a name alone would let `CHAT_MODEL=gpt-4` pass.
        # The trailing group allows the explanatory comments some lines carry.
        value = rf"{re.escape(default)}\s*(?:#|$)" if default is not None else ""
        if not re.search(rf"^#\s*{name}={value}", env_text, re.MULTILINE):
            missing.append(f".env.example (needs `# {name}={default or '<default>'}`)")
        if not has_table_row(readme_text, name, default):
            missing.append(f"README.md config table (needs a `{name}` row, {shown})")
        if missing:
            problems.append(f"  {name}: missing from " + "; ".join(missing))
    return problems
