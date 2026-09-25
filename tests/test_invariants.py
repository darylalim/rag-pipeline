"""Enforce the CLAUDE.md invariants across the tree, in CI, for everyone.

This is the enforcement layer — there is no other. A human editing in vim, and a
PR from a fork, are covered by these tests and nothing else.

The rules that live in `invariants.py` are only the ones about how source is
*written*. Their behavioral counterparts are asserted where the behavior is:
`test_cli.py` proves cli.py's imports stay cheap, `test_mlx_models.py` proves
generation decodes greedily, and `test_ingest.py` proves ingest leaves a shared
collection's foreign documents alone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rag_pipeline.config import ENV_VARS, Settings
from tests.invariants import (
    RULES,
    rules_problems,
    settings_defaults,
    settings_problems,
    violations,
)

ROOT = Path(__file__).resolve().parent.parent


def tracked_python_files() -> list[str]:
    """Every tracked .py file, or [] when this is not a git work tree.

    Globbed rather than listed because a hardcoded list fails by silently not
    covering a new file. Returning [] rather than raising matters: this runs at
    collection time, and an exception here takes down the whole suite — including
    the product tests — in a release tarball or a Docker build with no .git.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "*.py"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(result.stdout.split()) if result.returncode == 0 else []


TRACKED = tracked_python_files()


# --- the tree-wide sweep: the invariant, enforced for everyone ---------------


@pytest.mark.skipif(not TRACKED, reason="not a git work tree")
@pytest.mark.parametrize("relpath", TRACKED or ["<none>"])
def test_no_source_file_violates_an_invariant(relpath: str) -> None:
    """The committed tree is clean against every rule.

    This is what makes the rules real: it fails in CI regardless of who wrote
    the code or which editor they used.
    """
    assert violations(relpath, (ROOT / relpath).read_text()) == []


@pytest.mark.skipif(not TRACKED, reason="not a git work tree")
def test_the_sweep_actually_covers_the_tree() -> None:
    """Guard against the sweep silently covering nothing.

    A parametrized test over an empty list is a green test that asserts
    nothing — the exact failure mode a hardcoded file list had. Skipped rather
    than failed without git, so a release tarball or Docker build still runs
    the product tests; CI always has a work tree, which is where this bites.
    """
    assert len(TRACKED) >= 10
    assert "rag_pipeline/config.py" in TRACKED
    assert "app.py" in TRACKED


# --- the rules themselves, in-process ----------------------------------------

VIOLATIONS = [
    pytest.param("app.py", 'store = Chroma(collection_name="x")', id="inline-chroma"),
    pytest.param(
        "app.py",
        "store = Chroma.from_documents(docs, embedding=e)",
        id="chroma-classmethod-constructor",
    ),
    pytest.param(
        "rag_pipeline/pipeline.py",
        "client = chromadb.PersistentClient(path=p)",
        id="inline-persistent-client",
    ),
    pytest.param("app.py", "client = chromadb.Client()", id="inline-chroma-client"),
    pytest.param(
        # tests/ is not exempt: a test that needs a collection opens it through
        # open_store(), so a second, differently configured client never exists.
        "tests/test_ingest.py",
        "client = chromadb.PersistentClient(path=str(tmp_path))",
        id="store-in-tests",
    ),
    pytest.param(
        "app.py", "e = QwenVLEmbeddings(m, dimensions=32)", id="inline-embeddings"
    ),
    pytest.param(
        # Qualified by its module, the construction is still a construction.
        "rag_pipeline/pipeline.py",
        "e = mlx_models.QwenVLEmbeddings(m, dimensions=d)",
        id="qualified-embeddings",
    ),
    pytest.param(
        # Nothing in tests builds a real embedding model -- the adapter's own
        # tests go through build_embeddings() too. The rule catches the legacy
        # HuggingFace spelling here as well.
        "tests/test_pipeline.py",
        'emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")',
        id="embeddings-in-tests",
    ),
    pytest.param("rag_pipeline/config.py", "import os  # noqa: F401", id="suppression"),
    pytest.param("rag_pipeline/config.py", "x = y  # ty: ignore", id="ty-ignore"),
]

ALLOWED = [
    pytest.param(
        "rag_pipeline/ingest.py",
        "client = chromadb.PersistentClient(path=p)\nreturn Chroma(client=client)",
        id="ingest-is-the-factory-home",
    ),
    pytest.param(
        # Only chromadb's Client is the store; a bare `Client(` belongs to every
        # HTTP library, so the rule names the module rather than guess.
        "app.py",
        "http = httpx.Client(timeout=5)",
        id="another-librarys-client",
    ),
    pytest.param(
        "rag_pipeline/ingest.py",
        "return QwenVLEmbeddings(m, dimensions=d)",
        id="ingest-builds-the-embedder",
    ),
    pytest.param(
        # The definition has to spell the name followed by a paren, and builds
        # nothing -- without the rule's `class ` lookbehind the adapter module
        # itself would fail the sweep.
        "rag_pipeline/mlx_models.py",
        "class QwenVLEmbeddings(Embeddings):",
        id="the-embedder-class-definition",
    ),
    pytest.param(
        # Naming the class is not building it: the adapter's tests patch its
        # methods this way.
        "tests/test_mlx_models.py",
        'monkeypatch.setattr(mlx_models.QwenVLEmbeddings, "_pool", pool)',
        id="referencing-the-embedder-class",
    ),
    # Prose describing a rule must not trip it, or the rule cannot be documented.
    pytest.param(
        "app.py",
        "# Never construct Chroma(...) inline -- use open_store().",
        id="comment-describing-store-rule",
    ),
    pytest.param(
        "rag_pipeline/config.py",
        'DOC = "write # noqa and it gets rejected"',
        id="suppression-inside-a-string",
    ),
    pytest.param(
        # A banned construction inside a triple-quoted block, at column 0 so no
        # indentation hides it: the case that proves multi-line string masking
        # works, and not merely that the single-line kind above does.
        "app.py",
        'HELP = """\nchromadb.PersistentClient(path=p)\n"""',
        id="store-inside-a-docstring",
    ),
    pytest.param("README.md", "Never construct Chroma(...) inline.", id="not-python"),
]


@pytest.mark.parametrize(("relpath", "text"), VIOLATIONS)
def test_violations_are_reported(relpath: str, text: str) -> None:
    assert violations(relpath, text) != []


@pytest.mark.parametrize(("relpath", "text"), ALLOWED)
def test_legitimate_code_is_not_reported(relpath: str, text: str) -> None:
    assert violations(relpath, text) == []


def test_masking_is_linear_on_pathological_input() -> None:
    """An unterminated quote plus backslash-heavy text must not blow up.

    The earlier two-pass masking backtracked exponentially here: 8 such lines
    took 6.5s and 12 never finished, i.e. the sweep hung rather than failed.
    Kept as a regression guard on the alternation in `invariants.py`, which is
    only linear as long as no two branches can match a backslash.
    """
    fragment = "'''\n" + 'x = re.compile(r"\\d+\\w*\\s")\n' * 40
    assert violations("rag_pipeline/ingest.py", fragment) == []


def test_every_rule_has_a_case_in_both_directions() -> None:
    """A rule with no test is a rule that can rot unnoticed.

    Checked per rule, not by counting cases: each rule must be the one reported
    by some violating case, and must match the raw text of some allowed case --
    a near miss it has to let through by path, masking or lookbehind. Without
    the second half, an allowed case no rule could ever match would pass while
    testing nothing, and a rule's exemptions would go unexercised.
    """
    assert len(RULES) == 3
    assert len({rule.name for rule in RULES}) == len(RULES)

    reported = [violations(*map(str, case.values)) for case in VIOLATIONS]
    near_misses = [str(case.values[1]) for case in ALLOWED]
    for rule in RULES:
        assert any(rule.message in found for found in reported), (
            f"{rule.name}: no case in VIOLATIONS is reported by it"
        )
        assert any(rule.pattern.search(text) for text in near_misses), (
            f"{rule.name}: no case in ALLOWED is a near miss for it"
        )


def test_every_rule_is_documented() -> None:
    """The README's rule table lists every rule that actually exists.

    The counterpart to `test_every_setting_is_documented` below, and added for
    the same reason after the same failure: the prose had fallen two rules behind
    `RULES` with the whole suite green, because a rule's existence is invisible
    to every other check.
    """
    assert rules_problems(ROOT) == []


def test_an_undocumented_rule_is_reported(tmp_path: Path) -> None:
    """The check fails when a rule is missing, not merely when it is present.

    Without this, `rules_problems` returning [] unconditionally — a bad regex, a
    renamed table — would read as success forever.
    """
    (tmp_path / "README.md").write_text("| `store-factory` | only this one |\n")
    assert len(rules_problems(tmp_path)) == len(RULES) - 1


# --- the Settings documentation rule -----------------------------------------


def test_every_setting_is_documented() -> None:
    """`.env.example` and the README config table are complete.

    Nothing else notices when they go stale: ruff, ty and the whole suite stay
    green against a stale README.
    """
    assert settings_problems(ROOT) == []


def test_documented_defaults_match_the_declared_ones() -> None:
    """Every default in `Settings` renders, so none goes silently unchecked.

    `settings_problems` skips a field whose default it cannot render, which is
    right — an unrenderable default is one no document could state literally —
    but it also means a rendering bug would quietly stop checking values while
    still reporting the row as present.
    """
    declared = settings_defaults((ROOT / "rag_pipeline/config.py").read_text())

    assert set(declared) == set(ENV_VARS)
    assert all(value is not None for value in declared.values()), (
        f"unrenderable defaults go unchecked: "
        f"{[k for k, v in declared.items() if v is None]}"
    )


@pytest.mark.parametrize(
    ("site", "wrong"),
    [
        pytest.param("README.md", "| `RETRIEVAL_K` | `9` | Chunks |", id="readme"),
        pytest.param(".env.example", "# RETRIEVAL_K=9\n", id="env-example"),
    ],
)
def test_a_documented_default_that_lies_is_reported(
    tmp_path: Path, site: str, wrong: str
) -> None:
    """A row naming the right variable with the wrong value must not pass.

    The row-exists check would: it reads only the leading cell. A default is the
    one thing in these tables a reader acts on directly, and it is derivable, so
    it is checkable — unlike the prose beside it.
    """
    (tmp_path / "rag_pipeline").mkdir()
    (tmp_path / "rag_pipeline" / "config.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Settings:\n"
        "    retrieval_k: int = 4\n"
    )
    sites = {
        "README.md": "| `RETRIEVAL_K` | `4` | Chunks |",
        ".env.example": "# RETRIEVAL_K=4\n",
    }
    sites[site] = wrong
    for filename, text in sites.items():
        (tmp_path / filename).write_text(text)

    assert any("RETRIEVAL_K" in problem for problem in settings_problems(tmp_path))


@pytest.mark.parametrize(
    ("cell", "documented"),
    [
        pytest.param(" ", True, id="blank"),
        pytest.param(" `http://localhost:6006` ", False, id="a-value"),
        pytest.param(" `` ", False, id="empty-code-span"),
    ],
)
def test_an_empty_default_is_documented_as_a_blank_cell(
    tmp_path: Path, cell: str, documented: bool
) -> None:
    """An empty default has one honest rendering in a table: nothing.

    Checked both ways: a value in that cell would be a default the code does
    not have, and a pair of backticks is what Markdown shows for an "empty"
    code span.
    """
    (tmp_path / "rag_pipeline").mkdir()
    (tmp_path / "rag_pipeline" / "config.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Settings:\n"
        '    phoenix_collector_endpoint: str = ""\n'
    )
    (tmp_path / ".env.example").write_text("# PHOENIX_COLLECTOR_ENDPOINT=\n")
    (tmp_path / "README.md").write_text(
        f"| `PHOENIX_COLLECTOR_ENDPOINT` |{cell}| Tracing |\n"
    )

    assert (settings_problems(tmp_path) == []) is documented


@pytest.mark.parametrize(
    ("line", "documented"),
    [
        pytest.param("# PHOENIX_COLLECTOR_ENDPOINT=\n", True, id="bare"),
        pytest.param(
            "# PHOENIX_COLLECTOR_ENDPOINT=   # e.g. http://localhost:6006\n",
            False,
            id="trailing-comment",
        ),
    ],
)
def test_an_empty_default_is_a_bare_line_in_the_env_example(
    tmp_path: Path, line: str, documented: bool
) -> None:
    """Uncommented, the line must mean the default -- and for an empty one a
    trailing comment breaks that: python-dotenv takes `# e.g. ...` as the value,
    which the URL check then refuses, stopping both frontends."""
    (tmp_path / "rag_pipeline").mkdir()
    (tmp_path / "rag_pipeline" / "config.py").write_text(
        "from dataclasses import dataclass\n\n\n"
        "@dataclass(frozen=True)\n"
        "class Settings:\n"
        '    phoenix_collector_endpoint: str = ""\n'
    )
    (tmp_path / ".env.example").write_text(line)
    (tmp_path / "README.md").write_text(
        "| `PHOENIX_COLLECTOR_ENDPOINT` | | Tracing |\n"
    )

    assert (settings_problems(tmp_path) == []) is documented


def test_settings_extraction_matches_config_env_vars() -> None:
    """The text-level extraction agrees with the imported dataclass.

    Two mechanisms — `settings_defaults` parses config.py's AST, `ENV_VARS`
    reads `fields(Settings)` — and only the first decides what gets documented.
    A field the AST pass failed to see would go unchecked at both sites while
    every other test stayed green, so the disagreement is asserted directly.
    """
    declared = settings_defaults((ROOT / "rag_pipeline/config.py").read_text())

    assert tuple(declared) == ENV_VARS


@pytest.mark.parametrize("var", ENV_VARS)
def test_every_env_var_actually_overrides_its_field(
    var: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each name in ENV_VARS is one `from_env` really reads.

    ENV_VARS is derived from the dataclass fields, but `from_env` names its
    variables as literals — so a typo there would leave a name in ENV_VARS that
    overrides nothing, and the defaults test would clear a variable no one uses.
    Asserting the override behaviorally is what ties the two together.
    """
    field = next(
        f for f in Settings.__dataclass_fields__.values() if f.name.upper() == var
    )
    default = getattr(Settings, field.name)
    # isinstance, not type(): a Path default is a PosixPath, and bool must be
    # checked before int because bool subclasses it.
    if isinstance(default, Path):
        override = str(tmp_path)
    elif isinstance(default, bool):
        override = "1"
    elif isinstance(default, int):
        override = "7"
    else:
        # A URL, because one str setting must be one; every other str reader
        # passes any string through.
        override = "http://sentinel.invalid"

    monkeypatch.setenv(var, override)
    changed = getattr(Settings.from_env(), field.name)

    assert changed != default, f"{var} did not override {field.name}"
