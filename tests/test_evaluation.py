"""Tests for `rag eval`: the question set, the evaluators, the judge, the report.

All offline. LangSmith and Claude are stood in for by small fakes at the two
seams `evaluation.py` exposes -- the LangSmith client it is handed and the
`Judge` callable -- and the target runs the real pipeline over the conftest
fakes. What `langsmith.evaluate` itself does with these is LangSmith's to test.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import FakeListChatModel
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.utils import LangSmithError

from rag_pipeline import evaluation as ev
from rag_pipeline import ingest as ingest_mod
from rag_pipeline.pipeline import RAGPipeline

_ROOT = Path(__file__).resolve().parents[1]


def _q(
    id: str = "q1", sources: tuple[str, ...] = ("a.md",), category: str = "easy"
) -> ev.Question:
    return ev.Question(
        id=id,
        question=f"Question {id}?",
        answer="An answer.",
        sources=sources,
        category=category,
    )


# --- the question set ---------------------------------------------------------


def test_the_shipped_question_set_fits_the_eval_corpus():
    """Every expected source is a corpus file, so retrieval_hit can pass.

    A misspelled source would score every retrieval of that question a miss,
    which reads as a retrieval problem rather than a typo.
    """
    questions = ev.load_questions()
    corpus = {
        p.relative_to(ev.EVAL_CORPUS).as_posix() for p in ev.EVAL_CORPUS.rglob("*.md")
    }

    assert len(questions) >= 40
    assert {s for q in questions for s in q.sources} <= corpus
    # Both kinds, so `correct` is tested on declining as well as answering.
    assert any(q.answerable for q in questions)
    assert any(not q.answerable for q in questions)
    # Every category, so each one's line in the report means something.
    assert {q.category for q in questions} == set(ev.CATEGORIES)
    # A decoy is unanswerable by definition; a source on one would make it
    # an ordinary question scored as a failed decline.
    assert not any(q.answerable for q in questions if q.category == "decoy")


def test_the_eval_corpus_outnumbers_what_vector_search_fetches():
    """With fewer chunks than FETCH_K, every chunk reaches the reranker and the
    embedding model is never tested -- the 9-chunk sample corpus's problem."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag_pipeline.config import Settings

    defaults = Settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=defaults.chunk_size, chunk_overlap=defaults.chunk_overlap
    )
    chunks = sum(
        len(splitter.split_text(p.read_text(encoding="utf-8")))
        for p in ev.EVAL_CORPUS.rglob("*.md")
    )

    assert chunks >= 2 * defaults.fetch_k


def test_a_context_question_cites_chunks_that_do_not_name_their_subject():
    """What makes a `context` question hard: its document is long enough to
    split into several chunks, and most of them never name the service the
    file is about (its file name), so a chunk alone cannot be attributed to it.

    An edit that put the name back into every section would make the category
    quietly easy again, with every check still green.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag_pipeline.config import Settings

    defaults = Settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=defaults.chunk_size, chunk_overlap=defaults.chunk_overlap
    )
    cited = {
        source
        for q in ev.load_questions()
        if q.category == "context"
        for source in q.sources
    }

    assert cited, "the shipped set has no context questions"
    for source in sorted(cited):
        chunks = splitter.split_text((ev.EVAL_CORPUS / source).read_text("utf-8"))
        subject = Path(source).stem
        named = [chunk for chunk in chunks if subject.lower() in chunk.lower()]
        assert len(chunks) >= 4, source
        assert len(named) < len(chunks) / 2, (source, len(named), len(chunks))


def test_a_table_question_cites_a_table_split_from_its_header():
    """What makes a `table` question hard: its document's table is long
    enough that the splitter cuts it, leaving rows in a chunk without the
    header that says what each column means.

    A table shortened, or moved so it fit in one chunk, would make the
    category quietly easy again.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from rag_pipeline.config import Settings

    defaults = Settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=defaults.chunk_size, chunk_overlap=defaults.chunk_overlap
    )
    cited = {
        source
        for q in ev.load_questions()
        if q.category == "table"
        for source in q.sources
    }

    assert cited, "the shipped set has no table questions"
    for source in sorted(cited):
        chunks = splitter.split_text((ev.EVAL_CORPUS / source).read_text("utf-8"))
        headerless = [c for c in chunks if "\n| " in c and "| --- |" not in c]
        assert headerless, f"{source}: no table rows without their header"


def test_eval_settings_swap_the_corpus_and_collection_but_keep_the_models(settings):
    """The models are what is evaluated, so they are kept; tracing is on, so an
    experiment's rows carry each question's whole trace."""
    evaluated = ev.eval_settings(settings)

    assert evaluated.langsmith_tracing is True
    assert not settings.langsmith_tracing

    assert evaluated.data_dir == ev.EVAL_CORPUS
    assert evaluated.collection_name == ev.EVAL_COLLECTION != settings.collection_name
    assert evaluated.mongodb_db == settings.mongodb_db
    assert evaluated.embedding_model == settings.embedding_model
    assert evaluated.chat_model == settings.chat_model
    assert evaluated.retrieval_k == settings.retrieval_k


def test_the_eval_corpus_indexes_every_expected_source(settings, fake_embeddings):
    """Ingested as `rag eval` ingests it, beside -- not into -- the user's index."""
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    evaluated = ev.eval_settings(settings)

    ingest_mod.ingest(evaluated, embeddings=fake_embeddings)

    expected = {s for q in ev.load_questions() for s in q.sources}
    assert expected <= ingest_mod.indexed_sources(evaluated)
    # The user's collection is untouched by it.
    assert ingest_mod.indexed_sources(settings) == {"a.md", "sub/b.txt"}


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        pytest.param("not json", "Cannot read", id="not-json"),
        pytest.param("[]", "non-empty JSON list", id="empty"),
        pytest.param(
            '[{"id": "a", "question": "q?"}]', "exactly the keys", id="missing-key"
        ),
        pytest.param(
            '[{"id": "a", "question": " ", "answer": "x", "sources": [],'
            ' "category": "easy"}]',
            "must be text",
            id="blank-question",
        ),
        pytest.param(
            '[{"id": "a", "question": "q?", "answer": "x", "sources": "a.md",'
            ' "category": "easy"}]',
            "list of file names",
            id="sources-not-a-list",
        ),
        pytest.param(
            '[{"id": "a", "question": "q?", "answer": "x", "sources": [],'
            ' "category": "easy"},'
            ' {"id": "a", "question": "r?", "answer": "y", "sources": [],'
            ' "category": "easy"}]',
            "duplicate question ids",
            id="duplicate-id",
        ),
        pytest.param(
            '[{"id": "a", "question": "q?", "answer": "x", "sources": []}]',
            "exactly the keys",
            id="no-category",
        ),
        pytest.param(
            '[{"id": "a", "question": "q?", "answer": "x", "sources": [],'
            ' "category": "hard"}]',
            "category must be one of",
            id="unknown-category",
        ),
    ],
)
def test_a_malformed_question_set_is_a_value_error(tmp_path, content, problem):
    path = tmp_path / "questions.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=problem):
        ev.load_questions(path)


def test_the_dataset_name_follows_the_questions_content():
    """Same questions, same name; any edit, a new name -- so an experiment is
    never set beside one scored on different questions."""
    questions = [_q("q1"), _q("q2", ())]

    assert ev.dataset_name(questions) == ev.dataset_name(list(questions))
    edited = [questions[0], replace(questions[1], answer="Another answer.")]
    assert ev.dataset_name(edited) != ev.dataset_name(questions)
    assert ev.dataset_name(questions).startswith("rag-pipeline-qa-")


# --- the LangSmith dataset ----------------------------------------------------


class _FakeClient:
    """Records what sync_dataset asks of LangSmith."""

    def __init__(self, existing_count: int | None = None, fail_examples: bool = False):
        self.existing_count = existing_count
        self.fail_examples = fail_examples
        self.created: list[str] = []
        self.examples: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def has_dataset(self, *, dataset_name: str) -> bool:
        return self.existing_count is not None

    def read_dataset(self, *, dataset_name: str) -> Any:
        return SimpleNamespace(example_count=self.existing_count)

    def create_dataset(self, dataset_name: str, *, description: str) -> Any:
        self.created.append(dataset_name)
        return SimpleNamespace(id="ds-1")

    def create_examples(
        self, *, dataset_id: str, examples: list[dict[str, Any]]
    ) -> None:
        if self.fail_examples:
            raise LangSmithError("upload failed")
        self.examples.extend(examples)

    def delete_dataset(self, *, dataset_id: str) -> None:
        self.deleted.append(dataset_id)


def test_a_new_question_set_is_uploaded_once_with_references():
    questions = [_q("q1"), _q("q2", ())]
    client = _FakeClient()

    name = ev.sync_dataset(client, questions)

    assert client.created == [name]
    assert client.examples == [
        {
            "inputs": {"question": "Question q1?"},
            "outputs": {"answer": "An answer.", "sources": ["a.md"]},
            "metadata": {"id": "q1", "category": "easy"},
        },
        {
            "inputs": {"question": "Question q2?"},
            "outputs": {"answer": "An answer.", "sources": []},
            "metadata": {"id": "q2", "category": "easy"},
        },
    ]


def test_an_uploaded_question_set_is_reused():
    client = _FakeClient(existing_count=2)

    ev.sync_dataset(client, [_q("q1"), _q("q2")])

    assert client.created == []


def test_a_dataset_holding_other_examples_is_refused():
    with pytest.raises(RuntimeError, match="holds 5 examples, not 2"):
        ev.sync_dataset(_FakeClient(existing_count=5), [_q("q1"), _q("q2")])


def test_a_half_uploaded_dataset_is_deleted_and_reported_as_runtime_error():
    """Left behind, the next run would find it by name and score against it."""
    client = _FakeClient(fail_examples=True)

    with pytest.raises(RuntimeError, match="LangSmith refused"):
        ev.sync_dataset(client, [_q()])

    assert client.deleted == ["ds-1"]


# --- the target ---------------------------------------------------------------


def test_the_target_runs_the_pipeline_and_returns_what_was_read(
    settings, fake_embeddings, fake_reranker
):
    ingest_mod.ingest(settings, embeddings=fake_embeddings)
    ingest_mod.reset_store_cache()
    pipeline = RAGPipeline(
        settings,
        embeddings=fake_embeddings,
        llm=FakeListChatModel(responses=["Apples grow in orchards. (a.md)"]),
        reranker=fake_reranker,
    )

    outputs = ev.make_target(pipeline)({"question": "apples"})

    assert outputs["answer"] == "Apples grow in orchards. (a.md)"
    assert set(outputs["sources"]) <= {"a.md", "sub/b.txt"}
    assert outputs["contexts"]
    assert {c["source"] for c in outputs["contexts"]} == set(outputs["sources"])
    # The vector search's candidates, before the rerank: the fake reranker
    # keeps the first RETRIEVAL_K of them, in order.
    assert outputs["candidates"][: len(outputs["contexts"])] == [
        c["source"] for c in outputs["contexts"]
    ]
    # Plain dicts: LangSmith serializes the outputs as JSON.
    json.dumps(outputs)


# --- evaluators ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("outputs", "expected", "score"),
    [
        pytest.param({"sources": ["b.md", "a.md"]}, ["a.md"], 1, id="hit"),
        pytest.param({"sources": ["b.md"]}, ["a.md"], 0, id="miss"),
        pytest.param({"sources": ["b.md"]}, ["a.md", "b.md"], 1, id="any-of-two"),
        pytest.param({"sources": ["a.md"]}, [], None, id="unanswerable"),
        pytest.param({}, ["a.md"], None, id="run-failed"),
    ],
)
def test_retrieval_hit(outputs, expected, score):
    result = ev.retrieval_hit(outputs, {"answer": "x", "sources": expected})

    assert result["key"] == "retrieval_hit"
    assert result["score"] == score


@pytest.mark.parametrize(
    ("outputs", "expected", "score"),
    [
        pytest.param({"sources": ["a.md"]}, ["a.md"], 1.0, id="one-of-one"),
        pytest.param({"sources": ["a.md", "c.md"]}, ["a.md", "b.md"], 0.5, id="half"),
        pytest.param({"sources": ["b.md", "a.md"]}, ["a.md", "b.md"], 1.0, id="both"),
        pytest.param({"sources": ["c.md"]}, ["a.md"], 0.0, id="none"),
        pytest.param({"sources": ["a.md"]}, [], None, id="unanswerable"),
        pytest.param({}, ["a.md"], None, id="run-failed"),
    ],
)
def test_retrieval_recall(outputs, expected, score):
    """Where retrieval_hit passes a two-file question on one file, recall halves."""
    result = ev.retrieval_recall(outputs, {"answer": "x", "sources": expected})

    assert result["key"] == "retrieval_recall"
    assert result["score"] == score


def _contexts(*sources: str) -> dict:
    return {"contexts": [{"source": source, "text": "..."} for source in sources]}


@pytest.mark.parametrize(
    ("outputs", "expected", "score"),
    [
        pytest.param(_contexts("a.md", "b.md"), ["a.md"], 1.0, id="first"),
        pytest.param(_contexts("b.md", "a.md"), ["a.md"], 0.5, id="second"),
        # Ranked over chunks, not files: two chunks of b.md push a.md to third.
        pytest.param(_contexts("b.md", "b.md", "a.md"), ["a.md"], 1 / 3, id="by-chunk"),
        pytest.param(
            _contexts("c.md", "b.md", "a.md"), ["a.md", "b.md"], 0.5, id="any-of-two"
        ),
        pytest.param(_contexts("b.md", "c.md"), ["a.md"], 0.0, id="absent"),
        pytest.param(_contexts("a.md"), [], None, id="unanswerable"),
        pytest.param({}, ["a.md"], None, id="run-failed"),
    ],
)
def test_retrieval_rank(outputs, expected, score):
    result = ev.retrieval_rank(outputs, {"answer": "x", "sources": expected})

    assert result["key"] == "retrieval_rank"
    assert result["score"] == (pytest.approx(score) if score is not None else None)


@pytest.mark.parametrize(
    ("candidates", "expected", "score"),
    [
        pytest.param(["a.md", "b.md"], ["a.md"], 1.0, id="first"),
        pytest.param(["b.md", "b.md", "c.md", "a.md"], ["a.md"], 0.25, id="fourth"),
        pytest.param(["b.md", "c.md"], ["a.md"], 0.0, id="never-fetched"),
        pytest.param(["a.md"], [], None, id="unanswerable"),
        pytest.param(None, ["a.md"], None, id="run-failed"),
    ],
)
def test_candidate_rank(candidates, expected, score):
    """Ranked over the vector search's candidates, never the prompt's chunks,
    which here always hold the right file."""
    outputs = (
        {} if candidates is None else {"candidates": candidates, **_contexts("a.md")}
    )

    result = ev.candidate_rank(outputs, {"answer": "x", "sources": expected})

    assert result["key"] == "candidate_rank"
    assert result["score"] == (pytest.approx(score) if score is not None else None)


class _FakeJudge:
    def __init__(self, passed: bool = True):
        self.passed = passed
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> ev.Verdict:
        self.prompts.append(prompt)
        return ev.Verdict(reasoning="because", passed=self.passed)


def test_correct_shows_the_judge_question_reference_and_response():
    judge = _FakeJudge(passed=False)
    correct, _ = ev.make_judge_evaluators(judge)

    result = correct(
        {"question": "Why chunk?"},
        {"answer": "Context limits. (rag.md)"},
        {"answer": "Detail is lost and context is limited.", "sources": ["rag.md"]},
    )

    assert result == {"key": "correct", "score": 0, "comment": "because"}
    (prompt,) = judge.prompts
    assert "<question>\nWhy chunk?\n</question>" in prompt
    assert "<reference>\nDetail is lost and context is limited.\n</reference>" in prompt
    assert "<response>\nContext limits. (rag.md)\n</response>" in prompt


def test_grounded_shows_the_judge_the_passages_the_answer_came_from():
    judge = _FakeJudge()
    _, grounded = ev.make_judge_evaluators(judge)

    result = grounded(
        {
            "answer": "Apples. (a.md)",
            "contexts": [{"source": "a.md", "text": "Alpha topic about apples."}],
        }
    )

    assert result == {"key": "grounded", "score": 1, "comment": "because"}
    assert "[Source: a.md]\nAlpha topic about apples." in judge.prompts[0]


def test_a_failed_run_is_not_sent_to_the_judge():
    judge = _FakeJudge()
    correct, grounded = ev.make_judge_evaluators(judge)

    assert correct({"question": "q"}, {}, {"answer": "a"})["score"] is None
    assert grounded({})["score"] is None
    assert judge.prompts == []


# --- the Claude judge ---------------------------------------------------------


class _FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def _judge_over(
    messages: _FakeMessages, monkeypatch: pytest.MonkeyPatch
) -> ev.ClaudeJudge:
    """A judge over a real client with only its network call replaced."""
    client = anthropic.Anthropic(api_key="sk-ant-test")
    monkeypatch.setattr(client.messages, "parse", messages.parse)
    return ev.ClaudeJudge(client)


def _api_error() -> anthropic.APIError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIError("overloaded", request, body=None)


def test_the_judge_is_the_fixed_model_with_a_schema(monkeypatch):
    verdict = ev.Verdict(reasoning="r", passed=True)
    messages = _FakeMessages(
        SimpleNamespace(stop_reason="end_turn", parsed_output=verdict)
    )

    assert _judge_over(messages, monkeypatch)("grade this") == verdict
    (call,) = messages.calls
    assert call["model"] == ev.JUDGE_MODEL
    assert call["output_format"] is ev.Verdict
    assert call["output_config"] == {"effort": "medium"}
    assert call["messages"] == [{"role": "user", "content": "grade this"}]


@pytest.mark.parametrize(
    ("messages", "problem"),
    [
        pytest.param(
            _FakeMessages(SimpleNamespace(stop_reason="refusal", parsed_output=None)),
            "declined",
            id="refusal",
        ),
        pytest.param(
            _FakeMessages(
                SimpleNamespace(stop_reason="max_tokens", parsed_output=None)
            ),
            "no verdict",
            id="cut-off",
        ),
        pytest.param(_FakeMessages(error=_api_error()), "failed", id="api-error"),
    ],
)
def test_anything_but_a_verdict_is_a_runtime_error(monkeypatch, messages, problem):
    """Never a score: a refused or failed grade is recorded as unscored."""
    with pytest.raises(RuntimeError, match=problem):
        _judge_over(messages, monkeypatch)("grade this")


def test_the_judge_check_reports_an_unusable_key_as_runtime_error(monkeypatch):
    def retrieve(_model: str) -> None:
        raise _api_error()

    client = anthropic.Anthropic(api_key="sk-ant-test")
    monkeypatch.setattr(client.models, "retrieve", retrieve)
    judge = ev.ClaudeJudge(client)

    with pytest.raises(RuntimeError, match="Cannot reach the judge"):
        judge.check()


# --- summary, report and baseline -----------------------------------------------


def _scored(**scores: float | None) -> dict[str, float | None]:
    """Scores for every metric, 1 unless given."""
    return dict.fromkeys(ev.METRICS, 1) | scores


def test_summarize_averages_only_what_was_scored():
    rows: list[ev.Row] = [
        ("easy", _scored(retrieval_hit=1, retrieval_rank=0.5), False),
        ("easy", _scored(retrieval_hit=0, retrieval_rank=0.0, correct=0), False),
        # unanswerable
        ("easy", _scored(retrieval_hit=None, retrieval_rank=None), False),
        ("easy", {}, True),  # the run raised
    ]

    summary = ev.summarize(rows)

    assert summary.metrics["retrieval_hit"] == ev.Metric(mean=0.5, n=2)
    assert summary.metrics["retrieval_rank"] == ev.Metric(mean=0.25, n=2)
    assert summary.metrics["correct"] == ev.Metric(mean=2 / 3, n=3)
    assert summary.metrics["grounded"] == ev.Metric(mean=1.0, n=3)
    assert (summary.examples, summary.errors) == (4, 1)


def test_summarize_breaks_the_means_down_by_category_in_a_fixed_order():
    """A hard category's failures are seen in its own line, not diluted by the
    easy questions' passes in the overall mean."""
    rows: list[ev.Row] = [
        ("multi", _scored(retrieval_recall=0.5), False),
        *[("easy", _scored(), False) for _ in range(9)],
        ("multi", _scored(retrieval_recall=0.0), False),
    ]

    summary = ev.summarize(rows)

    assert list(summary.by_category) == ["easy", "multi"]
    assert summary.by_category["multi"].metrics["retrieval_recall"] == ev.Metric(
        mean=0.25, n=2
    )
    assert summary.by_category["easy"].examples == 9
    assert summary.metrics["retrieval_recall"].mean == pytest.approx(9.5 / 11)


def test_rows_read_langsmith_results():
    row = {
        "run": SimpleNamespace(error=None),
        "example": SimpleNamespace(metadata={"id": "q1", "category": "table"}),
        "evaluation_results": {
            "results": [
                EvaluationResult(key="retrieval_hit", score=None),
                EvaluationResult(key="correct", score=1),
            ]
        },
    }

    assert list(ev._rows([row])) == [
        ("table", {"retrieval_hit": None, "correct": 1}, False)
    ]


def test_incomplete_names_each_metric_short_of_every_question():
    questions = [_q("q1"), _q("q2", ())]
    summary = ev.summarize([("easy", _scored(), False)])

    assert ev.incomplete(summary, questions) == [
        "correct: 1 of 2 scored",
        "grounded: 1 of 2 scored",
    ]


def _summary(
    correct: float, rank: float = 1.0, *, categories: tuple[str, ...] = ()
) -> ev.Summary:
    metrics = {metric: ev.Metric(1.0, 2) for metric in ev.METRICS} | {
        "retrieval_rank": ev.Metric(rank, 2),
        "correct": ev.Metric(correct, 2),
    }
    by_category = {
        category: ev.Summary(metrics=metrics, examples=2, errors=0)
        for category in categories
    }
    return ev.Summary(metrics=metrics, examples=2, errors=0, by_category=by_category)


def test_the_report_compares_with_a_baseline_on_the_same_questions_and_judge(settings):
    baseline = ev.baseline_record(_summary(0.5), "ds", "exp", "https://x", settings)

    lines = ev.format_report(_summary(1.0), "ds", baseline)

    assert any("correct" in line and "change +50.0%" in line for line in lines)


def test_the_rank_is_reported_as_a_number_and_its_change_in_points(settings):
    baseline = ev.baseline_record(_summary(1.0), "ds", "exp", "https://x", settings)

    lines = ev.format_report(_summary(1.0, rank=0.75), "ds", baseline)

    (rank_line,) = [line for line in lines if line.startswith("  retrieval_rank")]
    assert "  0.75" in rank_line
    assert "change  -0.25" in rank_line


def test_a_metric_the_baseline_predates_is_reported_without_a_change(settings):
    baseline = ev.baseline_record(_summary(1.0), "ds", "exp", "https://x", settings)
    del baseline["scores"]["retrieval_rank"]

    lines = ev.format_report(_summary(1.0, rank=0.5), "ds", baseline)

    (rank_line,) = [line for line in lines if line.startswith("  retrieval_rank")]
    assert "baseline" not in rank_line


def test_each_category_is_a_line_with_its_change_beneath(settings):
    baseline = ev.baseline_record(
        _summary(0.5, categories=("easy", "table")), "ds", "exp", "https://x", settings
    )

    lines = ev.format_report(
        _summary(1.0, categories=("easy", "table")), "ds", baseline
    )

    header = next(
        i for i, line in enumerate(lines) if line.split()[:2] == ["category", "n"]
    )
    assert lines[header + 1].split()[:2] == ["easy", "2"]
    assert lines[header + 2].split()[0] == "change"
    assert "+50.0%" in lines[header + 2]
    assert lines[header + 3].split()[:2] == ["table", "2"]


def test_a_baseline_without_categories_is_compared_overall_only(settings):
    """The baseline from before categories existed has no breakdown: its
    categories have nothing to compare with, as a metric it predates does not."""
    baseline = ev.baseline_record(_summary(0.5), "ds", "exp", "https://x", settings)
    del baseline["scores_by_category"]

    lines = ev.format_report(_summary(1.0, categories=("easy",)), "ds", baseline)

    assert any(line.split()[:2] == ["easy", "2"] for line in lines)
    assert not any(line.split()[:1] == ["change"] for line in lines)
    assert any("correct" in line and "change +50.0%" in line for line in lines)


def test_a_metric_a_category_does_not_score_is_a_dash_not_zero():
    """decoy questions are unanswerable, so no retrieval metric scores them."""
    unscored = ev.Summary(
        metrics={
            metric: ev.Metric(0.0, 0)
            if metric in ev._RETRIEVAL_METRICS
            else ev.Metric(1.0, 3)
            for metric in ev.METRICS
        },
        examples=3,
        errors=0,
    )
    summary = replace(_summary(1.0), by_category={"decoy": unscored})

    (line,) = [
        line
        for line in ev.format_report(summary, "ds", None)
        if line.split()[:1] == ["decoy"]
    ]
    assert line.split()[2:] == ["-", "-", "-", "-", "100.0%", "100.0%"]


@pytest.mark.parametrize(
    "change",
    [{"dataset": "other-ds"}, {"judge": "another-model"}],
    ids=["other-questions", "other-judge"],
)
def test_the_report_refuses_to_compare_across_questions_or_judges(settings, change):
    baseline = ev.baseline_record(
        _summary(0.5, categories=("easy",)), "ds", "exp", "https://x", settings
    )

    lines = ev.format_report(
        _summary(1.0, categories=("easy",)), "ds", {**baseline, **change}
    )

    assert not any("change" in line for line in lines)
    assert any("no comparison" in line for line in lines)


def test_a_baseline_records_the_judge_settings_and_categories_but_no_paths(settings):
    record = ev.baseline_record(
        _summary(1.0, categories=("easy", "multi")), "ds", "exp", "https://x", settings
    )

    assert record["judge"] == ev.JUDGE_MODEL
    assert record["settings"]["retrieval_k"] == settings.retrieval_k
    assert "data_dir" not in record["settings"]
    assert "persist_dir" not in record["settings"]
    assert set(record["scores"]) == set(ev.METRICS)
    assert list(record["scores_by_category"]) == ["easy", "multi"]
    assert record["scores_by_category"]["multi"]["correct"] == {"mean": 1.0, "n": 2}
    json.dumps(record)


# --- the run, up to the first network call ------------------------------------


@pytest.mark.parametrize("key", ["LANGSMITH_API_KEY", "ANTHROPIC_API_KEY"])
def test_a_missing_key_stops_the_run_before_anything_loads(monkeypatch, settings, key):
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv(key)

    with pytest.raises(RuntimeError, match=f"^{key} is not set"):
        ev.run(settings)


def test_the_eval_index_is_rebuilt_for_a_new_width_and_kept_otherwise(
    settings, fake_embeddings
):
    """The eval's collection is the eval's alone, so a width change rebuilds it
    where ingest would refuse a user's collection: the old index is dropped and
    waited out, the old chunks deleted, and the next ingest builds afresh."""
    evaluated = ev.eval_settings(settings)
    ingest_mod.ingest(evaluated, embeddings=fake_embeddings)

    assert ev.fit_eval_index(evaluated) is False, "same width: nothing to rebuild"

    narrower = dataclasses.replace(evaluated, embedding_dimensions=16)
    assert ev.fit_eval_index(narrower) is True
    assert ingest_mod._collection(narrower).count_documents({}) == 0

    ingest_mod.ingest(narrower, embeddings=DeterministicFakeEmbedding(size=16))
    expected = {s for q in ev.load_questions() for s in q.sources}
    assert expected <= ingest_mod.indexed_sources(narrower)
