"""`rag eval`: score the pipeline on a fixed question set, as a LangSmith experiment.

The migration to hosted models swaps one service at a time, and each swap has
to be judged against the stack it replaced. Single answers cannot show that --
the hosted chat model does not decode greedily, so one answer differs from run
to run -- so every change is scored on the same questions, by the same judge,
and compared with a saved baseline.

Four scores per question, all but ``retrieval_rank`` 1 (pass) or 0 (fail):

- ``retrieval_hit``: an expected source file is among the chunks the answer was
  generated from. Not scored for a question the documents cannot answer.
- ``retrieval_rank``: how high the first chunk from an expected file ranks
  among them, as a reciprocal rank (1 for first, 1/2 for second, 0 if none).
  Scored like ``retrieval_hit``, but not 0 or 1: it sees a right file slipping
  down the prompt before it falls out of it.
- ``correct``: the answer matches the reference answer -- or, for a question
  the documents cannot answer, says so instead of answering from general
  knowledge. Judged by Claude.
- ``grounded``: every claim in the answer is supported by those chunks. Judged
  by Claude.

The questions are about ``evals/corpus/``: the engineering handbook of a
fictional company, Tallowmere, which ``rag eval`` indexes into a collection of
its own. Fictional so that a question can only be answered by retrieving the
right passage -- a model answering questions about real topics can pass from
what it already knows, whatever retrieval did -- and large and repetitive
enough (four services documented alike, a deprecated guide contradicting the
current one) that retrieval has real choices to get wrong. The three sample
documents in ``data/`` made 9 chunks, and every question scored 100%.

The question set (``evals/questions.json``) is uploaded to LangSmith under a
name derived from its content, so an edited set is a new dataset: an experiment
is only ever compared with others scored on the same questions.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import anthropic
from langsmith import Client
from langsmith import evaluate as langsmith_evaluate
from langsmith.utils import LangSmithError
from pydantic import BaseModel

from rag_pipeline.config import Settings, require_env_key

if TYPE_CHECKING:
    from rag_pipeline.pipeline import RAGPipeline

_ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = _ROOT / "evals" / "questions.json"
BASELINE_PATH = _ROOT / "evals" / "baseline.json"
EVAL_CORPUS = _ROOT / "evals" / "corpus"
# Its own collection, beside the user's index in the same store: the eval never
# reads, writes or retrieves the user's documents, and `rag ingest` never sees
# the eval corpus.
EVAL_COLLECTION = "rag_eval"

# The judge is part of the measurement, not a setting: scores from two judges
# are not comparable, so changing it invalidates the baseline as surely as
# editing the questions would. Kept fixed here for that reason, and recorded in
# every experiment and in the baseline. Opus because it grades answers that a
# Sonnet will write later in the migration, and a model grading its own family's
# answers is the bias an evaluation exists to avoid.
JUDGE_MODEL = "claude-opus-5-5"
# Opus 5.5 always thinks; effort is the only control (its default is medium,
# stated here so a change of default upstream does not move the scores).
_JUDGE_EFFORT = "medium"
# Room for the thinking and the verdict together; well under the SDK's
# non-streaming timeout.
_JUDGE_MAX_TOKENS = 16000

METRICS = ("retrieval_hit", "retrieval_rank", "correct", "grounded")
# Shown as a mean reciprocal rank rather than a pass rate.
_RANK_METRICS = frozenset({"retrieval_rank"})
_EXPERIMENT_PREFIX = "rag-pipeline"


@dataclass(frozen=True)
class Question:
    """One evaluation question, with what a correct answer must say."""

    id: str
    question: str
    answer: str
    # The files a correct answer draws on, relative to the data directory as a
    # chunk's `source` is. Empty for a question the documents cannot answer.
    sources: tuple[str, ...]

    @property
    def answerable(self) -> bool:
        return bool(self.sources)


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """Read and check the question set.

    Checked here, before anything is uploaded or any model loads: a malformed
    entry would otherwise surface as a LangSmith error, or as an example every
    evaluator skips.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the question set {path}: {exc}") from exc
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} must be a non-empty JSON list of questions.")

    fields = {"id", "question", "answer", "sources"}
    questions: list[Question] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != fields:
            raise ValueError(
                f"{path}, entry {i}: expected exactly the keys {sorted(fields)}."
            )
        text = [entry["id"], entry["question"], entry["answer"]]
        sources = entry["sources"]
        if not all(isinstance(value, str) and value.strip() for value in text):
            raise ValueError(
                f"{path}, entry {i}: id, question and answer must be text."
            )
        if not isinstance(sources, list) or not all(
            isinstance(source, str) and source for source in sources
        ):
            raise ValueError(
                f"{path}, entry {i}: sources must be a list of file names."
            )
        questions.append(Question(*text, sources=tuple(sources)))

    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"{path}: duplicate question ids {duplicated}.")
    return questions


def dataset_name(questions: Iterable[Question]) -> str:
    """The LangSmith dataset name for exactly this question set.

    A digest of the content rather than a fixed name: LangSmith compares
    experiments within a dataset, and editing a fixed-name dataset in place
    would set new scores beside old ones taken on different questions.
    """
    canonical = json.dumps([asdict(q) for q in questions], sort_keys=True)
    return f"rag-pipeline-qa-{hashlib.sha256(canonical.encode()).hexdigest()[:12]}"


class DatasetStore(Protocol):
    """The part of `langsmith.Client` that `sync_dataset` uses.

    Named so the dataset logic can be tested against a fake: the real client
    satisfies it structurally.
    """

    def has_dataset(self, *, dataset_name: str) -> bool: ...
    def read_dataset(self, *, dataset_name: str) -> Any: ...
    def create_dataset(self, dataset_name: str, *, description: str) -> Any: ...
    def create_examples(
        self, *, dataset_id: Any, examples: list[dict[str, Any]]
    ) -> Any: ...
    def delete_dataset(self, *, dataset_id: Any) -> None: ...


def sync_dataset(client: DatasetStore, questions: list[Question]) -> str:
    """Make sure LangSmith holds this question set, and return its name.

    Uploaded once per content digest. A dataset whose upload failed half-way is
    deleted rather than left behind, because the next run would find it by name
    and score against the questions that made it.
    """
    name = dataset_name(questions)
    try:
        if client.has_dataset(dataset_name=name):
            found = client.read_dataset(dataset_name=name)
            if found.example_count != len(questions):
                raise RuntimeError(
                    f"LangSmith dataset {name!r} holds {found.example_count} "
                    f"examples, not {len(questions)}. Delete it in LangSmith "
                    "and run `rag eval` again to re-upload it."
                )
            return name
        created = client.create_dataset(
            name,
            description="rag-pipeline evaluation questions (evals/questions.json).",
        )
        try:
            client.create_examples(
                dataset_id=created.id,
                examples=[
                    {
                        "inputs": {"question": q.question},
                        "outputs": {"answer": q.answer, "sources": list(q.sources)},
                        "metadata": {"id": q.id},
                    }
                    for q in questions
                ],
            )
        except BaseException:
            client.delete_dataset(dataset_id=created.id)
            raise
    except LangSmithError as exc:
        raise RuntimeError(f"LangSmith refused the question set: {exc}") from exc
    return name


# --- the target: one question through the real pipeline ---------------------


def make_target(pipeline: RAGPipeline) -> Callable[[dict], dict]:
    """The function LangSmith runs per question: the pipeline, end to end.

    Returns the chunks themselves as well as their sources, because groundedness
    is judged against exactly what the model read.
    """
    from rag_pipeline.pipeline import source_excerpts, unique_sources

    def target(inputs: dict) -> dict:
        answer = pipeline.answer(inputs["question"])
        return {
            "answer": answer.text,
            "sources": unique_sources(answer.sources),
            "contexts": [dict(excerpt) for excerpt in source_excerpts(answer.sources)],
        }

    return target


# --- evaluators ---------------------------------------------------------------


def retrieval_hit(outputs: dict, reference_outputs: dict) -> dict:
    """1 if any expected source file reached the prompt, else 0.

    `None` -- not scored -- for a question no source answers, and for a run that
    produced no output (its error is counted separately).
    """
    expected = reference_outputs.get("sources") or []
    if not expected:
        return {"key": "retrieval_hit", "score": None, "comment": "No source expected."}
    if "sources" not in outputs:
        return {"key": "retrieval_hit", "score": None, "comment": "No output."}
    found = [source for source in expected if source in outputs["sources"]]
    return {
        "key": "retrieval_hit",
        "score": int(bool(found)),
        "comment": f"Expected {expected}; retrieved {outputs['sources']}.",
    }


def retrieval_rank(outputs: dict, reference_outputs: dict) -> dict:
    """1/rank of the first chunk from an expected file in the prompt, else 0.

    Ranked over the chunks in the order the model read them -- the reranker's
    order -- not over distinct files, so two chunks of one wrong file ahead of
    the right one count as two places. Not scored when `retrieval_hit` is not.
    """
    expected = reference_outputs.get("sources") or []
    if not expected:
        return {
            "key": "retrieval_rank",
            "score": None,
            "comment": "No source expected.",
        }
    if "contexts" not in outputs:
        return {"key": "retrieval_rank", "score": None, "comment": "No output."}
    order = [context["source"] for context in outputs["contexts"]]
    rank = next((i for i, source in enumerate(order, 1) if source in expected), None)
    return {
        "key": "retrieval_rank",
        "score": 1 / rank if rank else 0.0,
        "comment": f"First expected chunk at rank {rank} of {len(order)}: {order}."
        if rank
        else f"No chunk from {expected} among {order}.",
    }


class Verdict(BaseModel):
    """The judge's grade. Reasoning comes first so the verdict follows from it."""

    reasoning: str
    passed: bool


Judge = Callable[[str], Verdict]

_JUDGE_SYSTEM = (
    "You grade the output of a question-answering system that must answer only "
    "from a fixed set of documents. Apply the criteria you are given strictly "
    "and literally. Give your reasoning, then the verdict."
)

_CORRECT_PROMPT = """\
Grade whether the response answers the question correctly, using the reference \
answer as the ground truth.

Pass if the response gives what the question asks for, as the reference \
states it, and contradicts nothing in the reference. Judge against the \
question: anything in the reference that the question does not ask for is \
context, and the response need not repeat it. Extra detail in the response is \
fine if it does not contradict the reference; wording does not matter; source \
file names in parentheses are citations, not part of the answer. If the \
reference says the documents do not contain the answer, pass only if the \
response says so and does not supply an answer from general knowledge.

<question>
{question}
</question>

<reference>
{reference}
</reference>

<response>
{response}
</response>"""

_GROUNDED_PROMPT = """\
Grade whether every factual claim in the response is supported by the \
retrieved passages the system was given.

Pass if each claim appears in, or directly follows from, the passages. Fail if \
any claim is absent from the passages or contradicts them, even if it is true \
in general. A statement that the documents do not contain the answer is \
grounded. Source file names in parentheses are citations, not claims.

<passages>
{passages}
</passages>

<response>
{response}
</response>"""


def make_judge_evaluators(judge: Judge) -> list[Callable[..., dict]]:
    """The two judged evaluators, over any `Judge` -- Claude in production, a
    fake in the tests."""

    def correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        if "answer" not in outputs:
            return {"key": "correct", "score": None, "comment": "No output."}
        verdict = judge(
            _CORRECT_PROMPT.format(
                question=inputs["question"],
                reference=reference_outputs["answer"],
                response=outputs["answer"],
            )
        )
        return {
            "key": "correct",
            "score": int(verdict.passed),
            "comment": verdict.reasoning,
        }

    def grounded(outputs: dict) -> dict:
        if "answer" not in outputs:
            return {"key": "grounded", "score": None, "comment": "No output."}
        passages = "\n\n".join(
            f"[Source: {context['source']}]\n{context['text']}"
            for context in outputs.get("contexts", [])
        )
        verdict = judge(
            _GROUNDED_PROMPT.format(passages=passages, response=outputs["answer"])
        )
        return {
            "key": "grounded",
            "score": int(verdict.passed),
            "comment": verdict.reasoning,
        }

    return [correct, grounded]


class ClaudeJudge:
    """Grades with `JUDGE_MODEL`, raising RuntimeError for anything but a verdict.

    No fallback model on a refusal, though the API offers one: a fallback would
    grade that example with another judge, and scores are only comparable under
    one. A refused or unparseable grade is an error LangSmith records against
    the example, which the summary then counts as unscored.
    """

    def __init__(self, client: anthropic.Anthropic) -> None:
        self._client = client

    def check(self) -> None:
        """Fail on a bad key or an unavailable model before the pipeline loads."""
        try:
            self._client.models.retrieve(JUDGE_MODEL)
        except anthropic.APIError as exc:
            raise RuntimeError(f"Cannot reach the judge, {JUDGE_MODEL}: {exc}") from exc

    def __call__(self, prompt: str) -> Verdict:
        try:
            response = self._client.messages.parse(
                model=JUDGE_MODEL,
                max_tokens=_JUDGE_MAX_TOKENS,
                system=_JUDGE_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": _JUDGE_EFFORT},
                output_format=Verdict,
            )
        except anthropic.APIError as exc:
            raise RuntimeError(f"The judge, {JUDGE_MODEL}, failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise RuntimeError(f"The judge, {JUDGE_MODEL}, declined to grade this.")
        if response.parsed_output is None:
            raise RuntimeError(
                f"The judge, {JUDGE_MODEL}, returned no verdict "
                f"(stop reason {response.stop_reason!r})."
            )
        return response.parsed_output


# --- summary, baseline and report ---------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A metric's mean over the examples it scored, and how many those were."""

    mean: float
    n: int


@dataclass(frozen=True)
class Summary:
    metrics: dict[str, Metric]
    examples: int
    # Examples whose pipeline run raised: none of their scores exist.
    errors: int


def summarize(rows: Iterable[tuple[Mapping[str, float | None], bool]]) -> Summary:
    """Mean of each metric over the examples that have a score for it.

    Each row is one example's scores by key, and whether its run raised.
    """
    scores: dict[str, list[float]] = {metric: [] for metric in METRICS}
    examples = errors = 0
    for by_key, failed in rows:
        examples += 1
        errors += failed
        for metric in METRICS:
            score = by_key.get(metric)
            if score is not None:
                scores[metric].append(float(score))
    return Summary(
        metrics={
            metric: Metric(
                mean=sum(values) / len(values) if values else 0.0, n=len(values)
            )
            for metric, values in scores.items()
        },
        examples=examples,
        errors=errors,
    )


def _rows(results: Iterable[Any]) -> Iterable[tuple[dict[str, float | None], bool]]:
    """Each LangSmith result row as (scores by key, whether the run raised)."""
    for row in results:
        by_key = {
            result.key: result.score for result in row["evaluation_results"]["results"]
        }
        yield by_key, row["run"].error is not None


def expected_counts(questions: list[Question]) -> dict[str, int]:
    """How many examples each metric scores when nothing fails."""
    answerable = sum(q.answerable for q in questions)
    return {
        "retrieval_hit": answerable,
        "retrieval_rank": answerable,
        "correct": len(questions),
        "grounded": len(questions),
    }


def incomplete(summary: Summary, questions: list[Question]) -> list[str]:
    """The metrics some example went unscored on, worded for a reader."""
    return [
        f"{metric}: {summary.metrics[metric].n} of {count} scored"
        for metric, count in expected_counts(questions).items()
        if summary.metrics[metric].n != count
    ]


def _comparable_settings(settings: Settings) -> dict[str, Any]:
    """The settings a score depends on, without this machine's paths."""
    return {
        key: value
        for key, value in asdict(settings).items()
        if not isinstance(value, Path)
    }


def baseline_record(
    summary: Summary, dataset: str, experiment: str, url: str, settings: Settings
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "experiment": experiment,
        "url": url,
        "date": datetime.now(UTC).date().isoformat(),
        "judge": JUDGE_MODEL,
        "settings": _comparable_settings(settings),
        "scores": {
            metric: {"mean": round(m.mean, 4), "n": m.n}
            for metric, m in summary.metrics.items()
        },
    }


def format_report(
    summary: Summary, dataset: str, baseline: Mapping[str, Any] | None
) -> list[str]:
    """One line per metric, with its change from the baseline when comparable."""
    comparable = (
        baseline is not None
        and baseline.get("dataset") == dataset
        and baseline.get("judge") == JUDGE_MODEL
    )
    lines = []
    for metric, m in summary.metrics.items():
        line = f"  {metric:<14} {_shown(metric, m.mean)}  ({m.n} scored)"
        # A metric the baseline predates has nothing to be compared with.
        before = (baseline or {}).get("scores", {}).get(metric) if comparable else None
        if before is not None:
            change = _shown(metric, m.mean - before["mean"], signed=True)
            line += f"   baseline {_shown(metric, before['mean'])}, change {change}"
        lines.append(line)
    if baseline is not None and not comparable:
        lines.append(
            "  The saved baseline used other questions or another judge: no comparison."
        )
    if summary.errors:
        lines.append(
            f"  {summary.errors} of {summary.examples} questions failed to run."
        )
    return lines


def _shown(metric: str, value: float, *, signed: bool = False) -> str:
    """A pass rate as a percentage; a mean reciprocal rank as a number."""
    sign = "+" if signed else ""
    if metric in _RANK_METRICS:
        return f"{value:{sign}6.2f}"
    return f"{value:{sign}6.1%}"


def read_baseline(path: Path = BASELINE_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read the baseline {path}: {exc}") from exc


# --- the run --------------------------------------------------------------------


def eval_settings(settings: Settings) -> Settings:
    """`settings` pointed at the eval corpus and its own collection.

    Everything else -- the models, chunking, k -- is kept, because that is what
    is being evaluated.
    """
    return replace(settings, data_dir=EVAL_CORPUS, collection_name=EVAL_COLLECTION)


def fit_eval_index(settings: Settings, timeout_s: float = 180.0) -> bool:
    """Clear the eval's collection if its index holds vectors of another width.

    Ingest refuses a width change -- for a user's collection, rightly, since
    the fix (a new COLLECTION_NAME) is theirs to choose. The eval's collection
    is the eval's alone, so a run after EMBEDDING_DIMENSIONS changed rebuilds it
    instead: the index is dropped and waited out (Atlas drops it
    asynchronously, and a new one of the same name cannot be created until it
    is gone), then the chunks are deleted. Returns whether it did.
    """
    from rag_pipeline import ingest as ingest_mod

    collection = ingest_mod._collection(settings)
    with ingest_mod.provider_errors_as_runtime():
        found = list(collection.list_search_indexes(settings.vector_index_name))
        if not found:
            return False
        fields = (found[0].get("latestDefinition") or {}).get("fields", [])
        widths = [f.get("numDimensions") for f in fields if f.get("type") == "vector"]
        if widths == [settings.embedding_dimensions]:
            return False
        collection.drop_search_index(settings.vector_index_name)
        deadline = time.monotonic() + timeout_s
        while list(collection.list_search_indexes(settings.vector_index_name)):
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"The eval's old vector index was not dropped within {timeout_s:.0f}s."
                )
            time.sleep(0.5)
        collection.delete_many({})
    return True


def run(settings: Settings, *, save_baseline: bool = False) -> list[str]:
    """Score the pipeline built from `settings`, and return the report's lines.

    Ordered cheapest failure first: the question set, both keys, the judge and
    the dataset are all checked before any model loads.
    """
    from rag_pipeline.ingest import indexed_sources, ingest
    from rag_pipeline.pipeline import RAGPipeline

    questions = load_questions()
    langsmith_key = require_env_key(
        "LANGSMITH_API_KEY", "`rag eval` keeps its questions and results in LangSmith"
    )
    anthropic_key = require_env_key(
        "ANTHROPIC_API_KEY", "`rag eval` grades answers with Claude"
    )

    judge = ClaudeJudge(anthropic.Anthropic(api_key=anthropic_key))
    judge.check()
    client = Client(api_key=langsmith_key)
    dataset = sync_dataset(client, questions)

    # Incremental like any ingest: after the first run, only a changed corpus
    # file or a changed model or chunk setting re-embeds anything.
    settings = eval_settings(settings)
    rebuilt = fit_eval_index(settings)
    ingest(settings)
    expected = {source for q in questions for source in q.sources}
    if missing := sorted(expected - indexed_sources(settings)):
        raise RuntimeError(
            f"The eval corpus did not index {missing}, which questions expect."
        )

    pipeline = RAGPipeline(settings)
    try:
        results = langsmith_evaluate(
            make_target(pipeline),
            data=dataset,
            evaluators=[retrieval_hit, retrieval_rank, *make_judge_evaluators(judge)],
            experiment_prefix=_EXPERIMENT_PREFIX,
            metadata={"judge": JUDGE_MODEL, **_comparable_settings(settings)},
            # One question at a time: the local models serialize on their locks
            # anyway, and the answers' order then matches the dataset's.
            max_concurrency=0,
            client=client,
        )
        summary = summarize(_rows(results))
    except LangSmithError as exc:
        raise RuntimeError(f"LangSmith failed during the run: {exc}") from exc

    lines = [
        *(
            ["The eval's index was rebuilt for the new EMBEDDING_DIMENSIONS."]
            if rebuilt
            else []
        ),
        f"Experiment {results.experiment_name} on {dataset}, judged by {JUDGE_MODEL}:",
        *format_report(summary, dataset, read_baseline()),
        f"Details: {results.url}",
    ]
    if save_baseline:
        if gaps := incomplete(summary, questions):
            raise RuntimeError(
                "Not saved as the baseline: a baseline needs every question "
                f"scored ({'; '.join(gaps)}). See {results.url}."
            )
        record = baseline_record(
            summary, dataset, results.experiment_name, results.url or "", settings
        )
        BASELINE_PATH.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        lines.append(f"Saved as the baseline: {BASELINE_PATH.relative_to(_ROOT)}")
    return lines
