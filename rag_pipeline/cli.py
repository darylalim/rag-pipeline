"""Command-line interface: `rag ingest`, `rag query "..."` and `rag eval`.

A thin wrapper over the core modules so the pipeline is scriptable from a
terminal. The same ``Settings`` and ``RAGPipeline`` back the Streamlit app.
"""

from __future__ import annotations

import argparse
import sys

from rag_pipeline.config import Settings


def cmd_ingest(settings: Settings) -> int:
    from rag_pipeline.ingest import ingest

    print(f"Ingesting documents from {settings.data_dir} ...")
    n_chunks = ingest(settings)
    print(
        f"Indexed {n_chunks} chunks into MongoDB Atlas, "
        f"{settings.mongodb_db}.{settings.collection_name}"
    )
    print('Ready. Ask a question with:  rag query "..."')
    return 0


def cmd_query(settings: Settings, question: str) -> int:
    from rag_pipeline.pipeline import RAGPipeline, unique_sources

    # Traced if LANGSMITH_TRACING is true. Nothing to flush here: runs not yet
    # sent when the answer ends are sent before the process exits.
    pipeline = RAGPipeline(settings)
    docs, chunks = pipeline.stream_answer(question)

    print(f"\nQ: {question}\n")
    # Printed as it arrives rather than after the full generation, so a long
    # answer starts appearing immediately. A model failure arrives as a
    # RuntimeError, which main() reports — note that a mid-stream failure leaves
    # the partial answer on screen above the error, which is the cost of
    # streaming at all.
    try:
        for chunk in chunks:
            print(chunk, end="", flush=True)
    finally:
        # Terminate the streamed line even when generation failed partway,
        # otherwise main()'s "Error: ..." collides with the partial answer on
        # the same line.
        print()

    print("\nSources:")
    for src in unique_sources(docs):
        print(f"  - {src}")
    return 0


def cmd_eval(settings: Settings, save_baseline: bool) -> int:
    from rag_pipeline.evaluation import run

    print(
        "Scoring the pipeline on evals/questions.json, over evals/corpus/ in "
        "its own collection. This loads the models and answers every "
        "question, so it takes several minutes."
    )
    for line in run(settings, save_baseline=save_baseline):
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="A RAG pipeline built with LangChain, MongoDB Atlas, Voyage AI and Claude.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load, chunk, embed, and index ./data")

    query_parser = subparsers.add_parser(
        "query", help="Ask a question against the indexed documents"
    )
    query_parser.add_argument("question", help="The question to answer")

    eval_parser = subparsers.add_parser(
        "eval",
        help="Score the pipeline on evals/questions.json, as a LangSmith experiment",
    )
    eval_parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save this run's scores as the baseline later runs are compared with",
    )

    args = parser.parse_args(argv)

    try:
        # Inside the try: a malformed numeric env var (e.g. CHUNK_SIZE=abc)
        # raises ValueError here, which the handler below turns into a friendly
        # message rather than a traceback.
        settings = Settings.from_env()
        if args.command == "ingest":
            return cmd_ingest(settings)
        if args.command == "eval":
            return cmd_eval(settings, args.save_baseline)
        # `required=True` guarantees a subcommand; "query" is the only other one.
        return cmd_query(settings, args.question)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Ctrl-C is how a slow local answer gets abandoned -- the model reads
        # the prompt for several seconds before it writes -- so it is an
        # ordinary way out, not a crash: one line and the shell's usual status
        # for SIGINT, not a traceback through the model's generation loop.
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
