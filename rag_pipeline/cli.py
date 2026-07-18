"""Command-line interface: `rag ingest` and `rag query "..."`.

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
    print(f"Indexed {n_chunks} chunks into {settings.persist_dir}")
    print("Ready. Ask a question with:  rag query \"...\"")
    return 0


def cmd_query(settings: Settings, question: str) -> int:
    from rag_pipeline.pipeline import RAGPipeline, unique_sources

    pipeline = RAGPipeline(settings)
    result = pipeline.answer(question)

    print(f"\nQ: {question}\n")
    print(result.text)
    print("\nSources:")
    for src in unique_sources(result.sources):
        print(f"  - {src}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="A local-embeddings RAG pipeline built with LangChain and Claude.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Load, chunk, embed, and index ./data")

    query_parser = subparsers.add_parser(
        "query", help="Ask a question against the indexed documents"
    )
    query_parser.add_argument("question", help="The question to answer")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    try:
        if args.command == "ingest":
            return cmd_ingest(settings)
        # `required=True` guarantees a subcommand; "query" is the only other one.
        return cmd_query(settings, args.question)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
