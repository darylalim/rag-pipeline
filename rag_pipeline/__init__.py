"""A local RAG pipeline built with LangChain, Chroma, and MLX models.

The pipeline has two phases, with a hard boundary between them:

    ingest:  load -> split -> embed -> store                (rag_pipeline.ingest)
    query:   embed question -> search -> rerank -> generate (rag_pipeline.pipeline)

Every model runs locally with MLX (Apple Silicon), loaded from the Hugging Face
cache, and the index is a Chroma collection persisted on disk, so nothing
reaches the network at runtime.

Configuration lives in rag_pipeline.config and is driven by environment
variables so the same code backs both the CLI and the Streamlit app.
"""

import os

# transformers arrives only as mlx-lm's tokenizer backend, and torch is
# deliberately absent. LangChain imports transformers opportunistically
# (langchain_text_splitters, and langchain_core's language models before 1.6)
# before mlx-lm gets the chance to silence it, as mlx_lm/__init__.py does for
# itself -- so without this every command and the app on a Mac would open with
# "PyTorch was not found. Models won't be available", which is false here:
# every model loads through MLX. Set in the package's __init__, which every
# entry point imports before anything touches langchain; setdefault, so an
# explicit value wins.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

__version__ = "0.1.0"
