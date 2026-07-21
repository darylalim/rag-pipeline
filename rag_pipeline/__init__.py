"""A RAG pipeline built with LangChain, Voyage AI embeddings, and Claude.

The pipeline has two phases, with a hard boundary between them:

    ingest:  load -> split -> embed -> store                  (rag_pipeline.ingest)
    query:   embed question -> search -> rerank -> ask Claude (rag_pipeline.pipeline)

Configuration lives in rag_pipeline.config and is driven by environment
variables so the same code backs both the CLI and the Streamlit app.
"""

__version__ = "0.1.0"
