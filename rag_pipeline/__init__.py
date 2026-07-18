"""A local-embeddings RAG pipeline built with LangChain and Claude.

The pipeline has three stages, mirroring the classic RAG data flow:

    ingest:  load -> split -> embed -> store   (rag_pipeline.ingest)
    query:   embed question -> search -> generate answer  (rag_pipeline.pipeline)

Configuration lives in rag_pipeline.config and is driven by environment
variables so the same code backs both the CLI and the Streamlit app.
"""

__version__ = "0.1.0"
