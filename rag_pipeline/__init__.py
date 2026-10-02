"""A RAG pipeline built with LangChain, MongoDB Atlas, Voyage AI and Claude.

The pipeline has two phases, with a hard boundary between them:

    ingest:  load -> split -> embed -> store                (rag_pipeline.ingest)
    query:   embed question -> search -> rerank -> generate (rag_pipeline.pipeline)

Voyage AI embeds and reranks, Claude answers, and the index lives in MongoDB
Atlas -- chunks, their vectors, and an Atlas Vector Search index -- each
reached through its own credential (MONGODB_URI, VOYAGE_API_KEY,
ANTHROPIC_API_KEY).

Configuration lives in rag_pipeline.config and is driven by environment
variables so the same code backs both the CLI and the Streamlit app.
"""
