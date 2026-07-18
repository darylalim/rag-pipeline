# Embeddings and Vector Stores

A vector store is the database at the heart of a RAG system. It holds the
embeddings of your document chunks and answers similarity queries: given the
embedding of a question, it returns the stored chunks whose vectors are closest.

## Embeddings

An embedding model converts a piece of text into a fixed-length list of numbers —
a vector — that captures the text's meaning. Passages about similar topics land
near each other in this vector space, which is what makes similarity search work.
The critical rule is consistency: the same embedding model must be used to index
the documents and to embed the questions, because vectors from different models
are not comparable. Each model also produces vectors of a fixed dimension, and
that dimension cannot be mixed within a single store.

Embeddings can be produced by a hosted API or by a local model. A local
sentence-transformers model such as all-MiniLM-L6-v2 runs entirely on-device,
requires no API key, and produces 384-dimensional vectors, which makes it a
convenient default for getting a pipeline running end-to-end.

## Choosing a vector store

Different vector stores suit different stages of a project:

- In-memory stores are simplest and are ideal for quick tests, but the index is
  lost when the process exits.
- FAISS is a high-performance local option that can be saved to and loaded from
  disk.
- Chroma is a developer-friendly local store that persists to a directory, so an
  index built once can be reloaded on later runs without re-embedding.
- Pinecone is a managed cloud store aimed at production workloads.

## Persistence matters

A common mistake is to build an index in memory and lose it on every restart.
Persisting the store — for example, by giving Chroma a persist directory — means
the expensive embedding step runs only during ingestion. Afterwards, querying
simply loads the existing index from disk, which is far faster and cheaper than
re-embedding the whole corpus on every question.
