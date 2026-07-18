# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation, or RAG, is a technique for improving the
answers a large language model gives by supplying it with relevant external
information at question time. Instead of relying only on what the model learned
during training, a RAG system searches a knowledge base for passages related to
the user's question and places those passages into the prompt as context. The
model then generates its answer grounded in that retrieved evidence.

## Why RAG helps

Language models have two well-known limitations that RAG addresses. First, their
knowledge is frozen at training time, so they cannot answer questions about
recent events or private documents they never saw. Second, when a model does not
know an answer it may "hallucinate" — produce a fluent but incorrect response.
By grounding generation in retrieved source text, RAG reduces hallucination and
lets a model answer questions about data that did not exist when it was trained.

## The pipeline

A RAG pipeline has two phases. The indexing phase runs once, ahead of time:
documents are loaded, split into smaller chunks, converted into numeric vectors
by an embedding model, and stored in a vector database. The query phase runs for
every question: the question is embedded with the same model, the vector store
returns the most similar chunks, and those chunks plus the question are sent to
the language model to generate a final answer.

## Chunking

Documents are split into chunks because embedding an entire long document into a
single vector loses detail, and because language models have a limited context
window. A common strategy is recursive character splitting, which tries to break
text on natural boundaries — paragraphs first, then sentences, then words. A
chunk size of 500 to 1500 characters usually works well. Adjacent chunks are
given a small overlap, typically 10 to 20 percent of the chunk size, so that a
sentence spanning a chunk boundary is not lost to either side.

## Retrieval quality

The number of chunks retrieved per question, often called k, trades off recall
against noise: too few chunks and the answer may miss relevant evidence; too many
and the prompt fills with irrelevant text. Similarity search returns the closest
chunks by vector distance, while Maximal Marginal Relevance (MMR) additionally
rewards diversity so the retrieved set is not redundant.
