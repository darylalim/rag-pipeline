"""Indexing phase: load -> split -> embed -> store.

Run once (via ``rag ingest``) whenever the documents in ``data/`` change. The
expensive embedding step happens here; querying later just reloads the persisted
Chroma index from disk.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline.config import Settings

# File extensions we know how to read into text.
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Construct the embedding model.

    Defined here (not in the pipeline) because it is the one component that
    *must* be identical for indexing and querying — vectors from different
    models are not comparable. Both stages import this single factory.
    """
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def _read_pdf(path: Path) -> str:
    """Extract text from every page of a PDF and join it."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(data_dir: Path) -> list[Document]:
    """Walk ``data_dir`` and read supported files into LangChain Documents.

    Each document records its file path (relative to ``data_dir``) under the
    ``source`` metadata key so answers can cite where evidence came from.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            text = _read_pdf(path)
        else:
            text = path.read_text(encoding="utf-8")

        if not text.strip():
            continue  # skip empty / unreadable files

        source = path.relative_to(data_dir).as_posix()
        documents.append(Document(page_content=text, metadata={"source": source}))

    return documents


def split_documents(documents: list[Document], settings: Settings) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Recursive character splitting breaks on the most natural boundary that fits
    (paragraph, then line, then space), so chunks stay coherent. The overlap
    carries a little context across boundaries so a sentence split between two
    chunks is not lost to either.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(settings: Settings, embeddings: Embeddings | None = None) -> int:
    """Run the full indexing phase and persist the vector store.

    The persist directory is wiped first so re-ingesting is idempotent — you get
    a fresh index rather than duplicate chunks appended to the old one. Returns
    the number of chunks indexed. ``embeddings`` is injectable so tests can
    substitute a lightweight fake; production callers leave it as None.
    """
    documents = load_documents(settings.data_dir)
    if not documents:
        raise ValueError(
            f"No readable documents found in {settings.data_dir} "
            f"(looked for {', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )

    chunks = split_documents(documents, settings)

    # Rebuild from scratch: remove any prior index so we don't append duplicates.
    if settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings or build_embeddings(settings),
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_dir),
    )
    return len(chunks)
