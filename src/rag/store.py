from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from src.config import get_settings
from src.rag.chunking import chunk_text
from src.rag.loaders import Document, iter_corpus


def embedding_fn() -> DefaultEmbeddingFunction:
    return DefaultEmbeddingFunction()


def get_collection(create: bool = True):
    s = get_settings()
    client = chromadb.PersistentClient(path=s.chroma_path)
    kwargs = {
        "name": s.collection_name,
        "embedding_function": embedding_fn(),
        "metadata": {"hnsw:space": "cosine"},
    }
    if create:
        return client.get_or_create_collection(**kwargs)
    return client.get_collection(name=s.collection_name, embedding_function=embedding_fn())


def chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}|{index}|{text}".encode("utf-8")).hexdigest()[:24]
    return f"{digest}-{index}"


def ingest_corpus(corpus_dir: Path | None = None) -> dict[str, Any]:
    s = get_settings()
    corpus_dir = Path(corpus_dir or s.corpus_dir)
    docs = iter_corpus(corpus_dir)
    col = get_collection(create=True)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_text(doc.text, s.chunk_size, s.chunk_overlap)
        for ch in chunks:
            cid = chunk_id(doc.source, ch.index, ch.text)
            ids.append(cid)
            documents.append(ch.text)
            metadatas.append(
                {
                    "source": doc.source,
                    "doc_type": doc.doc_type,
                    "chunk_index": ch.index,
                    "start": ch.start,
                    "end": ch.end,
                }
            )

    desired = set(ids)
    existing = set(col.get()["ids"])
    stale = list(existing - desired)
    if stale:
        for i in range(0, len(stale), 100):
            col.delete(ids=stale[i : i + 100])

    if ids:
        for i in range(0, len(ids), 64):
            col.upsert(
                ids=ids[i : i + 64],
                documents=documents[i : i + 64],
                metadatas=metadatas[i : i + 64],
            )

    return {
        "documents": len(docs),
        "chunks_upserted": len(ids),
        "stale_deleted": len(stale),
        "collection_count": col.count(),
        "chunk_size": s.chunk_size,
        "chunk_overlap": s.chunk_overlap,
        "embedding_model": s.embedding_model,
        "embedding_dim": s.embedding_dim,
    }
