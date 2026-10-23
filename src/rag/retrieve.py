from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import get_settings
from src.rag.store import get_collection


@dataclass
class Hit:
    id: str
    text: str
    source: str
    doc_type: str
    chunk_index: int
    distance: float
    similarity: float
    metadata: dict[str, Any] = field(default_factory=dict)


def retrieve(
    query: str,
    *,
    k: int | None = None,
    where: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> list[Hit]:
    s = get_settings()
    k = k if k is not None else s.top_k
    threshold = s.similarity_threshold if threshold is None else threshold
    col = get_collection(create=True)
    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": max(k, 1)}
    if where:
        kwargs["where"] = where
    raw = col.query(**kwargs)
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    hits: list[Hit] = []
    for i, cid in enumerate(ids):
        dist = float(dists[i]) if i < len(dists) and dists[i] is not None else 1.0
        sim = 1.0 - dist
        meta = metas[i] or {}
        hits.append(
            Hit(
                id=cid,
                text=docs[i] or "",
                source=str(meta.get("source") or ""),
                doc_type=str(meta.get("doc_type") or ""),
                chunk_index=int(meta.get("chunk_index") or 0),
                distance=dist,
                similarity=sim,
                metadata=meta,
            )
        )
    hits.sort(key=lambda h: h.similarity, reverse=True)
    return [h for h in hits if h.similarity >= threshold][:k]


def retrieve_unfiltered(query: str, k: int) -> list[Hit]:
    """Top-k without the similarity gate — used by the IR eval harness."""
    s = get_settings()
    old = s.similarity_threshold
    try:
        return retrieve(query, k=k, threshold=-1.0)
    finally:
        s.similarity_threshold = old
