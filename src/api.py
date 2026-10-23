from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag.generate import answer_question
from src.rag.store import get_collection

app = FastAPI(title="NimbusLedger RAG", version="1.0.0")


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, ge=1, le=20)
    doc_type: str | None = Field(default=None, description="Optional metadata filter, e.g. pdf|html|markdown")


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[dict[str, Any]]
    chunk_count: int
    latency_ms: dict[str, Any]
    tokens: dict[str, int]
    model: str | None


@app.get("/health")
def health() -> dict[str, Any]:
    s = get_settings()
    col = get_collection(create=True)
    return {
        "ok": True,
        "collection": s.collection_name,
        "count": col.count(),
        "generator_model": s.generator_model,
        "embedding_model": s.embedding_model,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict[str, Any]:
    where = {"doc_type": req.doc_type} if req.doc_type else None
    return answer_question(req.question, k=req.top_k, where=where)
