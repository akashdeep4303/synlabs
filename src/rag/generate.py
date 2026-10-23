from __future__ import annotations

import time
from typing import Any

from src.config import get_settings
from src.llm import LLMClient
from src.logging_utils import log_event
from src.rag.retrieve import Hit, retrieve

SYSTEM = """You are the NimbusLedger documentation assistant.
Answer using ONLY the numbered context chunks.
Rules:
- If the chunks do not contain the answer, reply exactly: I don't have that information in the NimbusLedger docs.
- Do not invent product facts, prices, SLAs, or API limits.
- Cite every claim with [source: FILENAME] using the source field of the chunk you used.
- Keep answers concise (under 120 words) unless the user asks for detail.
"""


def format_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        parts.append(f"[{i}] source={h.source} type={h.doc_type} sim={h.similarity:.3f}\n{h.text}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    *,
    k: int | None = None,
    where: dict[str, Any] | None = None,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    s = get_settings()
    t0 = time.perf_counter()
    hits = retrieve(question, k=k, where=where)
    retrieval_ms = (time.perf_counter() - t0) * 1000
    if not hits:
        result = {
            "question": question,
            "answer": "I don't have that information in the NimbusLedger docs.",
            "refused": True,
            "citations": [],
            "chunks": [],
            "chunk_count": 0,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 1),
                "generation": 0.0,
                "total": round(retrieval_ms, 1),
            },
            "tokens": {"prompt": 0, "completion": 0, "total": 0},
            "model": None,
        }
        log_event("query", {**result, "kind_detail": "no_context"}, "queries.jsonl")
        return result

    llm = llm or LLMClient("generator")
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}",
        },
    ]
    gen = llm.chat(messages, max_tokens=280)
    payload = {
        "question": question,
        "answer": gen.text.strip(),
        "refused": False,
        "citations": [
            {
                "source": h.source,
                "chunk_index": h.chunk_index,
                "similarity": round(h.similarity, 4),
                "doc_type": h.doc_type,
            }
            for h in hits
        ],
        "chunks": [{"id": h.id, "source": h.source, "text": h.text, "similarity": h.similarity} for h in hits],
        "chunk_count": len(hits),
        "latency_ms": {
            "retrieval": round(retrieval_ms, 1),
            "generation": round(gen.latency_ms, 1),
            "total": round(retrieval_ms + gen.latency_ms, 1),
        },
        "tokens": {
            "prompt": gen.prompt_tokens,
            "completion": gen.completion_tokens,
            "total": gen.total_tokens,
        },
        "model": gen.model,
        "top_k": k or s.top_k,
        "filter": where,
    }
    log_event(
        "query",
        {
            "question": question,
            "answer": payload["answer"],
            "chunk_count": payload["chunk_count"],
            "latency_ms": payload["latency_ms"],
            "tokens": payload["tokens"],
            "model": payload["model"],
            "citations": payload["citations"],
            "refused": False,
        },
        "queries.jsonl",
    )
    return payload
