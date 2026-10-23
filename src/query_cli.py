from __future__ import annotations

import argparse
import json
import time

from src.rag.generate import answer_question
from src.rag.retrieve import retrieve


def main() -> None:
    p = argparse.ArgumentParser(description="Query the NimbusLedger RAG index.")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--doc-type", default=None)
    p.add_argument("--retrieve-only", action="store_true")
    args = p.parse_args()
    where = {"doc_type": args.doc_type} if args.doc_type else None
    t0 = time.perf_counter()
    if args.retrieve_only:
        hits = retrieve(args.question, k=args.k, where=where)
        elapsed = (time.perf_counter() - t0) * 1000
        print(json.dumps({"latency_ms": round(elapsed, 1), "hits": [h.__dict__ for h in hits]}, indent=2, default=str))
        return
    t_ret0 = time.perf_counter()
    hits = retrieve(args.question, k=args.k, where=where)
    retrieval_ms = (time.perf_counter() - t_ret0) * 1000
    out = answer_question(args.question, k=args.k, where=where)
    out["latency_ms"]["retrieval"] = round(retrieval_ms, 1)
    out["latency_ms"]["total"] = round((time.perf_counter() - t0) * 1000, 1)
    slim = {k: out[k] for k in ("answer", "refused", "citations", "chunk_count", "latency_ms", "tokens", "model")}
    print(json.dumps(slim, indent=2))


if __name__ == "__main__":
    main()
