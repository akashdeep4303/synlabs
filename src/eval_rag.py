from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from src.config import ROOT, get_settings
from src.judge.pipeline import judge_pointwise
from src.logging_utils import write_json
from src.llm import LLMClient
from src.metrics import (
    context_precision,
    exact_match,
    hit_rate,
    mean_ndcg,
    mean_recall,
    mrr,
    ndcg_at_k,
    percentile,
    token_f1,
)
from src.rag.generate import answer_question
from src.rag.retrieve import retrieve
from src.rag.store import ingest_corpus


def load_questions(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def eval_retrieval(questions: list[dict], k: int) -> dict:
    ranks = []
    recalls = []
    ndcgs = []
    precisions = []
    latencies = []
    per_q = []
    for q in questions:
        t0 = time.perf_counter()
        hits = retrieve(q["question"], k=k, threshold=-1.0)
        latencies.append((time.perf_counter() - t0) * 1000)
        gold = set(q.get("relevant_sources") or [])
        sources = [h.source for h in hits[:k]]
        rel_flags = [1.0 if s in gold else 0.0 for s in sources]
        if not gold:
            # unanswerable: success is retrieving nothing above a high bar; IR metrics skip gold-empty
            rank = None
            rec = 1.0 if not any(rel_flags) else 1.0
            ranks.append(None)
            recalls.append(1.0)
            ndcgs.append(1.0)
            precisions.append(1.0)
        else:
            rank = next((i + 1 for i, s in enumerate(sources) if s in gold), None)
            n_rel_ret = len({s for s in sources if s in gold})
            rec = n_rel_ret / len(gold)
            ranks.append(rank)
            recalls.append(rec)
            ndcgs.append(ndcg_at_k(rel_flags, k))
            precisions.append(context_precision(sum(rel_flags), max(1, len(sources))))
        per_q.append(
            {
                "id": q["id"],
                "rank_first_relevant": rank if gold else "n/a_unanswerable",
                "recall": rec,
                "sources": sources,
                "gold": sorted(gold),
            }
        )
    answerable_ranks = [r for q, r in zip(questions, ranks) if q.get("relevant_sources")]
    return {
        "k": k,
        "hit_rate": round(hit_rate(answerable_ranks), 4),
        "recall_at_k": round(mean_recall([r for q, r in zip(questions, recalls) if q.get("relevant_sources")]), 4),
        "mrr": round(mrr(answerable_ranks), 4),
        "ndcg_at_k": round(mean_ndcg([n for q, n in zip(questions, ndcgs) if q.get("relevant_sources")]), 4),
        "context_precision": round(
            sum(p for q, p in zip(questions, precisions) if q.get("relevant_sources"))
            / max(1, sum(1 for q in questions if q.get("relevant_sources"))),
            4,
        ),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "mean": round(sum(latencies) / len(latencies), 2),
        },
        "per_question": per_q,
    }


def eval_answers(questions: list[dict], k: int, judge: bool, judge_limit: int = 8) -> dict:
    gen = LLMClient("generator")
    jllm = LLMClient("judge") if judge else None
    ems, f1s, faiths, rels = [], [], [], []
    e2e = []
    refuse_ok = []
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        out = answer_question(q["question"], k=k, llm=gen)
        e2e.append((time.perf_counter() - t0) * 1000)
        gold = q.get("gold") or ""
        pred = out["answer"]
        em = exact_match(pred, gold)
        _, _, f1 = token_f1(pred, gold)
        ems.append(em)
        f1s.append(f1)
        expected_refuse = bool(q.get("expect_refuse"))
        did_refuse = bool(out["refused"]) or "don't have that information" in pred.lower()
        if expected_refuse:
            refuse_ok.append(1.0 if did_refuse else 0.0)
        faith = rel = None
        if judge and jllm and (judge_limit <= 0 or len(faiths) < judge_limit):
            case = {
                "id": q["id"],
                "input": q["question"],
                "system_prompt": "Answer only from retrieved NimbusLedger chunks; refuse if missing.",
                "model_output": pred,
                "expected_output": gold,
            }
            rec = judge_pointwise(case, llm=jllm)
            scores = rec["verdict"]["scores"]
            faith = scores.get("faithfulness", {}).get("score")
            rel = scores.get("correctness", {}).get("score")
            faiths.append(faith or 0)
            rels.append(rel or 0)
        rows.append(
            {
                "id": q["id"],
                "answer": pred,
                "gold": gold,
                "em": em,
                "f1": round(f1, 4),
                "refused": did_refuse,
                "expect_refuse": expected_refuse,
                "faithfulness": faith,
                "relevance": rel,
                "chunk_count": out["chunk_count"],
                "tokens": out["tokens"],
                "citations": out["citations"],
            }
        )
    n = max(1, len(questions))
    return {
        "em": round(sum(ems) / n, 4),
        "f1": round(sum(f1s) / n, 4),
        "refusal_accuracy": round(sum(refuse_ok) / max(1, len(refuse_ok)), 4),
        "faithfulness_mean": round(sum(faiths) / len(faiths), 3) if faiths else None,
        "relevance_mean": round(sum(rels) / len(rels), 3) if rels else None,
        "e2e_latency_ms": {
            "p50": round(percentile(e2e, 50), 1),
            "p95": round(percentile(e2e, 95), 1),
        },
        "per_question": rows,
    }


def cost_table() -> dict:
    # Assumptions stated in README / submission.
    # Vectors: 384-d float32 + 200B metadata ≈ 1.8 KB on disk; HNSW RAM ≈ 1.5 KB extra.
    # Chroma on a single VM, no HA replica.
    # Managed comparison: Pinecone pod p1.x1 (~$0.096/hr ≈ $70/mo) which the assignment
    # background describes (always-on pods billed on stored capacity).
    return {
        "assumptions": {
            "embedding_dim": 384,
            "bytes_per_vector_disk": 1800,
            "chroma_vm_100k": {"spec": "1 vCPU / 2 GB / 20 GB SSD", "usd_mo": 6},
            "chroma_vm_1m": {"spec": "2 vCPU / 4 GB / 40 GB SSD", "usd_mo": 12},
            "chroma_vm_10m": {"spec": "8 vCPU / 32 GB / 200 GB SSD (HNSW in RAM)", "usd_mo": 80},
            "managed": "Pinecone p1 pod family, us-east, 1 replica, 0 queries included in pod price",
            "pinecone_p1_x1_usd_mo": 70,
            "pinecone_capacity_note": "p1.x1 ~100k x 768-d; 384-d roughly 2x vectors per pod. 1M needs ~5 p1.x1. 10M needs p2 or ~50 p1.",
            "query_volume": "lightly queried: 1 QPS peak, 2M queries/month — pod price still paid idle",
        },
        "rows": [
            {"vectors": "100K", "chroma_usd_mo": 6, "managed_usd_mo": 70, "savings_usd_mo": 64},
            {"vectors": "1M", "chroma_usd_mo": 12, "managed_usd_mo": 350, "savings_usd_mo": 338},
            {"vectors": "10M", "chroma_usd_mo": 80, "managed_usd_mo": 1400, "savings_usd_mo": 1320},
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default=str(ROOT / "data" / "eval" / "rag_questions.yaml"))
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--judge-limit", type=int, default=8, help="LLM-judge at most N answers (0 = all)")
    p.add_argument("--skip-ingest", action="store_true")
    args = p.parse_args()
    s = get_settings()
    if not args.skip_ingest:
        ingest = ingest_corpus()
    else:
        ingest = {"skipped": True}
    spec = load_questions(Path(args.questions))
    k = int(spec.get("k_default") or s.top_k)
    questions = spec["questions"]
    retrieval = eval_retrieval(questions, k)
    answers = eval_answers(questions, k, judge=not args.skip_judge, judge_limit=args.judge_limit)
    report = {
        "ingest": ingest,
        "retrieval": retrieval,
        "answers": answers,
        "cost": cost_table(),
        "config": {
            "chunk_size": s.chunk_size,
            "chunk_overlap": s.chunk_overlap,
            "top_k": k,
            "similarity_threshold": s.similarity_threshold,
            "embedding_model": s.embedding_model,
            "embedding_dim": s.embedding_dim,
            "generator_model": s.generator_model,
            "judge_model": s.judge_model,
            "store": "chromadb",
        },
    }
    out = Path(s.runs_dir) / "rag_eval.json"
    write_json(out, report)
    print(
        yaml.safe_dump(
            {
                "wrote": str(out),
                "hit_rate": retrieval["hit_rate"],
                "mrr": retrieval["mrr"],
                "ndcg": retrieval["ndcg_at_k"],
                "em": answers["em"],
                "f1": answers["f1"],
                "faithfulness": answers["faithfulness_mean"],
                "retrieval_p50_ms": retrieval["latency_ms"]["p50"],
                "retrieval_p95_ms": retrieval["latency_ms"]["p95"],
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
