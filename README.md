# NimbusLedger take-home — cost-efficient RAG + LLM-as-judge

Two runnable pipelines for the Applied AI / ML Engineering assignment.

| Problem | Store / models | Entry point |
| --- | --- | --- |
| 1. Cost-efficient RAG | **ChromaDB** (embedded) + MiniLM-L6-v2 (384-d) + `llama3.2:3b` | `POST /query` |
| 2. LLM-as-judge | Pointwise rubric + pairwise A/B with order swap | `python -m src.eval_judge` |

Generator and judge are independent config keys (`GENERATOR_MODEL`, `JUDGE_MODEL`). This machine only had disk for one chat model (`llama3.2:3b`); a second family (Qwen) was attempted and skipped. Validation against gold labels is the backstop for self-enhancement.

## Prerequisites

- macOS or Linux, Python 3.11+
- [Ollama](https://ollama.com) with `llama3.2:3b` (`ollama pull llama3.2:3b`)
- ~500 MB disk for the ONNX MiniLM weights (downloaded once by Chroma)

## Setup

```bash
cd synlabs
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
export PYTHONPATH=.
```

Environment variable **names** (no secrets in git): `LLM_BASE_URL`, `LLM_API_KEY`, `GENERATOR_MODEL`, `JUDGE_MODEL`, `CHROMA_PATH`, `COLLECTION_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `SIMILARITY_THRESHOLD`, `LOG_DIR`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`.

Default chunking: **size 800 / overlap 150** characters, snapped to paragraph or sentence breaks.

## Problem 1 — RAG

```bash
python -m scripts.build_pdf_corpus   # writes the two PDF docs
PYTHONPATH=. python -m src.ingest    # idempotent upsert; prints collection_count
PYTHONPATH=. python -m src.ingest    # run again — count must stay the same
PYTHONPATH=. python -m src.query_cli "What is the P1 first-response SLA?"
PYTHONPATH=. python scripts/serve.py
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the P1 first-response SLA?","top_k":5}'
```

Metadata filter example (PDF-only): `{"question":"...","doc_type":"pdf"}`.

If no chunk is above `SIMILARITY_THRESHOLD` (default 0.28 cosine similarity), the service returns *I don't have that information in the NimbusLedger docs.* and does not call the LLM.

Per-query latency, chunk count, and tokens are appended to `logs/queries.jsonl`.

### Evaluation

```bash
PYTHONPATH=. python -m src.eval_rag
```

Writes `data/runs/rag_eval.json` (retrieval IR metrics, answer EM/F1 + LLM faithfulness/relevance, p50/p95 latency, cost table).

Relevant chunk = chunk whose `source` is in the question’s `relevant_sources` list (`data/eval/rag_questions.yaml`, 22 items).

### Why Chroma

The assignment’s cost problem is **always-on pods billed on stored vectors** for a lightly queried index. Chroma embedded keeps vectors on local disk, uses the same cosine HNSW pattern as the managed stores, and needs no Docker. pgvector would be the production next step (SQL filters, backups); FAISS has no metadata filter without a sidecar; sqlite-vec is smaller but younger.

### Cost (see `data/runs/rag_eval.json` for the table)

Assumptions: 384-d float32 + metadata ≈ 1.8 KB/vector on disk; lightly queried (~1 QPS); Chroma on a single DigitalOcean-class VM, no replica; managed = Pinecone **p1 pod** family (always-on, US East, 1 replica). p1.x1 ≈ $70/mo and holds on the order of 100K 768-d vectors (~2× that at 384-d). 1M ≈ 5 pods; 10M ≈ p2 / ~20× p1. Idle query volume does not reduce pod rent.

| Vectors | Chroma VM | Pinecone pods | You keep |
| --- | ---: | ---: | ---: |
| 100K | $6 | $70 | $64 |
| 1M | $12 | $350 | $338 |
| 10M | $80 | $1,400 | $1,320 |

Switch back to managed when you need multi-AZ SLA, zero-ops upgrades, or query QPS that forces you to run a 32 GB HNSW box anyway.

### Discussion

Retrieval is the weaker link on near-duplicate policy questions (retention vs backups both mention RPO); generation is constrained by the refuse path and is mostly extraction. Chunk 800/150 beat 400/50 (too many partial tables) and 1600/200 (mixed pricing+SSO in one chunk, lower context precision) in a hand check during development.

## Problem 2 — Judge

```bash
PYTHONPATH=. python -m src.eval_judge
```

Writes `data/runs/judge_eval.json`. Every prompt and raw response is in `logs/judge.jsonl`.

- **Mode:** pointwise 1–5 rubric (reference-based when `expected_output` is present). Pairwise A-vs-B is used for position-bias measurement and the A/B winner call.
- **Rubric:** correctness 0.30, faithfulness 0.25, completeness 0.20, instruction-following 0.15, tone/safety 0.10. Pass ≥ 3.5.
- **Malformed JSON:** fenced JSON → brace match → quote repair → regex fallback → one retry.
- **Position bias:** each pair scored AB and BA; winner flipped → `tie_on_flip`; flip rate reported.
- **Verbosity:** padded-correct vs terse-correct vs verbose-but-wrong probes; length is not a rubric bonus.
- **Sycophancy:** confidently-wrong probe must fail.
- **Score clustering:** 1/3/5 few-shot anchors in the judge prompt; spread reported.

Would this gate a release? **No, not alone.** It can fail a CI check for *regressions vs a frozen gold set* (kappa with labels, flip rate < 15%). A human still reviews safety, pricing, and any case the judge flipped on order swap.

## Tests (no LLM)

```bash
PYTHONPATH=. python -m pytest tests/ -q
```

## Layout

```
src/rag/          ingest, chunk, chroma, retrieve, generate
src/judge/        rubric, JSON parse, pointwise + pairwise
src/eval_rag.py   src/eval_judge.py   src/api.py
data/corpus/      md + html + pdf
data/eval/        22 RAG questions, judge suite, pairwise pairs
data/runs/        evaluation JSON (generated)
logs/             query + judge JSONL (generated)
figures/          architecture diagrams
```

## AI usage

Cursor Grok 4.6 was used to scaffold the repo, wire Chroma/FastAPI, and draft eval harnesses. Gold questions, rubric anchors, cost assumptions, and the refusal policy were specified to match the PDF brief. Fill `submission.docx` candidate name/email before sending.
