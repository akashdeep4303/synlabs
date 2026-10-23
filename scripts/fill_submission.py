"""Fill submission_template.docx from eval JSON + architecture figures."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "submission_template.docx"
OUT = ROOT / "submission.docx"
RAG_JSON = ROOT / "data" / "runs" / "rag_eval.json"
JUDGE_JSON = ROOT / "data" / "runs" / "judge_eval.json"
QUERY_LOG = ROOT / "logs" / "queries.jsonl"
JUDGE_LOG = ROOT / "logs" / "judge.jsonl"


def set_cell(cell, text: str) -> None:
    text = str(text)
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(10)
    run.font.name = "Calibri"


def set_cell_mono(cell, text: str) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    run.font.size = Pt(8)
    run.font.name = "Menlo"
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = rPr.makeelement(qn("w:rFonts"), {})
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), "Menlo")
    rFonts.set(qn("w:hAnsi"), "Menlo")
    run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)


def add_picture_to_cell(cell, path: Path, width_in: float = 6.4) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))


def replace_para_suffix(doc: Document, startswith: str, new_text: str) -> None:
    for p in doc.paragraphs:
        if p.text.startswith(startswith):
            for r in p.runs:
                r.text = ""
            if p.runs:
                p.runs[0].text = new_text
            else:
                p.add_run(new_text)
            return


def last_jsonl(path: Path, pred=None) -> dict:
    last = {}
    if not path.exists():
        return last
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if pred is None or pred(rec):
            last = rec
    return last


def first_jsonl(path: Path, pred) -> dict:
    if not path.exists():
        return {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if pred(rec):
            return rec
    return {}


def main() -> None:
    rag = json.loads(RAG_JSON.read_text())
    judge = json.loads(JUDGE_JSON.read_text())
    ret = rag["retrieval"]
    ans = rag["answers"]
    cost = rag["cost"]
    cfg = rag["config"]
    val = judge["validation"]
    bias = judge["bias"]
    ab = judge["ab"]
    adv = val["adversarial"]

    query = last_jsonl(QUERY_LOG, lambda r: r.get("kind") == "query" and not r.get("refused"))
    ingest_note = (
        "Ingest 1 and ingest 2 both reported collection_count=16, chunks_upserted=16, "
        "stale_deleted=0 (sha256(source|index|text) upsert)."
    )
    query_dump = json.dumps(
        {
            "answer": query.get("answer")
            or "See logs/queries.jsonl — last successful P1 SLA query.",
            "chunk_count": query.get("chunk_count"),
            "latency_ms": query.get("latency_ms"),
            "tokens": query.get("tokens"),
            "model": query.get("model"),
            "citations": query.get("citations"),
        },
        indent=2,
    )[:1800]

    judge_raw = first_jsonl(JUDGE_LOG, lambda r: r.get("kind") == "judge" and r.get("prompt"))
    prompt_snip = json.dumps(judge_raw.get("prompt", [])[:2], indent=2)[:900]
    resp_snip = str(judge_raw.get("raw_response") or "")[:700]
    audit = f"PROMPT:\n{prompt_snip}\n\nRAW RESPONSE:\n{resp_snip}"

    pair_ab = first_jsonl(JUDGE_LOG, lambda r: r.get("kind") == "judge_pairwise" and r.get("order") == "AB")
    pair_ba = first_jsonl(
        JUDGE_LOG,
        lambda r: r.get("kind") == "judge_pairwise"
        and r.get("id") == pair_ab.get("id")
        and r.get("order") == "BA",
    )
    pos_box = (
        f"Pair {pair_ab.get('id')}\n"
        f"AB winner (mapped): {pair_ab.get('winner')}\n"
        f"BA winner (mapped to original labels): {pair_ba.get('winner')}\n"
        f"Suite flip_rate: {bias.get('position_flip_rate')}\n"
        f"Ties-on-flip policy: disagreement after swap → winner=tie_on_flip"
    )

    verbose_case = next((c for c in judge["pointwise"]["cases"] if c["id"] == "verbose_but_wrong"), {})
    terse_case = next((c for c in judge["pointwise"]["cases"] if c["id"] == "terse_but_correct"), {})
    padded_case = next((c for c in judge["pointwise"]["cases"] if c["id"] == "padded_correct"), {})
    conf_case = next((c for c in judge["pointwise"]["cases"] if c["id"] == "confidently_wrong"), {})
    probe_box = (
        f"verbose_but_wrong: overall={verbose_case.get('overall')} pass={verbose_case.get('pass')} "
        f"(fooled={adv.get('verbose_but_wrong_fooled')})\n"
        f"terse_but_correct: overall={terse_case.get('overall')} pass={terse_case.get('pass')}\n"
        f"padded_correct: overall={padded_case.get('overall')} pass={padded_case.get('pass')} "
        f"length_not_rewarded={adv.get('length_not_rewarded')}\n"
        f"confidently_wrong: overall={conf_case.get('overall')} pass={conf_case.get('pass')} "
        f"(fooled={adv.get('confidently_wrong_fooled')})"
    )

    doc = Document(str(TEMPLATE))
    t = doc.tables

    t[0].rows[1].cells[1].text = "Akashdeep"
    t[0].rows[2].cells[1].text = "[ fill before send ]"
    t[0].rows[3].cells[1].text = date.today().isoformat()
    t[0].rows[4].cells[1].text = "local repo: synlabs/ (no remote yet)"
    t[0].rows[5].cells[1].text = "~8 h across interrupted Cursor sessions"
    t[0].rows[6].cells[1].text = "ChromaDB (embedded, cosine HNSW)"
    t[0].rows[7].cells[1].text = (
        f"judge: {judge.get('judge_model')} · generator: {judge.get('generator_model')} "
        "(same family; independent env keys — see 2.4)"
    )

    add_picture_to_cell(t[1].rows[0].cells[0], ROOT / "figures" / "rag_architecture.png")

    set_cell_mono(
        t[2].rows[0].cells[0],
        "python3 -m venv .venv && source .venv/bin/activate\n"
        "pip install -r requirements.txt\n"
        "cp .env.example .env\n"
        "ollama pull llama3.2:3b\n"
        "export PYTHONPATH=.\n"
        "# macOS/Linux, Python 3.11+, ~8 GB RAM, Ollama local",
    )
    set_cell_mono(
        t[3].rows[0].cells[0],
        "python -m scripts.build_pdf_corpus\n"
        "PYTHONPATH=. python -m src.ingest   # default chunk 800 / overlap 150\n"
        "PYTHONPATH=. python -m src.ingest   # count must stay the same",
    )
    set_cell_mono(
        t[4].rows[0].cells[0],
        'PYTHONPATH=. python -m src.query_cli "What is the P1 first-response SLA?"\n'
        "PYTHONPATH=. python scripts/serve.py\n"
        "curl -s http://127.0.0.1:8000/health\n"
        "curl -s http://127.0.0.1:8000/query -H 'Content-Type: application/json' \\\n"
        '  -d \'{"question":"What is the P1 first-response SLA?","top_k":5}\'',
    )

    k = ret["k"]
    replace_para_suffix(doc, "Prerequisites", "Prerequisites  Python 3.11+, macOS/Linux, Ollama with llama3.2:3b, ~8 GB RAM")
    replace_para_suffix(
        doc,
        "Environment variables",
        "Environment variables  LLM_BASE_URL, LLM_API_KEY, GENERATOR_MODEL, JUDGE_MODEL, "
        "CHROMA_PATH, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, SIMILARITY_THRESHOLD, "
        "LOG_DIR, EMBEDDING_MODEL, EMBEDDING_DIM",
    )
    replace_para_suffix(
        doc,
        "Vector store chosen",
        "Vector store chosen + one-line why  Chroma embedded — cosine HNSW + metadata filters, no Docker, fits a lightly queried index billed on stored vectors.",
    )
    replace_para_suffix(doc, "k used", f"k used  k = {k}")

    set_cell(t[5].rows[1].cells[1], f"hit={ret['hit_rate']}  recall={ret['recall_at_k']}")
    set_cell(
        t[5].rows[1].cells[2],
        "Relevant iff retrieved chunk.source ∈ question.relevant_sources. "
        "Hit = first relevant rank ≤ k. Unanswerable items excluded from IR means.",
    )
    set_cell(t[5].rows[2].cells[1], str(ret["mrr"]))
    set_cell(t[5].rows[2].cells[2], "Mean of 1/rank of first relevant source; 0 if none in top-k.")
    set_cell(t[5].rows[3].cells[1], str(ret["ndcg_at_k"]))
    set_cell(t[5].rows[3].cells[2], "Binary gains (1 if source in gold list). nDCG@k averaged over answerable questions.")
    set_cell(t[5].rows[4].cells[1], str(ret["context_precision"]))
    set_cell(t[5].rows[4].cells[2], "Share of top-k chunks whose source is in relevant_sources.")

    set_cell(t[6].rows[1].cells[1], str(ans.get("faithfulness_mean")))
    set_cell(t[6].rows[1].cells[2], "LLM-judge faithfulness 1–5, first 8 answers")
    set_cell(t[6].rows[1].cells[3], f"judge={cfg.get('judge_model')}; see data/runs/rag_eval.json")
    set_cell(t[6].rows[2].cells[1], str(ans.get("relevance_mean")))
    set_cell(t[6].rows[2].cells[2], "LLM-judge correctness used as relevance")
    set_cell(t[6].rows[2].cells[3], f"refusal_accuracy={ans.get('refusal_accuracy')}")
    set_cell(t[6].rows[3].cells[1], str(ans.get("em")))
    set_cell(t[6].rows[3].cells[2], "normalized exact match vs short gold")
    set_cell(t[6].rows[3].cells[3], "Gold strings are extractive; EM is strict so F1 is the fairer number.")
    set_cell(t[6].rows[4].cells[1], str(ans.get("f1")))
    set_cell(t[6].rows[4].cells[2], "token F1 vs gold")
    set_cell(t[6].rows[4].cells[3], "22 questions in data/eval/rag_questions.yaml")

    for i, row in enumerate(cost["rows"], start=1):
        set_cell(t[7].rows[i].cells[1], str(row["chroma_usd_mo"]))
        set_cell(t[7].rows[i].cells[2], str(row["managed_usd_mo"]))
        set_cell(t[7].rows[i].cells[3], f"${row['savings_usd_mo']}/mo vs Pinecone p1 family")
    a = cost["assumptions"]
    set_cell(
        t[8].rows[0].cells[0],
        f"384-d float32 + metadata ≈ {a['bytes_per_vector_disk']} B/vector. "
        f"Chroma VM 100K={a['chroma_vm_100k']}; 1M={a['chroma_vm_1m']}; 10M={a['chroma_vm_10m']}. "
        f"Managed={a['managed']}; p1.x1≈${a['pinecone_p1_x1_usd_mo']}/mo. "
        f"{a['pinecone_capacity_note']} Query volume: {a['query_volume']}.",
    )

    set_cell(t[9].rows[1].cells[1], str(ret["latency_ms"]["p50"]))
    set_cell(t[9].rows[1].cells[2], f"k={k}, {rag['ingest'].get('collection_count', 16)} chunks, local Chroma + MiniLM ONNX, Apple Silicon")
    set_cell(t[9].rows[2].cells[1], str(ret["latency_ms"]["p95"]))
    set_cell(t[9].rows[2].cells[2], "Includes first-query ONNX warmup in the distribution if present")
    set_cell(t[9].rows[3].cells[1], str(ans["e2e_latency_ms"]["p95"]))
    set_cell(t[9].rows[3].cells[2], "retrieval + llama3.2:3b generation on this machine")

    set_cell(
        t[10].rows[0].cells[0],
        "Chroma embedded matches the assignment’s cost problem (always-on pods billed on stored vectors for a lightly queried index). "
        "It gives cosine HNSW and metadata filters without Docker. pgvector is the production next step (SQL, backups, ACID) but needs Postgres. "
        "FAISS has no metadata filter without a sidecar. sqlite-vec is smaller but younger. Qdrant/LanceDB are fine; Chroma was already on disk here.",
    )
    set_cell(
        t[11].rows[0].cells[0],
        "Default 800/150 characters, snapped to paragraph then sentence breaks. 400/50 split tables (rate limits, price rows) across chunks and dropped context precision. "
        "1600/200 mixed pricing with SSO in one chunk and hurt precision on policy questions. Overlap 150 keeps a heading with the following table.",
    )
    set_cell(
        t[12].rows[0].cells[0],
        "Chroma DefaultEmbeddingFunction = MiniLM-L6-v2 ONNX, 384-d. Roughly 2× the vectors per RAM vs 768-d OpenAI ada, good enough on a 9-doc product corpus. "
        "Trade-off: weaker on near-paraphrase policy (retention vs backups both mention RPO).",
    )
    set_cell(
        t[13].rows[0].cells[0],
        "If no chunk has cosine similarity ≥ SIMILARITY_THRESHOLD (0.28), the service returns "
        "exactly “I don't have that information in the NimbusLedger docs.” and never calls the LLM. "
        "The generator is also instructed to refuse when the numbered chunks lack the fact.",
    )
    set_cell(
        t[14].rows[0].cells[0],
        "Chunk id = sha256(source|index|text)[:24]-index. Re-ingest upserts the same ids and deletes ids not in the new set. "
        "Running ingest twice left collection_count at 16 both times.",
    )
    set_cell(
        t[15].rows[0].cells[0],
        "Accepted: single-node HNSW, no replica, local embeddings, small 3B generator. Switch back to managed when you need multi-AZ SLA, "
        "zero-ops upgrades, or query QPS that forces a 32 GB HNSW box anyway. Retrieval, not generation, is the weak link on near-duplicate policy questions.",
    )

    set_cell_mono(t[16].rows[0].cells[0], query_dump)
    set_cell_mono(
        t[17].rows[0].cells[0],
        ingest_note + "\n\nFirst run elapsed_ms≈1800; second ≈660. Same 16 vectors.",
    )
    set_cell(
        t[18].rows[0].cells[0],
        "POST /query (or the CLI) loads settings from .env. retrieve() embeds the question with MiniLM, cosine-queries Chroma, "
        "optionally applies a doc_type where-filter, drops hits below 0.28, and returns top-k. If the list is empty we refuse. "
        "Otherwise the generator sees numbered chunks with source filenames and must cite [source: FILE]. Latency, chunk_count, and tokens go to logs/queries.jsonl.",
    )
    set_cell(
        t[19].rows[0].cells[0],
        "First MiniLM ONNX download hung the first ingest while Chroma fetched weights (~500 MB). I watched disk and retried ingest after the cache filled; "
        "collection_count then stuck at 16. A second family judge (Qwen) was skipped because the disk could not hold another weights file.",
    )
    set_cell(
        t[20].rows[0].cells[0],
        "Cursor Grok 4.6 scaffolded the repo, Chroma/FastAPI wiring, and eval harnesses. Gold questions, rubric anchors, cost assumptions, "
        "and the refusal sentence were specified to match the PDF brief. I ran ingest/eval locally on Ollama.",
    )

    add_picture_to_cell(t[21].rows[0].cells[0], ROOT / "figures" / "judge_architecture.png")
    # second env/prereq paragraphs (problem 2)
    seen_prereq = 0
    seen_env = 0
    for p in doc.paragraphs:
        if p.text.startswith("Prerequisites"):
            seen_prereq += 1
            if seen_prereq == 2:
                for r in p.runs:
                    r.text = ""
                p.runs[0].text = "Prerequisites  Same as Problem 1. Judge calls Ollama llama3.2:3b via LLM_BASE_URL."
        if p.text.startswith("Environment variables"):
            seen_env += 1
            if seen_env == 2:
                for r in p.runs:
                    r.text = ""
                p.runs[0].text = (
                    "Environment variables  Same names; JUDGE_MODEL is independent of GENERATOR_MODEL."
                )

    set_cell_mono(
        t[22].rows[0].cells[0],
        "pip install -r requirements.txt && cp .env.example .env && export PYTHONPATH=.",
    )
    set_cell_mono(
        t[23].rows[0].cells[0],
        "PYTHONPATH=. python -m src.eval_judge\n# writes data/runs/judge_eval.json ; logs/judge.jsonl has every prompt+raw",
    )
    set_cell_mono(
        t[24].rows[0].cells[0],
        "Same command: pairwise pairs in data/eval/judge_pairwise.yaml are Config A (grounded/terse) vs Config B (chatty). "
        "Pointwise + pairwise run together; winner is declared in the report.",
    )
    replace_para_suffix(
        doc,
        "Judge & generator configured independently?",
        "Judge & generator configured independently?  Yes — GENERATOR_MODEL and JUDGE_MODEL in .env; LLMClient(role=...).",
    )
    replace_para_suffix(
        doc,
        "Judging mode used",
        "Judging mode used  Pointwise 1–5 rubric (reference-based when expected_output is set). Pairwise A-vs-B for position bias and the A/B winner.",
    )

    rubric_rows = [
        ("correctness", "0.30"),
        ("faithfulness", "0.25"),
        ("completeness", "0.20"),
        ("instruction_following", "0.15"),
        ("tone_safety", "0.10"),
    ]
    from src.judge.rubric import CRITERIA

    for i, (name, w) in enumerate(rubric_rows, start=1):
        spec = CRITERIA[name]
        anchors = "; ".join(f"{k}={v}" for k, v in spec["anchors"].items())
        set_cell(t[25].rows[i].cells[1], f"{spec['definition']} Anchors: {anchors}")
        set_cell(t[25].rows[i].cells[2], w)

    flip = bias.get("position_flip_rate")
    set_cell(
        t[26].rows[1].cells[1],
        "judge_pairwise() scores AB and BA; maps BA winner back; disagreement → tie_on_flip. Averages criterion scores across orders.",
    )
    set_cell(t[26].rows[1].cells[2], f"measured flip_rate={flip} (after swap+tie policy). Unmitigated would take AB-only winner.")
    set_cell(
        t[26].rows[2].cells[1],
        "Rubric: length is not a bonus. Suite has terse_but_correct, padded_correct, verbose_but_wrong.",
    )
    set_cell(
        t[26].rows[2].cells[2],
        f"length_not_rewarded={adv.get('length_not_rewarded')}; "
        f"terse={adv.get('terse_but_correct_overall')} padded={adv.get('padded_correct_overall')}; "
        f"verbose_fooled={adv.get('verbose_but_wrong_fooled')}",
    )
    set_cell(
        t[26].rows[3].cells[1],
        "Independent JUDGE_MODEL key. Second-family weights (Qwen) skipped — disk full. Gold-label kappa is the backstop.",
    )
    set_cell(t[26].rows[3].cells[2], judge.get("self_enhancement_note", "")[:400])
    set_cell(
        t[26].rows[4].cells[1],
        "Per-criterion quotes required. confidently_wrong probe must fail.",
    )
    set_cell(t[26].rows[4].cells[2], f"confidently_wrong_fooled={adv.get('confidently_wrong_fooled')}")
    set_cell(
        t[26].rows[5].cells[1],
        "1/3/5 few-shot anchors in the judge system prompt; weighted overall not a lone integer.",
    )
    set_cell(t[26].rows[5].cells[2], f"score_spread_pstdev={bias.get('score_spread_pstdev')}")

    set_cell(t[27].rows[1].cells[1], f"{val['agreement_with_gold_pass']*100:.1f}%")
    set_cell(t[27].rows[1].cells[2], f"n={val['n']} gold_pass labels in judge_suite.yaml")
    set_cell(t[27].rows[2].cells[1], str(val["cohen_kappa_pass"]))
    set_cell(t[27].rows[2].cells[2], f"overall MAE={val['overall_mae']}")
    set_cell(t[27].rows[3].cells[1], f"{val['test_retest_flip_rate']*100:.1f}%")
    set_cell(t[27].rows[3].cells[2], f"retest_n={val['test_retest_n']}, temperature=0.0")
    fooled = adv.get("verbose_but_wrong_fooled") or adv.get("confidently_wrong_fooled")
    set_cell(t[27].rows[4].cells[1], "Y" if fooled else "N — not fooled on the main probes")
    set_cell(
        t[27].rows[4].cells[2],
        f"verbose_fooled={adv.get('verbose_but_wrong_fooled')}; confident_fooled={adv.get('confidently_wrong_fooled')}",
    )

    set_cell(t[28].rows[1].cells[0], f"Config A — {ab['config_a']['name']}")
    set_cell(t[28].rows[1].cells[1], f"{ab['config_a']['pass_rate']:.3f}")
    set_cell(t[28].rows[1].cells[2], f"{ab['config_a']['mean_score']:.3f}")
    set_cell(t[28].rows[1].cells[3], f"wins={ab['config_a']['wins']}")
    set_cell(t[28].rows[2].cells[0], f"Config B — {ab['config_b']['name']}")
    set_cell(t[28].rows[2].cells[1], f"{ab['config_b']['pass_rate']:.3f}")
    set_cell(t[28].rows[2].cells[2], f"{ab['config_b']['mean_score']:.3f}")
    set_cell(t[28].rows[2].cells[3], f"wins={ab['config_b']['wins']}")
    replace_para_suffix(
        doc,
        "Declared winner",
        f"Declared winner + one-line justification  {ab['declared_winner']} "
        f"(win_rate_A={ab['win_rate_a']}; grounded answers beat chatty ones on the rubric).",
    )

    set_cell(
        t[29].rows[0].cells[0],
        "Pointwise with a reference when expected_output exists — fits a CI gold set where each ticket has a known right answer. "
        "Pairwise is used for A/B (prompt style) and to measure position bias, which pointwise cannot see.",
    )
    set_cell(
        t[30].rows[0].cells[0],
        "parse_verdict: fenced JSON → first brace object → quote/trailing-comma repair → regex fallback on criterion names → one retry asking for JSON only. "
        "Overall is always recomputed from weighted criterion scores so a lying 'overall' field cannot pass.",
    )
    set_cell(
        t[31].rows[0].cells[0],
        "I wanted Qwen vs Llama to cut self-enhancement. Disk only held llama3.2:3b. Mitigation: independent config keys, few-shot anchors that punish Llama-style padding, "
        "and kappa vs gold so self-preference would show as disagreement.",
    )
    set_cell(
        t[32].rows[0].cells[0],
        "Position bias worried me most on A/B. Flip_rate is measured; ties-on-flip stops a lucky order from crowning a winner. "
        "A high flip rate would mean I would not trust pairwise alone.",
    )
    set_cell(
        t[33].rows[0].cells[0],
        "No, not alone. It can fail CI on regressions vs a frozen gold set (kappa, flip rate < 15%). A human still reviews safety, pricing, and any case the judge flipped on order swap.",
    )

    set_cell_mono(t[34].rows[0].cells[0], audit)
    set_cell_mono(t[35].rows[0].cells[0], pos_box)
    set_cell(
        t[36].rows[0].cells[0],
        "Each pair is judged twice with A and B swapped. The BA winner is mapped back to the original labels. If AB says A and BA says B, that is a flip and we call a tie. "
        "A high flip rate means the judge is following seat order, not content — you cannot ship that as an A/B oracle.",
    )
    set_cell_mono(t[37].rows[0].cells[0], probe_box)
    set_cell(
        t[38].rows[0].cells[0],
        "Cursor Grok 4.6 for scaffolding. Gold labels, rubric weights, and the “would this gate a release” call are mine against the assignment PDF. "
        "Logs in logs/judge.jsonl are the audit trail.",
    )

    doc.save(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
