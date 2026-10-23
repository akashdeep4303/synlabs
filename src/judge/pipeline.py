from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.judge.parse import parse_verdict
from src.judge.rubric import CRITERIA, FEW_SHOT, PASS_THRESHOLD, weighted_overall
from src.llm import LLMClient
from src.logging_utils import log_event

JUDGE_SYSTEM = """You are an independent evaluation judge, not the model that wrote the answer.
Score each rubric criterion from 1 to 5 using the anchors.
Do not reward length. A short correct answer beats a long padded one.
Do not agree with the author just because the tone is confident.
Return ONLY JSON with this schema:
{
  "scores": {
    "correctness": {"score": <1-5>, "rationale": "<one sentence citing evidence>"},
    "faithfulness": {"score": <1-5>, "rationale": "..."},
    "completeness": {"score": <1-5>, "rationale": "..."},
    "instruction_following": {"score": <1-5>, "rationale": "..."},
    "tone_safety": {"score": <1-5>, "rationale": "..."}
  },
  "overall": <float>,
  "pass": <true|false>,
  "rationale": "<two sentences>"
}
Pass if overall >= 3.5.
"""


def _rubric_block() -> str:
    lines = []
    for name, spec in CRITERIA.items():
        lines.append(f"- {name} (weight {spec['weight']}): {spec['definition']}")
        for sc, txt in spec["anchors"].items():
            lines.append(f"    {sc}: {txt}")
    lines.append("Few-shot anchors:")
    for ex in FEW_SHOT:
        lines.append(json.dumps({"input": ex["input"], "output": ex["output"][:180], "scores": ex["scores"], "note": ex["note"]}))
    return "\n".join(lines)


def load_suite(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    return json.loads(text)


def pointwise_prompt(case: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "input": case.get("input"),
        "system_prompt": case.get("system_prompt"),
        "model_output": case.get("model_output"),
        "expected_output": case.get("expected_output"),
        "criteria": case.get("criteria") or list(CRITERIA),
        "rubric": _rubric_block(),
    }
    user = (
        "Judge this case. Ground every rationale in a quote from model_output or expected_output.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def pairwise_prompt(case: dict[str, Any], order: str) -> list[dict[str, str]]:
    a, b = case["output_a"], case["output_b"]
    if order == "BA":
        a, b = b, a
    payload = {
        "input": case.get("input"),
        "system_prompt": case.get("system_prompt"),
        "expected_output": case.get("expected_output"),
        "answer_A": a,
        "answer_B": b,
        "rubric": _rubric_block(),
        "order_tag": order,
    }
    user = (
        "Compare A vs B. For each criterion score A and B independently (1-5), then set winner to A, B, or tie.\n"
        "JSON schema: {\"A\": {\"scores\": {criterion: {\"score\": n, \"rationale\": s}}}, "
        "\"B\": {\"scores\": {...}}, \"winner\": \"A\"|\"B\"|\"tie\", \"rationale\": s}\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM + "\nThis call is pairwise, not pointwise."},
        {"role": "user", "content": user},
    ]


def judge_pointwise(case: dict[str, Any], llm: LLMClient | None = None) -> dict[str, Any]:
    llm = llm or LLMClient("judge")
    messages = pointwise_prompt(case)
    raw = llm.chat(messages, max_tokens=700)
    verdict = parse_verdict(raw.text)
    if verdict["parse_method"] != "json":
        retry = llm.chat(
            messages
            + [
                {"role": "assistant", "content": raw.text},
                {"role": "user", "content": "Your previous reply was not valid JSON. Reply with JSON only."},
            ],
            max_tokens=700,
        )
        raw = retry
        verdict = parse_verdict(raw.text)
    overall = weighted_overall({k: v["score"] for k, v in verdict["scores"].items()})
    verdict["overall"] = round(overall, 3)
    verdict["pass"] = overall >= PASS_THRESHOLD
    record = {
        "id": case.get("id"),
        "mode": "pointwise",
        "verdict": verdict,
        "judge_model": raw.model,
        "tokens": {"prompt": raw.prompt_tokens, "completion": raw.completion_tokens, "total": raw.total_tokens},
        "latency_ms": raw.latency_ms,
        "raw_response": raw.text,
        "prompt": messages,
    }
    log_event(
        "judge",
        {
            "id": case.get("id"),
            "mode": "pointwise",
            "prompt": messages,
            "raw_response": raw.text,
            "tokens": record["tokens"],
            "latency_ms": raw.latency_ms,
            "judge_model": raw.model,
            "verdict": verdict,
        },
        "judge.jsonl",
    )
    return record


def _extract_side(data: dict[str, Any], key: str) -> dict[str, float]:
    side = data.get(key) or {}
    scores = side.get("scores") or side
    out: dict[str, float] = {}
    if isinstance(scores, dict):
        for name in CRITERIA:
            cell = scores.get(name)
            if isinstance(cell, dict):
                out[name] = float(cell.get("score") or 0)
            elif isinstance(cell, (int, float)):
                out[name] = float(cell)
    return out


def judge_pairwise(case: dict[str, Any], llm: LLMClient | None = None) -> dict[str, Any]:
    """Run both A/B orders; average scores; record flip rate contribution."""
    llm = llm or LLMClient("judge")
    orders = {}
    for order in ("AB", "BA"):
        messages = pairwise_prompt(case, order)
        raw = llm.chat(messages, max_tokens=800)
        parsed = parse_verdict(raw.text)  # may fail schema; also try full JSON
        blob = raw.text
        try:
            from src.judge.parse import _first_brace_object

            obj = json.loads(_first_brace_object(blob) or "{}")
        except (json.JSONDecodeError, TypeError):
            obj = {}
        winner = str(obj.get("winner") or "").upper()
        if order == "BA":
            winner = {"A": "B", "B": "A", "TIE": "TIE"}.get(winner, winner)
        a_scores = _extract_side(obj, "A")
        b_scores = _extract_side(obj, "B")
        if order == "BA":
            a_scores, b_scores = b_scores, a_scores
        orders[order] = {
            "winner_mapped_to_original": winner,
            "A_scores": a_scores,
            "B_scores": b_scores,
            "raw": raw.text,
            "tokens": {"prompt": raw.prompt_tokens, "completion": raw.completion_tokens},
            "latency_ms": raw.latency_ms,
            "parse": parsed["parse_method"],
            "prompt": messages,
            "judge_model": raw.model,
        }
        if not winner:
            a_overall = weighted_overall(a_scores)
            b_overall = weighted_overall(b_scores)
            if a_overall > b_overall:
                orders[order]["winner_mapped_to_original"] = "A"
            elif b_overall > a_overall:
                orders[order]["winner_mapped_to_original"] = "B"
            else:
                orders[order]["winner_mapped_to_original"] = "TIE"
        log_event(
            "judge_pairwise",
            {
                "id": case.get("id"),
                "order": order,
                "prompt": messages,
                "raw_response": raw.text,
                "winner": orders[order]["winner_mapped_to_original"],
            },
            "judge.jsonl",
        )

    w_ab = orders["AB"]["winner_mapped_to_original"]
    w_ba = orders["BA"]["winner_mapped_to_original"]
    flipped = w_ab != w_ba and "TIE" not in {w_ab, w_ba} and w_ab and w_ba
    if flipped:
        final_winner = "tie_on_flip"
    elif w_ab == w_ba:
        final_winner = w_ab or "tie"
    else:
        final_winner = w_ab or w_ba or "tie"

    def avg_side(key: str) -> dict[str, float]:
        acc: dict[str, list[float]] = {n: [] for n in CRITERIA}
        for order in orders.values():
            side = order["A_scores" if key == "A" else "B_scores"]
            for n, v in side.items():
                if v:
                    acc[n].append(v)
        return {n: (sum(vs) / len(vs) if vs else 0.0) for n, vs in acc.items()}

    a_avg, b_avg = avg_side("A"), avg_side("B")
    return {
        "id": case.get("id"),
        "mode": "pairwise",
        "orders": {k: {kk: vv for kk, vv in v.items() if kk != "prompt"} for k, v in orders.items()},
        "flipped": flipped,
        "winner": final_winner,
        "A_overall": weighted_overall(a_avg),
        "B_overall": weighted_overall(b_avg),
        "A_scores": a_avg,
        "B_scores": b_avg,
        "prompts": {k: v["prompt"] for k, v in orders.items()},
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    point = [r for r in records if r.get("mode") == "pointwise"]
    pair = [r for r in records if r.get("mode") == "pairwise"]
    scores = [r["verdict"]["overall"] for r in point]
    passes = [r["verdict"]["pass"] for r in point]
    return {
        "n_pointwise": len(point),
        "n_pairwise": len(pair),
        "pass_rate": (sum(passes) / len(passes)) if passes else None,
        "mean_score": (sum(scores) / len(scores)) if scores else None,
        "pairwise_win_counts": {
            "A": sum(1 for r in pair if str(r.get("winner")).upper().startswith("A")),
            "B": sum(1 for r in pair if str(r.get("winner")).upper().startswith("B")),
            "tie": sum(1 for r in pair if "TIE" in str(r.get("winner")).upper()),
        },
        "flip_rate": (sum(1 for r in pair if r.get("flipped")) / len(pair)) if pair else None,
    }
