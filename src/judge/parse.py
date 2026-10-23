from __future__ import annotations

import json
import re
from typing import Any

from src.judge.rubric import CRITERIA, weighted_overall


SCORE_RE = re.compile(
    r"(correctness|faithfulness|completeness|instruction_following|tone_safety)[^0-9]{0,12}([1-5](?:\.\d)?)",
    re.I,
)


def parse_verdict(raw: str) -> dict[str, Any]:
    """Parse a structured verdict; recover from markdown fences and partial JSON."""
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    blob = _first_brace_object(text)
    data: dict[str, Any] | None = None
    if blob:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            repaired = re.sub(r",\s*([}\]])", r"\1", blob)
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                try:
                    data = json.loads(_single_quote_strings_to_json(repaired))
                except json.JSONDecodeError:
                    data = None
    if data is None:
        data = _regex_fallback(raw)
        data["parse_method"] = "regex_fallback"
    else:
        data["parse_method"] = "json"

    scores: dict[str, dict[str, Any]] = {}
    raw_scores = data.get("scores") or data.get("criteria") or {}
    if isinstance(raw_scores, dict):
        for name in CRITERIA:
            cell = raw_scores.get(name) or {}
            if isinstance(cell, (int, float)):
                score, rationale = float(cell), ""
            else:
                score = float(cell.get("score", cell.get("rating", 0)) or 0)
                rationale = str(cell.get("rationale") or cell.get("reason") or "")
            scores[name] = {"score": _clip(score), "rationale": rationale}
    overall = data.get("overall")
    try:
        overall_f = float(overall)
    except (TypeError, ValueError):
        overall_f = weighted_overall({k: v["score"] for k, v in scores.items()})
    passed = data.get("pass")
    if not isinstance(passed, bool):
        passed = overall_f >= 3.5
    return {
        "scores": scores,
        "overall": round(_clip(overall_f), 3),
        "pass": bool(passed),
        "rationale": str(data.get("rationale") or data.get("summary") or ""),
        "parse_method": data.get("parse_method", "json"),
        "raw_excerpt": raw[:2000],
    }


def _clip(x: float) -> float:
    return max(1.0, min(5.0, x)) if x else 0.0


def _first_brace_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _single_quote_strings_to_json(text: str) -> str:
    out = []
    in_double = False
    in_single = False
    escaped = False
    for char in text:
        if in_double:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_double = False
        elif in_single:
            if escaped:
                out.append("'" if char == "'" else "\\" + char)
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == "'":
                out.append('"')
                in_single = False
            elif char == '"':
                out.append('\\"')
            else:
                out.append(char)
        elif char == '"':
            out.append(char)
            in_double = True
        elif char == "'":
            out.append('"')
            in_single = True
        else:
            out.append(char)
    if escaped:
        out.append("\\")
    return "".join(out)


def extract_winner(raw: str) -> str:
    """Last winner label in the blob. Survives truncated JSON from small models."""
    matches = re.findall(r'"winner"\s*:\s*"?(A|B|TIE|tie)"?', raw, re.I)
    if not matches:
        return ""
    return matches[-1].upper()


def _regex_fallback(raw: str) -> dict[str, Any]:
    found = {m.group(1).lower(): float(m.group(2)) for m in SCORE_RE.finditer(raw)}
    scores = {name: found.get(name, 0.0) for name in CRITERIA}
    return {
        "scores": {k: {"score": v, "rationale": "regex fallback"} for k, v in scores.items()},
        "overall": weighted_overall(scores),
        "pass": weighted_overall(scores) >= 3.5,
        "rationale": "Recovered scores from unstructured judge text.",
    }
