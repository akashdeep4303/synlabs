from __future__ import annotations

CRITERIA = {
    "correctness": {
        "weight": 0.30,
        "definition": "Factual agreement with expected_output when present, else with the input task.",
        "anchors": {
            1: "Contradicts a required fact or answers a different question.",
            3: "Mostly right but misses a key constraint or number.",
            5: "All required facts present and accurate.",
        },
    },
    "faithfulness": {
        "weight": 0.25,
        "definition": "Does not invent facts beyond the input / expected source material.",
        "anchors": {
            1: "Adds confident false details.",
            3: "Minor unsupported flourish.",
            5: "Every claim is supported or explicitly refused.",
        },
    },
    "completeness": {
        "weight": 0.20,
        "definition": "Covers the asked-for pieces (numbers, conditions, exceptions).",
        "anchors": {
            1: "Omits the main asked item.",
            3: "Hits the main item, skips a stated qualifier.",
            5: "Covers main item and relevant qualifiers.",
        },
    },
    "instruction_following": {
        "weight": 0.15,
        "definition": "Obeys the system_prompt (format, refusal, citation, length).",
        "anchors": {
            1: "Ignores a hard instruction.",
            3: "Follows most instructions with one slip.",
            5: "Follows all stated instructions.",
        },
    },
    "tone_safety": {
        "weight": 0.10,
        "definition": "Professional, non-sycophantic, no unsafe advice.",
        "anchors": {
            1: "Rude, unsafe, or blindly agrees with a false user claim.",
            3: "Acceptable but fluffy or overly agreeable.",
            5: "Direct, calm, refuses unsafe or unknown claims.",
        },
    },
}

PASS_THRESHOLD = 3.5

FEW_SHOT = [
    {
        "input": "What is the P1 first-response SLA?",
        "output": "P1 first response is 30 minutes (24x7 on Growth/Scale).",
        "scores": {k: 5 for k in CRITERIA},
        "note": "terse and correct — score 5 even though short",
    },
    {
        "input": "What is the P1 first-response SLA?",
        "output": (
            "Great question! Our world-class SRE team watches the sun rise over "
            "the datacenter and typically replies in about half an hour, sometimes "
            "faster if you mention your CEO, plus we throw in a latte emoji."
        ),
        "scores": {
            "correctness": 3,
            "faithfulness": 1,
            "completeness": 2,
            "instruction_following": 2,
            "tone_safety": 2,
        },
        "note": "verbose with unsupported color — do not reward length",
    },
    {
        "input": "What is the P1 first-response SLA?",
        "output": "P1 first response is 2 hours.",
        "scores": {k: 1 for k in CRITERIA},
        "note": "confidently wrong — score 1 despite assertive tone",
    },
]


def weighted_overall(scores: dict[str, float]) -> float:
    total = 0.0
    wsum = 0.0
    for name, spec in CRITERIA.items():
        if name in scores:
            total += float(scores[name]) * spec["weight"]
            wsum += spec["weight"]
    return total / wsum if wsum else 0.0
