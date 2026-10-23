from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ROOT, get_settings
from src.judge.pipeline import aggregate, judge_pairwise, judge_pointwise, load_suite
from src.llm import LLMClient
from src.logging_utils import write_json
from src.metrics import cohen_kappa


def gold_bin(case: dict) -> int:
    return 1 if case.get("gold_pass") else 0


def pred_bin(rec: dict) -> int:
    return 1 if rec["verdict"]["pass"] else 0


def main() -> None:
    p = argparse.ArgumentParser(description="Run LLM-as-judge suite, bias checks, A/B, validation.")
    p.add_argument("--suite", default=str(ROOT / "data" / "eval" / "judge_suite.yaml"))
    p.add_argument("--pairs", default=str(ROOT / "data" / "eval" / "judge_pairwise.yaml"))
    p.add_argument("--retest", type=int, default=3)
    args = p.parse_args()
    s = get_settings()
    llm = LLMClient("judge")
    suite = load_suite(Path(args.suite))
    cases = suite["cases"]

    pointwise = [judge_pointwise(c, llm=llm) for c in cases]
    point_agg = aggregate(pointwise)

    gold_pass = [gold_bin(c) for c in cases]
    pred_pass = [pred_bin(r) for r in pointwise]
    agreement = sum(g == p for g, p in zip(gold_pass, pred_pass)) / len(cases)
    kappa = cohen_kappa(gold_pass, pred_pass)

    gold_overall = [float(c.get("gold_overall") or 0) for c in cases]
    pred_overall = [r["verdict"]["overall"] for r in pointwise]
    mae = sum(abs(a - b) for a, b in zip(gold_overall, pred_overall)) / len(cases)

    # Test-retest on the first N cases
    retest_n = min(args.retest, len(cases))
    retest = [judge_pointwise(c, llm=llm) for c in cases[:retest_n]]
    flips = 0
    for a, b in zip(pointwise[:retest_n], retest):
        if a["verdict"]["pass"] != b["verdict"]["pass"]:
            flips += 1
        elif abs(a["verdict"]["overall"] - b["verdict"]["overall"]) >= 1.0:
            flips += 1
    retest_flip = flips / max(1, retest_n)

    # Verbosity / length: compare terse_but_correct vs padded_correct vs verbose_but_wrong
    by_id = {c["id"]: r for c, r in zip(cases, pointwise)}
    terse = by_id.get("terse_but_correct", {}).get("verdict", {})
    padded = by_id.get("padded_correct", {}).get("verdict", {})
    verbose_wrong = by_id.get("verbose_but_wrong", {}).get("verdict", {})
    length_ok = True
    if terse and padded:
        length_ok = padded.get("overall", 5) <= terse.get("overall", 0) + 0.5
    fooled_verbose = bool(verbose_wrong.get("pass")) if verbose_wrong else None
    fooled_confident = bool(by_id.get("confidently_wrong", {}).get("verdict", {}).get("pass"))

    pairs_spec = load_suite(Path(args.pairs))
    pair_recs = [judge_pairwise(pcase, llm=llm) for pcase in pairs_spec["pairs"]]
    pair_agg = aggregate(pair_recs)
    gold_match = []
    for spec, rec in zip(pairs_spec["pairs"], pair_recs):
        gw = str(spec.get("gold_winner", "A")).upper()
        pw = str(rec.get("winner", "")).upper()
        gold_match.append(pw.startswith(gw[0]) if gw in {"A", "B"} else "TIE" in pw)

    # Config A vs B from pairwise averages (no extra LLM calls)
    a_scores = [r["A_overall"] for r in pair_recs]
    b_scores = [r["B_overall"] for r in pair_recs]
    mean_a = sum(a_scores) / max(1, len(a_scores))
    mean_b = sum(b_scores) / max(1, len(b_scores))
    pass_a = sum(1 for x in a_scores if x >= 3.5) / max(1, len(a_scores))
    pass_b = sum(1 for x in b_scores if x >= 3.5) / max(1, len(b_scores))
    wins_a = sum(1 for r in pair_recs if str(r.get("winner", "")).upper().startswith("A"))
    wins_b = sum(1 for r in pair_recs if str(r.get("winner", "")).upper().startswith("B"))
    winner = "Config A — grounded/terse" if mean_a >= mean_b else "Config B — chatty"
    agg_a = {"pass_rate": pass_a, "mean_score": mean_a}
    agg_b = {"pass_rate": pass_b, "mean_score": mean_b}

    # Score spread (clustering): stdev of overalls
    import statistics

    spread = statistics.pstdev(pred_overall) if len(pred_overall) > 1 else 0.0

    report = {
        "judge_model": s.judge_model,
        "generator_model": s.generator_model,
        "self_enhancement_note": (
            "Disk was too full to pull a second-family weights file. Judge and generator "
            "share llama3.2:3b but are configured independently (JUDGE_MODEL vs GENERATOR_MODEL). "
            "Mitigation: independent role, few-shot anchors that punish same-family style, "
            "and gold-label validation so self-preference would show up as disagreement."
        ),
        "pointwise": {
            "aggregate": point_agg,
            "cases": [
                {
                    "id": c["id"],
                    "overall": r["verdict"]["overall"],
                    "pass": r["verdict"]["pass"],
                    "gold_overall": c.get("gold_overall"),
                    "gold_pass": c.get("gold_pass"),
                    "parse": r["verdict"]["parse_method"],
                    "scores": r["verdict"]["scores"],
                }
                for c, r in zip(cases, pointwise)
            ],
        },
        "validation": {
            "n": len(cases),
            "agreement_with_gold_pass": round(agreement, 4),
            "cohen_kappa_pass": round(kappa, 4),
            "overall_mae": round(mae, 3),
            "test_retest_n": retest_n,
            "test_retest_flip_rate": round(retest_flip, 4),
            "adversarial": {
                "verbose_but_wrong_fooled": fooled_verbose,
                "terse_but_correct_overall": terse.get("overall"),
                "padded_correct_overall": padded.get("overall"),
                "length_not_rewarded": length_ok,
                "confidently_wrong_fooled": fooled_confident,
            },
        },
        "bias": {
            "position_flip_rate": pair_agg["flip_rate"],
            "pairwise_wins": pair_agg["pairwise_win_counts"],
            "pairwise_gold_agreement": round(sum(gold_match) / max(1, len(gold_match)), 4),
            "verbosity_length_not_rewarded": length_ok,
            "score_spread_pstdev": round(spread, 3),
        },
        "ab": {
            "config_a": {"name": "grounded/terse", "pass_rate": agg_a["pass_rate"], "mean_score": agg_a["mean_score"], "wins": wins_a},
            "config_b": {"name": "chatty/salesy", "pass_rate": agg_b["pass_rate"], "mean_score": agg_b["mean_score"], "wins": wins_b},
            "win_rate_a": round(wins_a / max(1, wins_a + wins_b), 4),
            "declared_winner": winner,
        },
        "pairwise_detail": [
            {"id": r["id"], "winner": r["winner"], "flipped": r["flipped"], "A": r["A_overall"], "B": r["B_overall"]}
            for r in pair_recs
        ],
    }
    out = Path(s.runs_dir) / "judge_eval.json"
    write_json(out, report)
    print(f"wrote {out}")
    print(
        {
            "agreement": report["validation"]["agreement_with_gold_pass"],
            "kappa": report["validation"]["cohen_kappa_pass"],
            "flip_rate": report["bias"]["position_flip_rate"],
            "winner": winner,
            "verbose_fooled": fooled_verbose,
        }
    )


if __name__ == "__main__":
    main()
