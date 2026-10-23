import json
from types import SimpleNamespace

from src.judge.pipeline import judge_pairwise
from src.metrics import cohen_kappa, exact_match, mrr, ndcg_at_k, token_f1
from src.rag.chunking import chunk_text
from src.judge.parse import parse_verdict


def test_mrr():
    assert abs(mrr([1, 2, None]) - (1 + 0.5 + 0) / 3) < 1e-9


def test_ndcg_perfect():
    assert abs(ndcg_at_k([1, 0, 0], 3) - 1.0) < 1e-9


def test_f1_and_em():
    assert exact_match("7 years", "7 years") == 1
    _, _, f1 = token_f1("7 years from posting", "7 years from the posting date")
    assert 0.4 < f1 < 1


def test_kappa_perfect():
    assert cohen_kappa([1, 0, 1], [1, 0, 1]) == 1


def test_chunk_overlap():
    text = ("alpha beta gamma. " * 80).strip()
    chunks = chunk_text(text, chunk_size=120, overlap=30)
    assert len(chunks) > 1
    assert chunks[0].index == 0


def test_parse_json_and_fallback():
    raw = '{"scores": {"correctness": {"score": 4, "rationale": "ok"}, "faithfulness": {"score": 5, "rationale": "x"}, "completeness": {"score": 3, "rationale": "x"}, "instruction_following": {"score": 4, "rationale": "x"}, "tone_safety": {"score": 5, "rationale": "x"}}, "overall": 4.2, "pass": true, "rationale": "fine"}'
    v = parse_verdict(raw)
    assert v["parse_method"] == "json"
    assert v["scores"]["correctness"]["score"] == 4
    messy = "correctness is 2 and faithfulness 1 completeness 1 instruction_following 2 tone_safety 3"
    v2 = parse_verdict(messy)
    assert v2["parse_method"] == "regex_fallback"
    assert v2["scores"]["correctness"]["score"] == 2


def test_parse_repairs_trailing_comma_without_corrupting_apostrophes():
    raw = '''{"scores": {"correctness": {"score": 3, "rationale": "The model output 'Growth is $199 per month.' misses an entity."}, "faithfulness": {"score": 5}, "completeness": {"score": 3}, "instruction_following": {"score": 1}, "tone_safety": {"score": 5},}, "overall": 3.2, "pass": false,}'''

    verdict = parse_verdict(raw)

    assert verdict["parse_method"] == "json"
    assert verdict["scores"]["correctness"]["score"] == 3
    assert "'Growth is $199 per month.'" in verdict["scores"]["correctness"]["rationale"]

    single_quoted = r"{'scores': {'correctness': {'score': 4, 'rationale': 'It\'s correct'},}, 'overall': 4, 'pass': true,}"
    repaired = parse_verdict(single_quoted)
    assert repaired["parse_method"] == "json"
    assert repaired["scores"]["correctness"]["rationale"] == "It's correct"


def test_pairwise_infers_missing_winner_from_scores(monkeypatch):
    monkeypatch.setattr("src.judge.pipeline.log_event", lambda *args, **kwargs: None)
    high = {name: {"score": 5} for name in (
        "correctness",
        "faithfulness",
        "completeness",
        "instruction_following",
        "tone_safety",
    )}
    low = {name: {"score": 1} for name in high}
    responses = [
        json.dumps({"A": {"scores": high, "winner": "A"}, "B": {"scores": low, "winner": "B"}}),
        json.dumps({"A": {"scores": low, "winner": "A"}, "B": {"scores": high, "winner": "B"}}),
    ]

    class FakeLLM:
        def chat(self, *args, **kwargs):
            return SimpleNamespace(
                text=responses.pop(0),
                model="test",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1.0,
            )

    result = judge_pairwise(
        {
            "id": "missing_winner",
            "input": "Which answer is correct?",
            "output_a": "Correct answer",
            "output_b": "Incorrect answer",
        },
        llm=FakeLLM(),
    )

    assert result["winner"] == "A"
    assert result["flipped"] is False
    assert result["A_overall"] > result["B_overall"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
