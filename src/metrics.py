from __future__ import annotations

import math
from typing import Iterable, Sequence


def hit_rate(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None) / len(ranks)


def mean_recall(recalls: Sequence[float]) -> float:
    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def mrr(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        total += 0.0 if r is None else 1.0 / r
    return total / len(ranks)


def dcg(gains: Sequence[float]) -> float:
    score = 0.0
    for i, g in enumerate(gains, 1):
        score += g / math.log2(i + 1)
    return score


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    rel = list(relevances)[:k]
    ideal = sorted(rel, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(rel) / denom


def mean_ndcg(per_query: Sequence[float]) -> float:
    if not per_query:
        return 0.0
    return sum(per_query) / len(per_query)


def context_precision(n_relevant: int, n_retrieved: int) -> float:
    if n_retrieved == 0:
        return 0.0
    return n_relevant / n_retrieved


def percentile(values: Iterable[float], p: float) -> float:
    xs = sorted(values)
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def token_f1(pred: str, gold: str) -> tuple[float, float, float]:
    p = pred.lower().split()
    g = gold.lower().split()
    if not p and not g:
        return 1.0, 1.0, 1.0
    if not p or not g:
        return 0.0, 0.0, 0.0
    from collections import Counter

    pc, gc = Counter(p), Counter(g)
    overlap = sum((pc & gc).values())
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(gc.values()))
    if precision + recall == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def exact_match(pred: str, gold: str) -> float:
    def norm(s: str) -> str:
        return " ".join(s.lower().split())

    return 1.0 if norm(pred) == norm(gold) else 0.0


def cohen_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    labels = sorted(set(a) | set(b))
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)
