"""Retrieval metrics (recall/MRR/nDCG @k) and bootstrap confidence intervals."""
import math

import numpy as np

from evaltypes import Qrels, Run


def per_query(run: Run, qrels: Qrels, k: int = 10) -> dict[str, dict[str, float]]:
    """Score every query in qrels; a query missing from the run scores 0.
    Only grade > 0 counts as relevant. Ranking is the run order as given."""
    out = {}
    for qid, grades in qrels.items():
        rel = {d: g for d, g in grades.items() if g > 0}
        top = [d for d, _ in run.get(qid, [])[:k]]
        hits = [d for d in top if d in rel]
        first = next((i + 1 for i, d in enumerate(top) if d in rel), None)
        dcg = sum(rel[d] / math.log2(i + 2) for i, d in enumerate(top) if d in rel)
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        out[qid] = {
            f"recall@{k}": len(hits) / len(rel) if rel else 0.0,
            f"mrr@{k}": 1 / first if first else 0.0,
            f"ndcg@{k}": dcg / idcg if idcg else 0.0,
        }
    return out


def mean(pq: dict[str, dict[str, float]]) -> dict[str, float]:
    """Macro average of each metric over queries."""
    keys = next(iter(pq.values())).keys() if pq else []
    return {m: float(np.mean([v[m] for v in pq.values()])) for m in keys}


def bootstrap_ci(values, n: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean over queries."""
    v = np.asarray(values, dtype=float)
    idx = np.random.default_rng(seed).integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [50 * (1 - level), 100 - 50 * (1 - level)])
    return float(lo), float(hi)


def paired_bootstrap(a, b, n: int = 10_000, seed: int = 0) -> dict:
    """Paired bootstrap of mean(b)-mean(a), same resampled queries for both.
    p = fraction of resamples with delta <= 0 (one-sided, H1: B > A)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape, "a and b must be per-query, same order"
    d = b - a
    idx = np.random.default_rng(seed).integers(0, len(d), size=(n, len(d)))
    deltas = d[idx].mean(axis=1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta": float(d.mean()), "ci": (float(lo), float(hi)), "p": float((deltas <= 0).mean())}


if __name__ == "__main__":
    close = lambda x, y: abs(x - y) < 1e-12
    L = math.log2

    # q1: relevant {a, c}; run a,x,c,y -> rel at ranks 1 and 3.
    #   recall = 2/2 = 1; mrr = 1/1 = 1
    #   dcg = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    #   idcg = 1/log2(2) + 1/log2(3) = 1 + 0.6309297... ; ndcg = 1.5 / (1 + 1/log2(3))
    # q2: relevant {z}; run has none -> all 0
    # q3: relevant {m}; missing from run -> all 0
    # q4: graded {a:3, b:1}; run b,a. dcg = 1/1 + 3/log2(3); idcg = 3/1 + 1/log2(3)
    # q5: relevant {a}, at rank 2 beyond k=1 -> recall 0; grade-0 doc ignored
    run = {
        "q1": [("a", 3.0), ("x", 2.0), ("c", 1.0), ("y", 0.5)],
        "q2": [("p", 1.0), ("q", 0.5)],
        "q4": [("b", 2.0), ("a", 1.0)],
        "q5": [("x", 2.0), ("a", 1.0)],
    }
    qrels = {"q1": {"a": 1, "c": 1}, "q2": {"z": 1}, "q3": {"m": 1},
             "q4": {"a": 3, "b": 1, "n": 0}, "q5": {"a": 1, "g0": 0}}
    pq = per_query(run, qrels)
    assert set(pq) == set(qrels)
    r = pq["q1"]
    assert close(r["recall@10"], 1) and close(r["mrr@10"], 1)
    assert close(r["ndcg@10"], 1.5 / (1 + 1 / L(3)))
    assert pq["q2"] == {"recall@10": 0, "mrr@10": 0, "ndcg@10": 0}
    assert pq["q3"] == {"recall@10": 0, "mrr@10": 0, "ndcg@10": 0}
    r = pq["q4"]
    assert close(r["recall@10"], 1) and close(r["mrr@10"], 1)
    assert close(r["ndcg@10"], (1 + 3 / L(3)) / (3 + 1 / L(3)))
    assert per_query(run, qrels, k=1)["q5"] == {"recall@1": 0, "mrr@1": 0, "ndcg@1": 0}
    assert close(per_query(run, qrels)["q5"]["mrr@10"], 0.5)
    m = mean(pq)
    assert close(m["mrr@10"], (1 + 0 + 0 + 1 + 0.5) / 5)

    rng = np.random.default_rng(1)
    a = rng.random(200)
    lo, hi = bootstrap_ci(a)
    assert lo < a.mean() < hi and bootstrap_ci(a) == (lo, hi)
    s = paired_bootstrap(a, a)
    assert s["delta"] == 0 and s["ci"] == (0.0, 0.0) and s["p"] == 1.0
    s = paired_bootstrap(a, a + 0.1)
    assert close(s["delta"], 0.1) and s["p"] < 0.001 and s["ci"][0] > 0
    print("ok")
