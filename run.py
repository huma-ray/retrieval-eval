"""Runs the retrieval ladder on FiQA-2018 and writes results/ and RESULTS.md.

Every choice (fusion method, hybrid weight) is made on the dev split.
The test split is only scored, never used to pick anything.
"""
import csv
import json
import platform
import statistics
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import torch

from data import load_fiqa
from metrics import bootstrap_ci, mean, paired_bootstrap, per_query
from retrievers import BM25, Dense, Reranker, rrf, weighted

OUT = Path("results")
ALPHAS = [round(a / 10, 1) for a in range(11)]
METRICS = ["recall@10", "mrr@10", "ndcg@10"]


def ndcg(run, qrels):
    return mean(per_query(run, qrels))["ndcg@10"]


def timed(fn):
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def main():
    corpus, queries, qrels = load_fiqa()
    dev_q = {q: queries[q] for q in qrels["dev"]}
    test_q = {q: queries[q] for q in qrels["test"]}
    bm25, dense, reranker = BM25(corpus), Dense(corpus), Reranker()

    # --- choices, on dev only
    dev_sparse, dev_dense = bm25.search(dev_q), dense.search(dev_q)
    alpha_dev = {a: ndcg(weighted(dev_sparse, dev_dense, a), qrels["dev"]) for a in ALPHAS}
    alpha = max(alpha_dev, key=alpha_dev.get)
    rrf_dev = ndcg(rrf([dev_sparse, dev_dense]), qrels["dev"])
    fusion = "weighted" if alpha_dev[alpha] >= rrf_dev else "rrf"
    dev_hybrid = weighted(dev_sparse, dev_dense, alpha) if fusion == "weighted" else rrf([dev_sparse, dev_dense])
    rerank_dev = ndcg(reranker.rerank(dev_hybrid, dev_q, corpus), qrels["dev"])
    hybrid_dev = max(alpha_dev[alpha], rrf_dev)
    keep_rerank = rerank_dev > hybrid_dev

    # --- the ladder, on test
    sparse, dense_run = bm25.search(test_q), dense.search(test_q)
    hybrid_rrf = rrf([sparse, dense_run])
    hybrid_w = weighted(sparse, dense_run, alpha)
    hybrid = hybrid_w if fusion == "weighted" else hybrid_rrf
    reranked = reranker.rerank(hybrid, test_q, corpus)
    fused_name = f"Hybrid weighted (alpha={alpha})" if fusion == "weighted" else "Hybrid RRF"
    configs = {
        "BM25": sparse,
        "Dense (bge-small-en-v1.5)": dense_run,
        "Hybrid RRF": hybrid_rrf,
        f"Hybrid weighted (alpha={alpha})": hybrid_w,
        f"{fused_name} + rerank": reranked,
    }
    pq = {name: per_query(run, qrels["test"]) for name, run in configs.items()}
    qids = sorted(qrels["test"])

    # --- per-query latency, one query at a time, stages in sequence
    stage = {"bm25": [], "dense": [], "fuse": [], "rerank": []}
    for q in [qids[0]] + qids:  # first call is a warm-up, dropped below
        one = {q: test_q[q]}
        s, d = {}, {}
        stage["bm25"].append(timed(lambda: s.update(bm25.search(one))))
        stage["dense"].append(timed(lambda: d.update(dense.search(one))))
        fused = {}
        stage["fuse"].append(timed(lambda: fused.update(
            weighted(s, d, alpha) if fusion == "weighted" else rrf([s, d]))))
        stage["rerank"].append(timed(lambda: reranker.rerank(fused, one, corpus)))
    stage = {k: v[1:] for k, v in stage.items()}
    paths = {
        "BM25": ["bm25"],
        "Dense (bge-small-en-v1.5)": ["dense"],
        "Hybrid RRF": ["bm25", "dense", "fuse"],
        f"Hybrid weighted (alpha={alpha})": ["bm25", "dense", "fuse"],
        f"{fused_name} + rerank": ["bm25", "dense", "fuse", "rerank"],
    }

    def latency(parts):
        totals = sorted(sum(stage[p][i] for p in parts) for i in range(len(qids)))
        return {"p50_ms": statistics.median(totals), "p95_ms": totals[int(0.95 * (len(totals) - 1))]}

    # --- summaries
    def column(name, metric):
        return [pq[name][q][metric] for q in qids]

    summary = {
        name: {
            **{m: {"mean": statistics.fmean(column(name, m)), "ci95": bootstrap_ci(column(name, m))} for m in METRICS},
            **latency(paths[name]),
        }
        for name in configs
    }
    names = list(configs)
    shipped = names[4] if keep_rerank else fused_name
    steps = [
        ("BM25 -> Dense", names[0], names[1]),
        (f"Dense -> {fused_name}", names[1], fused_name),
        (f"{fused_name} -> + rerank", fused_name, names[4]),
        (f"BM25 -> {shipped} (before / after)", names[0], shipped),
    ]
    paired = [
        {"change": label, **{m: paired_bootstrap(column(a, m), column(b, m)) for m in ("ndcg@10", "recall@10")}}
        for label, a, b in steps
    ]
    moved = [pq[names[4]][q]["ndcg@10"] - pq[fused_name][q]["ndcg@10"] for q in qids]
    rerank_effect = {
        "helped": sum(x > 1e-9 for x in moved),
        "hurt": sum(x < -1e-9 for x in moved),
        "unchanged": sum(abs(x) <= 1e-9 for x in moved),
    }

    results = {
        "dataset": {"name": "BEIR FiQA-2018", "docs": len(corpus), "dev_queries": len(dev_q), "test_queries": len(test_q)},
        "dev_choices": {"alpha_ndcg@10": alpha_dev, "rrf_ndcg@10": rrf_dev, "alpha": alpha, "fusion": fusion,
                        "hybrid_ndcg@10": hybrid_dev, "rerank_ndcg@10": rerank_dev, "keep_rerank": keep_rerank,
                        "shipped": shipped},
        "test": summary,
        "paired": paired,
        "rerank_effect": rerank_effect,
        "environment": {
            "date": date.today().isoformat(),
            "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "packages": {p: version(p) for p in ("bm25s", "sentence-transformers", "torch", "numpy")},
            "models": {"dense": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "per_query.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query"] + [f"{n} | {m}" for n in names for m in ("ndcg@10", "recall@10")])
        for q in qids:
            w.writerow([q, test_q[q]] + [round(pq[n][q][m], 4) for n in names for m in ("ndcg@10", "recall@10")])
    Path("RESULTS.md").write_text(render(results))
    print(Path("RESULTS.md").read_text())


def render(r):
    def ci(x):
        return f"{x['mean']:.3f} [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"

    def delta(x):
        # 10,000 resamples: a p of 0 means no resample went the other way, i.e. p < 0.0001
        return f"{x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]", f"{x['p']:.4f}" if x["p"] else "< 0.0001"

    d, env, dc = r["dataset"], r["environment"], r["dev_choices"]
    lines = [
        "# Results",
        "",
        f"Generated by `run.py` on {env['date']}. Do not edit by hand.",
        "",
        f"{d['name']}: {d['docs']:,} documents, {d['test_queries']} test queries (scored), "
        f"{d['dev_queries']} dev queries (used only to choose settings).",
        "",
        "## Test split",
        "",
        "Mean over queries, 95% bootstrap confidence interval in brackets.",
        "",
        "| Config | Recall@10 | MRR@10 | nDCG@10 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in r["test"].items():
        lines.append(f"| {name} | {ci(s['recall@10'])} | {ci(s['mrr@10'])} | {ci(s['ndcg@10'])} "
                     f"| {s['p50_ms']:.0f} | {s['p95_ms']:.0f} |")
    lines += [
        "",
        "## Is each step a real improvement?",
        "",
        "Paired bootstrap over the same test queries. p = share of resamples where the change did not help.",
        "",
        "| Change | Δ nDCG@10 | p | Δ Recall@10 | p |",
        "|---|---|---|---|---|",
    ]
    for p in r["paired"]:
        n, n_p = delta(p["ndcg@10"])
        rc, rc_p = delta(p["recall@10"])
        lines.append(f"| {p['change']} | {n} | {n_p} | {rc} | {rc_p} |")
    e = r["rerank_effect"]
    lines += [
        "",
        f"Reranking changed nDCG@10 for {e['helped'] + e['hurt']} of {d['test_queries']} queries: "
        f"{e['helped']} better, {e['hurt']} worse, {e['unchanged']} unchanged. "
        "Per-query scores are in `results/per_query.csv`.",
        "",
        "## Settings chosen on dev",
        "",
        f"RRF dev nDCG@10: {dc['rrf_ndcg@10']:.3f}. Chosen: **{dc['fusion']}**"
        + (f", alpha = {dc['alpha']}" if dc["fusion"] == "weighted" else "") + ".",
        "",
        f"Rerank on dev: nDCG@10 {dc['hybrid_ndcg@10']:.3f} without, {dc['rerank_ndcg@10']:.3f} with. "
        f"Reranker **{'kept' if dc['keep_rerank'] else 'rejected'}**. Shipped config: **{dc['shipped']}**.",
        "",
        "| alpha (weight on dense) | dev nDCG@10 |",
        "|---|---|",
        *[f"| {a} | {v:.3f} |" for a, v in dc["alpha_ndcg@10"].items()],
        "",
        "## Environment",
        "",
        f"- Machine: {env['machine']}, device: {env['device']}",
        f"- Models: dense `{env['models']['dense']}`, reranker `{env['models']['reranker']}`",
        "- Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
        "- Latency: one query at a time, stages run in sequence in one process, query encoding included.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
