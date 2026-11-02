"""BM25, dense (bge), fusion (RRF / weighted) and cross-encoder rerank over evaltypes shapes."""
import re
import tempfile
import time
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from evaltypes import Corpus, Queries, Run

BGE_PREFIX = "Represent this sentence for searching relevant passages: "
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _text(d: dict[str, str]) -> str:
    return (d.get("title", "") + " " + d.get("text", "")).strip()


def _topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise indices of the top k scores, best first."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
    return np.take_along_axis(idx, order, 1)


class BM25:
    def __init__(self, corpus: Corpus):
        self.ids = list(corpus)
        self.stemmer = Stemmer.Stemmer("english")
        tokens = bm25s.tokenize([_text(corpus[i]) for i in self.ids], stopwords="en",
                                stemmer=self.stemmer, show_progress=False)
        self.index = bm25s.BM25()
        self.index.index(tokens, show_progress=False)

    def search(self, queries: Queries, k: int = 100) -> Run:
        qids = list(queries)
        qt = bm25s.tokenize([queries[q] for q in qids], stopwords="en", stemmer=self.stemmer,
                            return_ids=False, show_progress=False)
        k = min(k, len(self.ids))
        docs, scores = self.index.retrieve(qt, k=k, show_progress=False)
        return {q: [(self.ids[d], float(s)) for d, s in zip(dr, sr)]
                for q, dr, sr in zip(qids, docs, scores)}


class Dense:
    def __init__(self, corpus: Corpus, model: str = "BAAI/bge-small-en-v1.5", cache: Path = Path("cache")):
        self.model = SentenceTransformer(model, device=DEVICE)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        emb_f, ids_f = Path(cache) / f"{safe}.npy", Path(cache) / f"{safe}.ids.txt"
        ids = list(corpus)
        # ponytail: cache keyed by model + doc-id order only; delete cache/ if corpus text changes
        if emb_f.exists() and ids_f.exists() and ids_f.read_text().split("\n") == ids:
            self.emb, self.loaded = np.load(emb_f), True
        else:
            self.emb = self.model.encode([_text(corpus[i]) for i in ids], batch_size=64,
                                         normalize_embeddings=True, show_progress_bar=len(ids) > 1000)
            Path(cache).mkdir(parents=True, exist_ok=True)
            np.save(emb_f, self.emb)
            ids_f.write_text("\n".join(ids))
            self.loaded = False
        self.ids = ids

    def search(self, queries: Queries, k: int = 100) -> Run:
        qids = list(queries)
        q = self.model.encode([BGE_PREFIX + queries[i] for i in qids], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=False)
        scores = q @ self.emb.T
        top = _topk(scores, k)
        return {qid: [(self.ids[j], float(scores[r, j])) for j in top[r]] for r, qid in enumerate(qids)}


def rrf(runs: list[Run], k: int = 60, top: int = 100) -> Run:
    out = {}
    for qid in runs[0]:
        s: dict[str, float] = {}
        for run in runs:
            for rank, (d, _) in enumerate(run.get(qid, []), 1):
                s[d] = s.get(d, 0.0) + 1.0 / (k + rank)
        out[qid] = sorted(s.items(), key=lambda x: -x[1])[:top]
    return out


def _norm(hits: list[tuple[str, float]]) -> dict[str, float]:
    if not hits:
        return {}
    lo, hi = min(s for _, s in hits), max(s for _, s in hits)
    return {d: (s - lo) / (hi - lo) if hi > lo else 1.0 for d, s in hits}


def weighted(sparse: Run, dense: Run, alpha: float, top: int = 100) -> Run:
    out = {}
    for qid in sparse:
        sp, de = _norm(sparse[qid]), _norm(dense.get(qid, []))
        s = {d: alpha * de.get(d, 0.0) + (1 - alpha) * sp.get(d, 0.0) for d in sp.keys() | de.keys()}
        out[qid] = sorted(s.items(), key=lambda x: -x[1])[:top]
    return out


class Reranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model, device=DEVICE)

    def rerank(self, run: Run, queries: Queries, corpus: Corpus, top: int = 50) -> Run:
        pairs, spans = [], {}
        for qid, hits in run.items():
            hits = hits[:top]
            spans[qid] = (len(pairs), hits)
            pairs += [(queries[qid], _text(corpus[d])) for d, _ in hits]
        scores = self.model.predict(pairs, batch_size=64, show_progress_bar=False) if pairs else []
        out = {}
        for qid, (start, hits) in spans.items():
            sc = scores[start:start + len(hits)]
            out[qid] = sorted(((d, float(s)) for (d, _), s in zip(hits, sc)), key=lambda x: -x[1])
        return out


if __name__ == "__main__":
    corpus = {
        "budget": {"title": "", "text": "How to build a monthly household budget and track every expense in a spreadsheet."},
        "mortgage": {"title": "", "text": "Fixed rate mortgage versus variable rate home loan: which has lower interest repayments."},
        "index": {"title": "", "text": "Index funds and ETFs give cheap diversified exposure to the stock market."},
        "tax": {"title": "", "text": "Filing your annual tax return: deductions, refunds and the deadline to lodge."},
        "credit": {"title": "", "text": "Credit card interest, late fees and how to pay off your balance."},
        "super": {"title": "", "text": "Superannuation retirement fund contributions and employer super guarantee."},
        "insurance": {"title": "", "text": "Choosing car, home and health insurance policies and comparing premiums."},
        "car": {"title": "", "text": "Saving up a down payment to buy a new car without taking a loan."},
    }
    queries = {"q1": "mortgage interest rate home loan", "q2": "index fund stock market ETF",
               "q3": "tax return refund deductions"}
    want = {"q1": "mortgage", "q2": "index", "q3": "tax"}
    top1 = lambda run: {q: h[0][0] for q, h in run.items()}

    bm = BM25(corpus).search(queries, k=5)
    with tempfile.TemporaryDirectory() as tmp:
        t = time.time()
        d = Dense(corpus, cache=Path(tmp))
        dt = time.time() - t
        assert not d.loaded
        d2 = Dense(corpus, cache=Path(tmp))
        assert d2.loaded and np.allclose(d.emb, d2.emb)
    print(f"device={DEVICE}; load+encode {len(corpus)} docs in {dt:.2f}s ({len(corpus)/dt:.1f} docs/s incl. model load)")
    t = time.time(); de = d.search(queries, k=5); print(f"dense search 3 queries {time.time()-t:.3f}s")
    one = d.search({"q1": queries["q1"]}, k=5)
    assert one["q1"][0][0] == de["q1"][0][0]

    r = rrf([bm, de], top=5)
    w = weighted(bm, de, 0.5, top=5)
    rr = Reranker().rerank(r, queries, corpus, top=5)
    for name, run in [("bm25", bm), ("dense", de), ("rrf", r), ("weighted", w), ("rerank", rr)]:
        assert top1(run) == want, (name, top1(run))
        for hits in run.values():
            assert len(hits) == 5 and [s for _, s in hits] == sorted((s for _, s in hits), reverse=True), name
    bm, de = BM25(corpus).search(queries, k=8), d.search(queries, k=8)  # full lists: no truncation ties
    ids = lambda run: {q: [x for x, _ in h] for q, h in run.items()}
    # alpha=0 follows BM25 order (BM25 itself has ties/zeros, so compare BM25 scores along the fused order); alpha=1 is dense order
    for q, order in ids(weighted(bm, de, 0.0, top=8)).items():
        sc = [dict(bm[q]).get(x, 0.0) for x in order]
        assert sc == sorted(sc, reverse=True), q
    assert ids(weighted(bm, de, 1.0, top=8)) == ids(de)
    print("ok")
