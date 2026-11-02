"""FiQA-2018 (financial QA) from the BEIR benchmark.

Source: https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
Corpus: StackExchange posts under the Investment topic, 2009-2017 (BEIR paper, appx D.2).
Labels: binary, taken from FiQA's own question-to-document matches
(https://sites.google.com/view/fiqa/), not written for this harness.
BEIR sampled 500 dev and 648 test queries from FiQA's original training split.
"""
import csv
import json
import urllib.request
import zipfile
from pathlib import Path

from evaltypes import Corpus, Qrels, Queries

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


def _jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _qrels(path: Path) -> Qrels:
    out: Qrels = {}
    with open(path, encoding="utf-8", newline="") as f:
        rows = csv.reader(f, delimiter="\t")
        next(rows)  # header: query-id, corpus-id, score
        for qid, did, score in rows:
            out.setdefault(qid, {})[did] = int(score)
    return out


def load_fiqa(root: Path = Path("data")) -> tuple[Corpus, Queries, dict[str, Qrels]]:
    root = Path(root)
    if not (root / "fiqa").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, root / "fiqa.zip")
        with zipfile.ZipFile(root / "fiqa.zip") as z:
            z.extractall(root)
    d = root / "fiqa"
    corpus = {r["_id"]: {"title": r.get("title", ""), "text": r["text"]} for r in _jsonl(d / "corpus.jsonl")}
    queries = {r["_id"]: r["text"] for r in _jsonl(d / "queries.jsonl")}
    return corpus, queries, {s: _qrels(d / "qrels" / f"{s}.tsv") for s in ("dev", "test")}


if __name__ == "__main__":
    corpus, queries, qrels = load_fiqa()
    test = qrels["test"]
    print(f"corpus: {len(corpus)} docs, {len(queries)} queries")
    print(f"dev queries: {len(qrels['dev'])}, test queries: {len(test)}")
    print(f"avg relevant docs per test query: {sum(map(len, test.values())) / len(test):.2f}")
    qid = next(iter(test))
    did = next(iter(test[qid]))
    print(f"sample query {qid}: {queries[qid]}")
    print(f"relevant doc {did}: {corpus[did]['text'][:200]}")
