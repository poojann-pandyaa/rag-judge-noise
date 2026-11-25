#!/usr/bin/env python3
"""Week-1 setup: download pinned CLAPNQ dev data + passage corpus, print stats,
freeze a seeded, stratified eval set. Stdlib only (Python 3.9+).

    python3 setup_clapnq.py [--n 250] [--answerable-frac 0.7] [--drop-gold-frac 0.2] [--seed 1234]

Outputs (under ./data):
    raw/*                downloaded files (skipped if already present)
    eval_set.jsonl       the frozen eval set -- never regenerate after week 1
    manifest.json        upstream commit, sha256 of every file, sampling params
"""
import argparse, csv, hashlib, io, json, random, re, statistics, sys, urllib.request, zipfile
from pathlib import Path

# Pinned upstream commit (primeqa/clapnq, 2025-01-21). Bump only deliberately.
COMMIT = "c508d78d8c5c2eecbf172fcf5f789004206e11fb"
BASE = f"https://raw.githubusercontent.com/primeqa/clapnq/{COMMIT}/"
FILES = [
    "annotated_data/dev/clapnq_dev_answerable.jsonl",
    "annotated_data/dev/clapnq_dev_unanswerable.jsonl",
    "retrieval/dev/question_dev_answerable.tsv",
    "retrieval/passages.tsv.zip",
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel, raw):
    dst = raw / Path(rel).name
    if not dst.exists():
        print(f"downloading {rel}")
        urllib.request.urlretrieve(BASE + rel, dst)
    return dst


def jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--answerable-frac", type=float, default=0.7)
    ap.add_argument("--drop-gold-frac", type=float, default=0.2,
                    help="fraction of sampled answerable items flagged drop_gold "
                         "(pipeline removes the gold passage from the index for them)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    out = Path(a.out); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    paths = {rel: fetch(rel, raw) for rel in FILES}
    p_ans, p_una, p_q, p_zip = (paths[r] for r in FILES)

    ans, una = jsonl(p_ans), jsonl(p_una)
    with zipfile.ZipFile(p_zip) as z:
        name = next(n for n in z.namelist() if n.endswith(".tsv"))
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8"), delimiter="\t"))
    assert {"id", "text", "title"} <= set(rows[0]), f"unexpected corpus columns: {list(rows[0])}"
    corpus_ids = {r["id"] for r in rows}

    with open(p_q, encoding="utf-8") as f:
        gold_ids = {r["id"]: [g for g in re.split(r"[,\s]+", r["doc-id-list"]) if g]
                    for r in csv.DictReader(f, delimiter="\t")}

    # ---- stats (report these in the paper's testbed section) ----
    print(f"\ndev answerable={len(ans)} unanswerable={len(una)} (expect 300/300)")
    print(f"corpus passages={len(rows)} unique ids={len(corpus_ids)} "
          f"median words={statistics.median(len(r['text'].split()) for r in rows):.0f}")
    covered = sum(all(g in corpus_ids for g in gold_ids.get(x["id"], [])) and bool(gold_ids.get(x["id"]))
                  for x in ans)
    print(f"answerable items with all gold passages present in corpus: {covered}/{len(ans)}")

    def gold_answers(x):
        return [o["answer"] for o in x["output"] if o["answer"].strip()]

    ans = [x for x in ans if gold_answers(x) and gold_ids.get(x["id"])]  # drop skipped/unmapped items
    print(f"answerable usable after filtering: {len(ans)}")
    print(f"gold answer words median={statistics.median(len(g.split()) for x in ans for g in gold_answers(x)):.0f}")

    # ---- frozen eval set ----
    rng = random.Random(a.seed)
    n_ans = round(a.n * a.answerable_frac); n_una = a.n - n_ans
    assert n_ans <= len(ans) and n_una <= len(una), "requested sample larger than pool"
    s_ans = rng.sample(sorted(ans, key=lambda x: x["id"]), n_ans)
    s_una = rng.sample(sorted(una, key=lambda x: x["id"]), n_una)
    dropped = set(rng.sample([x["id"] for x in s_ans], round(n_ans * a.drop_gold_frac)))

    recs = [dict(id=x["id"], question=x["input"], answerable=True, gold_answers=gold_answers(x),
                 gold_passage_ids=gold_ids[x["id"]], drop_gold=x["id"] in dropped) for x in s_ans]
    recs += [dict(id=x["id"], question=x["input"], answerable=False, gold_answers=[],
                  gold_passage_ids=[], drop_gold=False) for x in s_una]
    rng.shuffle(recs)
    ev = out / "eval_set.jsonl"
    ev.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")
    print(f"\nwrote {ev}: {len(recs)} items ({n_ans} answerable, {n_una} unanswerable, {len(dropped)} drop_gold)")

    (out / "manifest.json").write_text(json.dumps(dict(
        upstream=f"https://github.com/primeqa/clapnq @ {COMMIT}",
        params=vars(a),
        sha256={str(p.name): sha256(p) for p in [*paths.values(), ev]},
    ), indent=1))
    print("wrote manifest.json")


if __name__ == "__main__":
    sys.exit(main())
