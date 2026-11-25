#!/usr/bin/env python3
"""Faithfulness VALIDITY check against human labels (RAGTruth QA task, MS MARCO passages).

    python3 ragtruth.py selftest
    python3 ragtruth.py prepare [--n 150]        # -> data/ragtruth_eval.jsonl  (needs data/raw/ragtruth/*)
    python3 ragtruth.py run --judge qwen2.5-7b   # -> data/judgments/ragtruth_<judge>.jsonl   (or --judge all)
    python3 ragtruth.py report                   # -> data/results/ragtruth_validity.csv

Each response is split into sentences with the same regex as the main study. A sentence is "hallucinated" if a
human-labelled span overlaps it. Judges answer the base faithfulness rubric (variant 0) per sentence.
Test split only, quality == 'good', half the responses contain a hallucination, fixed seed.
"""
import argparse, json, random, re, sys, time
from pathlib import Path

import os
import judge as J
EXT = os.environ.get("RJN_EXT") == "1"  # include the bigger/control judges; write to data/results_ext

DATA = Path("data")
RAW = DATA / "raw" / "ragtruth"
SEP = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(`])|\n+")  # identical to judge.split_sents


def spans(t):
    """[(start, end, text)] for the sentences judge.split_sents would return."""
    out, pos = [], 0
    for m in list(SEP.finditer(t)) + [None]:
        seg = t[pos:m.start() if m else len(t)]
        if len(seg.strip()) > 3:
            out.append((pos + len(seg) - len(seg.lstrip()), pos + len(seg.rstrip()), seg.strip()))
        pos = m.end() if m else len(t)
    return out


def overlaps(labels, a, b):
    return any(l["start"] < b and l["end"] > a for l in labels)


def prepare(n):
    src = {s["source_id"]: s for s in map(json.loads, open(RAW / "source_info.jsonl"))}
    rs = [r for r in map(json.loads, open(RAW / "response.jsonl"))
          if r["split"] == "test" and r["quality"] == "good" and src[r["source_id"]]["task_type"] == "QA"]
    bad, ok = [r for r in rs if r["labels"]], [r for r in rs if not r["labels"]]
    rng = random.Random(1234)
    pick = rng.sample(sorted(bad, key=lambda r: r["id"]), n // 2) + rng.sample(sorted(ok, key=lambda r: r["id"]), n - n // 2)
    units = []
    for r in pick:
        passages = src[r["source_id"]]["source_info"]["passages"]  # QA source_info is a dict: question, passages
        for u, (a, b, s) in enumerate(spans(r["response"])):
            units.append(dict(id=r["id"], unit=u, model=r["model"], passages=passages, claim=s,
                              hallucinated=int(overlaps(r["labels"], a, b)), response_has_halluc=int(bool(r["labels"]))))
    (DATA / "ragtruth_eval.jsonl").write_text("".join(json.dumps(u) + "\n" for u in units))
    print(f"{len(pick)} responses ({len(bad)} hallucinated / {len(ok)} clean available), {len(units)} sentences, "
          f"{sum(u['hallucinated'] for u in units)} hallucinated sentences")


def run(judge_name):
    units = [json.loads(l) for l in (DATA / "ragtruth_eval.jsonl").read_text().splitlines()]
    out = J.OUT / f"ragtruth_{judge_name}.jsonl"
    done = {(r["id"], r["unit"]) for r in map(json.loads, out.read_text().splitlines())} if out.exists() else set()
    todo = [u for u in units if (u["id"], u["unit"]) not in done]
    if not todo:
        return
    jd = J.Judge(judge_name); t0 = time.time()
    for i in range(0, len(todo), 16):
        chunk = todo[i:i + 16]
        res = [jd.forward([jd.encode("faith", 0, dict(passages=u["passages"], claim=u["claim"]))])[0] for u in chunk]
        with open(out, "a") as f:
            for u, (d, mass) in zip(chunk, res):
                f.write(json.dumps(dict(id=u["id"], unit=u["unit"], d=d, mass=mass, y0=int(d > 0))) + "\n")
        jd.mx.clear_cache()
        print(f"{judge_name}: {i + len(chunk)}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def report():
    import numpy as np, pandas as pd
    from sklearn.metrics import roc_auc_score
    units = {(u["id"], u["unit"]): u for u in map(json.loads, (DATA / "ragtruth_eval.jsonl").read_text().splitlines())}
    rows = []
    for j in (J.ALL_JUDGES if EXT else J.ROSTER):
        p = J.OUT / f"ragtruth_{j}.jsonl"
        if not p.exists():
            continue
        R = [json.loads(l) for l in p.read_text().splitlines()]
        if len(R) < len(units):
            continue
        d = np.array([r["d"] for r in R]); sup = np.array([1 - units[(r["id"], r["unit"])]["hallucinated"] for r in R])
        y = np.array([r["y0"] for r in R])
        resp = {}
        for r in R:
            resp.setdefault(r["id"], []).append(r["d"])
        rid = list(resp); rd = np.array([min(resp[i]) for i in rid])
        rc = np.array([1 - next(u["response_has_halluc"] for (k, _), u in units.items() if k == i) for i in rid])
        tpr, tnr = (y[sup == 1] == 1).mean(), (y[sup == 0] == 0).mean()
        rows.append(dict(judge=j, n_sentences=len(R), halluc_sentences=int((sup == 0).sum()),
                         sentence_AUC=roc_auc_score(sup, d), sentence_balanced_acc=0.5 * (tpr + tnr),
                         recall_of_hallucinations=tnr, supported_yes_rate=float(y.mean()),
                         response_AUC=roc_auc_score(rc, rd), n_responses=len(rid)))
    out_dir = DATA / ("results_ext" if EXT else "results"); out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out_dir / "ragtruth_validity.csv", index=False)
    print(df.round(3).to_string(index=False))


def selftest():
    t = "First one here. Second one!\nThird line"
    s = spans(t)
    assert [x[2] for x in s] == J.split_sents(t), (s, J.split_sents(t))
    assert all(t[a:b] == x for a, b, x in s)
    assert overlaps([dict(start=17, end=22)], *s[1][:2]) and not overlaps([dict(start=0, end=5)], *s[1][:2])
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "prepare", "run", "report"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--judge", choices=list(J.ALL_JUDGES) + ["all"])
    a = ap.parse_args()
    {"selftest": selftest, "prepare": lambda: prepare(a.n), "report": report,
     "run": lambda: [run(j) for j in (J.ROSTER if a.judge == "all" else [a.judge])]}[a.cmd]()
