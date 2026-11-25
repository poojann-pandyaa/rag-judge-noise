#!/usr/bin/env python3
"""Reference-based metrics for the frozen answers -> data/refmetrics.jsonl (CPU only, so it can run beside a sweep).

Per item: hit@1/3/5/10 (gold passage retrieved), ROUGE-L F1 and BERTScore F1 (best over the gold answers),
abstain_rule (generator answer starts with the abstention sentence). Gold-based fields are None for
unanswerable items; hit@k is None for drop_gold items (gold was removed from the index on purpose).
"""
import json
from pathlib import Path

DATA = Path("data")
ev = {x["id"]: x for x in map(json.loads, (DATA / "eval_set.jsonl").read_text(encoding="utf-8").splitlines())}
ans = [json.loads(l) for l in (DATA / "answers.jsonl").read_text(encoding="utf-8").splitlines()]

from rouge_score import rouge_scorer
from bert_score import BERTScorer

rl = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
bs = BERTScorer(lang="en", device="cpu")  # roberta-large layer 17, no baseline rescaling

rows = []
for a in ans:
    e = ev[a["id"]]
    r = dict(id=a["id"], answerable=e["answerable"], drop_gold=e["drop_gold"],
             abstain_rule=int(a["answer"].lower().startswith("i cannot answer")),
             n_words=len(a["answer"].split()))
    gold = e["gold_answers"]
    for k in (1, 3, 5, 10):
        r[f"hit@{k}"] = None if (not e["answerable"] or e["drop_gold"]) else int(bool(set(a["retrieved_ids"][:k]) & set(e["gold_passage_ids"])))
    if gold:
        r["rougeL"] = max(rl.score(g, a["answer"])["rougeL"].fmeasure for g in gold)
        _, _, f = bs.score([a["answer"]] * len(gold), gold)
        r["bertscore"] = float(f.max())
    else:
        r["rougeL"] = r["bertscore"] = None
    rows.append(r)

(DATA / "refmetrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
done = [r for r in rows if r["rougeL"] is not None]
print(f"wrote {len(rows)} rows; gold-scored {len(done)}; mean ROUGE-L={sum(r['rougeL'] for r in done) / len(done):.3f} "
      f"BERTScore={sum(r['bertscore'] for r in done) / len(done):.3f}")
for k in (1, 3, 5, 10):
    v = [r[f"hit@{k}"] for r in rows if r[f"hit@{k}"] is not None]
    print(f"Recall@{k}={sum(v) / len(v):.3f} (n={len(v)})", end="  ")
print()
