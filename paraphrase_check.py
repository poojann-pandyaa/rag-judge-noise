#!/usr/bin/env python3
"""Automated sanity check of the rubric paraphrases -> data/results/paraphrase_check.csv.

Embeds each template with its placeholders blanked (so only the instruction wording is compared) and reports
cosine similarity of every paraphrase to the base rubric, plus length ratio. This is NOT human validation: it
catches paraphrases that drifted far from the base wording. Anything below 0.80 is flagged for manual review.
"""
import csv, re
import numpy as np
import judge as J
from sentence_transformers import SentenceTransformer
import rag

m = SentenceTransformer(rag.hf_path(rag.EMB, rag.EMB_REV, ["*.json", "*.txt", "model.safetensors", "1_Pooling/*"]), device="cpu")
rows = []
for kind, ts in J.TEMPLATES.items():
    blank = [re.sub(r"\{\w+\}", "[...]", t) for t in ts]
    E = m.encode(blank, normalize_embeddings=True)
    for v in range(1, 4):
        sim = float(E[0] @ E[v])
        rows.append(dict(kind=kind, variant=v, cosine_to_base=round(sim, 3),
                         length_ratio=round(len(blank[v]) / len(blank[0]), 2), flagged=sim < 0.80))
with open("data/results/paraphrase_check.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(r)
print("min cosine:", min(r["cosine_to_base"] for r in rows), "| flagged:", sum(r["flagged"] for r in rows))
