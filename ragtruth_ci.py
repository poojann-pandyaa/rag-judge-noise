"""Response-cluster bootstrap CIs for RAGTruth sentence AUC + stability-vs-validity rank correlation across judges."""
import json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import judge as J
U = [json.loads(l) for l in open("data/ragtruth_eval.jsonl")]
key = {(u["id"], u["unit"]): u for u in U}
ids = sorted({u["id"] for u in U}); rng = np.random.default_rng(1234)
rows = []
for j in J.ROSTER:
    R = [json.loads(l) for l in open(f"data/judgments/ragtruth_{j}.jsonl")]
    by = {}
    for r in R: by.setdefault(r["id"], []).append((r["d"], 1 - key[(r["id"], r["unit"])]["hallucinated"]))
    pt = roc_auc_score([s for v in by.values() for _, s in v], [d for v in by.values() for d, _ in v])
    bs = []
    for _ in range(2000):
        v = [x for i in rng.choice(len(ids), len(ids)) for x in by[ids[i]]]
        y = [s for _, s in v]
        if 0 < sum(y) < len(y): bs.append(roc_auc_score(y, [d for d, _ in v]))
    rows.append(dict(judge=j, sentence_AUC=pt, ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
df = pd.DataFrame(rows); df.to_csv("data/results/ragtruth_auc_ci.csv", index=False); print(df.round(3).to_string(index=False))
nf = pd.read_csv("data/results/noise_floor.csv"); nf = nf[(nf["T"] == 0.3) & (nf.metric == "faith")].set_index("judge").mean_item_SD
print("spearman(faith seed noise T0.3, ragtruth AUC) across 7 judges:", spearmanr(nf[df.judge], df.sentence_AUC))
am = [json.loads(l) for l in open("data/refmetrics.jsonl")]
print("abstain_rule positives:", sum(bool(a["abstain_rule"]) for a in am), "of", len(am))
