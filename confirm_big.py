#!/usr/bin/env python3
"""Do the article's four findings hold for a bigger judge? -> data/results_ext/confirmation.md (+ printed)

PRE-REGISTERED CRITERIA (written 2026-11-25, before any 14B / 4-bit data existed):
  Control (Qwen2.5-7B 8-bit vs 4-bit, base rubric): quantisation counts as "small" if T=0 verdicts agree on >= 90% of
    units in every metric AND the RAGTruth AUCs differ by < 0.03. Only then is the 14B (4-bit) compared with the 8-bit
    ladder without a caveat; otherwise every 14B comparison is reported as confounded with quantisation.
  F1 (determinism): holds for a judge if the logit margin spread across the 4 execution conditions is exactly 0.
  F2 (wording > seed): holds for the 14B if within-item SD across the 4 wordings exceeds the seed SD (T=0.3) in at
    least 5 of 6 metrics AND its correctness-verdict flip rate across wordings exceeds that across seeds.
  F3 (judges disagree): holds if Krippendorff's alpha for the correctness verdict over all 8 judges stays below 0.667
    (the usual lowest bar for "tentative" reliability) AND at least one judge dissents on more than half of the items.
    Also reported: dissent among the three largest judges (Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B).
  F4 (bigger = more correct, not clearly steadier): "more correct" is supported if the 14B's RAGTruth AUC point estimate
    is >= the 7B's (0.795); reported as "clearly better" only if its 95% interval lies above 0.795. "Not clearly
    steadier" is reported descriptively (14B vs 7B seed flips and mean seed SD) plus the 5-judge Qwen trend p-value.
"""
import json, numpy as np, pandas as pd
import judge as J

RX = "data/results_ext/"
out = []
say = lambda s="": (print(s), out.append(s))


def load(j):
    return [json.loads(l) for l in open(f"data/judgments/{j}.jsonl")]


def det_spread(R):
    g = {}
    for r in R:
        if r["variant"] == 0: g.setdefault((r["id"], r["kind"], r["unit"]), []).append(r["d"])
    multi = [v for v in g.values() if len(v) > 1]
    return len(multi), max(max(v) - min(v) for v in multi)


def base(R):
    return {(r["id"], r["kind"], r["unit"]): r for r in R if r["variant"] == 0 and r["cond"] == "serial" and r["rep"] == 0}


auc = pd.read_csv(RX + "ragtruth_auc_ci.csv").set_index("judge")
say("# Bigger-judge confirmation (Qwen2.5-14B, 4-bit) and 4-bit control\n")

# ---------------- control: quantisation
b8, b4 = base(load("qwen2.5-7b")), base(load("qwen2.5-7b-4bit"))
keys = sorted(set(b8) & set(b4))
say("## Control: Qwen2.5-7B at 8-bit vs 4-bit (base rubric, T = 0)\n")
say("| metric | units | T=0 verdict agreement | mean abs. margin change |")
say("|---|---|---|---|")
agree_all = []
for k in J.KINDS:
    ks = [x for x in keys if x[1] == k]
    a = np.mean([b8[x]["y0"] == b4[x]["y0"] for x in ks]); agree_all.append(a)
    say(f"| {k} | {len(ks)} | {100 * a:.1f}% | {np.mean([abs(b8[x]['d'] - b4[x]['d']) for x in ks]):.2f} |")
d_auc = abs(auc.loc["qwen2.5-7b-4bit", "sentence_AUC"] - auc.loc["qwen2.5-7b", "sentence_AUC"])
small_quant = min(agree_all) >= 0.90 and d_auc < 0.03
say(f"\nRAGTruth AUC: 8-bit {auc.loc['qwen2.5-7b', 'sentence_AUC']:.3f} vs 4-bit {auc.loc['qwen2.5-7b-4bit', 'sentence_AUC']:.3f} "
    f"(difference {d_auc:.3f}). Quantisation effect small by the pre-registered rule: **{'yes' if small_quant else 'no'}**"
    f" (lowest metric agreement {100 * min(agree_all):.1f}%).\n")

# ---------------- F1
say("## F1. Temperature 0 is deterministic (serial / batch of 4)\n")
f1 = {}
for j in ("qwen2.5-7b-4bit", "qwen2.5-14b-4bit"):
    n, sp = det_spread(load(j)); f1[j] = sp == 0
    say(f"- {j}: {n} units x 4 conditions, max margin spread {sp} -> {'HOLDS' if sp == 0 else 'DOES NOT HOLD'}")

# ---------------- F2
ps = pd.read_csv(RX + "paraphrase_sensitivity.csv"); vf = pd.read_csv(RX + "verdict_flips.csv")
p14 = ps[ps.judge == "qwen2.5-14b-4bit"]
wins = int((p14.mean_item_SD_across_paraphrases > p14.mean_item_SD_across_seeds_T03).sum())
fl = lambda j, src, T=None: vf[(vf.kind == "corr") & (vf.source == src) & (vf.judge == j) & ((vf["T"] == T) if T is not None else True)].flip_rate.iloc[0]
s14, w14 = fl("qwen2.5-14b-4bit", "seeds", 0.3), fl("qwen2.5-14b-4bit", "paraphrase")
f2 = wins >= 5 and w14 > s14
say("\n## F2. Wording moves verdicts more than the seed\n")
say("| metric | SD across wordings | SD across seeds (T=0.3) |"); say("|---|---|---|")
for r in p14.itertuples(): say(f"| {r.metric} | {r.mean_item_SD_across_paraphrases:.3f} | {r.mean_item_SD_across_seeds_T03:.3f} |")
say(f"\nWording > seed in {wins} of 6 metrics; correctness verdicts change for {100 * w14:.0f}% of answers across wordings vs {100 * s14:.0f}% across seeds "
    f"(Qwen2.5-7B 8-bit: {100 * fl('qwen2.5-7b', 'paraphrase'):.0f}% vs {100 * fl('qwen2.5-7b', 'seeds', 0.3):.0f}%). -> **{'HOLDS' if f2 else 'DOES NOT HOLD'}**\n")

# ---------------- F3
panel = vf[(vf.source == "judges:all_judges") & (vf.kind == "corr")].iloc[0]
V = {}
for j in list(J.ROSTER) + ["qwen2.5-14b-4bit"]:
    for r in load(j):
        if r["kind"] == "corr" and r["variant"] == 0 and r["cond"] == "serial" and r["rep"] == 0: V.setdefault(r["id"], {})[j] = r["y0"]
M8 = np.array([[v[j] for j in list(J.ROSTER) + ["qwen2.5-14b-4bit"]] for v in V.values() if len(v) == 8])
big = [list(J.ROSTER).index("qwen2.5-7b"), list(J.ROSTER).index("llama3.1-8b"), 7]
dissent_big = np.mean(M8[:, big].min(1) != M8[:, big].max(1))
pa = pd.read_csv(RX + "pairwise_agreement_corr.csv", index_col=0)
f3 = panel.krippendorff_alpha < 0.667 and panel.flip_rate > 0.5
say("## F3. Judges disagree\n")
say(f"- All 8 judges (7 original + 14B): alpha = {panel.krippendorff_alpha:.3f} (original 7: 0.269); items with at least one dissent: {100 * panel.flip_rate:.0f}% (original: 83%).")
say(f"- The three largest judges (Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B) disagree on {100 * dissent_big:.0f}% of the {len(M8)} correctness items.")
say("- Qwen2.5-14B agreement with each other judge: " + ", ".join(f"{c} {100 * pa.loc['qwen2.5-14b-4bit', c]:.0f}%" for c in pa.columns if c != "qwen2.5-14b-4bit"))
say(f"-> **{'HOLDS' if f3 else 'DOES NOT HOLD'}**\n")

# ---------------- F4
nf = pd.read_csv(RX + "noise_floor.csv"); nf = nf[nf["T"] == 0.3]
msd = lambda j: nf[nf.judge == j].mean_item_SD.mean()
a14, lo14, hi14 = auc.loc["qwen2.5-14b-4bit", ["sentence_AUC", "ci_lo", "ci_hi"]]
a7 = auc.loc["qwen2.5-7b", "sentence_AUC"]
cap = pd.read_csv(RX + "capacity_trend.csv")
ct = cap[(cap.scope == "qwen_only")].set_index("outcome")
say("## F4. Bigger judges: more correct, not clearly steadier\n")
say("| judge | RAGTruth AUC [95% CI] | corr. seed flips (T=0.3) | mean seed SD, 6 metrics |"); say("|---|---|---|---|")
for j in ("qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "qwen2.5-7b-4bit", "qwen2.5-14b-4bit"):
    if j in auc.index:
        x = auc.loc[j]
        sf = f"{100 * fl(j, 'seeds', 0.3):.0f}%" if ((vf.judge == j) & (vf.source == "seeds")).any() else "-"
        sd = f"{msd(j):.3f}" if (nf.judge == j).any() else "-"
        say(f"| {j} | {x.sentence_AUC:.3f} [{x.ci_lo:.3f}, {x.ci_hi:.3f}] | {sf} | {sd} |")
verdict = "clearly better" if lo14 > a7 else ("higher (intervals overlap)" if a14 >= a7 else "not higher")
say(f"\n14B vs 7B AUC: {a14:.3f} vs {a7:.3f} -> {verdict}. Qwen ladder (5 judges) seed-noise trend p-values (judge-level OLS): "
    + ", ".join(f"{o.replace('noise_SD_', '')} {ct.loc[o, 'p_ols']:.2f}" for o in ct.index if o.startswith("noise_SD_")))
if not small_quant: say("\nCaveat: the control shows a non-trivial quantisation effect, so 14B-vs-7B differences may be partly due to 4-bit weights.")
open(RX + "confirmation.md", "w").write("\n".join(out) + "\n")
print("\nwrote", RX + "confirmation.md")
