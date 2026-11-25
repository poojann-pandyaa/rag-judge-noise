#!/usr/bin/env python3
"""Three simple, print-friendly figures for the OSFY article -> article_figures/Figure1.png ... Figure3.png (300 dpi).
Every number is read from data/results/*.csv (see build_supp.py / analyze.py); nothing is typed in by hand."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path("article_figures"); OUT.mkdir(exist_ok=True)
R = "data/results/"
BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#1f2933", "#5f6b76", "#e3e3df", "#ffffff"
ORD = ["qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b", "llama3.2-1b", "llama3.2-3b", "llama3.1-8b"]
LAB = {"qwen2.5-0.5b": "Qwen2.5\n0.5B", "qwen2.5-1.5b": "Qwen2.5\n1.5B", "qwen2.5-3b": "Qwen2.5\n3B", "qwen2.5-7b": "Qwen2.5\n7B",
       "llama3.2-1b": "Llama-3.2\n1B", "llama3.2-3b": "Llama-3.2\n3B", "llama3.1-8b": "Llama-3.1\n8B"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "savefig.facecolor": SURF})

# ------------------------------------------------------------------ Figure 1: seeds vs rewordings (correctness verdict)
vf = pd.read_csv(R + "verdict_flips.csv")
seed = [100 * vf[(vf.kind == "corr") & (vf.source == "seeds") & (vf.judge == j) & (vf["T"] == 0.3)].flip_rate.iloc[0] for j in ORD]
word = [100 * vf[(vf.kind == "corr") & (vf.source == "paraphrase") & (vf.judge == j)].flip_rate.iloc[0] for j in ORD]
x = np.array([0, 1, 2, 3, 4.7, 5.7, 6.7]); w = 0.36
fig, ax = plt.subplots(figsize=(7.4, 4.1))
ax.bar(x - w / 2 - 0.01, seed, w, color=BLUE, label="Across 10 random seeds (T = 0.3)", zorder=3)
ax.bar(x + w / 2 + 0.01, word, w, color=ORANGE, hatch="////", edgecolor="white", linewidth=0, label="Across 4 rewordings of the question", zorder=3)
ax.set_xticks(x); ax.set_xticklabels([LAB[j] for j in ORD], fontsize=8.5)
ax.set_ylim(0, 60); ax.set_yticks(range(0, 61, 10)); ax.yaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.set_ylabel("Answers whose verdict changes (%)")
ax.axvline(3.85, color=GRID, lw=1.2, zorder=1)
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=1, bbox_to_anchor=(0.0, 1.0))
ax.text(1.5, -0.20, "Qwen2.5 family", transform=ax.get_xaxis_transform(), ha="center", color=MUTED, fontsize=9)
ax.text(5.7, -0.20, "Llama family", transform=ax.get_xaxis_transform(), ha="center", color=MUTED, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "Figure1.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure 2: judge-vs-judge agreement (correctness verdict)
pa = pd.read_csv(R + "pairwise_agreement_corr.csv", index_col=0).loc[ORD, ORD].values * 100
M = np.ma.masked_array(pa, mask=np.eye(len(ORD), dtype=bool))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("blue_seq", ["#e8f0fb", "#9cc0ec", "#2a78d6", "#154a8f"])
cmap.set_bad("#efefec")
fig, ax = plt.subplots(figsize=(6.4, 5.5))
ax.imshow(M, cmap=cmap, vmin=30, vmax=90)
ax.set_xticks(range(7)); ax.set_xticklabels([LAB[j].replace("\n", " ") for j in ORD], rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(7)); ax.set_yticklabels([LAB[j].replace("\n", " ") for j in ORD], fontsize=9)
for i in range(7):
    for j in range(7):
        if i == j: ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=10); continue
        v = pa[i, j]; ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=10, color="white" if v >= 68 else INK)
ax.set_xticks(np.arange(-.5, 7, 1), minor=True); ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="both", length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xlabel("Each number is the % of questions on which\nthe two judges gave the same Yes/No verdict", fontsize=9, color=MUTED, labelpad=10)
fig.tight_layout(); fig.savefig(OUT / "Figure2.png", dpi=300, bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure 3: AUC on human-labelled hallucinations (RAGTruth)
rc = pd.read_csv(R + "ragtruth_auc_ci.csv").set_index("judge").loc[ORD]
y = np.array([7.0, 6.0, 5.0, 4.0, 2.3, 1.3, 0.3])
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.axvline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.text(0.505, 7.75, "coin flip (0.5)", color=MUTED, fontsize=8.5, va="bottom")
for yy, (j, r) in zip(y, rc.iterrows()):
    ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=BLUE, lw=2.2, solid_capstyle="round", zorder=2)
    ax.scatter(r.sentence_AUC, yy, s=70, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
    ax.text(r.ci_hi + 0.012, yy, f"{r.sentence_AUC:.2f}", va="center", fontsize=9)
ax.set_yticks(y); ax.set_yticklabels([LAB[j].replace("\n", " ") for j in ORD], fontsize=9)
ax.set_xlim(0.40, 0.92); ax.set_ylim(-0.4, 8.1); ax.xaxis.grid(True, color=GRID, zorder=0); ax.set_axisbelow(True)
ax.set_xlabel("How well the judge separates hallucinated from supported sentences  (0.5 = coin flip, 1.0 = perfect)", fontsize=8.7)
ax.tick_params(axis="y", length=0)
fig.tight_layout(); fig.savefig(OUT / "Figure3.png", dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved:", sorted(p.name for p in OUT.iterdir()))
