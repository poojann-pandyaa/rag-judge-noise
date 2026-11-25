# How stable is RAG evaluation with small local judges?

**Noise floor, capacity scaling and cross-metric consistency on a fully version-pinned, open-weight, local stack**

Single-run study, Apple M4 / 16 GB, 7 judges (0.5B–8B), 250 CLAPNQ questions, 6 metrics, 10 seeds × 3 temperatures × 4 rubric wordings. Everything below was produced by the scripts in this folder; every number can be traced to a CSV in `data/results/`.

---

## 0. Summary — what we actually found

| | Finding | Strength of evidence |
|---|---|---|
| **RQ1** | **T=0 is deterministic for the way this study was run**: 431 units × 4 execution conditions (serial ×3 in shuffled order, padded batch of 4) gave bit-identical Yes/No logit margins for all 7 judges. A later stress test on the 0.5B judge found identical values at batch sizes 1 and 8 but one-bf16-step differences (≤ 0.0156 logits, 4 of 96 units) at batch size ≥ 32 — so "deterministic" holds for serial / small-batch scoring, not for arbitrary batch compositions (§3.1). | Strong within the tested conditions; resolution-limited (bf16 logits — see §8). |
| **RQ1** | **Sampling noise is real and sizeable at T>0** and depends on the judge at least as much as on the metric. At the pre-declared T=0.3 the correctness verdict flips across seeds for 4–50 % of items depending on judge (Qwen2.5-1.5B 4 %, Llama-3.1-8B 11 %, Llama-3.2-1B 50 %). | Strong (item-bootstrap CIs, 10 seeds). |
| **RQ1** | **Prompt wording matters more than seeds.** At T=0.3, within-item SD across 4 rubric wordings exceeds within-item SD across seeds in **40 of 42** judge×metric cells (median ratio 2.7×, up to 14×). Correctness verdicts flip across wordings for 25–45 % of items for *every* judge, including the 7B and 8B ones. | Strong for "wording > seed"; but the 3 paraphrases are **author-drafted and only automatically checked** (§8). |
| **RQ2** | **No statistically supportable capacity trend.** Point estimates say seed noise falls with size (≈ −0.01 to −0.03 SD per doubling), but with 7 judges the judge-level OLS p-values are 0.05–0.87 (Holm-adjusted: all ≥ 0.95). The picture is non-monotone (Qwen2.5-3B is *noisier* than 1.5B and 7B) and **family matters more than size**: Llama is 2–5× noisier than Qwen at comparable size. Paraphrase sensitivity does not shrink with size at all. | Honest negative. n = 7 judges cannot support a scaling law. |
| **RQ2 and validity** | **Stability ≠ validity.** The quietest judges are often the least informative: Qwen2.5-1.5B never says "abstain" (noise 0.001, but zero recall), Llama-3.2-1B says "not relevant" to everything. Validity on human labels (RAGTruth) *does* rise with size: sentence AUC 0.50 → 0.80. Across the 7 judges, lower faithfulness-seed-noise goes with higher AUC (Spearman ρ = −0.71, p = 0.07, n = 7: suggestive only). | AUC trend: moderate (CIs in §5.4). Correlation: weak (n = 7). |
| **RQ3** | **Judge identity is the largest unstable component.** 83 % of correctness items get at least one dissenting judge among the 7 (Krippendorff α = 0.27); for abstention α = 0.15. Variance decomposition: item×judge interaction is 31–53 % of variance at T=0.3, versus 10–29 % for seed noise. | Strong. |
| **RQ3** | **Cross-metric agreement is limited and judge-dependent.** For the 7B judge only retrieval-style metrics line up with reference-based ones (context recall ↔ gold-passage hit@3 ρ = 0.82; correctness ↔ BERTScore ρ = 0.52); faithfulness is only loosely linked to the reference measures (absolute ρ ≤ 0.20). The 0.5B judge shows a "yes-bias" cluster (faithfulness/relevance/correctness ρ = 0.53–0.71) that is uncorrelated with reference metrics (≤ 0.23). | Descriptive, one run. |
| **RQ4** | **Not run** (frontier-API anchor needs a paid external key that was not provided). | — |

**Practical reading:** on this stack, (i) pin the decoding (T=0) and the rubric wording — wording is the dominant *controllable* noise source; (ii) never rely on a single small judge — validity and cross-judge agreement are both poor below ~3B and still imperfect at 7–8B; (iii) report seed/wording/judge sensitivity alongside any RAG score.

---

## 1. Motivation and research questions

RAG systems are increasingly scored by LLM judges. If the judge is itself noisy — across decoding seeds, rubric wording, or which judge is used — a reported score difference between two RAG pipelines can be an artefact. We measure that noise on an open-weight, version-pinned, locally runnable stack so that it can be reproduced exactly, as a function of judge capacity (≈0.5B–8B parameters).

- **RQ1 — noise floor.** How much do verdicts vary with decoding seed and prompt paraphrase? Is T=0 actually deterministic?
- **RQ2 — capacity scaling.** Does that noise shrink with judge size?
- **RQ3 — cross-metric consistency.** Do the standard RAG metrics (faithfulness, relevance, context precision/recall, correctness, abstention) agree with each other, with reference-based metrics, and across judges?
- **RQ4 (optional) — frontier anchor.** Compare to a frontier judge. *Not run.*

## 2. Experimental design

### 2.1 Data
- **Questions:** CLAPNQ dev set (answerable and unanswerable; no public test split exists). 250 items: 70 % answerable, 20 % with the gold passage removed at query time (`drop_gold`), the rest unanswerable; seed 1234. Files and checksums are in `data/manifest.json`.
- **Validity set for faithfulness:** RAGTruth, QA task, test split only, `quality == good`, 150 responses (75 with a human-labelled hallucination span, 75 clean), 1,092 sentences, 164 hallucinated. Pinned to commit `c103204b9ce28d6bbad859304bf30de72b8ed8fe` (MIT licence).

### 2.2 The RAG fixture being judged (fixed, not varied)
- **Retrieval:** BM25 (SQLite FTS5) + dense `BAAI/bge-base-en-v1.5` (768-d), fused by reciprocal-rank fusion (k = 60, pool 50). Top-10 IDs stored; top-3 go to the generator. Recall@1/3/5/10 = 0.286 / 0.543 / 0.693 / 0.843.
- **Generator:** `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, greedy, seed 1234, fixed abstention sentence "I cannot answer this from the provided passages." Median answer 43 words; 20 of 250 answers contain the abstention sentence.
- **Reference-based metrics** (for cross-metric comparison): ROUGE-L F1 (mean 0.301), BERTScore F1 with roberta-large (mean 0.873), hit@k.

### 2.3 Judges (all 8-bit MLX, same quantisation, pinned by revision)

| Judge | Family | Params (B) | HF repo revision |
|---|---|---|---|
| qwen2.5-0.5b | Qwen2.5 | 0.49 | `6474d4c1a05b…` |
| qwen2.5-1.5b | Qwen2.5 | 1.54 | `c7e1ff25efda…` |
| qwen2.5-3b | Qwen2.5 | 3.09 | `439033055dac…` |
| qwen2.5-7b | Qwen2.5 | 7.62 | `df11cb390bd6…` |
| llama3.2-1b | Llama | 1.24 | `d48cdf0a4ea2…` |
| llama3.2-3b | Llama | 3.21 | `ff0548996090…` |
| llama3.1-8b | Llama | 8.03 | `142d42800404…` |

Full 40-character revisions: `data/results/table1_roster.csv`.

### 2.4 Metrics and how a judgement is made
Six metrics, all computed from constrained **single-token Yes/No** judgements: faithfulness and context recall (per sentence of the answer / gold answer), context precision (rank-weighted average precision over the 3 passages), and single-call answer relevance, answer correctness (vs ≤3 gold references) and abstention. Sentences are split by a deterministic regex.

For each call the harness stores the logit margin `d = logit(Yes) − logit(No)` and the probability mass on {Yes, No}. The T=0 label is `d > 0`. **Seeded T>0 verdicts are sampled from `sigmoid(d/T)`** — i.e. they are draws from the judge's own Yes/No distribution, not independent forward passes (this is exact for a single constrained token and makes 10 seeds × 3 temperatures affordable; see §8).

### 2.5 Variance sources manipulated
- **Seed:** 10 seeds × T ∈ {0.3 (pre-declared primary), 0.7, 1.0}.
- **Prompt paraphrase:** 4 rubric wordings per metric (variant 0 = base; 1–3 = paraphrases; automated check: minimum cosine similarity of blanked templates to the base = 0.826, none below the 0.80 flag threshold).
- **Judge identity:** 7 judges.
- Total: 2,147 forward passes per judge per wording; 9,881 stored records per judge.

### 2.6 Statistics
Item-cluster bootstrap CIs (2,000 resamples); Krippendorff's α (nominal); verdict-flip rates; Spearman matrices with PCA participation ratio and hierarchical clustering; capacity trend as a slope per doubling of parameters with the **judge-level OLS p-value** (the item bootstrap treats judges as fixed and overstates significance with n = 7), Holm-corrected over all trend tests; variance components from random-effects ANOVA (item × judge × seed with replication; item × judge × wording at T=0 without replication). The estimators recover known variances in simulation (`python3 analyze.py selftest`).

## 3. RQ1 — Noise floor

### 3.1 Is T=0 deterministic?
**Yes at the resolution we can measure.** 431 units per judge were scored under four conditions (serial run ×3 in shuffled order; one padded batch of 4). All four gave identical `d` (max |Δd| = 0.0) and identical labels (agreement 1.0, α = 1.0) for **all 7 judges** (`table2_determinism.csv`). Right-padding plus a causal mask means padding cannot reach real tokens, so batching being a no-op is expected. Caveat: logits come out of the MLX forward in bf16, so differences below ~0.01–0.1 logits would be invisible; "bit-identical at logit precision" is the accurate statement, not "mathematically deterministic".

**Stress test (added after the main run, Qwen2.5-0.5B, 96 random units with 82–1,056 tokens).** A fresh batch-of-1 rerun reproduced the stored sweep values exactly (max |Δd| = 0). Batches of 8 (different neighbours and padding) were also identical. Batches of 32 and 96 (random and length-sorted) differed from batch-of-1 in 4 of 96 units by exactly one bf16 step (0.0156 logits; mean |Δd| = 0.0007). A verdict can only flip if a unit lies within that step of the boundary; this applies to ≤ 0.6 % of units for every judge (Llama-3.2-3B 0.61 %, Llama-3.2-1B 0.37 %, Qwen2.5-3B 0.19 %, others ≤ 0.14 %). The study itself scored every unit serially (batch of 1), so none of the reported results is exposed to this; it matters for anyone who re-runs the evaluation with large batches. Only the 0.5B judge was stress-tested.

### 3.2 Seed noise (Figure 1)
![Figure 1](figures/fig1_noise_vs_size.png)

Mean within-item SD of the metric across the 10 seeds, T = 0.3 (full table with all T and CIs: `noise_floor.csv`):

| Judge | faith | rel | prec | rec | corr | abst |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.062 | 0.093 | 0.001 | 0.084 | 0.068 | 0.156 |
| qwen2.5-1.5b | 0.015 | 0.022 | 0.019 | 0.034 | 0.015 | 0.001 |
| qwen2.5-3b | 0.025 | 0.045 | 0.029 | 0.013 | 0.057 | 0.025 |
| qwen2.5-7b | 0.018 | 0.018 | 0.012 | 0.006 | 0.022 | 0.016 |
| llama3.2-1b | 0.200 | 0.033 | 0.138 | 0.214 | 0.205 | 0.174 |
| llama3.2-3b | 0.093 | 0.122 | 0.062 | 0.080 | 0.097 | 0.115 |
| llama3.1-8b | 0.072 | 0.083 | 0.042 | 0.062 | 0.041 | 0.034 |

Noise grows steeply with temperature (e.g. Llama-3.2-1B correctness-verdict flip rate across seeds: 0.50 / 0.78 / 0.90 at T = 0.3 / 0.7 / 1.0). Because seeded draws come from `sigmoid(d/T)`, seed noise is essentially a measure of how close the judge's margin sits to zero relative to T: a judge that is *confidently* wrong is perfectly "stable".

### 3.3 Verdict flips (Figure 2, right)
![Figure 2](figures/fig2_agreement_flips_vs_size.png)

Correctness verdict, fraction of items whose binary verdict is not constant (`verdict_flips.csv`):

| Judge | across seeds (T=0.3) | across 4 wordings (T=0) |
|---|---|---|
| qwen2.5-0.5b | 0.18 | 0.25 |
| qwen2.5-1.5b | 0.04 | 0.31 |
| qwen2.5-3b | 0.14 | 0.44 |
| qwen2.5-7b | 0.06 | 0.29 |
| llama3.2-1b | 0.50 | 0.45 |
| llama3.2-3b | 0.24 | 0.43 |
| llama3.1-8b | 0.11 | 0.34 |

Abstention verdicts flip across wordings for 12–34 % of items (median 0.29). **A single base-rubric score therefore carries a wording-dependent verdict uncertainty of roughly a quarter to a half of items, even for the largest judges tested, and this is much larger than the seed noise at T = 0.3.**

## 4. RQ2 — Capacity scaling

**Result: no trend that survives correction.** `capacity_trend.csv` gives slope per doubling of parameters, for all judges and for Qwen alone (cleaner ladder: same recipe at every size).

- Point estimates: seed-noise SD falls with size for 5 of 6 metrics (all-judge slopes: faith −0.013, rec −0.022, corr −0.017, abst −0.032 per doubling); correctness seed-flip rate −0.042 per doubling; inter-judge agreement on correctness +0.028 per doubling.
- Inference: the item-bootstrap intervals exclude zero for most of these, but that bootstrap resamples *items only*, treating the 7 judges as the population — it cannot support a statement about "judges of larger size in general". The judge-level OLS p-values are 0.12–0.87 (all judges) and 0.05–0.66 (Qwen only; the single p = 0.047 for context-recall noise becomes 0.945 after Holm correction over 18 tests).
- The data are visibly non-monotone: Qwen2.5-3B is noisier than 1.5B and 7B on faith/prec/corr; Llama-3.2-3B is noisier than Llama-3.2-1B on relevance.
- **Family dominates size.** Llama is 2–5× noisier than Qwen at similar size (faithfulness SD at T=0.3: Llama-1.2B 0.20, Llama-3.2B 0.09, Llama-8B 0.07 vs. Qwen 0.015–0.025 at 1.5–7B). Part of this is probably calibration/sharpness differences between the two training recipes, not capacity.
- **Wording sensitivity does not decrease with size** (mean SD across wordings, averaged over metrics: Qwen 0.13 / 0.14 / 0.19 / 0.11 for 0.5 / 1.5 / 3 / 7B; Llama 0.18 / 0.23 / 0.14 for 1.2 / 3.2 / 8B).
- **Validity does improve with size** (§5.4): RAGTruth sentence AUC Qwen 0.70 → 0.73 → 0.77 → 0.80; Llama 0.50 → 0.66 → 0.76.

What this does and does not support: *seed* noise tends to be lower in larger judges; *wording* noise is not; *validity* is higher in larger judges; and with 7 judges in two families no scaling law can be claimed. Adding judges (Gemma, Phi, Mistral families at several sizes) is the obvious way to make RQ2 testable.

## 5. RQ3 — Consistency across judges and metrics

### 5.1 Judge identity (Figure 2 left; `pairwise_agreement_*.csv`, `verdict_flips.csv`)
At T=0 and base wording:

| Items where all judges agree | Krippendorff α | all 7 judges | Qwen ladder (4) | Llama ladder (3) |
|---|---|---|---|---|
| Correctness (n=175) | α | 0.27 | 0.33 | 0.14 |
| Abstention (n=250) | α | 0.15 | 0.02 | 0.23 |

At least one dissent exists on 83 % of correctness items and 36 % of abstention items. Mean pairwise correctness agreement with other judges rises from 0.50 (Llama-3.2-1B) to 0.70 (Qwen2.5-7B) but is non-monotone within the Qwen ladder. (Abstention agreement is inflated by a base rate of ~8 % positives: most pairs agree simply by both saying "No".)

### 5.2 Where the variance comes from (Figure 5)
![Figure 5](figures/fig5_variance_components.png)

Shares of total variance (`variance_components.csv`), item × judge × seed design, T = 0.3: item 14–40 %, judge main effect 7–38 %, **item × judge 31–53 %**, seed noise 10–29 %. At T = 0.7 seed noise grows to 21–50 %. In the T = 0 wording design, the wording main effect is small (0–8 %) but **judge × wording interactions (3–16 %) and the residual (19–46 %)** are not: each judge reacts differently to rewording, and in a way that varies by item.

### 5.3 Cross-metric agreement (Figure 3; `crossmetric_<judge>.csv`, `redundancy.csv`)
![Figure 3](figures/fig3_crossmetric_heatmap.png)

Spearman correlations of per-item scores (T = 0 base wording) between the six LLM-judged metrics and ROUGE-L, BERTScore and gold-passage hit@3 (three judges shown; the others are in the CSVs):

- **Qwen2.5-7B / Llama-3.1-8B:** context recall ↔ hit@3 = 0.82 / 0.64; correctness ↔ BERTScore = 0.52 / 0.37 and ↔ ROUGE-L = 0.49 / 0.35; context precision ↔ hit@3 = 0.45 / 0.34. Faithfulness is only loosely linked to the reference measures (|ρ| ≤ 0.20 for Qwen-7B, ≤ 0.13 for Llama-8B) and to the other judged metrics (|ρ| ≤ 0.40 and ≤ 0.41). Abstention correlates *negatively* with relevance (ρ = −0.56 / −0.59), the expected sign (an abstaining answer is judged "not relevant").
- **Qwen2.5-0.5B:** faithfulness, relevance and correctness form a tight cluster (ρ = 0.53–0.71) — a shared yes-bias — that is uncorrelated with the reference metrics (≤ 0.23). Agreement among a small judge's metrics here reflects a common response bias, not shared validity.
- **Redundancy:** PCA participation ratio of the six metrics ranges from 3.0 (Qwen2.5-1.5B, which has a constant metric) to 5.2 (Qwen2.5-3B); there is no monotone relation with size, and we make no inferential claim about it.
- Constant metrics: Qwen2.5-1.5B abstention is constant (always "No"); Llama-3.2-1B relevance is constant (always "No"). Their correlations are undefined and are shown blank.

### 5.4 Validity anchors (so that stability is never read alone)

**Faithfulness vs human labels (RAGTruth, 1,092 sentences, 164 hallucinated; `ragtruth_validity.csv`, `ragtruth_auc_ci.csv`).** Sentence-level AUC for detecting hallucinated sentences from the Yes/No margin, with 95 % response-cluster bootstrap CIs:

| Judge | AUC | 95 % CI | response-level AUC | balanced acc. | note |
|---|---|---|---|---|---|
| qwen2.5-0.5b | 0.700 | 0.63–0.76 | 0.67 | 0.56 | says "supported" for 91 % of sentences |
| qwen2.5-1.5b | 0.732 | 0.68–0.78 | 0.68 | 0.55 | says "supported" for 91 % |
| qwen2.5-3b | 0.774 | 0.72–0.82 | 0.77 | 0.68 | |
| qwen2.5-7b | **0.795** | 0.75–0.83 | 0.76 | **0.71** | |
| llama3.2-1b | 0.498 | 0.43–0.57 | 0.59 | 0.49 | chance level |
| llama3.2-3b | 0.657 | 0.60–0.71 | 0.66 | 0.55 | flags 98 % of hallucinations but only 11 % of supported sentences are passed |
| llama3.1-8b | 0.761 | 0.71–0.80 | 0.76 | 0.70 | |

Note how the AUC (ranking) can be decent while the 0-threshold decision is not (0.5B/1.5B Qwen: AUC 0.70–0.73 but balanced accuracy ≈ 0.55) — the T=0 verdict depends on an uncalibrated threshold.

**Abstention vs the regex label** (generator answer contains the abstention sentence; 20 of 250 items = 8 %; `validity.csv`). Predicting "never abstains" scores 0.92 accuracy. Only Qwen2.5-7B (base wording: precision 0.45, recall 0.95) and, with poor precision, Llama-3.1-8B (precision 0.30, recall 1.00) detect abstentions. **The same judge's behaviour changes sharply with wording**: Qwen2.5-7B accuracy 0.90 / 0.95 / 0.75 / 0.93 over the four wordings; Qwen2.5-3B recall 0.30 / 0.00 / 0.75 / 0.00. This is a task a string match solves, so it is a sanity anchor rather than a hard test — and most judges below 7B fail it.

**Correctness vs ROUGE-L quartile extremes** (weak proxy, not human labels): balanced accuracy 0.50–0.82; best is Qwen2.5-7B (0.82 base wording), ≈ 0.57–0.61 for the 0.5B/1B judges.

**Interpreting stability with validity.** Among the seven judges, lower faithfulness seed noise tends to co-occur with higher RAGTruth AUC (ρ = −0.71, p = 0.07, n = 7), i.e. capacity may buy stability and validity together — but the extremely stable small judges are stable partly because they are constant or biased. Reporting seed variance without a validity anchor would rank Qwen2.5-1.5B (abstention SD 0.001) above Qwen2.5-7B.

## 6. RQ4 — Frontier anchor
Not run. It requires a paid external API key and authorisation to send CLAPNQ/RAGTruth items to a third-party service, neither of which was provided. The harness is model-agnostic at the prompt level; adding an API judge would need a new backend (logprobs are not available for all APIs, so verdict sampling at T>0 would use real repeated calls instead of `sigmoid(d/T)`).

## 7. Deviations from the original plan (all deliberate; please review)
1. **Seeded samples are draws from the judge's own Yes/No distribution**, not independent forward passes at T>0. Exact for one constrained token; much cheaper. It would not capture noise in free-text reasoning judges.
2. **Single-token constrained Yes/No judgements**, not free-text chain-of-thought judges. Yes/No probability mass is ≥ 0.9998 for the median call and ≥ 0.9957 at the 5th percentile for every judge (minimum over all calls: Qwen2.5-7B 0.64, Llama-3.2-1B 0.77, Llama-3.2-3B 0.87; fewer than 0.4 % of calls fall below 0.9 for any judge), so format failure is not a hidden source of noise.
3. **No true T=0 "repeat" noise to report** because T=0 is deterministic; the determinism check is table 2.
4. **Same 8-bit quantisation for all judges**, so size is not confounded with quantisation level; but the 8-bit MLX weights are not the full-precision models.
5. **Only 7 judges in two families.** Llama-3.2 1B/3B are distilled/pruned from larger models while the Qwen2.5 ladder uses one recipe, so "size" is confounded with "recipe" within the Llama family. Qwen2.5-3B carries a non-Apache ("qwen-research") licence.
6. **A memory-safety patch was applied mid-sweep** (MLX cache limit 512 MB and `clear_cache()` after each chunk) after the OS killed the first run twice with out-of-memory (a concurrent BERTScore job was the likely trigger). It affects speed only; the first two Qwen2.5-0.5B attempts were restarted and resumed from saved records. Integrity check after the run: every judge file has exactly 9,881 records (3,009 base-rubric serial + 431 batch-of-4 + 3 × 2,147 paraphrase), **no duplicate records, no non-finite values, and identical record keys across all 7 judges**; every RAGTruth file has 1,092 unique sentence records. The post-processing job (`finish.sh`) was also interrupted once (when this tool session restarted, with Llama-3.1-8B RAGTruth at 960/1,092) and was relaunched; it resumed from saved records, and the final analysis ran after all judgements were complete.
7. **Bug fixed during analysis:** the pairwise-agreement matrix for correctness initially counted items a judge had not scored (175 of 250 have correctness) as disagreements, giving a diagonal of 0.70; corrected and re-run (Figure 2 and all numbers above use the corrected version). Other analysis tables were not affected.

## 7b. Independent re-verification (done after the report was first drafted)
Because the results were produced by one analysis script, the headline numbers were re-derived with separate code straight from the raw `.jsonl` files:

| Check | Result |
|---|---|
| Raw-data integrity | 7 × 9,881 records, no duplicates / non-finite values, identical key sets across judges; 7 × 1,092 RAGTruth records, all unique |
| T=0 determinism from raw | 431 units × 4 conditions per judge, max spread of d = 0.0 (reproduced); stress test above qualifies it |
| Paraphrases are real, distinct prompts | 4 distinct templates per metric with identical placeholders; d differs from base for > 99 % of units (r with base 0.52–0.95) |
| Yes/No token mapping | single tokens "Yes"/"No", correct ids for both tokenizer families; the sign is also confirmed empirically (RAGTruth: supported sentences have higher d than hallucinated ones for every judge except the chance-level Llama-3.2-1B) |
| Seeded draws vs theory | observed seed-flip rates and Yes-rates match `sigmoid(d/T)` predictions for all 7 judges × 3 single-call metrics (e.g. Qwen2.5-7B correctness flip 0.057 vs 0.046 predicted; Yes-rate 0.618 vs 0.619) |
| Judge disagreement | independent Krippendorff-α code: correctness 0.269 (n = 175), abstention 0.152 (n = 250); dissent on 83.4 % / 35.6 % of items — identical to the report |
| Wording vs seed | recomputed from raw for the three single-call metrics: wording SD exceeds seed SD in 19 of 21 cells; the two exceptions are Llama-3.2-1B and Qwen2.5-0.5B abstention (≈ equal or seed larger) |
| Capacity trend | OLS slopes/p-values recomputed independently from `noise_floor.csv`: faith −0.013 (p = 0.54), rec −0.022 (0.30), corr −0.017 (0.40), abst −0.032 (0.12) — identical |
| RAGTruth AUC | recomputed from raw span labels: within 0.004 of the report for every judge. One sentence differed (165 vs 164 hallucinated) because that sentence's text appears twice in one response and a naive text search hit the first copy; the study uses exact character positions, which are correct |
| Qualitative spot check | six random 7B correctness verdicts read against the question, gold answer and generated answer: all six plausible (e.g. "No, Eugene is not a scientist" → correct; "the passages do not specify the rock type" vs a gold that names gneiss/granite → incorrect) |

## 8. Limitations
- **Paraphrases are not human-validated.** They were drafted by the author and checked only by embedding similarity (min cosine 0.826). Some "wording noise" could be a genuine change of meaning. A human check, or an LLM-and-human equivalence check, is the most important missing step before claiming "wording sensitivity" in a paper.
- **No human labels for correctness, relevance or context metrics.** RAGTruth (faithfulness) and a regex abstention label are the only human/rule-based anchors; ROUGE-L extremes are a proxy.
- **bf16 logit resolution** limits the determinism claim (§3.1).
- **Two gold-answer outliers** (items with 7 references; the correctness prompt includes at most 3) — 2 of 175 correctness items.
- **One run.** There is a single pass over a single 250-item sample; CIs reflect item-sampling, not repeated experiments on new data.
- **Ranking/validity checks have wide intervals** (164 hallucinated sentences; CIs ±0.04–0.06); neighbouring judges in §5.4 are not separable.
- **Citations not verified.** Related-work candidates found by search (e.g. "Rating Roulette", arXiv 2510.27106; arXiv 2609.13824 and 2606.19544 on consistency vs reliability) were not read in full and are deliberately not cited here.
- **Generalisation.** One RAG fixture (Mistral-7B generator, hybrid retriever), one dataset family (Wikipedia QA), English only.

## 9. Reproduction
Hardware used: Apple M4, 16 GB, macOS, system Python 3.9.6, `mlx-lm 0.29.1`. All model weights and datasets are pinned by commit/revision hash in the scripts and manifests; network is needed only to fetch them once.

```bash
python3 setup_clapnq.py --n 250 --answerable-frac 0.7 --drop-gold-frac 0.2 --seed 1234
python3 rag.py index && python3 rag.py generate          # retrieval index + generator answers
python3 refmetrics.py                                    # ROUGE-L / BERTScore / hit@k
caffeinate -ims bash sweep.sh                            # all judges: base rubric + determinism, then paraphrases (~16 h on this Mac)
python3 ragtruth.py prepare && bash finish.sh            # RAGTruth judgements (~2 h), then analysis
python3 paraphrase_check.py && python3 ragtruth_ci.py    # automated paraphrase check; AUC CIs
python3 analyze.py run                                   # tables -> data/results, figures -> figures
```
`python3 analyze.py selftest`, `judge.py selftest`, `rag.py selftest`, `ragtruth.py selftest` run the unit checks. Everything is resumable: re-running skips finished records.

Timing on this machine: stage 1 (base rubric, 7 judges) 6 h 15 min; stage 2 (3 paraphrases, 7 judges) ≈ 10 h; RAGTruth ≈ 2 h; analysis ≈ 4 min.

## 10. File index
| Path | Contents |
|---|---|
| `data/eval_set.jsonl`, `data/answers.jsonl` | the 250 questions; retrieved contexts and generator answers with model pins |
| `data/judgments/<judge>.jsonl` | 9,881 raw judgements per judge (d, mass, T0 label, seeded samples where stored) |
| `data/judgments/ragtruth_<judge>.jsonl` | RAGTruth sentence judgements |
| `data/refmetrics.jsonl` | ROUGE-L, BERTScore, hit@k per item |
| `data/results/*.csv`, `results.json` | every table cited above |
| `figures/fig1…fig5` | Figures 1, 2, 3 and 5 (Figure 4 would be the RQ4 anchor — not made) |
| `judge.py`, `rag.py`, `analyze.py`, `ragtruth.py`, `refmetrics.py`, `paraphrase_check.py`, `ragtruth_ci.py`, `sweep.sh`, `finish.sh` | code |
| `sweep.log`, `finish.log`, `finish2.log`, `analyze2.log` | run logs |
