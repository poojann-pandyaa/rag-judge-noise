# rag-judge-noise

How far can you trust a small local LLM judge for RAG evaluation?

This repository contains the code, pinned model revisions, raw judgments and result tables behind the article
*"How Far Can You Trust a Small Local LLM Judge for RAG Evaluation?"* We scored the answers of one fixed RAG pipeline
(250 CLAPnq questions) with seven open-weight judge models (0.5B to 8B parameters, all 8-bit, run with MLX on a 16 GB Apple M4 laptop)
and measured how stable and how correct their verdicts are.

## What we found (short version)

1. **Temperature 0 is deterministic if you score one prompt at a time.** 431 prompts per judge scored four ways gave identical logit margins.
   With large batches (32 or more) a few margins moved by one half-precision step (0.0156), which can only flip a verdict that sits on the borderline (about 0.6% of items or fewer).
2. **Rewording the question moves verdicts more than changing the random seed.** Across the six metrics, rewording had more effect in 40 of 42 judge-metric cells.
   For the correctness verdict, 25-45% of answers changed under rewording for every judge.
3. **Judges disagree.** On 83% of answers at least one of the seven judges disagreed with the rest (Krippendorff alpha 0.27).
4. **Bigger judges are not clearly steadier, but they are more correct.** On human-labelled hallucinations (RAGTruth) the faithfulness AUC rises from 0.50 (Llama-3.2-1B) to 0.80 (Qwen2.5-7B).
   Some small judges are very "stable" only because they answer the same thing every time.

**Bigger-judge check.** We repeated the full protocol with Qwen2.5-14B (4-bit, since the 8-bit 14B does not fit in 16 GB), plus a 4-bit Qwen2.5-7B
control to separate "bigger" from "4-bit". Pass/fail criteria were written before any of these judgments existed (`confirm_big.py`). All four findings held:
identical margins at T = 0; wording beats seed in 6 of 6 metrics (correctness flips 19% vs 3%); with 8 judges 85% of answers still have a dissenter (alpha 0.27);
the 14B has the best RAGTruth AUC, 0.81 [0.77, 0.85], overlapping the 7B's. The 4-bit control agrees with 8-bit on 93-97% of verdicts (AUC 0.78 vs 0.80).
Full report: `data/results_ext/confirmation.md`. These results are kept separate; no number for the original seven judges changes.

**Important caveat about "seeds".** Each judgment was computed once. The model's Yes/No score margin `d = logit(Yes) - logit(No)` is the temperature-0 result.
The "seeded" verdicts at temperature T are random draws with probability `sigmoid(d / T)` computed from that margin. They are therefore *not* independent runs of the model;
they show how close each call is to the borderline. A judge that writes out its reasoning before answering would add more variation than we measured.

## Layout

```
setup_clapnq.py        download the pinned CLAPnq files and sample the 250 evaluation questions
rag.py                 build the retrieval index (BM25 + bge-base-en-v1.5, reciprocal rank fusion) and generate answers (Mistral-7B)
refmetrics.py          reference metrics: hit@k, ROUGE-L, BERTScore
judge.py               the judge harness (Yes/No logit margins, seeded draws, 4 rubric wordings, determinism check)
sweep.sh, finish.sh    resumable drivers for the full run
ragtruth.py            human-labelled faithfulness check (RAGTruth)
paraphrase_check.py    automated similarity check of the rubric paraphrases
ragtruth_ci.py         confidence intervals for the RAGTruth AUCs
analyze.py             all tables and figures from the raw judgments
make_article_figures.py  the three figures used in the article
examples/judge_margin_demo.py   the short margin demo shown in the article
data/judgments/        raw judgments: one file per judge (9,881 records each) + RAGTruth judgments (1,092 each)
data/results/          every result table as CSV
sweep_big.sh, confirm_big.py   bigger-judge check: driver and pre-registered confirmation report
data/results_ext/      result tables with the 14B added (RJN_EXT=1), plus confirmation.md
figures/, article_figures/   figures
docs/                  longer technical report and process notes
```

## Reproduce

Tested with Python 3.9 on an Apple M4 (macOS). See `requirements.txt` for the versions used.

**1. Analysis only (about 5 minutes, any machine with the packages installed).** Uses the raw judgments shipped here:

```
python3 analyze.py selftest
python3 analyze.py run          # writes data/results/*.csv and figures/*.png
python3 ragtruth.py report
python3 ragtruth_ci.py
python3 make_article_figures.py
```

**2. Re-run the judges (about 18 hours on a 16 GB M4).** Needs the full datasets (steps in section 3 regenerate the text files that are not shipped for RAGTruth):

```
caffeinate -ims bash sweep.sh        # 7 judges: base rubric + determinism check, then 3 paraphrases
python3 ragtruth.py prepare          # rebuilds data/ragtruth_eval.jsonl with the source text, needs data/raw/ragtruth
bash finish.sh                       # RAGTruth judgments, then the analysis
```

**Bigger-judge check (about 13 hours on a 16 GB M4).** `caffeinate -ims bash sweep_big.sh` runs the 4-bit control and the 14B, then
`RJN_EXT=1 python3 analyze.py run`, `RJN_EXT=1 python3 ragtruth.py report`, `RJN_EXT=1 python3 ragtruth_ci.py` and `python3 confirm_big.py`.
With the shipped judgments, just run those last four commands.

**3. From scratch** (download the data, build the index, generate the answers):

```
python3 setup_clapnq.py --n 250 --answerable-frac 0.7 --drop-gold-frac 0.2 --seed 1234
python3 rag.py index && python3 rag.py generate
python3 refmetrics.py
```

All model revisions and dataset commits are pinned in the scripts and in `data/manifest.json`. The sweep is resumable: re-running skips finished records.

## Data, licences and what is not included

See `NOTICE.md`. In short: the code is MIT-licensed; the 250-question CLAPnq sample (questions, reference answers, passages) is redistributed under the dataset's Apache-2.0 licence
with attribution; RAGTruth source text is not redistributed (only ids and human labels); model weights are not included and carry their own licences.

## Limits of the study

Seven judges in two families; 250 English, Wikipedia-based questions; rubric paraphrases written by the author and checked only automatically; single-token Yes/No judging (not reasoning judges);
no comparison with a large commercial judge; only faithfulness has human labels. Process notes (a memory cap added mid-run, a fixed table bug, the checks done) are in `docs/PROCESS_NOTES.md`.
