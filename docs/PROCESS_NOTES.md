# Process notes

Notes on how the experiment was run, kept out of the article so that the article can stay focused. None of these changes the reported results.

## Things that went wrong or were fixed

* **Memory cap added mid-sweep.** The operating system killed the first two attempts on the smallest judge ("Killed: 9", out of memory) while a separate BERTScore job was running.
  We then capped the MLX cache at 512 MB and cleared it after every chunk (see `Judge.__init__` and `run` in `judge.py`). This only affects speed. The interrupted runs resumed from their saved records.
* **A table bug, found and fixed during analysis.** The pairwise judge-agreement table for correctness first counted items a judge had not scored (175 of 250 items have correctness) as disagreements,
  which gave a diagonal of 0.70 instead of 1.0. It is fixed in `analyze.py`; only that table and the figure drawn from it were affected, and the numbers in the article use the corrected version.
* **Interrupted post-processing job.** The queued RAGTruth/analysis job was stopped once when the working session restarted (Llama-3.1-8B RAGTruth at 960 of 1,092 sentences).
  It was relaunched and resumed from the saved records; the final analysis ran after all judgments were complete.

## Checks done after the run

* Every judge file has exactly 9,881 records (3,009 base-rubric serial + 431 batch-of-four + 3 x 2,147 paraphrase), no duplicate keys, no non-finite values, and identical key sets across the seven judges.
  Every RAGTruth file has 1,092 unique sentence records.
* The headline numbers were re-derived with separate code straight from the raw records: determinism spread, wording vs seed spread, Krippendorff alpha and dissent rate, capacity-trend slopes,
  and RAGTruth AUCs (labels rebuilt from the raw span annotations). All matched; the single difference (165 vs 164 hallucinated sentences) was caused by a sentence that appears twice in one response and a naive text search in the check, not by the study code.
* The seeded draws match the theoretical `sigmoid(d / T)` flip rates and Yes-rates for all seven judges.
* Determinism stress test (Qwen2.5-0.5B, 96 random prompts of 82 to 1,056 tokens): a fresh batch-of-1 run reproduced the stored values exactly; batch size 8 was identical;
  batch sizes 32 and 96 changed 4 of 96 margins by one bf16 step (0.0156). Only the 0.5B judge was stress-tested.
* `python3 examples/judge_margin_demo.py` reproduces the study harness's margin exactly on the same prompt (12.0781 for the Qwen2.5-1.5B example).

## Bigger-judge check (25-26 November 2026)

* Qwen2.5-14B-Instruct-4bit and a Qwen2.5-7B-Instruct-4bit control, revisions pinned in `judge.py` (`EXTRA`). The 8-bit 14B (15.7 GB) does not fit in 16 GB of RAM.
* Pass/fail criteria were written into `confirm_big.py` before any of these judgments existed. Driver: `sweep_big.sh` (about 13 hours; no failures or retries).
* Both new judges have the full 9,881 + 1,092 records with no duplicates. Results are written to `data/results_ext/`; `data/results/` and `figures/` are untouched.
  For the original seven judges every point estimate is unchanged. Some bootstrap interval bounds in the extended tables differ by up to about 0.01,
  because the extra judge shifts the shared random-number stream. The RAGTruth AUC intervals are identical.
* Outcome: all four findings held, and the 4-bit effect was small by the pre-set rule. See `data/results_ext/confirmation.md`.

## Known gaps

The three alternative rubric wordings were written by the author and checked only by embedding similarity (minimum cosine 0.826 to the base rubric); no human check of equivalence was done.
