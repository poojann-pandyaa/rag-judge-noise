# Bigger-judge confirmation (Qwen2.5-14B, 4-bit) and 4-bit control

## Control: Qwen2.5-7B at 8-bit vs 4-bit (base rubric, T = 0)

| metric | units | T=0 verdict agreement | mean abs. margin change |
|---|---|---|---|
| faith | 574 | 94.8% | 1.75 |
| rel | 250 | 95.2% | 1.79 |
| prec | 525 | 93.0% | 2.28 |
| rec | 373 | 97.1% | 2.03 |
| corr | 175 | 95.4% | 2.25 |
| abst | 250 | 95.6% | 2.34 |

RAGTruth AUC: 8-bit 0.795 vs 4-bit 0.779 (difference 0.016). Quantisation effect small by the pre-registered rule: **yes** (lowest metric agreement 93.0%).

## F1. Temperature 0 is deterministic (serial / batch of 4)

- qwen2.5-7b-4bit: 431 units x 4 conditions, max margin spread 0.0 -> HOLDS
- qwen2.5-14b-4bit: 431 units x 4 conditions, max margin spread 0.0 -> HOLDS

## F2. Wording moves verdicts more than the seed

| metric | SD across wordings | SD across seeds (T=0.3) |
|---|---|---|
| faith | 0.079 | 0.015 |
| rel | 0.243 | 0.009 |
| prec | 0.055 | 0.009 |
| rec | 0.040 | 0.006 |
| corr | 0.102 | 0.015 |
| abst | 0.096 | 0.010 |

Wording > seed in 6 of 6 metrics; correctness verdicts change for 19% of answers across wordings vs 3% across seeds (Qwen2.5-7B 8-bit: 29% vs 6%). -> **HOLDS**

## F3. Judges disagree

- All 8 judges (7 original + 14B): alpha = 0.274 (original 7: 0.269); items with at least one dissent: 85% (original: 83%).
- The three largest judges (Qwen2.5-7B, Llama-3.1-8B, Qwen2.5-14B) disagree on 49% of the 175 correctness items.
- Qwen2.5-14B agreement with each other judge: qwen2.5-0.5b 46%, qwen2.5-1.5b 59%, qwen2.5-3b 78%, qwen2.5-7b 66%, llama3.2-1b 72%, llama3.2-3b 71%, llama3.1-8b 54%
-> **HOLDS**

## F4. Bigger judges: more correct, not clearly steadier

| judge | RAGTruth AUC [95% CI] | corr. seed flips (T=0.3) | mean seed SD, 6 metrics |
|---|---|---|---|
| qwen2.5-0.5b | 0.700 [0.627, 0.762] | 18% | 0.077 |
| qwen2.5-1.5b | 0.732 [0.677, 0.782] | 4% | 0.018 |
| qwen2.5-3b | 0.774 [0.721, 0.820] | 14% | 0.032 |
| qwen2.5-7b | 0.795 [0.750, 0.833] | 6% | 0.015 |
| qwen2.5-7b-4bit | 0.779 [0.730, 0.822] | - | - |
| qwen2.5-14b-4bit | 0.813 [0.768, 0.850] | 3% | 0.011 |

14B vs 7B AUC: 0.813 vs 0.795 -> higher (intervals overlap). Qwen ladder (5 judges) seed-noise trend p-values (judge-level OLS): faith 0.12, rel 0.07, prec 0.76, rec 0.03, corr 0.21, abst 0.15
