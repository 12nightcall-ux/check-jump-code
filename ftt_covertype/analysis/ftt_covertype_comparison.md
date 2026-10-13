# Comparison: ftt_covertype

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.6177 ± 0.0130 | 0.7346 ± 0.0053 | 0.7307 ± 0.0047 | 0.7277 ± 0.0072 | 76 ± 0 | 0.809 ± 0.022 |
| adam | 5 | 0.5230 ± 0.0182 | 0.7804 ± 0.0110 | 0.7782 ± 0.0124 | 0.7722 ± 0.0162 | 76 ± 0 | 1.000 ± 0.000 |
| adamw | 5 | 0.4073 ± 0.0096 | 0.8328 ± 0.0034 | 0.8329 ± 0.0063 | 0.8302 ± 0.0051 | 77 ± 0 | 1.000 ± 0.000 |
| rmsprop | 5 | 0.3942 ± 0.0144 | 0.8358 ± 0.0066 | 0.8371 ± 0.0046 | 0.8348 ± 0.0047 | 76 ± 0 | 0.902 ± 0.017 |
| nadam | 5 | 0.5010 ± 0.0105 | 0.7918 ± 0.0044 | 0.7902 ± 0.0067 | 0.7869 ± 0.0019 | 77 ± 0 | 1.000 ± 0.000 |
| b25 | 5 | 0.4084 ± 0.0056 | 0.8311 ± 0.0039 | 0.8327 ± 0.0026 | 0.8306 ± 0.0051 | 82 ± 0 | 1.000 ± 0.000 |
| b25_plain | 5 | 0.4084 ± 0.0056 | 0.8311 ± 0.0039 | 0.8327 ± 0.0026 | 0.8306 ± 0.0051 | 81 ± 0 | 1.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00115 | [-0.00513, 0.00743] | 0.51 (4) | 0.6383 | 0.23 | 2/5 |
| val loss at end (exploratory) | 5 | 0.00057 | [-0.00440, 0.00555] | 0.32 (4) | 0.7646 | 0.14 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00020 | [-0.00540, 0.00500] | -0.11 (4) | 0.9201 | -0.05 | 2/5 |
| test acc @end (exploratory) | 5 | 0.00040 | [-0.00134, 0.00214] | 0.64 (4) | 0.5579 | 0.29 | 2/5 |
| wall-clock s (exploratory) | 5 | 5.22610 | [4.96627, 5.48593] | 55.84 (4) | 0.0000 | 24.97 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.20929 | [-0.22581, -0.19278] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.11460 | [-0.13415, -0.09505] | 0.0001 | 0.0002 | 5/5 |
| rmsprop | 5 | 0.01422 | [0.00154, 0.02690] | 0.0358 | 0.0358 | 1/5 |
| nadam | 5 | -0.09253 | [-0.10244, -0.08263] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00000, 95% CI [0.00000, 0.00000], p 1.0000, B25 better in 0/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 75.6 | 50.2 | 24.8 | 0.6 | 0.0 | 0.0 | 0.0 | 0.000 | n/a | 0.0 |
| b25_plain | 5 | 75.6 | 75.0 | 0.0 | 0.6 | 0.0 | 0.0 | 0.0 | 0.000 | n/a | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6173 | 5/5, 275 | 5/5, 275 | 5/5, 275 | both 5: B25 faster 0, slower 0; mean saving 0 units (95% CI 0 to 0); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.5331 | 5/5, 500 | 5/5, 500 | 5/5, 500 | both 5: B25 faster 2, slower 1; mean saving 10 units (95% CI -25 to 45); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.4640 | 5/5, 750 | 5/5, 750 | 5/5, 750 | both 5: B25 faster 2, slower 1; mean saving 5 units (95% CI -21 to 31); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.4176 | 4/5, 950 | 5/5, 975 | 5/5, 975 | both 4: B25 faster 2, slower 2; mean saving 0 units (95% CI -46 to 46); only B25 1, only AdamW 0, neither 0 | saving NOT supported |
| 100% | 0.4073 | 2/5, 988 | 2/5, 988 | 2/5, 988 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 1, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6173 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 25% | 1% | 0.6234 | 5/5 / 5/5 | -10 (-38 to 18) | 0 / 1 | no | no |
| 25% | 2% | 0.6296 | 5/5 / 5/5 | 0 (-22 to 22) | 1 / 1 | no | no |
| 25% | 5% | 0.6481 | 5/5 / 5/5 | 0 (-38 to 38) | 2 / 1 | no | yes |
| 50% | 0% | 0.5331 | 5/5 / 5/5 | 10 (-25 to 45) | 2 / 1 | no | no |
| 50% | 1% | 0.5384 | 5/5 / 5/5 | -5 (-19 to 9) | 0 / 1 | no | no |
| 50% | 2% | 0.5438 | 5/5 / 5/5 | 5 (-29 to 39) | 1 / 1 | no | no |
| 50% | 5% | 0.5597 | 5/5 / 5/5 | 0 (-66 to 66) | 1 / 1 | no | yes |
| 75% | 0% | 0.4640 | 5/5 / 5/5 | 5 (-21 to 31) | 2 / 1 | no | no |
| 75% | 1% | 0.4687 | 5/5 / 5/5 | -5 (-51 to 41) | 1 / 2 | no | no |
| 75% | 2% | 0.4733 | 5/5 / 5/5 | -20 (-60 to 20) | 1 / 3 | no | no |
| 75% | 5% | 0.4872 | 5/5 / 5/5 | -10 (-66 to 46) | 2 / 3 | no | yes |
| 95% | 0% | 0.4176 | 4/5 / 5/5 | 0 (-46 to 46) | 2 / 2 | no | no |
| 95% | 1% | 0.4218 | 5/5 / 5/5 | 0 (-31 to 31) | 2 / 2 | no | no |
| 95% | 2% | 0.4260 | 5/5 / 5/5 | -15 (-62 to 32) | 1 / 2 | no | no |
| 95% | 5% | 0.4385 | 5/5 / 5/5 | 0 (-22 to 22) | 1 / 1 | no | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6173 ± 0.0074 | 0.5331 ± 0.0123 | 0.4640 ± 0.0092 | 0.4073 ± 0.0096 | 0.5468 ± 0.0047 |
| b25 | 0.6196 ± 0.0045 | 0.5373 ± 0.0138 | 0.4584 ± 0.0074 | 0.4084 ± 0.0056 | 0.5476 ± 0.0045 |
| b25_plain | 0.6196 ± 0.0045 | 0.5373 ± 0.0138 | 0.4584 ± 0.0074 | 0.4084 ± 0.0056 | 0.5476 ± 0.0045 |
| sgd | 0.7006 ± 0.0141 | 0.6627 ± 0.0110 | 0.6419 ± 0.0120 | 0.6177 ± 0.0130 | 0.6946 ± 0.0156 |
| adam | 0.6435 ± 0.0035 | 0.5944 ± 0.0072 | 0.5582 ± 0.0085 | 0.5230 ± 0.0182 | 0.6110 ± 0.0060 |
| rmsprop | 0.5913 ± 0.0073 | 0.5039 ± 0.0089 | 0.4398 ± 0.0109 | 0.3942 ± 0.0144 | 0.5213 ± 0.0072 |
| nadam | 0.6425 ± 0.0035 | 0.5884 ± 0.0122 | 0.5384 ± 0.0107 | 0.5010 ± 0.0105 | 0.6023 ± 0.0069 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00236 | [-0.00219, 0.00692] | 0.2231 |
| at 50% of the budget | 5 | 0.00416 | [-0.00477, 0.01309] | 0.2658 |
| at 75% of the budget | 5 | -0.00560 | [-0.01159, 0.00038] | 0.0600 |
| at 100% of the budget | 5 | 0.00115 | [-0.00513, 0.00743] | 0.6383 |
| mean over the whole budget (area under the curve) | 5 | 0.00081 | [-0.00255, 0.00418] | 0.5398 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.