# Comparison: lstm_ptbword

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 6.5636 ± 0.0032 | 0.0886 ± 0.0026 | 0.0890 ± 0.0024 | 0.0890 ± 0.0024 | 33 ± 1 | 0.017 ± 0.001 |
| adam | 5 | 6.0140 ± 0.0123 | 0.1477 ± 0.0014 | 0.1480 ± 0.0013 | 0.1481 ± 0.0011 | 34 ± 0 | 0.071 ± 0.013 |
| adamw | 5 | 6.1659 ± 0.0855 | 0.1259 ± 0.0071 | 0.1280 ± 0.0067 | 0.1280 ± 0.0067 | 34 ± 0 | 0.007 ± 0.001 |
| rmsprop | 5 | 5.2544 ± 0.0138 | 0.2023 ± 0.0020 | 0.2008 ± 0.0020 | 0.2011 ± 0.0017 | 33 ± 0 | 0.004 ± 0.001 |
| nadam | 5 | 6.1087 ± 0.0130 | 0.1379 ± 0.0012 | 0.1386 ± 0.0015 | 0.1377 ± 0.0037 | 34 ± 0 | 0.021 ± 0.004 |
| b25 | 5 | 6.0213 ± 0.0331 | 0.1354 ± 0.0044 | 0.1374 ± 0.0048 | 0.1374 ± 0.0048 | 36 ± 0 | 0.006 ± 0.002 |
| b25_plain | 5 | 6.0322 ± 0.0255 | 0.1339 ± 0.0044 | 0.1362 ± 0.0050 | 0.1362 ± 0.0050 | 36 ± 0 | 0.006 ± 0.002 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.14454 | [-0.26508, -0.02400] | -3.33 (4) | 0.0291 | -1.49 | 4/5 |
| val loss at end (exploratory) | 5 | -0.14454 | [-0.26508, -0.02400] | -3.33 (4) | 0.0291 | -1.49 | 4/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00940 | [-0.00382, 0.02263] | 1.97 (4) | 0.1195 | 0.88 | 4/5 |
| test acc @end (exploratory) | 5 | 0.00940 | [-0.00382, 0.02263] | 1.97 (4) | 0.1195 | 0.88 | 4/5 |
| wall-clock s (exploratory) | 5 | 2.83821 | [2.72309, 2.95332] | 68.45 (4) | 0.0000 | 30.61 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.54225 | [-0.58088, -0.50361] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | 0.00735 | [-0.03684, 0.05155] | 0.6681 | 0.6681 | 2/5 |
| rmsprop | 5 | 0.76693 | [0.71445, 0.81942] | 0.0000 | 0.0000 | 0/5 |
| nadam | 5 | -0.08742 | [-0.13901, -0.03582] | 0.0093 | 0.0186 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.01089, 95% CI [-0.03263, 0.01086], p 0.2369, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 74.6 | 39.4 | 30.8 | 2.6 | 1.8 | 0.4 | 1.4 | 0.024 | 0.222 | 1.8 |
| b25_plain | 5 | 72.0 | 57.6 | 0.0 | 6.8 | 7.6 | 3.6 | 4.0 | 0.106 | 0.474 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 7.1188 | 5/5, 275 | 5/5, 200 | 5/5, 200 | both 5: B25 faster 5, slower 0; mean saving 70 units (95% CI 44 to 96); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 50% | 6.7434 | 5/5, 500 | 5/5, 325 | 5/5, 325 | both 5: B25 faster 5, slower 0; mean saving 190 units (95% CI 130 to 250); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 75% | 6.3912 | 5/5, 750 | 5/5, 500 | 5/5, 501 | both 5: B25 faster 5, slower 0; mean saving 250 units (95% CI 206 to 294); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 95% | 6.2004 | 3/5, 1000 | 5/5, 676 | 5/5, 678 | both 3: B25 faster 3, slower 0; mean saving 266 units (95% CI 66 to 466); only B25 2, only AdamW 0, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 6.1659 | 1/5, 850 | 5/5, 726 | 5/5, 751 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 4, only AdamW 0, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 7.1188 | 5/5 / 5/5 | 70 (44 to 96) | 5 / 0 | YES | yes |
| 25% | 1% | 7.1900 | 5/5 / 5/5 | 80 (66 to 94) | 5 / 0 | YES | yes |
| 25% | 2% | 7.2612 | 5/5 / 5/5 | 75 (53 to 97) | 5 / 0 | YES | yes |
| 25% | 5% | 7.4748 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | yes |
| 50% | 0% | 6.7434 | 5/5 / 5/5 | 190 (130 to 250) | 5 / 0 | YES | yes |
| 50% | 1% | 6.8108 | 5/5 / 5/5 | 140 (84 to 196) | 5 / 0 | YES | yes |
| 50% | 2% | 6.8783 | 5/5 / 5/5 | 115 (68 to 162) | 5 / 0 | YES | yes |
| 50% | 5% | 7.0806 | 5/5 / 5/5 | 70 (36 to 104) | 5 / 0 | YES | yes |
| 75% | 0% | 6.3912 | 5/5 / 5/5 | 250 (206 to 294) | 5 / 0 | YES | yes |
| 75% | 1% | 6.4551 | 5/5 / 5/5 | 270 (229 to 310) | 5 / 0 | YES | yes |
| 75% | 2% | 6.5190 | 5/5 / 5/5 | 260 (232 to 288) | 5 / 0 | YES | yes |
| 75% | 5% | 6.7107 | 5/5 / 5/5 | 205 (131 to 279) | 5 / 0 | YES | yes |
| 95% | 0% | 6.2004 | 3/5 / 5/5 | 266 (66 to 466) | 3 / 0 | no | no |
| 95% | 1% | 6.2624 | 5/5 / 5/5 | 279 (185 to 374) | 5 / 0 | YES | no |
| 95% | 2% | 6.3244 | 5/5 / 5/5 | 269 (198 to 341) | 5 / 0 | YES | no |
| 95% | 5% | 6.5105 | 5/5 / 5/5 | 255 (228 to 281) | 5 / 0 | YES | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 7.1188 ± 0.0361 | 6.7434 ± 0.0498 | 6.3912 ± 0.0875 | 6.1659 ± 0.0855 | 6.7530 ± 0.0423 |
| b25 | 6.9856 ± 0.0695 | 6.3912 ± 0.0314 | 6.1536 ± 0.0246 | 6.0213 ± 0.0331 | 6.5329 ± 0.0238 |
| b25_plain | 6.9856 ± 0.0695 | 6.4252 ± 0.0379 | 6.1763 ± 0.0280 | 6.0322 ± 0.0255 | 6.5491 ± 0.0252 |
| sgd | 7.5665 ± 0.0140 | 6.9216 ± 0.0060 | 6.7002 ± 0.0046 | 6.5636 ± 0.0032 | 7.2658 ± 0.0092 |
| adam | 6.3562 ± 0.0238 | 6.1336 ± 0.0094 | 6.0535 ± 0.0083 | 6.0140 ± 0.0123 | 6.2325 ± 0.0082 |
| rmsprop | 5.7917 ± 0.0216 | 5.5042 ± 0.0233 | 5.3586 ± 0.0175 | 5.2544 ± 0.0138 | 5.6166 ± 0.0220 |
| nadam | 6.3388 ± 0.0113 | 6.2001 ± 0.0167 | 6.1350 ± 0.0105 | 6.1087 ± 0.0130 | 6.2629 ± 0.0050 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.13321 | [-0.24517, -0.02125] | 0.0298 |
| at 50% of the budget | 5 | -0.35216 | [-0.40361, -0.30071] | 0.0000 |
| at 75% of the budget | 5 | -0.23756 | [-0.32596, -0.14916] | 0.0017 |
| at 100% of the budget | 5 | -0.14454 | [-0.26508, -0.02400] | 0.0291 |
| mean over the whole budget (area under the curve) | 5 | -0.22007 | [-0.26473, -0.17541] | 0.0002 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.