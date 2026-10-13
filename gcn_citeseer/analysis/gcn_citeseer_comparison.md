# Comparison: gcn_citeseer

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 1.7425 ± 0.0135 | 0.2904 ± 0.0446 | 0.2648 ± 0.0317 | 0.2544 ± 0.0138 | 8 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 1.0604 ± 0.0086 | 0.7216 ± 0.0089 | 0.7160 ± 0.0076 | 0.7156 ± 0.0062 | 8 ± 0 | 0.000 ± 0.000 |
| adamw | 5 | 1.0290 ± 0.0251 | 0.6700 ± 0.0127 | 0.6630 ± 0.0109 | 0.6230 ± 0.0159 | 8 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 0.9527 ± 0.0185 | 0.6844 ± 0.0048 | 0.6906 ± 0.0073 | 0.6904 ± 0.0069 | 8 ± 0 | 0.000 ± 0.000 |
| nadam | 5 | 1.0598 ± 0.0059 | 0.7184 ± 0.0070 | 0.7160 ± 0.0050 | 0.7138 ± 0.0047 | 8 ± 0 | 0.000 ± 0.000 |
| b25 | 5 | 1.0223 ± 0.0262 | 0.6732 ± 0.0084 | 0.6684 ± 0.0148 | 0.6120 ± 0.0141 | 9 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 1.0175 ± 0.0295 | 0.6712 ± 0.0144 | 0.6642 ± 0.0079 | 0.6136 ± 0.0113 | 9 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00674 | [-0.01509, 0.00161] | -2.24 (4) | 0.0885 | -1.00 | 4/5 |
| val loss at end (exploratory) | 5 | 0.42486 | [0.35167, 0.49806] | 16.12 (4) | 0.0001 | 7.21 | 0/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00540 | [-0.00467, 0.01547] | 1.49 (4) | 0.2108 | 0.67 | 4/5 |
| test acc @end (exploratory) | 5 | -0.01100 | [-0.01840, -0.00360] | -4.13 (4) | 0.0145 | -1.85 | 0/5 |
| wall-clock s (exploratory) | 5 | 0.69615 | [0.60740, 0.78491] | 21.78 (4) | 0.0000 | 9.74 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.72024 | [-0.75808, -0.68240] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.03813 | [-0.06367, -0.01260] | 0.0143 | 0.0286 | 5/5 |
| rmsprop | 5 | 0.06964 | [0.04453, 0.09476] | 0.0015 | 0.0046 | 0/5 |
| nadam | 5 | -0.03750 | [-0.06415, -0.01086] | 0.0174 | 0.0286 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00480, 95% CI [-0.00439, 0.01399], p 0.2206, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 69.6 | 4.8 | 44.8 | 0.0 | 20.0 | 19.2 | 0.8 | 0.287 | 0.960 | 20.0 |
| b25_plain | 5 | 59.8 | 4.8 | 0.0 | 0.0 | 55.0 | 49.2 | 5.8 | 0.920 | 0.895 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 1.2573 | 5/5, 275 | 5/5, 150 | 5/5, 150 | both 5: B25 faster 5, slower 0; mean saving 120 units (95% CI 106 to 134); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 50% | 1.0290 | 2/5, 400 | 2/5, 200 | 2/5, 200 | both 2: B25 faster 2, slower 0; mean saving 200 units (95% CI -112 to 511); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 1.0290 | 2/5, 400 | 2/5, 200 | 2/5, 200 | both 2: B25 faster 2, slower 0; mean saving 200 units (95% CI -112 to 511); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 1.0290 | 2/5, 400 | 2/5, 200 | 2/5, 200 | both 2: B25 faster 2, slower 0; mean saving 200 units (95% CI -112 to 511); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 1.0290 | 2/5, 400 | 2/5, 200 | 2/5, 200 | both 2: B25 faster 2, slower 0; mean saving 200 units (95% CI -112 to 511); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 1.2573 | 5/5 / 5/5 | 120 (106 to 134) | 5 / 0 | YES | yes |
| 25% | 1% | 1.2699 | 5/5 / 5/5 | 120 (106 to 134) | 5 / 0 | YES | yes |
| 25% | 2% | 1.2825 | 5/5 / 5/5 | 125 (103 to 147) | 5 / 0 | YES | yes |
| 25% | 5% | 1.3202 | 5/5 / 5/5 | 110 (82 to 137) | 5 / 0 | YES | yes |
| 50% | 0% | 1.0290 | 2/5 / 2/5 | 200 (-112 to 511) | 2 / 0 | no | no |
| 50% | 1% | 1.0393 | 3/5 / 3/5 | 200 (138 to 262) | 3 / 0 | no | no |
| 50% | 2% | 1.0496 | 4/5 / 5/5 | 219 (169 to 269) | 4 / 0 | YES | no |
| 50% | 5% | 1.0805 | 5/5 / 5/5 | 185 (168 to 202) | 5 / 0 | YES | no |
| 75% | 0% | 1.0290 | 2/5 / 2/5 | 200 (-112 to 511) | 2 / 0 | no | no |
| 75% | 1% | 1.0393 | 3/5 / 3/5 | 200 (138 to 262) | 3 / 0 | no | no |
| 75% | 2% | 1.0496 | 4/5 / 5/5 | 219 (169 to 269) | 4 / 0 | YES | no |
| 75% | 5% | 1.0805 | 5/5 / 5/5 | 185 (168 to 202) | 5 / 0 | YES | no |
| 95% | 0% | 1.0290 | 2/5 / 2/5 | 200 (-112 to 511) | 2 / 0 | no | no |
| 95% | 1% | 1.0393 | 3/5 / 3/5 | 200 (138 to 262) | 3 / 0 | no | no |
| 95% | 2% | 1.0496 | 4/5 / 5/5 | 219 (169 to 269) | 4 / 0 | YES | no |
| 95% | 5% | 1.0805 | 5/5 / 5/5 | 185 (168 to 202) | 5 / 0 | YES | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 1.2573 ± 0.0502 | 1.0290 ± 0.0251 | 1.0290 ± 0.0251 | 1.0290 ± 0.0251 | 1.1686 ± 0.0281 |
| b25 | 1.0341 ± 0.0330 | 1.0223 ± 0.0262 | 1.0223 ± 0.0262 | 1.0223 ± 0.0262 | 1.1087 ± 0.0259 |
| b25_plain | 1.0175 ± 0.0295 | 1.0175 ± 0.0295 | 1.0175 ± 0.0295 | 1.0175 ± 0.0295 | 1.1039 ± 0.0265 |
| sgd | 1.7835 ± 0.0041 | 1.7768 ± 0.0039 | 1.7664 ± 0.0067 | 1.7425 ± 0.0135 | 1.7740 ± 0.0041 |
| adam | 1.5927 ± 0.0306 | 1.2973 ± 0.0312 | 1.1335 ± 0.0202 | 1.0604 ± 0.0086 | 1.3527 ± 0.0197 |
| rmsprop | 1.1766 ± 0.0450 | 1.0344 ± 0.0238 | 0.9791 ± 0.0213 | 0.9527 ± 0.0185 | 1.1093 ± 0.0296 |
| nadam | 1.5233 ± 0.0115 | 1.2516 ± 0.0143 | 1.1186 ± 0.0084 | 1.0598 ± 0.0059 | 1.3203 ± 0.0097 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.22321 | [-0.26554, -0.18088] | 0.0001 |
| at 50% of the budget | 5 | -0.00674 | [-0.01509, 0.00161] | 0.0885 |
| at 75% of the budget | 5 | -0.00674 | [-0.01509, 0.00161] | 0.0885 |
| at 100% of the budget | 5 | -0.00674 | [-0.01509, 0.00161] | 0.0885 |
| mean over the whole budget (area under the curve) | 5 | -0.05993 | [-0.07224, -0.04761] | 0.0002 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.