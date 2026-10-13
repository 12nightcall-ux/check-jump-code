# Comparison: distilbert_cola

GPU: NVIDIA A100-SXM4-40GB | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val MCC @best-val ckpt | test MCC @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.5022 ± 0.0093 | 0.7576 ± 0.0039 | 0.7593 ± 0.0163 | 0.7595 ± 0.0125 | 0.4172 ± 0.0177 | 0.4458 ± 0.0346 | 189 ± 0 | 0.957 ± 0.012 |
| adam | 5 | 0.4617 ± 0.0099 | 0.7937 ± 0.0124 | 0.7953 ± 0.0034 | 0.7981 ± 0.0071 | 0.4977 ± 0.0349 | 0.5164 ± 0.0061 | 190 ± 0 | 0.930 ± 0.008 |
| adamw | 5 | 0.4610 ± 0.0094 | 0.7937 ± 0.0117 | 0.7942 ± 0.0046 | 0.7967 ± 0.0064 | 0.4976 ± 0.0329 | 0.5137 ± 0.0093 | 190 ± 0 | 0.929 ± 0.012 |
| rmsprop | 5 | 0.4574 ± 0.0086 | 0.7845 ± 0.0085 | 0.7832 ± 0.0087 | 0.7916 ± 0.0058 | 0.4792 ± 0.0280 | 0.4913 ± 0.0207 | 187 ± 0 | 0.909 ± 0.009 |
| nadam | 5 | 0.4568 ± 0.0056 | 0.7927 ± 0.0058 | 0.7825 ± 0.0179 | 0.7935 ± 0.0033 | 0.5088 ± 0.0207 | 0.4990 ± 0.0296 | 189 ± 0 | 0.900 ± 0.006 |
| b25 | 5 | 0.4596 ± 0.0086 | 0.7881 ± 0.0147 | 0.7895 ± 0.0060 | 0.7946 ± 0.0085 | 0.4852 ± 0.0384 | 0.5051 ± 0.0241 | 205 ± 1 | 0.947 ± 0.008 |
| b25_plain | 5 | 0.4584 ± 0.0069 | 0.7931 ± 0.0139 | 0.7848 ± 0.0144 | 0.7942 ± 0.0046 | 0.4996 ± 0.0451 | 0.4935 ± 0.0393 | 202 ± 0 | 0.937 ± 0.008 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00143 | [-0.00833, 0.00548] | -0.57 (4) | 0.5973 | -0.26 | 2/5 |
| val loss at end (exploratory) | 5 | -0.14696 | [-0.26042, -0.03349] | -3.60 (4) | 0.0228 | -1.61 | 5/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00468 | [-0.01648, 0.00712] | -1.10 (4) | 0.3327 | -0.49 | 1/5 |
| test acc @end (exploratory) | 5 | -0.00211 | [-0.01265, 0.00844] | -0.55 (4) | 0.6088 | -0.25 | 3/5 |
| test MCC @best-val ckpt (exploratory; CoLA) | 5 | -0.00867 | [-0.04630, 0.02895] | -0.64 (4) | 0.5570 | -0.29 | 3/5 |
| wall-clock s (exploratory) | 5 | 15.15209 | [14.39854, 15.90564] | 55.83 (4) | 0.0000 | 24.97 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.04262 | [-0.06252, -0.02273] | 0.0040 | 0.0160 | 5/5 |
| adam | 5 | -0.00212 | [-0.00955, 0.00531] | 0.4717 | 1.0000 | 2/5 |
| rmsprop | 5 | 0.00218 | [-0.00932, 0.01368] | 0.6269 | 1.0000 | 1/5 |
| nadam | 5 | 0.00276 | [-0.00507, 0.01060] | 0.3828 | 1.0000 | 3/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00123, 95% CI [-0.00533, 0.00778], p 0.6313, B25 better in 1/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 71.8 | 34.6 | 24.2 | 1.8 | 11.2 | 3.6 | 7.6 | 0.156 | 0.321 | 11.2 |
| b25_plain | 5 | 69.6 | 47.6 | 0.0 | 3.4 | 18.6 | 4.6 | 14.0 | 0.267 | 0.247 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.4670 | 3/5, 225 | 4/5, 276 | 4/5, 252 | both 3: B25 faster 1, slower 2; mean saving 16 units (95% CI -277 to 309); only B25 1, only AdamW 0, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.4610 | 2/5, 250 | 3/5, 300 | 4/5, 263 | both 2: B25 faster 0, slower 2; mean saving -26 units (95% CI -26 to -25); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.4610 | 2/5, 250 | 3/5, 300 | 4/5, 263 | both 2: B25 faster 0, slower 2; mean saving -26 units (95% CI -26 to -25); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.4610 | 2/5, 250 | 3/5, 300 | 4/5, 263 | both 2: B25 faster 0, slower 2; mean saving -26 units (95% CI -26 to -25); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.4610 | 2/5, 250 | 3/5, 300 | 4/5, 263 | both 2: B25 faster 0, slower 2; mean saving -26 units (95% CI -26 to -25); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.4670 | 3/5 / 4/5 | 16 (-277 to 309) | 1 / 2 | no | no |
| 25% | 1% | 0.4716 | 5/5 / 5/5 | -1 (-115 to 113) | 1 / 4 | no | no |
| 25% | 2% | 0.4763 | 5/5 / 5/5 | -1 (-115 to 113) | 1 / 4 | no | no |
| 25% | 5% | 0.4903 | 5/5 / 5/5 | -16 (-58 to 26) | 0 / 4 | no | no |
| 50% | 0% | 0.4610 | 2/5 / 3/5 | -26 (-26 to -25) | 0 / 2 | no | no |
| 50% | 1% | 0.4656 | 3/5 / 4/5 | 16 (-277 to 309) | 1 / 2 | no | no |
| 50% | 2% | 0.4702 | 5/5 / 4/5 | 6 (-70 to 81) | 1 / 3 | no | no |
| 50% | 5% | 0.4841 | 5/5 / 5/5 | -31 (-82 to 21) | 1 / 3 | no | no |
| 75% | 0% | 0.4610 | 2/5 / 3/5 | -26 (-26 to -25) | 0 / 2 | no | no |
| 75% | 1% | 0.4656 | 3/5 / 4/5 | 16 (-277 to 309) | 1 / 2 | no | no |
| 75% | 2% | 0.4702 | 5/5 / 4/5 | 6 (-70 to 81) | 1 / 3 | no | no |
| 75% | 5% | 0.4841 | 5/5 / 5/5 | -31 (-82 to 21) | 1 / 3 | no | no |
| 95% | 0% | 0.4610 | 2/5 / 3/5 | -26 (-26 to -25) | 0 / 2 | no | no |
| 95% | 1% | 0.4656 | 3/5 / 4/5 | 16 (-277 to 309) | 1 / 2 | no | no |
| 95% | 2% | 0.4702 | 5/5 / 4/5 | 6 (-70 to 81) | 1 / 3 | no | no |
| 95% | 5% | 0.4841 | 5/5 / 5/5 | -31 (-82 to 21) | 1 / 3 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.4670 ± 0.0087 | 0.4610 ± 0.0094 | 0.4610 ± 0.0094 | 0.4610 ± 0.0094 | 0.4752 ± 0.0078 |
| b25 | 0.4818 ± 0.0090 | 0.4604 ± 0.0073 | 0.4596 ± 0.0086 | 0.4596 ± 0.0086 | 0.4769 ± 0.0068 |
| b25_plain | 0.4819 ± 0.0023 | 0.4584 ± 0.0069 | 0.4584 ± 0.0069 | 0.4584 ± 0.0069 | 0.4759 ± 0.0055 |
| sgd | 0.5246 ± 0.0077 | 0.5022 ± 0.0093 | 0.5022 ± 0.0093 | 0.5022 ± 0.0093 | 0.5200 ± 0.0069 |
| adam | 0.4669 ± 0.0087 | 0.4617 ± 0.0099 | 0.4617 ± 0.0099 | 0.4617 ± 0.0099 | 0.4756 ± 0.0080 |
| rmsprop | 0.4574 ± 0.0086 | 0.4574 ± 0.0086 | 0.4574 ± 0.0086 | 0.4574 ± 0.0086 | 0.4691 ± 0.0085 |
| nadam | 0.4579 ± 0.0069 | 0.4568 ± 0.0056 | 0.4568 ± 0.0056 | 0.4568 ± 0.0056 | 0.4681 ± 0.0057 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.01488 | [0.00413, 0.02563] | 0.0184 |
| at 50% of the budget | 5 | -0.00060 | [-0.00842, 0.00722] | 0.8420 |
| at 75% of the budget | 5 | -0.00143 | [-0.00833, 0.00548] | 0.5973 |
| at 100% of the budget | 5 | -0.00143 | [-0.00833, 0.00548] | 0.5973 |
| mean over the whole budget (area under the curve) | 5 | 0.00168 | [-0.00454, 0.00789] | 0.4956 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.