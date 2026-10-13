# Comparison: lstm_warpeace

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 2.5067 ± 0.0030 | 0.3090 ± 0.0032 | 0.3100 ± 0.0035 | 0.3100 ± 0.0035 | 18 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 1.9768 ± 0.0071 | 0.4345 ± 0.0026 | 0.4348 ± 0.0017 | 0.4332 ± 0.0037 | 18 ± 0 | 0.099 ± 0.015 |
| adamw | 5 | 1.6764 ± 0.0121 | 0.5046 ± 0.0026 | 0.5017 ± 0.0025 | 0.5005 ± 0.0023 | 19 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 1.5329 ± 0.0089 | 0.5415 ± 0.0030 | 0.5383 ± 0.0030 | 0.5376 ± 0.0034 | 18 ± 0 | 0.002 ± 0.001 |
| nadam | 5 | 2.0145 ± 0.0156 | 0.4260 ± 0.0043 | 0.4272 ± 0.0038 | 0.4171 ± 0.0065 | 19 ± 0 | 0.039 ± 0.008 |
| b25 | 5 | 1.6364 ± 0.0094 | 0.5143 ± 0.0023 | 0.5114 ± 0.0034 | 0.5114 ± 0.0034 | 20 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 1.6501 ± 0.0138 | 0.5116 ± 0.0026 | 0.5078 ± 0.0036 | 0.5078 ± 0.0036 | 20 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.03995 | [-0.04590, -0.03400] | -18.64 (4) | 0.0000 | -8.34 | 5/5 |
| val loss at end (exploratory) | 5 | -0.04214 | [-0.04770, -0.03659] | -21.06 (4) | 0.0000 | -9.42 | 5/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00977 | [0.00655, 0.01298] | 8.43 (4) | 0.0011 | 3.77 | 5/5 |
| test acc @end (exploratory) | 5 | 0.01093 | [0.00726, 0.01461] | 8.26 (4) | 0.0012 | 3.69 | 5/5 |
| wall-clock s (exploratory) | 5 | 1.24046 | [1.00940, 1.47151] | 14.91 (4) | 0.0001 | 6.67 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.87026 | [-0.88537, -0.85515] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.34041 | [-0.35403, -0.32679] | 0.0000 | 0.0000 | 5/5 |
| rmsprop | 5 | 0.10357 | [0.08544, 0.12170] | 0.0001 | 0.0001 | 0/5 |
| nadam | 5 | -0.37812 | [-0.39505, -0.36118] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.01368, 95% CI [-0.02532, -0.00204], p 0.0310, B25 better in 5/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 73.2 | 53.0 | 10.8 | 4.2 | 5.2 | 4.8 | 0.4 | 0.071 | 0.923 | 5.2 |
| b25_plain | 5 | 72.4 | 61.0 | 0.0 | 3.8 | 7.6 | 7.0 | 0.6 | 0.105 | 0.921 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 2.1028 | 5/5, 250 | 5/5, 250 | 5/5, 250 | both 5: B25 faster 4, slower 1; mean saving 20 units (95% CI 6 to 33); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 50% | 1.8565 | 5/5, 500 | 5/5, 426 | 5/5, 475 | both 5: B25 faster 5, slower 0; mean saving 85 units (95% CI 49 to 120); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 75% | 1.7442 | 5/5, 750 | 5/5, 626 | 5/5, 675 | both 5: B25 faster 5, slower 0; mean saving 125 units (95% CI 103 to 146); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 95% | 1.6903 | 4/5, 950 | 5/5, 776 | 5/5, 825 | both 4: B25 faster 4, slower 0; mean saving 162 units (95% CI 93 to 231); only B25 1, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 100% | 1.6764 | 2/5, 975 | 5/5, 850 | 5/5, 900 | both 2: B25 faster 2, slower 0; mean saving 162 units (95% CI 2 to 322); only B25 3, only AdamW 0, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 2.1028 | 5/5 / 5/5 | 20 (6 to 33) | 4 / 1 | YES | yes |
| 25% | 1% | 2.1238 | 5/5 / 5/5 | 15 (-2 to 31) | 3 / 2 | no | yes |
| 25% | 2% | 2.1449 | 5/5 / 5/5 | 10 (-7 to 26) | 2 / 3 | no | yes |
| 25% | 5% | 2.2080 | 5/5 / 5/5 | 5 (-9 to 18) | 1 / 4 | no | yes |
| 50% | 0% | 1.8565 | 5/5 / 5/5 | 85 (49 to 120) | 5 / 0 | YES | yes |
| 50% | 1% | 1.8750 | 5/5 / 5/5 | 70 (56 to 83) | 5 / 0 | YES | yes |
| 50% | 2% | 1.8936 | 5/5 / 5/5 | 55 (29 to 80) | 5 / 0 | YES | yes |
| 50% | 5% | 1.9493 | 5/5 / 5/5 | 35 (18 to 51) | 5 / 0 | YES | yes |
| 75% | 0% | 1.7442 | 5/5 / 5/5 | 125 (103 to 146) | 5 / 0 | YES | yes |
| 75% | 1% | 1.7616 | 5/5 / 5/5 | 105 (79 to 130) | 5 / 0 | YES | yes |
| 75% | 2% | 1.7791 | 5/5 / 5/5 | 110 (82 to 137) | 5 / 0 | YES | yes |
| 75% | 5% | 1.8314 | 5/5 / 5/5 | 75 (53 to 96) | 5 / 0 | YES | yes |
| 95% | 0% | 1.6903 | 4/5 / 5/5 | 162 (93 to 231) | 4 / 0 | YES | yes |
| 95% | 1% | 1.7072 | 5/5 / 5/5 | 145 (111 to 179) | 5 / 0 | YES | yes |
| 95% | 2% | 1.7241 | 5/5 / 5/5 | 155 (129 to 180) | 5 / 0 | YES | yes |
| 95% | 5% | 1.7748 | 5/5 / 5/5 | 105 (79 to 130) | 5 / 0 | YES | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 2.1028 ± 0.0181 | 1.8565 ± 0.0175 | 1.7442 ± 0.0136 | 1.6764 ± 0.0121 | 1.9684 ± 0.0165 |
| b25 | 2.1135 ± 0.0214 | 1.8255 ± 0.0152 | 1.7071 ± 0.0097 | 1.6364 ± 0.0094 | 1.9637 ± 0.0150 |
| b25_plain | 2.1317 ± 0.0195 | 1.8500 ± 0.0199 | 1.7272 ± 0.0162 | 1.6501 ± 0.0138 | 1.9807 ± 0.0190 |
| sgd | 3.1571 ± 0.0026 | 2.9447 ± 0.0306 | 2.6666 ± 0.0235 | 2.5067 ± 0.0030 | 3.0105 ± 0.0148 |
| adam | 2.1900 ± 0.0067 | 2.0600 ± 0.0067 | 2.0078 ± 0.0090 | 1.9768 ± 0.0071 | 2.1507 ± 0.0068 |
| rmsprop | 1.8003 ± 0.0350 | 1.6467 ± 0.0121 | 1.5762 ± 0.0107 | 1.5329 ± 0.0089 | 1.7333 ± 0.0286 |
| nadam | 2.2167 ± 0.0203 | 2.0945 ± 0.0128 | 2.0472 ± 0.0146 | 2.0145 ± 0.0156 | 2.1690 ± 0.0123 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.01068 | [-0.00975, 0.03111] | 0.2203 |
| at 50% of the budget | 5 | -0.03101 | [-0.04154, -0.02047] | 0.0012 |
| at 75% of the budget | 5 | -0.03706 | [-0.04623, -0.02789] | 0.0004 |
| at 100% of the budget | 5 | -0.03995 | [-0.04590, -0.03400] | 0.0000 |
| mean over the whole budget (area under the curve) | 5 | -0.00473 | [-0.01404, 0.00457] | 0.2307 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.