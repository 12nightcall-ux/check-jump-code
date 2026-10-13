# Comparison: gcn_cora

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 1.7904 ± 0.0427 | 0.4172 ± 0.0606 | 0.3848 ± 0.0682 | 0.3732 ± 0.0580 | 8 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 0.8503 ± 0.0056 | 0.7924 ± 0.0017 | 0.8100 ± 0.0054 | 0.8100 ± 0.0054 | 8 ± 0 | 0.000 ± 0.000 |
| adamw | 5 | 0.6642 ± 0.0148 | 0.7876 ± 0.0104 | 0.8052 ± 0.0075 | 0.7894 ± 0.0118 | 8 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 0.6628 ± 0.0086 | 0.7968 ± 0.0039 | 0.8138 ± 0.0061 | 0.8120 ± 0.0048 | 8 ± 0 | 0.000 ± 0.000 |
| nadam | 5 | 0.8506 ± 0.0088 | 0.7892 ± 0.0039 | 0.8098 ± 0.0008 | 0.8092 ± 0.0008 | 8 ± 0 | 0.000 ± 0.000 |
| b25 | 5 | 0.6609 ± 0.0109 | 0.7884 ± 0.0107 | 0.8072 ± 0.0060 | 0.7836 ± 0.0110 | 9 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 0.6598 ± 0.0117 | 0.7892 ± 0.0102 | 0.8056 ± 0.0068 | 0.7826 ± 0.0081 | 8 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00336 | [-0.01496, 0.00823] | -0.81 (4) | 0.4657 | -0.36 | 3/5 |
| val loss at end (exploratory) | 5 | 0.18106 | [0.13858, 0.22353] | 11.83 (4) | 0.0003 | 5.29 | 0/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00200 | [-0.00469, 0.00869] | 0.83 (4) | 0.4530 | 0.37 | 3/5 |
| test acc @end (exploratory) | 5 | -0.00580 | [-0.00901, -0.00259] | -5.01 (4) | 0.0074 | -2.24 | 0/5 |
| wall-clock s (exploratory) | 5 | 0.69062 | [0.63696, 0.74429] | 35.73 (4) | 0.0000 | 15.98 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -1.12957 | [-1.17980, -1.07934] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.18943 | [-0.20563, -0.17322] | 0.0000 | 0.0000 | 5/5 |
| rmsprop | 5 | -0.00194 | [-0.02117, 0.01729] | 0.7932 | 0.7932 | 2/5 |
| nadam | 5 | -0.18968 | [-0.21124, -0.16812] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00106, 95% CI [-0.00622, 0.00833], p 0.7076, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 69.8 | 3.2 | 46.8 | 0.0 | 19.8 | 19.4 | 0.4 | 0.284 | 0.980 | 19.8 |
| b25_plain | 5 | 59.8 | 5.2 | 0.0 | 0.4 | 54.2 | 49.6 | 4.6 | 0.906 | 0.915 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 1.2696 | 5/5, 250 | 5/5, 125 | 5/5, 125 | both 5: B25 faster 5, slower 0; mean saving 130 units (95% CI 116 to 144); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 50% | 0.6824 | 4/5, 500 | 5/5, 301 | 5/5, 226 | both 4: B25 faster 4, slower 0; mean saving 211 units (95% CI 187 to 235); only B25 1, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 75% | 0.6642 | 3/5, 525 | 2/5, 288 | 2/5, 250 | both 2: B25 faster 2, slower 0; mean saving 237 units (95% CI 77 to 396); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.6642 | 3/5, 525 | 2/5, 288 | 2/5, 250 | both 2: B25 faster 2, slower 0; mean saving 237 units (95% CI 77 to 396); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.6642 | 3/5, 525 | 2/5, 288 | 2/5, 250 | both 2: B25 faster 2, slower 0; mean saving 237 units (95% CI 77 to 396); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 1.2696 | 5/5 / 5/5 | 130 (116 to 144) | 5 / 0 | YES | yes |
| 25% | 1% | 1.2823 | 5/5 / 5/5 | 130 (116 to 144) | 5 / 0 | YES | yes |
| 25% | 2% | 1.2950 | 5/5 / 5/5 | 130 (116 to 143) | 5 / 0 | YES | yes |
| 25% | 5% | 1.3331 | 5/5 / 5/5 | 120 (94 to 146) | 5 / 0 | YES | yes |
| 50% | 0% | 0.6824 | 4/5 / 5/5 | 211 (187 to 235) | 4 / 0 | YES | yes |
| 50% | 1% | 0.6892 | 5/5 / 5/5 | 219 (193 to 245) | 5 / 0 | YES | yes |
| 50% | 2% | 0.6960 | 5/5 / 5/5 | 219 (193 to 245) | 5 / 0 | YES | yes |
| 50% | 5% | 0.7165 | 5/5 / 5/5 | 214 (186 to 242) | 5 / 0 | YES | yes |
| 75% | 0% | 0.6642 | 3/5 / 2/5 | 237 (77 to 396) | 2 / 0 | no | no |
| 75% | 1% | 0.6709 | 3/5 / 5/5 | 216 (180 to 252) | 3 / 0 | no | no |
| 75% | 2% | 0.6775 | 4/5 / 5/5 | 205 (187 to 223) | 4 / 0 | YES | no |
| 75% | 5% | 0.6975 | 5/5 / 5/5 | 209 (192 to 226) | 5 / 0 | YES | no |
| 95% | 0% | 0.6642 | 3/5 / 2/5 | 237 (77 to 396) | 2 / 0 | no | no |
| 95% | 1% | 0.6709 | 3/5 / 5/5 | 216 (180 to 252) | 3 / 0 | no | no |
| 95% | 2% | 0.6775 | 4/5 / 5/5 | 205 (187 to 223) | 4 / 0 | YES | no |
| 95% | 5% | 0.6975 | 5/5 / 5/5 | 209 (192 to 226) | 5 / 0 | YES | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 1.2696 ± 0.0696 | 0.6824 ± 0.0200 | 0.6642 ± 0.0148 | 0.6642 ± 0.0148 | 0.9589 ± 0.0261 |
| b25 | 0.7367 ± 0.0327 | 0.6609 ± 0.0109 | 0.6609 ± 0.0109 | 0.6609 ± 0.0109 | 0.8249 ± 0.0151 |
| b25_plain | 0.6794 ± 0.0274 | 0.6598 ± 0.0117 | 0.6598 ± 0.0117 | 0.6598 ± 0.0117 | 0.8185 ± 0.0167 |
| sgd | 1.9366 ± 0.0064 | 1.9144 ± 0.0084 | 1.8680 ± 0.0132 | 1.7904 ± 0.0427 | 1.8970 ± 0.0109 |
| adam | 1.5894 ± 0.0472 | 1.1272 ± 0.0211 | 0.9271 ± 0.0074 | 0.8503 ± 0.0056 | 1.2436 ± 0.0190 |
| rmsprop | 1.0048 ± 0.0744 | 0.7571 ± 0.0352 | 0.6869 ± 0.0181 | 0.6628 ± 0.0086 | 0.8916 ± 0.0417 |
| nadam | 1.4935 ± 0.0259 | 1.0747 ± 0.0131 | 0.9099 ± 0.0158 | 0.8506 ± 0.0088 | 1.2023 ± 0.0110 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.53289 | [-0.58672, -0.47905] | 0.0000 |
| at 50% of the budget | 5 | -0.02152 | [-0.04160, -0.00144] | 0.0409 |
| at 75% of the budget | 5 | -0.00336 | [-0.01496, 0.00823] | 0.4657 |
| at 100% of the budget | 5 | -0.00336 | [-0.01496, 0.00823] | 0.4657 |
| mean over the whole budget (area under the curve) | 5 | -0.13404 | [-0.15157, -0.11651] | 0.0000 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.