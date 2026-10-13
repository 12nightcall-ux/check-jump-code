# Comparison: bilstm_mrpc

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val F1 @best-val ckpt | test F1 @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.5889 ± 0.0089 | 0.7039 ± 0.0087 | 0.6916 ± 0.0097 | 0.6986 ± 0.0074 | 0.8129 ± 0.0098 | 0.7979 ± 0.0109 | 808 ± 2 | 0.004 ± 0.004 |
| adam | 5 | 0.5246 ± 0.0109 | 0.7412 ± 0.0045 | 0.6970 ± 0.0122 | 0.7063 ± 0.0113 | 0.8237 ± 0.0095 | 0.7900 ± 0.0099 | 813 ± 10 | 0.438 ± 0.016 |
| adamw | 5 | 0.5056 ± 0.0046 | 0.7480 ± 0.0091 | 0.7041 ± 0.0057 | 0.7166 ± 0.0277 | 0.8271 ± 0.0047 | 0.7922 ± 0.0108 | 832 ± 1 | 0.624 ± 0.024 |
| rmsprop | 5 | 0.5313 ± 0.0051 | 0.7441 ± 0.0045 | 0.6992 ± 0.0076 | 0.7079 ± 0.0097 | 0.8273 ± 0.0048 | 0.7886 ± 0.0083 | 790 ± 5 | 0.635 ± 0.015 |
| nadam | 5 | 0.5280 ± 0.0111 | 0.7377 ± 0.0118 | 0.7074 ± 0.0138 | 0.6975 ± 0.0104 | 0.8230 ± 0.0064 | 0.7962 ± 0.0089 | 854 ± 2 | 0.439 ± 0.024 |
| b25 | 5 | 0.5099 ± 0.0105 | 0.7510 ± 0.0227 | 0.6959 ± 0.0132 | 0.7275 ± 0.0109 | 0.8279 ± 0.0172 | 0.7849 ± 0.0162 | 870 ± 2 | 0.624 ± 0.028 |
| b25_plain | 5 | 0.5120 ± 0.0106 | 0.7529 ± 0.0117 | 0.7063 ± 0.0152 | 0.7281 ± 0.0174 | 0.8341 ± 0.0097 | 0.7975 ± 0.0165 | 865 ± 2 | 0.618 ± 0.020 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00434 | [-0.00946, 0.01815] | 0.87 (4) | 0.4317 | 0.39 | 2/5 |
| val loss at end (exploratory) | 5 | 0.04685 | [-0.23272, 0.32642] | 0.47 (4) | 0.6659 | 0.21 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00817 | [-0.02349, 0.00714] | -1.48 (4) | 0.2126 | -0.66 | 2/5 |
| test acc @end (exploratory) | 5 | 0.01090 | [-0.01350, 0.03530] | 1.24 (4) | 0.2826 | 0.55 | 4/5 |
| test F1 @best-val ckpt (exploratory; binary tasks) | 5 | -0.00727 | [-0.02721, 0.01267] | -1.01 (4) | 0.3686 | -0.45 | 3/5 |
| wall-clock s (exploratory) | 5 | 37.91920 | [35.23226, 40.60613] | 39.18 (4) | 0.0000 | 17.52 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.07895 | [-0.08446, -0.07345] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.01465 | [-0.03662, 0.00732] | 0.1379 | 0.2455 | 4/5 |
| rmsprop | 5 | -0.02141 | [-0.03811, -0.00471] | 0.0236 | 0.0708 | 5/5 |
| nadam | 5 | -0.01805 | [-0.04372, 0.00763] | 0.1227 | 0.2455 | 4/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00205, 95% CI [-0.01345, 0.00935], p 0.6437, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 73.6 | 44.8 | 20.8 | 3.4 | 4.6 | 0.6 | 4.0 | 0.062 | 0.130 | 4.6 |
| b25_plain | 5 | 72.2 | 60.0 | 0.0 | 4.0 | 8.2 | 1.4 | 6.8 | 0.114 | 0.171 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.5288 | 5/5, 250 | 5/5, 253 | 5/5, 300 | both 5: B25 faster 2, slower 3; mean saving -1 units (95% CI -86 to 84); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.5056 | 4/5, 375 | 1/5, 426 | 1/5, 426 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 3, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.5056 | 4/5, 375 | 1/5, 426 | 1/5, 426 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 3, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.5056 | 4/5, 375 | 1/5, 426 | 1/5, 426 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 3, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.5056 | 4/5, 375 | 1/5, 426 | 1/5, 426 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 3, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.5288 | 5/5 / 5/5 | -1 (-86 to 84) | 2 / 3 | no | no |
| 25% | 1% | 0.5341 | 5/5 / 5/5 | 14 (-72 to 100) | 3 / 2 | no | no |
| 25% | 2% | 0.5394 | 5/5 / 5/5 | 24 (-57 to 106) | 2 / 3 | no | yes |
| 25% | 5% | 0.5552 | 5/5 / 5/5 | -0 (-1 to 0) | 0 / 3 | no | yes |
| 50% | 0% | 0.5056 | 4/5 / 1/5 | n/a | 0 / 1 | no | no |
| 50% | 1% | 0.5106 | 4/5 / 2/5 | n/a | 0 / 1 | no | no |
| 50% | 2% | 0.5157 | 5/5 / 3/5 | -26 (-274 to 223) | 1 / 2 | no | no |
| 50% | 5% | 0.5309 | 5/5 / 5/5 | 14 (-73 to 101) | 3 / 2 | no | yes |
| 75% | 0% | 0.5056 | 4/5 / 1/5 | n/a | 0 / 1 | no | no |
| 75% | 1% | 0.5106 | 4/5 / 2/5 | n/a | 0 / 1 | no | no |
| 75% | 2% | 0.5157 | 5/5 / 3/5 | -26 (-274 to 223) | 1 / 2 | no | no |
| 75% | 5% | 0.5309 | 5/5 / 5/5 | 14 (-73 to 101) | 3 / 2 | no | yes |
| 95% | 0% | 0.5056 | 4/5 / 1/5 | n/a | 0 / 1 | no | no |
| 95% | 1% | 0.5106 | 4/5 / 2/5 | n/a | 0 / 1 | no | no |
| 95% | 2% | 0.5157 | 5/5 / 3/5 | -26 (-274 to 223) | 1 / 2 | no | no |
| 95% | 5% | 0.5309 | 5/5 / 5/5 | 14 (-73 to 101) | 3 / 2 | no | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.5288 ± 0.0098 | 0.5056 ± 0.0046 | 0.5056 ± 0.0046 | 0.5056 ± 0.0046 | 0.5210 ± 0.0039 |
| b25 | 0.5284 ± 0.0126 | 0.5099 ± 0.0105 | 0.5099 ± 0.0105 | 0.5099 ± 0.0105 | 0.5248 ± 0.0075 |
| b25_plain | 0.5417 ± 0.0087 | 0.5120 ± 0.0106 | 0.5120 ± 0.0106 | 0.5120 ± 0.0106 | 0.5277 ± 0.0065 |
| sgd | 0.6242 ± 0.0006 | 0.6236 ± 0.0005 | 0.6166 ± 0.0071 | 0.5889 ± 0.0089 | 0.6188 ± 0.0030 |
| adam | 0.5529 ± 0.0124 | 0.5338 ± 0.0112 | 0.5254 ± 0.0118 | 0.5246 ± 0.0109 | 0.5432 ± 0.0084 |
| rmsprop | 0.5517 ± 0.0036 | 0.5344 ± 0.0029 | 0.5313 ± 0.0051 | 0.5313 ± 0.0051 | 0.5451 ± 0.0023 |
| nadam | 0.5569 ± 0.0062 | 0.5359 ± 0.0095 | 0.5288 ± 0.0111 | 0.5280 ± 0.0111 | 0.5443 ± 0.0068 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.00038 | [-0.02422, 0.02346] | 0.9666 |
| at 50% of the budget | 5 | 0.00434 | [-0.00946, 0.01815] | 0.4317 |
| at 75% of the budget | 5 | 0.00434 | [-0.00946, 0.01815] | 0.4317 |
| at 100% of the budget | 5 | 0.00434 | [-0.00946, 0.01815] | 0.4317 |
| mean over the whole budget (area under the curve) | 5 | 0.00387 | [-0.00640, 0.01414] | 0.3547 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.