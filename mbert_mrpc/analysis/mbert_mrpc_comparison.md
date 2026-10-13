# Comparison: mbert_mrpc

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val F1 @best-val ckpt | test F1 @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.3849 ± 0.0109 | 0.8235 ± 0.0185 | 0.8082 ± 0.0151 | 0.8109 ± 0.0099 | 0.8740 ± 0.0148 | 0.8579 ± 0.0094 | 468 ± 0 | 0.992 ± 0.008 |
| adam | 5 | 0.3690 ± 0.0309 | 0.8412 ± 0.0197 | 0.8153 ± 0.0199 | 0.8349 ± 0.0151 | 0.8825 ± 0.0182 | 0.8586 ± 0.0175 | 495 ± 0 | 0.543 ± 0.036 |
| adamw | 5 | 0.3483 ± 0.0180 | 0.8564 ± 0.0126 | 0.8294 ± 0.0199 | 0.8371 ± 0.0138 | 0.8958 ± 0.0103 | 0.8718 ± 0.0209 | 517 ± 0 | 0.530 ± 0.018 |
| rmsprop | 5 | 0.4091 ± 0.1242 | 0.8142 ± 0.0762 | 0.7907 ± 0.0741 | 0.7995 ± 0.0797 | 0.8749 ± 0.0387 | 0.8548 ± 0.0360 | 503 ± 5 | 0.532 ± 0.118 |
| nadam | 5 | 0.3548 ± 0.0169 | 0.8402 ± 0.0129 | 0.8262 ± 0.0106 | 0.8409 ± 0.0195 | 0.8830 ± 0.0108 | 0.8674 ± 0.0093 | 532 ± 1 | 0.484 ± 0.028 |
| b25 | 5 | 0.3558 ± 0.0262 | 0.8358 ± 0.0195 | 0.8300 ± 0.0185 | 0.8327 ± 0.0079 | 0.8778 ± 0.0158 | 0.8694 ± 0.0175 | 585 ± 13 | 0.556 ± 0.048 |
| b25_plain | 5 | 0.3427 ± 0.0141 | 0.8564 ± 0.0123 | 0.8322 ± 0.0081 | 0.8360 ± 0.0170 | 0.8963 ± 0.0098 | 0.8748 ± 0.0070 | 561 ± 1 | 0.561 ± 0.029 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00754 | [-0.00933, 0.02441] | 1.24 (4) | 0.2824 | 0.56 | 2/5 |
| val loss at end (exploratory) | 5 | 0.05836 | [-0.07312, 0.18984] | 1.23 (4) | 0.2853 | 0.55 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00054 | [-0.03315, 0.03424] | 0.04 (4) | 0.9663 | 0.02 | 2/5 |
| test acc @end (exploratory) | 5 | -0.00436 | [-0.01646, 0.00774] | -1.00 (4) | 0.3739 | -0.45 | 2/5 |
| test F1 @best-val ckpt (exploratory; binary tasks) | 5 | -0.00247 | [-0.03020, 0.02526] | -0.25 (4) | 0.8167 | -0.11 | 2/5 |
| wall-clock s (exploratory) | 5 | 68.38821 | [52.97683, 83.79958] | 12.32 (4) | 0.0002 | 5.51 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.02904 | [-0.05218, -0.00591] | 0.0252 | 0.1009 | 5/5 |
| adam | 5 | -0.01318 | [-0.04783, 0.02147] | 0.3506 | 1.0000 | 3/5 |
| rmsprop | 5 | -0.05323 | [-0.22804, 0.12158] | 0.4455 | 1.0000 | 3/5 |
| nadam | 5 | 0.00108 | [-0.04399, 0.04614] | 0.9504 | 1.0000 | 2/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.01314, 95% CI [-0.01713, 0.04341], p 0.2946, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 73.4 | 31.0 | 34.4 | 1.8 | 6.2 | 1.8 | 4.4 | 0.084 | 0.290 | 6.2 |
| b25_plain | 5 | 69.6 | 46.2 | 0.0 | 11.2 | 12.2 | 4.2 | 8.0 | 0.175 | 0.344 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.3487 | 3/5, 175 | 2/5, 176 | 4/5, 226 | both 2: B25 faster 0, slower 2; mean saving -1 units (95% CI -1 to -0); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.3483 | 3/5, 175 | 2/5, 176 | 4/5, 226 | both 2: B25 faster 0, slower 2; mean saving -1 units (95% CI -1 to -0); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.3483 | 3/5, 175 | 2/5, 176 | 4/5, 226 | both 2: B25 faster 0, slower 2; mean saving -1 units (95% CI -1 to -0); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.3483 | 3/5, 175 | 2/5, 176 | 4/5, 226 | both 2: B25 faster 0, slower 2; mean saving -1 units (95% CI -1 to -0); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.3483 | 3/5, 175 | 2/5, 176 | 4/5, 226 | both 2: B25 faster 0, slower 2; mean saving -1 units (95% CI -1 to -0); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.3487 | 3/5 / 2/5 | -1 (-1 to -0) | 0 / 2 | no | no |
| 25% | 1% | 0.3522 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 25% | 2% | 0.3557 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 25% | 5% | 0.3661 | 4/5 / 3/5 | -9 (-81 to 63) | 1 / 2 | no | no |
| 50% | 0% | 0.3483 | 3/5 / 2/5 | -1 (-1 to -0) | 0 / 2 | no | no |
| 50% | 1% | 0.3518 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 50% | 2% | 0.3553 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 50% | 5% | 0.3657 | 4/5 / 3/5 | -9 (-81 to 63) | 1 / 2 | no | no |
| 75% | 0% | 0.3483 | 3/5 / 2/5 | -1 (-1 to -0) | 0 / 2 | no | no |
| 75% | 1% | 0.3518 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 75% | 2% | 0.3553 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 75% | 5% | 0.3657 | 4/5 / 3/5 | -9 (-81 to 63) | 1 / 2 | no | no |
| 95% | 0% | 0.3483 | 3/5 / 2/5 | -1 (-1 to -0) | 0 / 2 | no | no |
| 95% | 1% | 0.3518 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 95% | 2% | 0.3553 | 3/5 / 3/5 | -9 (-45 to 27) | 0 / 3 | no | no |
| 95% | 5% | 0.3657 | 4/5 / 3/5 | -9 (-81 to 63) | 1 / 2 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.3487 ± 0.0179 | 0.3483 ± 0.0180 | 0.3483 ± 0.0180 | 0.3483 ± 0.0180 | 0.3639 ± 0.0153 |
| b25 | 0.3558 ± 0.0262 | 0.3558 ± 0.0262 | 0.3558 ± 0.0262 | 0.3558 ± 0.0262 | 0.3756 ± 0.0216 |
| b25_plain | 0.3427 ± 0.0141 | 0.3427 ± 0.0141 | 0.3427 ± 0.0141 | 0.3427 ± 0.0141 | 0.3640 ± 0.0132 |
| sgd | 0.4588 ± 0.0178 | 0.3942 ± 0.0061 | 0.3849 ± 0.0109 | 0.3849 ± 0.0109 | 0.4302 ± 0.0067 |
| adam | 0.3690 ± 0.0309 | 0.3690 ± 0.0309 | 0.3690 ± 0.0309 | 0.3690 ± 0.0309 | 0.3789 ± 0.0284 |
| rmsprop | 0.4128 ± 0.1210 | 0.4091 ± 0.1242 | 0.4091 ± 0.1242 | 0.4091 ± 0.1242 | 0.4216 ± 0.1164 |
| nadam | 0.3548 ± 0.0169 | 0.3548 ± 0.0169 | 0.3548 ± 0.0169 | 0.3548 ± 0.0169 | 0.3665 ± 0.0154 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00715 | [-0.01059, 0.02490] | 0.3257 |
| at 50% of the budget | 5 | 0.00754 | [-0.00933, 0.02441] | 0.2824 |
| at 75% of the budget | 5 | 0.00754 | [-0.00933, 0.02441] | 0.2824 |
| at 100% of the budget | 5 | 0.00754 | [-0.00933, 0.02441] | 0.2824 |
| mean over the whole budget (area under the curve) | 5 | 0.01166 | [-0.00232, 0.02563] | 0.0815 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.