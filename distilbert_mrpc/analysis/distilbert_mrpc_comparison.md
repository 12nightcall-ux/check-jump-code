# Comparison: distilbert_mrpc

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val F1 @best-val ckpt | test F1 @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.3600 ± 0.0185 | 0.8471 ± 0.0142 | 0.8038 ± 0.0109 | 0.8109 ± 0.0123 | 0.8871 ± 0.0115 | 0.8476 ± 0.0090 | 535 ± 1 | 0.746 ± 0.040 |
| adam | 5 | 0.3660 ± 0.0041 | 0.8461 ± 0.0063 | 0.8180 ± 0.0232 | 0.8272 ± 0.0068 | 0.8882 ± 0.0066 | 0.8598 ± 0.0208 | 545 ± 1 | 0.494 ± 0.019 |
| adamw | 5 | 0.3661 ± 0.0055 | 0.8466 ± 0.0064 | 0.8196 ± 0.0224 | 0.8300 ± 0.0115 | 0.8885 ± 0.0068 | 0.8609 ± 0.0202 | 548 ± 1 | 0.491 ± 0.030 |
| rmsprop | 5 | 0.3515 ± 0.0150 | 0.8402 ± 0.0061 | 0.8136 ± 0.0074 | 0.8245 ± 0.0176 | 0.8823 ± 0.0048 | 0.8542 ± 0.0068 | 527 ± 1 | 0.461 ± 0.009 |
| nadam | 5 | 0.3675 ± 0.0056 | 0.8397 ± 0.0156 | 0.8054 ± 0.0076 | 0.8272 ± 0.0103 | 0.8831 ± 0.0135 | 0.8499 ± 0.0093 | 536 ± 1 | 0.492 ± 0.017 |
| b25 | 5 | 0.3667 ± 0.0138 | 0.8407 ± 0.0105 | 0.8104 ± 0.0071 | 0.8316 ± 0.0104 | 0.8852 ± 0.0080 | 0.8536 ± 0.0058 | 572 ± 2 | 0.502 ± 0.016 |
| b25_plain | 5 | 0.3702 ± 0.0131 | 0.8407 ± 0.0143 | 0.8098 ± 0.0101 | 0.8305 ± 0.0143 | 0.8852 ± 0.0126 | 0.8534 ± 0.0104 | 562 ± 1 | 0.542 ± 0.039 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00060 | [-0.02087, 0.02207] | 0.08 (4) | 0.9418 | 0.03 | 2/5 |
| val loss at end (exploratory) | 5 | 0.01095 | [-0.09510, 0.11700] | 0.29 (4) | 0.7886 | 0.13 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00926 | [-0.03958, 0.02105] | -0.85 (4) | 0.4440 | -0.38 | 2/5 |
| test acc @end (exploratory) | 5 | 0.00163 | [-0.02272, 0.02599] | 0.19 (4) | 0.8612 | 0.08 | 4/5 |
| test F1 @best-val ckpt (exploratory; binary tasks) | 5 | -0.00725 | [-0.03692, 0.02241] | -0.68 (4) | 0.5345 | -0.30 | 3/5 |
| wall-clock s (exploratory) | 5 | 23.67443 | [21.06334, 26.28552] | 25.17 (4) | 0.0000 | 11.26 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | 0.00670 | [-0.01308, 0.02648] | 0.4002 | 1.0000 | 2/5 |
| adam | 5 | 0.00064 | [-0.01975, 0.02103] | 0.9349 | 1.0000 | 2/5 |
| rmsprop | 5 | 0.01513 | [-0.00551, 0.03578] | 0.1116 | 0.4464 | 1/5 |
| nadam | 5 | -0.00082 | [-0.01691, 0.01527] | 0.8943 | 1.0000 | 2/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00350, 95% CI [-0.01881, 0.01182], p 0.5607, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 72.6 | 29.0 | 33.4 | 1.4 | 8.8 | 3.4 | 5.4 | 0.121 | 0.386 | 8.8 |
| b25_plain | 5 | 68.6 | 43.4 | 0.0 | 5.8 | 19.4 | 6.2 | 13.2 | 0.283 | 0.320 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.3661 | 2/5, 175 | 2/5, 202 | 3/5, 175 | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 2, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.3661 | 2/5, 175 | 2/5, 202 | 3/5, 175 | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 2, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.3661 | 2/5, 175 | 2/5, 202 | 3/5, 175 | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 2, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.3661 | 2/5, 175 | 2/5, 202 | 3/5, 175 | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 2, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.3661 | 2/5, 175 | 2/5, 202 | 3/5, 175 | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 2, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.3661 | 2/5 / 2/5 | n/a | 0 / 0 | no | no |
| 25% | 1% | 0.3697 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 25% | 2% | 0.3734 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 25% | 5% | 0.3844 | 5/5 / 5/5 | -6 (-46 to 35) | 1 / 4 | no | no |
| 50% | 0% | 0.3661 | 2/5 / 2/5 | n/a | 0 / 0 | no | no |
| 50% | 1% | 0.3697 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 50% | 2% | 0.3734 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 50% | 5% | 0.3844 | 5/5 / 5/5 | -6 (-46 to 35) | 1 / 4 | no | no |
| 75% | 0% | 0.3661 | 2/5 / 2/5 | n/a | 0 / 0 | no | no |
| 75% | 1% | 0.3697 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 75% | 2% | 0.3734 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 75% | 5% | 0.3844 | 5/5 / 5/5 | -6 (-46 to 35) | 1 / 4 | no | no |
| 95% | 0% | 0.3661 | 2/5 / 2/5 | n/a | 0 / 0 | no | no |
| 95% | 1% | 0.3697 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 95% | 2% | 0.3734 | 4/5 / 3/5 | -26 (-31 to -21) | 0 / 2 | no | no |
| 95% | 5% | 0.3844 | 5/5 / 5/5 | -6 (-46 to 35) | 1 / 4 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.3661 ± 0.0055 | 0.3661 ± 0.0055 | 0.3661 ± 0.0055 | 0.3661 ± 0.0055 | 0.3844 ± 0.0048 |
| b25 | 0.3667 ± 0.0138 | 0.3667 ± 0.0138 | 0.3667 ± 0.0138 | 0.3667 ± 0.0138 | 0.3917 ± 0.0138 |
| b25_plain | 0.3702 ± 0.0131 | 0.3702 ± 0.0131 | 0.3702 ± 0.0131 | 0.3702 ± 0.0131 | 0.3930 ± 0.0111 |
| sgd | 0.3762 ± 0.0329 | 0.3600 ± 0.0185 | 0.3600 ± 0.0185 | 0.3600 ± 0.0185 | 0.3893 ± 0.0175 |
| adam | 0.3660 ± 0.0041 | 0.3660 ± 0.0041 | 0.3660 ± 0.0041 | 0.3660 ± 0.0041 | 0.3843 ± 0.0037 |
| rmsprop | 0.3515 ± 0.0150 | 0.3515 ± 0.0150 | 0.3515 ± 0.0150 | 0.3515 ± 0.0150 | 0.3625 ± 0.0151 |
| nadam | 0.3675 ± 0.0056 | 0.3675 ± 0.0056 | 0.3675 ± 0.0056 | 0.3675 ± 0.0056 | 0.3824 ± 0.0054 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00060 | [-0.02087, 0.02207] | 0.9418 |
| at 50% of the budget | 5 | 0.00060 | [-0.02087, 0.02207] | 0.9418 |
| at 75% of the budget | 5 | 0.00060 | [-0.02087, 0.02207] | 0.9418 |
| at 100% of the budget | 5 | 0.00060 | [-0.02087, 0.02207] | 0.9418 |
| mean over the whole budget (area under the curve) | 5 | 0.00738 | [-0.01233, 0.02709] | 0.3573 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.