# Comparison: mbert_rte

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.6200 ± 0.0255 | 0.6643 ± 0.0238 | 0.6538 ± 0.0321 | 0.6819 ± 0.0185 | 360 ± 0 | 0.734 ± 0.046 |
| adam | 5 | 0.6192 ± 0.0177 | 0.6773 ± 0.0137 | 0.6570 ± 0.0169 | 0.6578 ± 0.0131 | 389 ± 0 | 0.381 ± 0.053 |
| adamw | 5 | 0.6197 ± 0.0186 | 0.6773 ± 0.0137 | 0.6554 ± 0.0167 | 0.6707 ± 0.0102 | 384 ± 5 | 0.371 ± 0.026 |
| rmsprop | 5 | 0.6438 ± 0.0207 | 0.6260 ± 0.0325 | 0.6313 ± 0.0419 | 0.6378 ± 0.0435 | 373 ± 0 | 0.995 ± 0.006 |
| nadam | 5 | 0.6197 ± 0.0057 | 0.6693 ± 0.0104 | 0.6562 ± 0.0202 | 0.6747 ± 0.0117 | 391 ± 0 | 0.404 ± 0.053 |
| b25 | 5 | 0.6168 ± 0.0099 | 0.6729 ± 0.0214 | 0.6442 ± 0.0109 | 0.6554 ± 0.0118 | 470 ± 3 | 0.407 ± 0.034 |
| b25_plain | 5 | 0.6145 ± 0.0104 | 0.6852 ± 0.0160 | 0.6538 ± 0.0211 | 0.6731 ± 0.0179 | 454 ± 1 | 0.448 ± 0.039 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00290 | [-0.02851, 0.02271] | -0.31 (4) | 0.7689 | -0.14 | 3/5 |
| val loss at end (exploratory) | 5 | -0.01502 | [-0.21139, 0.18135] | -0.21 (4) | 0.8422 | -0.09 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.01124 | [-0.03932, 0.01683] | -1.11 (4) | 0.3285 | -0.50 | 1/5 |
| test acc @end (exploratory) | 5 | -0.01526 | [-0.03564, 0.00512] | -2.08 (4) | 0.1061 | -0.93 | 1/5 |
| wall-clock s (exploratory) | 5 | 86.78145 | [78.37146, 95.19145] | 28.65 (4) | 0.0000 | 12.81 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.00316 | [-0.04285, 0.03653] | 0.8361 | 1.0000 | 3/5 |
| adam | 5 | -0.00243 | [-0.02732, 0.02247] | 0.8002 | 1.0000 | 3/5 |
| rmsprop | 5 | -0.02695 | [-0.06054, 0.00664] | 0.0898 | 0.3593 | 3/5 |
| nadam | 5 | -0.00286 | [-0.01887, 0.01314] | 0.6457 | 1.0000 | 2/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00231, 95% CI [-0.00515, 0.00977], p 0.4384, B25 better in 4/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 72.8 | 26.4 | 36.4 | 2.4 | 7.6 | 2.6 | 5.0 | 0.104 | 0.342 | 7.6 |
| b25_plain | 5 | 68.4 | 41.2 | 0.0 | 10.4 | 16.8 | 7.6 | 9.2 | 0.246 | 0.452 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6197 | 3/5, 100 | 2/5, 113 | 3/5, 126 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.6197 | 3/5, 100 | 2/5, 113 | 3/5, 126 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.6197 | 3/5, 100 | 2/5, 113 | 3/5, 126 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.6197 | 3/5, 100 | 2/5, 113 | 3/5, 126 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.6197 | 3/5, 100 | 2/5, 113 | 3/5, 126 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6197 | 3/5 / 2/5 | n/a | 0 / 1 | no | no |
| 25% | 1% | 0.6259 | 4/5 / 4/5 | -34 (-129 to 60) | 0 / 3 | no | no |
| 25% | 2% | 0.6321 | 4/5 / 5/5 | -20 (-79 to 39) | 0 / 4 | no | no |
| 25% | 5% | 0.6507 | 5/5 / 5/5 | -1 (-22 to 21) | 1 / 2 | no | no |
| 50% | 0% | 0.6197 | 3/5 / 2/5 | n/a | 0 / 1 | no | no |
| 50% | 1% | 0.6259 | 4/5 / 4/5 | -34 (-129 to 60) | 0 / 3 | no | no |
| 50% | 2% | 0.6321 | 4/5 / 5/5 | -20 (-79 to 39) | 0 / 4 | no | no |
| 50% | 5% | 0.6507 | 5/5 / 5/5 | -1 (-22 to 21) | 1 / 2 | no | no |
| 75% | 0% | 0.6197 | 3/5 / 2/5 | n/a | 0 / 1 | no | no |
| 75% | 1% | 0.6259 | 4/5 / 4/5 | -34 (-129 to 60) | 0 / 3 | no | no |
| 75% | 2% | 0.6321 | 4/5 / 5/5 | -20 (-79 to 39) | 0 / 4 | no | no |
| 75% | 5% | 0.6507 | 5/5 / 5/5 | -1 (-22 to 21) | 1 / 2 | no | no |
| 95% | 0% | 0.6197 | 3/5 / 2/5 | n/a | 0 / 1 | no | no |
| 95% | 1% | 0.6259 | 4/5 / 4/5 | -34 (-129 to 60) | 0 / 3 | no | no |
| 95% | 2% | 0.6321 | 4/5 / 5/5 | -20 (-79 to 39) | 0 / 4 | no | no |
| 95% | 5% | 0.6507 | 5/5 / 5/5 | -1 (-22 to 21) | 1 / 2 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6197 ± 0.0186 | 0.6197 ± 0.0186 | 0.6197 ± 0.0186 | 0.6197 ± 0.0186 | 0.6220 ± 0.0178 |
| b25 | 0.6168 ± 0.0099 | 0.6168 ± 0.0099 | 0.6168 ± 0.0099 | 0.6168 ± 0.0099 | 0.6209 ± 0.0085 |
| b25_plain | 0.6145 ± 0.0104 | 0.6145 ± 0.0104 | 0.6145 ± 0.0104 | 0.6145 ± 0.0104 | 0.6189 ± 0.0097 |
| sgd | 0.6375 ± 0.0124 | 0.6200 ± 0.0255 | 0.6200 ± 0.0255 | 0.6200 ± 0.0255 | 0.6343 ± 0.0187 |
| adam | 0.6192 ± 0.0177 | 0.6192 ± 0.0177 | 0.6192 ± 0.0177 | 0.6192 ± 0.0177 | 0.6216 ± 0.0171 |
| rmsprop | 0.6461 ± 0.0200 | 0.6438 ± 0.0207 | 0.6438 ± 0.0207 | 0.6438 ± 0.0207 | 0.6479 ± 0.0197 |
| nadam | 0.6197 ± 0.0057 | 0.6197 ± 0.0057 | 0.6197 ± 0.0057 | 0.6197 ± 0.0057 | 0.6215 ± 0.0059 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.00290 | [-0.02851, 0.02271] | 0.7689 |
| at 50% of the budget | 5 | -0.00290 | [-0.02851, 0.02271] | 0.7689 |
| at 75% of the budget | 5 | -0.00290 | [-0.02851, 0.02271] | 0.7689 |
| at 100% of the budget | 5 | -0.00290 | [-0.02851, 0.02271] | 0.7689 |
| mean over the whole budget (area under the curve) | 5 | -0.00117 | [-0.02369, 0.02135] | 0.8921 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.