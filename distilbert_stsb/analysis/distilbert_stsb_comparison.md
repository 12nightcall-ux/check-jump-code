# Comparison: distilbert_stsb

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

Regression task (stsb): the 'loss' columns are mean squared error, and the columns named 'Pearson r' are Pearson correlations (higher is better).

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val Pearson r | test Pearson r @best-val ckpt | test Pearson r @end | val Spearman @best-val ckpt | test Spearman @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.5974 ± 0.0503 | 0.8587 ± 0.0132 | 0.8332 ± 0.0127 | 0.8322 ± 0.0071 | 0.8552 ± 0.0129 | 0.7994 ± 0.0170 | 853 ± 3 | 1.000 ± 0.000 |
| adam | 5 | 0.5393 ± 0.0051 | 0.8729 ± 0.0018 | 0.8536 ± 0.0029 | 0.8527 ± 0.0048 | 0.8698 ± 0.0017 | 0.8251 ± 0.0041 | 870 ± 1 | 1.000 ± 0.000 |
| adamw | 5 | 0.5572 ± 0.0178 | 0.8689 ± 0.0055 | 0.8434 ± 0.0048 | 0.8490 ± 0.0023 | 0.8666 ± 0.0051 | 0.8125 ± 0.0103 | 870 ± 1 | 0.998 ± 0.001 |
| rmsprop | 5 | 0.5459 ± 0.0094 | 0.8709 ± 0.0021 | 0.8549 ± 0.0065 | 0.8579 ± 0.0018 | 0.8677 ± 0.0017 | 0.8260 ± 0.0070 | 882 ± 1 | 1.000 ± 0.000 |
| nadam | 5 | 0.5285 ± 0.0047 | 0.8751 ± 0.0015 | 0.8579 ± 0.0049 | 0.8586 ± 0.0033 | 0.8724 ± 0.0016 | 0.8310 ± 0.0059 | 913 ± 44 | 1.000 ± 0.000 |
| b25 | 5 | 0.5380 ± 0.0060 | 0.8726 ± 0.0015 | 0.8522 ± 0.0035 | 0.8535 ± 0.0043 | 0.8698 ± 0.0016 | 0.8235 ± 0.0057 | 924 ± 2 | 1.000 ± 0.000 |
| b25_plain | 5 | 0.5366 ± 0.0094 | 0.8732 ± 0.0022 | 0.8528 ± 0.0024 | 0.8515 ± 0.0016 | 0.8701 ± 0.0020 | 0.8241 ± 0.0043 | 920 ± 1 | 1.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.01921 | [-0.04090, 0.00247] | -2.46 (4) | 0.0697 | -1.10 | 5/5 |
| val loss at end (exploratory) | 5 | -0.07738 | [-0.15532, 0.00057] | -2.76 (4) | 0.0510 | -1.23 | 5/5 |
| test Pearson r @best-val ckpt (exploratory) | 5 | 0.00878 | [0.00053, 0.01702] | 2.96 (4) | 0.0417 | 1.32 | 4/5 |
| test Pearson r @end (exploratory) | 5 | 0.00446 | [-0.00318, 0.01211] | 1.62 (4) | 0.1802 | 0.73 | 4/5 |
| test Spearman @best-val ckpt (exploratory; regression tasks) | 5 | 0.01092 | [-0.00499, 0.02682] | 1.91 (4) | 0.1294 | 0.85 | 4/5 |
| wall-clock s (exploratory) | 5 | 53.98268 | [51.58998, 56.37539] | 62.64 (4) | 0.0000 | 28.01 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.05941 | [-0.12154, 0.00272] | 0.0567 | 0.1700 | 5/5 |
| adam | 5 | -0.00133 | [-0.00524, 0.00257] | 0.3970 | 0.3970 | 4/5 |
| rmsprop | 5 | -0.00794 | [-0.02140, 0.00552] | 0.1769 | 0.3538 | 4/5 |
| nadam | 5 | 0.00945 | [0.00531, 0.01359] | 0.0032 | 0.0127 | 0/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00134, 95% CI [-0.00746, 0.01013], p 0.6949, B25 better in 1/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 74.2 | 38.8 | 30.4 | 1.6 | 3.4 | 1.0 | 2.4 | 0.046 | 0.294 | 3.4 |
| b25_plain | 5 | 73.4 | 66.0 | 0.0 | 2.2 | 5.2 | 1.0 | 4.2 | 0.071 | 0.192 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6227 | 5/5, 275 | 5/5, 201 | 5/5, 201 | both 5: B25 faster 5, slower 0; mean saving 89 units (95% CI -10 to 189); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.5874 | 5/5, 475 | 5/5, 251 | 5/5, 276 | both 5: B25 faster 5, slower 0; mean saving 269 units (95% CI -26 to 565); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.5613 | 3/5, 650 | 5/5, 326 | 5/5, 326 | both 3: B25 faster 3, slower 0; mean saving 283 units (95% CI 92 to 473); only B25 2, only AdamW 0, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.5572 | 3/5, 650 | 5/5, 326 | 5/5, 501 | both 3: B25 faster 3, slower 0; mean saving 174 units (95% CI -233 to 582); only B25 2, only AdamW 0, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.5572 | 3/5, 650 | 5/5, 326 | 5/5, 501 | both 3: B25 faster 3, slower 0; mean saving 174 units (95% CI -233 to 582); only B25 2, only AdamW 0, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6227 | 5/5 / 5/5 | 89 (-10 to 189) | 5 / 0 | no | no |
| 25% | 1% | 0.6290 | 5/5 / 5/5 | 69 (-27 to 166) | 4 / 1 | no | no |
| 25% | 2% | 0.6352 | 5/5 / 5/5 | 29 (-11 to 70) | 3 / 2 | no | no |
| 25% | 5% | 0.6539 | 5/5 / 5/5 | 34 (-41 to 109) | 3 / 2 | no | no |
| 50% | 0% | 0.5874 | 5/5 / 5/5 | 269 (-26 to 565) | 5 / 0 | no | no |
| 50% | 1% | 0.5932 | 5/5 / 5/5 | 219 (57 to 381) | 5 / 0 | YES | no |
| 50% | 2% | 0.5991 | 5/5 / 5/5 | 229 (53 to 405) | 5 / 0 | YES | no |
| 50% | 5% | 0.6167 | 5/5 / 5/5 | 114 (-75 to 304) | 4 / 1 | no | yes |
| 75% | 0% | 0.5613 | 3/5 / 5/5 | 283 (92 to 473) | 3 / 0 | no | no |
| 75% | 1% | 0.5669 | 4/5 / 5/5 | 218 (-17 to 453) | 4 / 0 | no | no |
| 75% | 2% | 0.5725 | 4/5 / 5/5 | 218 (-55 to 491) | 4 / 0 | no | no |
| 75% | 5% | 0.5893 | 5/5 / 5/5 | 204 (47 to 362) | 5 / 0 | YES | no |
| 95% | 0% | 0.5572 | 3/5 / 5/5 | 174 (-233 to 582) | 3 / 0 | no | no |
| 95% | 1% | 0.5628 | 4/5 / 5/5 | 324 (159 to 490) | 4 / 0 | YES | no |
| 95% | 2% | 0.5683 | 4/5 / 5/5 | 205 (-57 to 468) | 4 / 0 | no | no |
| 95% | 5% | 0.5851 | 4/5 / 5/5 | 174 (-23 to 372) | 4 / 0 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6227 ± 0.0131 | 0.5874 ± 0.0212 | 0.5613 ± 0.0155 | 0.5572 ± 0.0178 | 0.6203 ± 0.0096 |
| b25 | 0.5879 ± 0.0133 | 0.5489 ± 0.0121 | 0.5418 ± 0.0091 | 0.5380 ± 0.0060 | 0.6412 ± 0.0195 |
| b25_plain | 0.5919 ± 0.0027 | 0.5560 ± 0.0079 | 0.5428 ± 0.0072 | 0.5366 ± 0.0094 | 0.6439 ± 0.0224 |
| sgd | 1.0550 ± 0.6284 | 0.6538 ± 0.0880 | 0.6152 ± 0.0686 | 0.5974 ± 0.0503 | 1.0676 ± 0.1825 |
| adam | 0.5717 ± 0.0123 | 0.5547 ± 0.0089 | 0.5462 ± 0.0069 | 0.5393 ± 0.0051 | 0.6229 ± 0.0056 |
| rmsprop | 0.5717 ± 0.0173 | 0.5508 ± 0.0092 | 0.5468 ± 0.0112 | 0.5459 ± 0.0094 | 0.5904 ± 0.0075 |
| nadam | 0.5746 ± 0.0125 | 0.5469 ± 0.0076 | 0.5350 ± 0.0064 | 0.5285 ± 0.0047 | 0.5915 ± 0.0100 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.03482 | [-0.06453, -0.00512] | 0.0312 |
| at 50% of the budget | 5 | -0.03850 | [-0.05687, -0.02012] | 0.0043 |
| at 75% of the budget | 5 | -0.01946 | [-0.04530, 0.00637] | 0.1046 |
| at 100% of the budget | 5 | -0.01921 | [-0.04090, 0.00247] | 0.0697 |
| mean over the whole budget (area under the curve) | 5 | 0.02083 | [0.00214, 0.03952] | 0.0364 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.