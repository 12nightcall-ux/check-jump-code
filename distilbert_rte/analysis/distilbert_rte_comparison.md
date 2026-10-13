# Comparison: distilbert_rte

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.6640 ± 0.0068 | 0.5993 ± 0.0128 | 0.5767 ± 0.0228 | 0.5783 ± 0.0295 | 172 ± 0 | 0.758 ± 0.016 |
| adam | 5 | 0.6706 ± 0.0047 | 0.5921 ± 0.0057 | 0.5695 ± 0.0100 | 0.5791 ± 0.0060 | 182 ± 0 | 0.347 ± 0.026 |
| adamw | 5 | 0.6655 ± 0.0047 | 0.6000 ± 0.0214 | 0.5671 ± 0.0253 | 0.5679 ± 0.0078 | 184 ± 0 | 0.838 ± 0.010 |
| rmsprop | 5 | 0.6692 ± 0.0034 | 0.5928 ± 0.0123 | 0.5719 ± 0.0225 | 0.5703 ± 0.0320 | 180 ± 0 | 0.722 ± 0.018 |
| nadam | 5 | 0.6667 ± 0.0058 | 0.5957 ± 0.0175 | 0.5647 ± 0.0225 | 0.5839 ± 0.0282 | 191 ± 0 | 0.399 ± 0.022 |
| b25 | 5 | 0.6768 ± 0.0041 | 0.5819 ± 0.0183 | 0.5406 ± 0.0273 | 0.5783 ± 0.0173 | 214 ± 1 | 0.337 ± 0.029 |
| b25_plain | 5 | 0.6743 ± 0.0076 | 0.5819 ± 0.0252 | 0.5365 ± 0.0239 | 0.5815 ± 0.0233 | 207 ± 0 | 0.370 ± 0.040 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.01136 | [0.00076, 0.02196] | 2.97 (4) | 0.0410 | 1.33 | 0/5 |
| val loss at end (exploratory) | 5 | 1.21490 | [0.94523, 1.48457] | 12.51 (4) | 0.0002 | 5.59 | 0/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.02651 | [-0.06905, 0.01604] | -1.73 (4) | 0.1587 | -0.77 | 2/5 |
| test acc @end (exploratory) | 5 | 0.01044 | [-0.00303, 0.02391] | 2.15 (4) | 0.0978 | 0.96 | 4/5 |
| wall-clock s (exploratory) | 5 | 29.46003 | [27.95708, 30.96298] | 54.42 (4) | 0.0000 | 24.34 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | 0.01285 | [0.00459, 0.02111] | 0.0124 | 0.0498 | 0/5 |
| adam | 5 | 0.00627 | [0.00089, 0.01164] | 0.0318 | 0.0953 | 0/5 |
| rmsprop | 5 | 0.00767 | [-0.00098, 0.01631] | 0.0695 | 0.1217 | 1/5 |
| nadam | 5 | 0.01011 | [-0.00074, 0.02096] | 0.0609 | 0.1217 | 1/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00258, 95% CI [-0.00561, 0.01076], p 0.4318, B25 better in 1/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 72.4 | 27.2 | 33.8 | 2.4 | 9.0 | 4.2 | 4.8 | 0.124 | 0.467 | 9.0 |
| b25_plain | 5 | 68.4 | 43.4 | 0.0 | 6.4 | 18.6 | 8.6 | 10.0 | 0.272 | 0.462 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6796 | 5/5, 275 | 4/5, 113 | 4/5, 101 | both 4: B25 faster 4, slower 0; mean saving 143 units (95% CI 105 to 181); only B25 0, only AdamW 1, neither 0 | SAVING SUPPORTED |
| 50% | 0.6662 | 3/5, 500 | 0/5, n/a | 0/5, n/a | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 0, only AdamW 3, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.6655 | 2/5, 475 | 0/5, n/a | 0/5, n/a | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 0, only AdamW 2, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.6655 | 2/5, 475 | 0/5, n/a | 0/5, n/a | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 0, only AdamW 2, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.6655 | 2/5, 475 | 0/5, n/a | 0/5, n/a | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 0, only AdamW 2, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6796 | 5/5 / 4/5 | 143 (105 to 181) | 4 / 0 | YES | yes |
| 25% | 1% | 0.6864 | 5/5 / 5/5 | 59 (31 to 87) | 5 / 0 | YES | yes |
| 25% | 2% | 0.6932 | 5/5 / 5/5 | -11 (-53 to 31) | 1 / 4 | no | yes |
| 25% | 5% | 0.7136 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | yes |
| 50% | 0% | 0.6662 | 3/5 / 0/5 | n/a | 0 / 0 | no | no |
| 50% | 1% | 0.6728 | 5/5 / 1/5 | n/a | 1 / 0 | no | no |
| 50% | 2% | 0.6795 | 5/5 / 4/5 | 143 (105 to 181) | 4 / 0 | YES | no |
| 50% | 5% | 0.6995 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 75% | 0% | 0.6655 | 2/5 / 0/5 | n/a | 0 / 0 | no | no |
| 75% | 1% | 0.6721 | 5/5 / 1/5 | n/a | 1 / 0 | no | no |
| 75% | 2% | 0.6788 | 5/5 / 4/5 | 149 (93 to 206) | 4 / 0 | YES | no |
| 75% | 5% | 0.6988 | 5/5 / 5/5 | 5 (-9 to 19) | 1 / 0 | no | no |
| 95% | 0% | 0.6655 | 2/5 / 0/5 | n/a | 0 / 0 | no | no |
| 95% | 1% | 0.6721 | 5/5 / 1/5 | n/a | 1 / 0 | no | no |
| 95% | 2% | 0.6788 | 5/5 / 4/5 | 149 (93 to 206) | 4 / 0 | YES | no |
| 95% | 5% | 0.6988 | 5/5 / 5/5 | 5 (-9 to 19) | 1 / 0 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6796 ± 0.0031 | 0.6662 ± 0.0039 | 0.6655 ± 0.0047 | 0.6655 ± 0.0047 | 0.6722 ± 0.0021 |
| b25 | 0.6768 ± 0.0041 | 0.6768 ± 0.0041 | 0.6768 ± 0.0041 | 0.6768 ± 0.0041 | 0.6784 ± 0.0039 |
| b25_plain | 0.6743 ± 0.0076 | 0.6743 ± 0.0076 | 0.6743 ± 0.0076 | 0.6743 ± 0.0076 | 0.6760 ± 0.0066 |
| sgd | 0.6842 ± 0.0016 | 0.6754 ± 0.0037 | 0.6691 ± 0.0030 | 0.6640 ± 0.0068 | 0.6764 ± 0.0027 |
| adam | 0.6706 ± 0.0047 | 0.6706 ± 0.0047 | 0.6706 ± 0.0047 | 0.6706 ± 0.0047 | 0.6721 ± 0.0043 |
| rmsprop | 0.6861 ± 0.0020 | 0.6787 ± 0.0030 | 0.6733 ± 0.0019 | 0.6692 ± 0.0034 | 0.6792 ± 0.0019 |
| nadam | 0.6667 ± 0.0058 | 0.6667 ± 0.0058 | 0.6667 ± 0.0058 | 0.6667 ± 0.0058 | 0.6688 ± 0.0052 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.00279 | [-0.00456, -0.00103] | 0.0117 |
| at 50% of the budget | 5 | 0.01068 | [0.00117, 0.02020] | 0.0356 |
| at 75% of the budget | 5 | 0.01136 | [0.00076, 0.02196] | 0.0410 |
| at 100% of the budget | 5 | 0.01136 | [0.00076, 0.02196] | 0.0410 |
| mean over the whole budget (area under the curve) | 5 | 0.00619 | [-0.00060, 0.01298] | 0.0647 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.