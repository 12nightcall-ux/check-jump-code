# Comparison: mbert_cola

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val MCC @best-val ckpt | test MCC @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.5178 ± 0.0026 | 0.7507 ± 0.0046 | 0.7415 ± 0.0084 | 0.7439 ± 0.0078 | 0.3748 ± 0.0113 | 0.3715 ± 0.0206 | 749 ± 1 | 0.996 ± 0.007 |
| adam | 5 | 0.5248 ± 0.0134 | 0.7526 ± 0.0072 | 0.7469 ± 0.0110 | 0.7530 ± 0.0101 | 0.3890 ± 0.0228 | 0.4012 ± 0.0325 | 763 ± 0 | 0.991 ± 0.017 |
| adamw | 5 | 0.5082 ± 0.0082 | 0.7523 ± 0.0100 | 0.7535 ± 0.0100 | 0.7450 ± 0.0186 | 0.3886 ± 0.0251 | 0.4096 ± 0.0222 | 783 ± 1 | 1.000 ± 0.000 |
| rmsprop | 5 | 0.5072 ± 0.0046 | 0.7565 ± 0.0092 | 0.7520 ± 0.0048 | 0.7626 ± 0.0122 | 0.3961 ± 0.0219 | 0.4092 ± 0.0202 | 748 ± 0 | 0.998 ± 0.001 |
| nadam | 5 | 0.5001 ± 0.0069 | 0.7595 ± 0.0176 | 0.7591 ± 0.0049 | 0.7705 ± 0.0075 | 0.4102 ± 0.0367 | 0.4270 ± 0.0274 | 775 ± 1 | 0.996 ± 0.003 |
| b25 | 5 | 0.5066 ± 0.0094 | 0.7590 ± 0.0045 | 0.7455 ± 0.0168 | 0.7537 ± 0.0043 | 0.4032 ± 0.0280 | 0.3924 ± 0.0394 | 880 ± 10 | 1.000 ± 0.000 |
| b25_plain | 5 | 0.5077 ± 0.0082 | 0.7588 ± 0.0118 | 0.7518 ± 0.0138 | 0.7485 ± 0.0142 | 0.3966 ± 0.0223 | 0.3978 ± 0.0283 | 845 ± 1 | 1.000 ± 0.001 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00158 | [-0.00819, 0.00503] | -0.66 (4) | 0.5426 | -0.30 | 3/5 |
| val loss at end (exploratory) | 5 | -0.06774 | [-0.15495, 0.01946] | -2.16 (4) | 0.0972 | -0.96 | 4/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00795 | [-0.01761, 0.00170] | -2.29 (4) | 0.0841 | -1.02 | 1/5 |
| test acc @end (exploratory) | 5 | 0.00865 | [-0.01670, 0.03401] | 0.95 (4) | 0.3969 | 0.42 | 3/5 |
| test MCC @best-val ckpt (exploratory; CoLA) | 5 | -0.01712 | [-0.04125, 0.00701] | -1.97 (4) | 0.1201 | -0.88 | 0/5 |
| wall-clock s (exploratory) | 5 | 97.04782 | [84.62041, 109.47523] | 21.68 (4) | 0.0000 | 9.70 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.01122 | [-0.02404, 0.00159] | 0.0718 | 0.2451 | 4/5 |
| adam | 5 | -0.01818 | [-0.04163, 0.00527] | 0.0977 | 0.2451 | 4/5 |
| rmsprop | 5 | -0.00066 | [-0.00740, 0.00608] | 0.7992 | 0.7992 | 3/5 |
| nadam | 5 | 0.00644 | [-0.00049, 0.01337] | 0.0613 | 0.2451 | 0/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00114, 95% CI [-0.00744, 0.00515], p 0.6407, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 70.2 | 34.0 | 17.2 | 3.6 | 15.4 | 6.0 | 9.4 | 0.219 | 0.390 | 15.4 |
| b25_plain | 5 | 68.4 | 43.8 | 0.0 | 3.2 | 21.4 | 8.2 | 13.2 | 0.313 | 0.383 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.5327 | 5/5, 300 | 5/5, 251 | 5/5, 275 | both 5: B25 faster 3, slower 2; mean saving -5 units (95% CI -65 to 54); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.5082 | 2/5, 388 | 3/5, 500 | 3/5, 400 | both 2: B25 faster 0, slower 2; mean saving -39 units (95% CI -185 to 107); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.5082 | 2/5, 388 | 3/5, 500 | 3/5, 400 | both 2: B25 faster 0, slower 2; mean saving -39 units (95% CI -185 to 107); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.5082 | 2/5, 388 | 3/5, 500 | 3/5, 400 | both 2: B25 faster 0, slower 2; mean saving -39 units (95% CI -185 to 107); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.5082 | 2/5, 388 | 3/5, 500 | 3/5, 400 | both 2: B25 faster 0, slower 2; mean saving -39 units (95% CI -185 to 107); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.5327 | 5/5 / 5/5 | -5 (-65 to 54) | 3 / 2 | no | no |
| 25% | 1% | 0.5380 | 5/5 / 5/5 | -10 (-27 to 7) | 1 / 4 | no | no |
| 25% | 2% | 0.5433 | 5/5 / 5/5 | -25 (-47 to -4) | 0 / 5 | no | yes |
| 25% | 5% | 0.5593 | 5/5 / 5/5 | -21 (-55 to 13) | 0 / 5 | no | yes |
| 50% | 0% | 0.5082 | 2/5 / 3/5 | -39 (-185 to 107) | 0 / 2 | no | no |
| 50% | 1% | 0.5132 | 3/5 / 4/5 | -26 (-191 to 139) | 1 / 2 | no | no |
| 50% | 2% | 0.5183 | 5/5 / 4/5 | -26 (-83 to 31) | 1 / 3 | no | no |
| 50% | 5% | 0.5336 | 5/5 / 5/5 | -5 (-65 to 54) | 3 / 2 | no | no |
| 75% | 0% | 0.5082 | 2/5 / 3/5 | -39 (-185 to 107) | 0 / 2 | no | no |
| 75% | 1% | 0.5132 | 3/5 / 4/5 | -26 (-191 to 139) | 1 / 2 | no | no |
| 75% | 2% | 0.5183 | 5/5 / 4/5 | -26 (-83 to 31) | 1 / 3 | no | no |
| 75% | 5% | 0.5336 | 5/5 / 5/5 | -5 (-65 to 54) | 3 / 2 | no | no |
| 95% | 0% | 0.5082 | 2/5 / 3/5 | -39 (-185 to 107) | 0 / 2 | no | no |
| 95% | 1% | 0.5132 | 3/5 / 4/5 | -26 (-191 to 139) | 1 / 2 | no | no |
| 95% | 2% | 0.5183 | 5/5 / 4/5 | -26 (-83 to 31) | 1 / 3 | no | no |
| 95% | 5% | 0.5336 | 5/5 / 5/5 | -5 (-65 to 54) | 3 / 2 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.5327 ± 0.0051 | 0.5082 ± 0.0082 | 0.5082 ± 0.0082 | 0.5082 ± 0.0082 | 0.5248 ± 0.0055 |
| b25 | 0.5370 ± 0.0069 | 0.5106 ± 0.0089 | 0.5066 ± 0.0094 | 0.5066 ± 0.0094 | 0.5275 ± 0.0059 |
| b25_plain | 0.5385 ± 0.0034 | 0.5101 ± 0.0094 | 0.5077 ± 0.0082 | 0.5077 ± 0.0082 | 0.5279 ± 0.0058 |
| sgd | 0.5595 ± 0.0130 | 0.5290 ± 0.0048 | 0.5189 ± 0.0016 | 0.5178 ± 0.0026 | 0.5430 ± 0.0045 |
| adam | 0.5510 ± 0.0043 | 0.5267 ± 0.0125 | 0.5248 ± 0.0134 | 0.5248 ± 0.0134 | 0.5414 ± 0.0106 |
| rmsprop | 0.5197 ± 0.0053 | 0.5072 ± 0.0046 | 0.5072 ± 0.0046 | 0.5072 ± 0.0046 | 0.5193 ± 0.0036 |
| nadam | 0.5217 ± 0.0072 | 0.5001 ± 0.0069 | 0.5001 ± 0.0069 | 0.5001 ± 0.0069 | 0.5169 ± 0.0048 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00434 | [-0.00492, 0.01360] | 0.2631 |
| at 50% of the budget | 5 | 0.00248 | [-0.00130, 0.00626] | 0.1426 |
| at 75% of the budget | 5 | -0.00158 | [-0.00819, 0.00503] | 0.5426 |
| at 100% of the budget | 5 | -0.00158 | [-0.00819, 0.00503] | 0.5426 |
| mean over the whole budget (area under the curve) | 5 | 0.00270 | [-0.00130, 0.00670] | 0.1344 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.