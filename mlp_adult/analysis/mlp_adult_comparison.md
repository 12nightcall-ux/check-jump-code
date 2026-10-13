# Comparison: mlp_adult

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val F1 @best-val ckpt | test F1 @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.3104 ± 0.0005 | 0.8536 ± 0.0013 | 0.8602 ± 0.0010 | 0.8590 ± 0.0015 | 0.6640 ± 0.0071 | 0.6777 ± 0.0056 | 16 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 0.3076 ± 0.0013 | 0.8550 ± 0.0016 | 0.8617 ± 0.0018 | 0.8610 ± 0.0014 | 0.6632 ± 0.0073 | 0.6772 ± 0.0067 | 16 ± 0 | 0.000 ± 0.000 |
| adamw | 5 | 0.3099 ± 0.0007 | 0.8533 ± 0.0015 | 0.8603 ± 0.0007 | 0.8581 ± 0.0015 | 0.6617 ± 0.0129 | 0.6764 ± 0.0113 | 15 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 0.3100 ± 0.0020 | 0.8549 ± 0.0017 | 0.8603 ± 0.0014 | 0.8573 ± 0.0027 | 0.6706 ± 0.0079 | 0.6804 ± 0.0057 | 16 ± 0 | 0.003 ± 0.002 |
| nadam | 5 | 0.3071 ± 0.0005 | 0.8561 ± 0.0012 | 0.8614 ± 0.0007 | 0.8622 ± 0.0009 | 0.6729 ± 0.0075 | 0.6822 ± 0.0056 | 16 ± 0 | 0.000 ± 0.001 |
| b25 | 5 | 0.3103 ± 0.0004 | 0.8544 ± 0.0018 | 0.8596 ± 0.0016 | 0.8573 ± 0.0019 | 0.6633 ± 0.0131 | 0.6714 ± 0.0118 | 17 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 0.3104 ± 0.0006 | 0.8531 ± 0.0022 | 0.8597 ± 0.0017 | 0.8585 ± 0.0014 | 0.6641 ± 0.0152 | 0.6772 ± 0.0134 | 17 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00032 | [-0.00039, 0.00102] | 1.24 (4) | 0.2835 | 0.55 | 1/5 |
| val loss at end (exploratory) | 5 | -0.01371 | [-0.02355, -0.00387] | -3.87 (4) | 0.0180 | -1.73 | 5/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00074 | [-0.00211, 0.00064] | -1.49 (4) | 0.2108 | -0.67 | 1/5 |
| test acc @end (exploratory) | 5 | -0.00082 | [-0.00429, 0.00266] | -0.65 (4) | 0.5487 | -0.29 | 2/5 |
| test F1 @best-val ckpt (exploratory; binary tasks) | 5 | -0.00495 | [-0.02233, 0.01244] | -0.79 (4) | 0.4738 | -0.35 | 2/5 |
| wall-clock s (exploratory) | 5 | 1.47179 | [1.25183, 1.69175] | 18.58 (4) | 0.0000 | 8.31 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.00010 | [-0.00066, 0.00045] | 0.6272 | 1.0000 | 3/5 |
| adam | 5 | 0.00262 | [0.00095, 0.00428] | 0.0120 | 0.0360 | 0/5 |
| rmsprop | 5 | 0.00022 | [-0.00219, 0.00263] | 0.8118 | 1.0000 | 1/5 |
| nadam | 5 | 0.00320 | [0.00235, 0.00405] | 0.0005 | 0.0019 | 0/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00019, 95% CI [-0.00055, 0.00017], p 0.2248, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 74.6 | 48.4 | 22.2 | 1.6 | 2.4 | 0.2 | 2.2 | 0.032 | 0.083 | 2.4 |
| b25_plain | 5 | 74.2 | 69.6 | 0.0 | 1.8 | 2.8 | 0.0 | 2.8 | 0.038 | 0.000 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.3116 | 5/5, 275 | 5/5, 376 | 5/5, 376 | both 5: B25 faster 1, slower 4; mean saving -86 units (95% CI -185 to 14); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.3099 | 2/5, 375 | 1/5, 401 | 1/5, 401 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.3099 | 2/5, 375 | 1/5, 401 | 1/5, 401 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.3099 | 2/5, 375 | 1/5, 401 | 1/5, 401 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.3099 | 2/5, 375 | 1/5, 401 | 1/5, 401 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.3116 | 5/5 / 5/5 | -86 (-185 to 14) | 1 / 4 | no | no |
| 25% | 1% | 0.3147 | 5/5 / 5/5 | -31 (-57 to -4) | 0 / 4 | no | yes |
| 25% | 2% | 0.3178 | 5/5 / 5/5 | -5 (-65 to 55) | 2 / 3 | no | yes |
| 25% | 5% | 0.3272 | 5/5 / 5/5 | 5 (-22 to 31) | 2 / 1 | no | yes |
| 50% | 0% | 0.3099 | 2/5 / 1/5 | n/a | 0 / 1 | no | no |
| 50% | 1% | 0.3130 | 5/5 / 5/5 | -46 (-116 to 25) | 1 / 4 | no | no |
| 50% | 2% | 0.3161 | 5/5 / 5/5 | -15 (-51 to 20) | 1 / 3 | no | no |
| 50% | 5% | 0.3254 | 5/5 / 5/5 | 10 (-18 to 38) | 3 / 1 | no | no |
| 75% | 0% | 0.3099 | 2/5 / 1/5 | n/a | 0 / 1 | no | no |
| 75% | 1% | 0.3130 | 5/5 / 5/5 | -46 (-116 to 25) | 1 / 4 | no | no |
| 75% | 2% | 0.3161 | 5/5 / 5/5 | -15 (-51 to 20) | 1 / 3 | no | no |
| 75% | 5% | 0.3254 | 5/5 / 5/5 | 10 (-18 to 38) | 3 / 1 | no | no |
| 95% | 0% | 0.3099 | 2/5 / 1/5 | n/a | 0 / 1 | no | no |
| 95% | 1% | 0.3130 | 5/5 / 5/5 | -46 (-116 to 25) | 1 / 4 | no | no |
| 95% | 2% | 0.3161 | 5/5 / 5/5 | -15 (-51 to 20) | 1 / 3 | no | no |
| 95% | 5% | 0.3254 | 5/5 / 5/5 | 10 (-18 to 38) | 3 / 1 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.3116 ± 0.0009 | 0.3099 ± 0.0007 | 0.3099 ± 0.0007 | 0.3099 ± 0.0007 | 0.3121 ± 0.0005 |
| b25 | 0.3125 ± 0.0006 | 0.3103 ± 0.0005 | 0.3103 ± 0.0004 | 0.3103 ± 0.0004 | 0.3126 ± 0.0006 |
| b25_plain | 0.3127 ± 0.0011 | 0.3108 ± 0.0012 | 0.3104 ± 0.0006 | 0.3104 ± 0.0006 | 0.3127 ± 0.0008 |
| sgd | 0.3438 ± 0.0051 | 0.3174 ± 0.0009 | 0.3119 ± 0.0007 | 0.3104 ± 0.0005 | 0.3524 ± 0.0027 |
| adam | 0.3134 ± 0.0008 | 0.3106 ± 0.0013 | 0.3091 ± 0.0003 | 0.3076 ± 0.0013 | 0.3124 ± 0.0003 |
| rmsprop | 0.3116 ± 0.0011 | 0.3100 ± 0.0020 | 0.3100 ± 0.0020 | 0.3100 ± 0.0020 | 0.3118 ± 0.0014 |
| nadam | 0.3130 ± 0.0006 | 0.3105 ± 0.0012 | 0.3091 ± 0.0008 | 0.3071 ± 0.0005 | 0.3120 ± 0.0007 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00092 | [-0.00072, 0.00256] | 0.1947 |
| at 50% of the budget | 5 | 0.00040 | [-0.00031, 0.00111] | 0.1896 |
| at 75% of the budget | 5 | 0.00032 | [-0.00039, 0.00102] | 0.2835 |
| at 100% of the budget | 5 | 0.00032 | [-0.00039, 0.00102] | 0.2835 |
| mean over the whole budget (area under the curve) | 5 | 0.00051 | [-0.00025, 0.00127] | 0.1354 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.