# Comparison: ftt_adult

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val F1 @best-val ckpt | test F1 @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.3101 ± 0.0017 | 0.8525 ± 0.0020 | 0.8625 ± 0.0010 | 0.8581 ± 0.0052 | 0.6550 ± 0.0153 | 0.6752 ± 0.0121 | 172 ± 0 | 0.632 ± 0.059 |
| adam | 5 | 0.3085 ± 0.0005 | 0.8536 ± 0.0011 | 0.8627 ± 0.0007 | 0.8616 ± 0.0018 | 0.6613 ± 0.0123 | 0.6781 ± 0.0087 | 172 ± 0 | 0.794 ± 0.021 |
| adamw | 5 | 0.3044 ± 0.0013 | 0.8575 ± 0.0018 | 0.8638 ± 0.0012 | 0.8624 ± 0.0011 | 0.6730 ± 0.0077 | 0.6846 ± 0.0090 | 172 ± 0 | 0.635 ± 0.044 |
| rmsprop | 5 | 0.3100 ± 0.0003 | 0.8534 ± 0.0013 | 0.8613 ± 0.0014 | 0.8592 ± 0.0035 | 0.6542 ± 0.0120 | 0.6716 ± 0.0114 | 171 ± 1 | 0.051 ± 0.017 |
| nadam | 5 | 0.3071 ± 0.0008 | 0.8549 ± 0.0008 | 0.8634 ± 0.0009 | 0.8581 ± 0.0042 | 0.6580 ± 0.0119 | 0.6761 ± 0.0113 | 172 ± 0 | 0.664 ± 0.038 |
| b25 | 5 | 0.3044 ± 0.0014 | 0.8579 ± 0.0008 | 0.8636 ± 0.0009 | 0.8610 ± 0.0023 | 0.6736 ± 0.0072 | 0.6828 ± 0.0087 | 178 ± 0 | 0.636 ± 0.046 |
| b25_plain | 5 | 0.3051 ± 0.0010 | 0.8567 ± 0.0017 | 0.8638 ± 0.0014 | 0.8598 ± 0.0017 | 0.6640 ± 0.0170 | 0.6784 ± 0.0138 | 177 ± 0 | 0.641 ± 0.042 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00002 | [-0.00052, 0.00049] | -0.09 (4) | 0.9314 | -0.04 | 3/5 |
| val loss at end (exploratory) | 5 | 0.00069 | [-0.00302, 0.00441] | 0.52 (4) | 0.6314 | 0.23 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00019 | [-0.00245, 0.00207] | -0.24 (4) | 0.8257 | -0.11 | 2/5 |
| test acc @end (exploratory) | 5 | -0.00142 | [-0.00459, 0.00175] | -1.24 (4) | 0.2813 | -0.56 | 1/5 |
| test F1 @best-val ckpt (exploratory; binary tasks) | 5 | -0.00178 | [-0.02161, 0.01806] | -0.25 (4) | 0.8158 | -0.11 | 3/5 |
| wall-clock s (exploratory) | 5 | 5.62324 | [5.43260, 5.81387] | 81.90 (4) | 0.0000 | 36.63 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.00573 | [-0.00799, -0.00346] | 0.0022 | 0.0065 | 5/5 |
| adam | 5 | -0.00413 | [-0.00592, -0.00233] | 0.0031 | 0.0065 | 5/5 |
| rmsprop | 5 | -0.00564 | [-0.00755, -0.00373] | 0.0012 | 0.0048 | 5/5 |
| nadam | 5 | -0.00272 | [-0.00502, -0.00043] | 0.0302 | 0.0302 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00073, 95% CI [-0.00189, 0.00043], p 0.1540, B25 better in 4/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 75.2 | 40.6 | 32.8 | 0.4 | 1.4 | 0.0 | 1.4 | 0.019 | 0.000 | 1.4 |
| b25_plain | 5 | 74.2 | 68.6 | 0.0 | 2.4 | 3.2 | 0.0 | 3.2 | 0.043 | 0.000 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.3170 | 5/5, 200 | 5/5, 226 | 5/5, 200 | both 5: B25 faster 3, slower 2; mean saving 9 units (95% CI -105 to 124); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.3101 | 5/5, 450 | 5/5, 400 | 5/5, 501 | both 5: B25 faster 3, slower 2; mean saving 14 units (95% CI -33 to 62); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.3067 | 5/5, 700 | 5/5, 676 | 5/5, 651 | both 5: B25 faster 3, slower 2; mean saving 19 units (95% CI -164 to 203); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.3044 | 2/5, 838 | 2/5, 875 | 1/5, 776 | both 2: B25 faster 0, slower 1; mean saving -38 units (95% CI -520 to 444); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.3044 | 2/5, 838 | 2/5, 875 | 1/5, 776 | both 2: B25 faster 0, slower 1; mean saving -38 units (95% CI -520 to 444); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.3170 | 5/5 / 5/5 | 9 (-105 to 124) | 3 / 2 | no | no |
| 25% | 1% | 0.3202 | 5/5 / 5/5 | -26 (-151 to 100) | 2 / 3 | no | no |
| 25% | 2% | 0.3233 | 5/5 / 5/5 | 9 (-101 to 120) | 3 / 2 | no | no |
| 25% | 5% | 0.3328 | 5/5 / 5/5 | 14 (-14 to 42) | 2 / 2 | no | yes |
| 50% | 0% | 0.3101 | 5/5 / 5/5 | 14 (-33 to 62) | 3 / 2 | no | no |
| 50% | 1% | 0.3132 | 5/5 / 5/5 | -26 (-179 to 128) | 2 / 3 | no | no |
| 50% | 2% | 0.3163 | 5/5 / 5/5 | 9 (-105 to 124) | 3 / 2 | no | yes |
| 50% | 5% | 0.3256 | 5/5 / 5/5 | 4 (-118 to 127) | 2 / 3 | no | yes |
| 75% | 0% | 0.3067 | 5/5 / 5/5 | 19 (-164 to 203) | 3 / 2 | no | no |
| 75% | 1% | 0.3098 | 5/5 / 5/5 | 14 (-33 to 62) | 3 / 2 | no | yes |
| 75% | 2% | 0.3128 | 5/5 / 5/5 | 19 (-178 to 217) | 3 / 2 | no | yes |
| 75% | 5% | 0.3220 | 5/5 / 5/5 | -6 (-107 to 96) | 2 / 3 | no | yes |
| 95% | 0% | 0.3044 | 2/5 / 2/5 | -38 (-520 to 444) | 0 / 1 | no | no |
| 95% | 1% | 0.3075 | 5/5 / 5/5 | -6 (-163 to 151) | 3 / 2 | no | no |
| 95% | 2% | 0.3105 | 5/5 / 5/5 | 74 (-35 to 184) | 4 / 1 | no | no |
| 95% | 5% | 0.3197 | 5/5 / 5/5 | -26 (-151 to 100) | 2 / 3 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.3170 ± 0.0064 | 0.3101 ± 0.0020 | 0.3067 ± 0.0006 | 0.3044 ± 0.0013 | 0.3154 ± 0.0021 |
| b25 | 0.3184 ± 0.0066 | 0.3095 ± 0.0007 | 0.3063 ± 0.0003 | 0.3044 ± 0.0014 | 0.3157 ± 0.0017 |
| b25_plain | 0.3147 ± 0.0039 | 0.3099 ± 0.0016 | 0.3066 ± 0.0012 | 0.3051 ± 0.0010 | 0.3157 ± 0.0021 |
| sgd | 0.3320 ± 0.0100 | 0.3137 ± 0.0019 | 0.3129 ± 0.0021 | 0.3101 ± 0.0017 | 0.3329 ± 0.0067 |
| adam | 0.3168 ± 0.0055 | 0.3110 ± 0.0010 | 0.3089 ± 0.0003 | 0.3085 ± 0.0005 | 0.3191 ± 0.0035 |
| rmsprop | 0.3242 ± 0.0074 | 0.3148 ± 0.0030 | 0.3115 ± 0.0010 | 0.3100 ± 0.0003 | 0.3242 ± 0.0042 |
| nadam | 0.3230 ± 0.0056 | 0.3128 ± 0.0033 | 0.3084 ± 0.0011 | 0.3071 ± 0.0008 | 0.3255 ± 0.0012 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00140 | [-0.01224, 0.01504] | 0.7900 |
| at 50% of the budget | 5 | -0.00060 | [-0.00224, 0.00104] | 0.3680 |
| at 75% of the budget | 5 | -0.00044 | [-0.00099, 0.00011] | 0.0884 |
| at 100% of the budget | 5 | -0.00002 | [-0.00052, 0.00049] | 0.9314 |
| mean over the whole budget (area under the curve) | 5 | 0.00034 | [-0.00115, 0.00183] | 0.5590 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.