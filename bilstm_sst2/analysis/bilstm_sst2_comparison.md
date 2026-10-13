# Comparison: bilstm_sst2

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.3306 ± 0.0051 | 0.8566 ± 0.0054 | 0.7959 ± 0.0026 | 0.7899 ± 0.0166 | 374 ± 1 | 0.226 ± 0.017 |
| adam | 5 | 0.3137 ± 0.0024 | 0.8674 ± 0.0035 | 0.8117 ± 0.0045 | 0.7961 ± 0.0247 | 388 ± 1 | 0.321 ± 0.009 |
| adamw | 5 | 0.2616 ± 0.0144 | 0.8928 ± 0.0034 | 0.8275 ± 0.0099 | 0.8287 ± 0.0164 | 387 ± 1 | 0.769 ± 0.049 |
| rmsprop | 5 | 0.2741 ± 0.0166 | 0.8860 ± 0.0131 | 0.8190 ± 0.0106 | 0.8108 ± 0.0127 | 378 ± 1 | 0.770 ± 0.041 |
| nadam | 5 | 0.3137 ± 0.0023 | 0.8691 ± 0.0022 | 0.8179 ± 0.0039 | 0.8055 ± 0.0151 | 393 ± 0 | 0.326 ± 0.012 |
| b25 | 5 | 0.2755 ± 0.0025 | 0.8835 ± 0.0017 | 0.8307 ± 0.0043 | 0.8330 ± 0.0085 | 418 ± 3 | 0.344 ± 0.021 |
| b25_plain | 5 | 0.2752 ± 0.0046 | 0.8870 ± 0.0057 | 0.8317 ± 0.0084 | 0.8310 ± 0.0031 | 412 ± 1 | 0.346 ± 0.017 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.01386 | [-0.00518, 0.03290] | 2.02 (4) | 0.1134 | 0.90 | 1/5 |
| val loss at end (exploratory) | 5 | -0.00511 | [-0.02840, 0.01818] | -0.61 (4) | 0.5752 | -0.27 | 3/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00321 | [-0.01392, 0.02034] | 0.52 (4) | 0.6302 | 0.23 | 3/5 |
| test acc @end (exploratory) | 5 | 0.00436 | [-0.01711, 0.02582] | 0.56 (4) | 0.6030 | 0.25 | 2/5 |
| wall-clock s (exploratory) | 5 | 31.89916 | [28.50478, 35.29354] | 26.09 (4) | 0.0000 | 11.67 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.05518 | [-0.06003, -0.05033] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.03827 | [-0.04166, -0.03487] | 0.0000 | 0.0000 | 5/5 |
| rmsprop | 5 | 0.00141 | [-0.02100, 0.02381] | 0.8702 | 0.8702 | 3/5 |
| nadam | 5 | -0.03823 | [-0.04303, -0.03342] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00027, 95% CI [-0.00308, 0.00362], p 0.8358, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 73.0 | 36.0 | 27.8 | 2.2 | 7.0 | 1.6 | 5.4 | 0.096 | 0.229 | 7.0 |
| b25_plain | 5 | 70.8 | 53.6 | 0.0 | 3.2 | 14.0 | 3.2 | 10.8 | 0.198 | 0.229 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.3824 | 5/5, 250 | 5/5, 101 | 5/5, 101 | both 5: B25 faster 5, slower 0; mean saving 144 units (95% CI 34 to 255); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 50% | 0.3042 | 5/5, 500 | 5/5, 501 | 5/5, 526 | both 5: B25 faster 3, slower 2; mean saving 9 units (95% CI -134 to 153); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.2775 | 4/5, 700 | 4/5, 938 | 3/5, 875 | both 3: B25 faster 0, slower 3; mean saving -200 units (95% CI -546 to 145); only B25 1, only AdamW 1, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.2626 | 3/5, 775 | 0/5, n/a | 0/5, n/a | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 0, only AdamW 3, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.2616 | 3/5, 775 | 0/5, n/a | 0/5, n/a | both 0: B25 faster 0, slower 0; mean saving n/a units; only B25 0, only AdamW 3, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.3824 | 5/5 / 5/5 | 144 (34 to 255) | 5 / 0 | YES | yes |
| 25% | 1% | 0.3862 | 5/5 / 5/5 | 144 (34 to 255) | 5 / 0 | YES | yes |
| 25% | 2% | 0.3900 | 5/5 / 5/5 | 144 (71 to 218) | 5 / 0 | YES | yes |
| 25% | 5% | 0.4015 | 5/5 / 5/5 | 144 (64 to 224) | 5 / 0 | YES | yes |
| 50% | 0% | 0.3042 | 5/5 / 5/5 | 9 (-134 to 153) | 3 / 2 | no | no |
| 50% | 1% | 0.3072 | 5/5 / 5/5 | 19 (-125 to 164) | 3 / 2 | no | no |
| 50% | 2% | 0.3102 | 5/5 / 5/5 | 24 (-74 to 123) | 3 / 2 | no | no |
| 50% | 5% | 0.3194 | 5/5 / 5/5 | 69 (-46 to 184) | 4 / 1 | no | no |
| 75% | 0% | 0.2775 | 4/5 / 4/5 | -200 (-546 to 145) | 0 / 3 | no | no |
| 75% | 1% | 0.2803 | 4/5 / 5/5 | -238 (-449 to -27) | 0 / 4 | no | no |
| 75% | 2% | 0.2830 | 4/5 / 5/5 | -213 (-429 to 3) | 0 / 4 | no | no |
| 75% | 5% | 0.2914 | 5/5 / 5/5 | -86 (-245 to 73) | 1 / 4 | no | no |
| 95% | 0% | 0.2626 | 3/5 / 0/5 | n/a | 0 / 0 | no | no |
| 95% | 1% | 0.2653 | 3/5 / 0/5 | n/a | 0 / 0 | no | no |
| 95% | 2% | 0.2679 | 3/5 / 0/5 | n/a | 0 / 0 | no | no |
| 95% | 5% | 0.2758 | 4/5 / 4/5 | -217 (-628 to 195) | 0 / 3 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.3824 ± 0.0573 | 0.3042 ± 0.0193 | 0.2775 ± 0.0160 | 0.2616 ± 0.0144 | 0.3498 ± 0.0329 |
| b25 | 0.3367 ± 0.0023 | 0.3048 ± 0.0067 | 0.2885 ± 0.0023 | 0.2755 ± 0.0025 | 0.3221 ± 0.0036 |
| b25_plain | 0.3395 ± 0.0023 | 0.3070 ± 0.0071 | 0.2907 ± 0.0043 | 0.2752 ± 0.0046 | 0.3233 ± 0.0031 |
| sgd | 0.6744 ± 0.0042 | 0.4246 ± 0.0081 | 0.3508 ± 0.0084 | 0.3306 ± 0.0051 | 0.4872 ± 0.0114 |
| adam | 0.3568 ± 0.0069 | 0.3313 ± 0.0026 | 0.3189 ± 0.0009 | 0.3137 ± 0.0024 | 0.3484 ± 0.0029 |
| rmsprop | 0.5164 ± 0.0418 | 0.3917 ± 0.0524 | 0.3120 ± 0.0227 | 0.2741 ± 0.0166 | 0.4412 ± 0.0392 |
| nadam | 0.3557 ± 0.0088 | 0.3312 ± 0.0039 | 0.3165 ± 0.0027 | 0.3137 ± 0.0023 | 0.3465 ± 0.0023 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.04566 | [-0.11501, 0.02369] | 0.1416 |
| at 50% of the budget | 5 | 0.00066 | [-0.01590, 0.01722] | 0.9171 |
| at 75% of the budget | 5 | 0.01100 | [-0.00917, 0.03117] | 0.2045 |
| at 100% of the budget | 5 | 0.01386 | [-0.00518, 0.03290] | 0.1134 |
| mean over the whole budget (area under the curve) | 5 | -0.02771 | [-0.06571, 0.01030] | 0.1130 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.