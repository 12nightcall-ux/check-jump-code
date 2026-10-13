# Comparison: mlp_california

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

Regression task (california): the 'loss' columns are mean squared error, and the columns named 'Pearson r' are Pearson correlations (higher is better).

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val Pearson r | test Pearson r @best-val ckpt | test Pearson r @end | val Spearman @best-val ckpt | test Spearman @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.2214 ± 0.0022 | 0.8832 ± 0.0017 | 0.8825 ± 0.0022 | 0.8801 ± 0.0039 | 0.8868 ± 0.0013 | 0.8850 ± 0.0021 | 10 ± 0 | 0.243 ± 0.045 |
| adam | 5 | 0.2154 ± 0.0039 | 0.8872 ± 0.0018 | 0.8837 ± 0.0013 | 0.8823 ± 0.0048 | 0.8905 ± 0.0037 | 0.8872 ± 0.0026 | 10 ± 0 | 0.167 ± 0.014 |
| adamw | 5 | 0.2088 ± 0.0032 | 0.8907 ± 0.0019 | 0.8887 ± 0.0013 | 0.8890 ± 0.0020 | 0.8947 ± 0.0037 | 0.8927 ± 0.0027 | 10 ± 0 | 0.075 ± 0.010 |
| rmsprop | 5 | 0.2076 ± 0.0020 | 0.8918 ± 0.0010 | 0.8898 ± 0.0008 | 0.8883 ± 0.0011 | 0.8970 ± 0.0014 | 0.8952 ± 0.0009 | 10 ± 0 | 0.218 ± 0.046 |
| nadam | 5 | 0.2109 ± 0.0034 | 0.8902 ± 0.0010 | 0.8857 ± 0.0015 | 0.8852 ± 0.0013 | 0.8937 ± 0.0007 | 0.8896 ± 0.0010 | 10 ± 0 | 0.224 ± 0.008 |
| b25 | 5 | 0.2054 ± 0.0025 | 0.8922 ± 0.0016 | 0.8886 ± 0.0023 | 0.8882 ± 0.0014 | 0.8967 ± 0.0017 | 0.8953 ± 0.0030 | 11 ± 0 | 0.083 ± 0.009 |
| b25_plain | 5 | 0.2052 ± 0.0023 | 0.8922 ± 0.0016 | 0.8889 ± 0.0020 | 0.8884 ± 0.0013 | 0.8963 ± 0.0021 | 0.8952 ± 0.0031 | 11 ± 0 | 0.083 ± 0.011 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00334 | [-0.00851, 0.00182] | -1.80 (4) | 0.1468 | -0.80 | 4/5 |
| val loss at end (exploratory) | 5 | -0.00027 | [-0.00892, 0.00837] | -0.09 (4) | 0.9339 | -0.04 | 3/5 |
| test Pearson r @best-val ckpt (exploratory) | 5 | -0.00010 | [-0.00284, 0.00264] | -0.10 (4) | 0.9237 | -0.05 | 3/5 |
| test Pearson r @end (exploratory) | 5 | -0.00085 | [-0.00336, 0.00166] | -0.94 (4) | 0.4023 | -0.42 | 3/5 |
| test Spearman @best-val ckpt (exploratory; regression tasks) | 5 | 0.00258 | [-0.00226, 0.00742] | 1.48 (4) | 0.2128 | 0.66 | 3/5 |
| wall-clock s (exploratory) | 5 | 0.69167 | [0.53704, 0.84629] | 12.42 (4) | 0.0002 | 5.55 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.01598 | [-0.01938, -0.01259] | 0.0002 | 0.0008 | 5/5 |
| adam | 5 | -0.00994 | [-0.01464, -0.00525] | 0.0042 | 0.0126 | 5/5 |
| rmsprop | 5 | -0.00216 | [-0.00626, 0.00194] | 0.2173 | 0.2173 | 4/5 |
| nadam | 5 | -0.00545 | [-0.00937, -0.00153] | 0.0181 | 0.0362 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00018, 95% CI [-0.00033, 0.00069], p 0.3739, B25 better in 0/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 75.4 | 48.8 | 25.8 | 0.6 | 0.2 | 0.0 | 0.2 | 0.003 | 0.000 | 0.2 |
| b25_plain | 5 | 75.4 | 74.4 | 0.0 | 0.6 | 0.4 | 0.0 | 0.4 | 0.005 | 0.000 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.2326 | 5/5, 250 | 5/5, 250 | 5/5, 250 | both 5: B25 faster 3, slower 2; mean saving -15 units (95% CI -115 to 85); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.2197 | 5/5, 525 | 5/5, 475 | 5/5, 500 | both 5: B25 faster 3, slower 2; mean saving 15 units (95% CI -85 to 114); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.2114 | 4/5, 700 | 5/5, 751 | 5/5, 750 | both 4: B25 faster 1, slower 2; mean saving 12 units (95% CI -167 to 192); only B25 1, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.2096 | 4/5, 700 | 5/5, 775 | 5/5, 776 | both 4: B25 faster 1, slower 3; mean saving -44 units (95% CI -306 to 219); only B25 1, only AdamW 0, neither 0 | saving NOT supported |
| 100% | 0.2088 | 4/5, 800 | 5/5, 850 | 5/5, 801 | both 4: B25 faster 2, slower 2; mean saving -69 units (95% CI -295 to 157); only B25 1, only AdamW 0, neither 0 | saving NOT supported |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.2326 | 5/5 / 5/5 | -15 (-115 to 85) | 3 / 2 | no | no |
| 25% | 1% | 0.2350 | 5/5 / 5/5 | 20 (-36 to 76) | 4 / 1 | no | no |
| 25% | 2% | 0.2373 | 5/5 / 5/5 | -5 (-92 to 81) | 3 / 2 | no | no |
| 25% | 5% | 0.2443 | 5/5 / 5/5 | -45 (-123 to 33) | 0 / 3 | no | no |
| 50% | 0% | 0.2197 | 5/5 / 5/5 | 15 (-85 to 114) | 3 / 2 | no | no |
| 50% | 1% | 0.2219 | 5/5 / 5/5 | -30 (-97 to 37) | 1 / 2 | no | no |
| 50% | 2% | 0.2241 | 5/5 / 5/5 | -60 (-173 to 53) | 1 / 3 | no | yes |
| 50% | 5% | 0.2307 | 5/5 / 5/5 | -45 (-137 to 47) | 1 / 4 | no | yes |
| 75% | 0% | 0.2114 | 4/5 / 5/5 | 12 (-167 to 192) | 1 / 2 | no | no |
| 75% | 1% | 0.2135 | 4/5 / 5/5 | 156 (-40 to 352) | 3 / 0 | no | no |
| 75% | 2% | 0.2156 | 5/5 / 5/5 | 110 (-49 to 269) | 3 / 0 | no | no |
| 75% | 5% | 0.2219 | 5/5 / 5/5 | -30 (-97 to 37) | 1 / 2 | no | no |
| 95% | 0% | 0.2096 | 4/5 / 5/5 | -44 (-306 to 219) | 1 / 3 | no | no |
| 95% | 1% | 0.2117 | 4/5 / 5/5 | 12 (-167 to 192) | 1 / 2 | no | no |
| 95% | 2% | 0.2138 | 4/5 / 5/5 | 156 (-40 to 352) | 3 / 0 | no | no |
| 95% | 5% | 0.2201 | 5/5 / 5/5 | -30 (-97 to 37) | 1 / 2 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.2326 ± 0.0053 | 0.2197 ± 0.0019 | 0.2114 ± 0.0039 | 0.2088 ± 0.0032 | 0.2320 ± 0.0019 |
| b25 | 0.2351 ± 0.0122 | 0.2179 ± 0.0038 | 0.2116 ± 0.0017 | 0.2054 ± 0.0025 | 0.2323 ± 0.0038 |
| b25_plain | 0.2368 ± 0.0125 | 0.2185 ± 0.0043 | 0.2109 ± 0.0016 | 0.2052 ± 0.0023 | 0.2325 ± 0.0039 |
| sgd | 0.3163 ± 0.0186 | 0.2559 ± 0.0119 | 0.2290 ± 0.0039 | 0.2214 ± 0.0022 | 0.2971 ± 0.0047 |
| adam | 0.2382 ± 0.0061 | 0.2247 ± 0.0022 | 0.2185 ± 0.0017 | 0.2154 ± 0.0039 | 0.2384 ± 0.0015 |
| rmsprop | 0.2391 ± 0.0035 | 0.2206 ± 0.0057 | 0.2119 ± 0.0021 | 0.2076 ± 0.0020 | 0.2389 ± 0.0066 |
| nadam | 0.2493 ± 0.0080 | 0.2236 ± 0.0035 | 0.2165 ± 0.0015 | 0.2109 ± 0.0034 | 0.2407 ± 0.0016 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00245 | [-0.01152, 0.01642] | 0.6517 |
| at 50% of the budget | 5 | -0.00184 | [-0.00610, 0.00241] | 0.2956 |
| at 75% of the budget | 5 | 0.00023 | [-0.00379, 0.00424] | 0.8828 |
| at 100% of the budget | 5 | -0.00334 | [-0.00851, 0.00182] | 0.1468 |
| mean over the whole budget (area under the curve) | 5 | 0.00030 | [-0.00229, 0.00289] | 0.7632 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.