# Comparison: distilbert_sst2

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.2285 ± 0.0098 | 0.9106 ± 0.0087 | 0.8828 ± 0.0049 | 0.8739 ± 0.0137 | 320 ± 3 | 0.987 ± 0.002 |
| adam | 5 | 0.1890 ± 0.0035 | 0.9284 ± 0.0020 | 0.9021 ± 0.0066 | 0.8986 ± 0.0062 | 327 ± 0 | 0.990 ± 0.004 |
| adamw | 5 | 0.1888 ± 0.0035 | 0.9281 ± 0.0024 | 0.9018 ± 0.0055 | 0.8984 ± 0.0042 | 340 ± 1 | 0.990 ± 0.004 |
| rmsprop | 5 | 0.1844 ± 0.0049 | 0.9297 ± 0.0027 | 0.8979 ± 0.0066 | 0.8938 ± 0.0071 | 328 ± 6 | 0.994 ± 0.002 |
| nadam | 5 | 0.1788 ± 0.0032 | 0.9309 ± 0.0019 | 0.8991 ± 0.0096 | 0.8959 ± 0.0128 | 331 ± 0 | 0.991 ± 0.003 |
| b25 | 5 | 0.1857 ± 0.0072 | 0.9295 ± 0.0044 | 0.8888 ± 0.0075 | 0.8869 ± 0.0097 | 364 ± 1 | 0.989 ± 0.003 |
| b25_plain | 5 | 0.1874 ± 0.0086 | 0.9280 ± 0.0056 | 0.8913 ± 0.0105 | 0.8865 ± 0.0095 | 358 ± 16 | 0.991 ± 0.001 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00306 | [-0.01181, 0.00568] | -0.97 (4) | 0.3860 | -0.43 | 3/5 |
| val loss at end (exploratory) | 5 | 0.00469 | [-0.00760, 0.01698] | 1.06 (4) | 0.3491 | 0.47 | 1/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.01307 | [-0.02071, -0.00543] | -4.75 (4) | 0.0090 | -2.12 | 0/5 |
| test acc @end (exploratory) | 5 | -0.01147 | [-0.02212, -0.00081] | -2.99 (4) | 0.0404 | -1.34 | 0/5 |
| wall-clock s (exploratory) | 5 | 24.17448 | [22.56934, 25.77962] | 41.82 (4) | 0.0000 | 18.70 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.04275 | [-0.05929, -0.02621] | 0.0020 | 0.0080 | 5/5 |
| adam | 5 | -0.00325 | [-0.01187, 0.00538] | 0.3546 | 0.7093 | 3/5 |
| rmsprop | 5 | 0.00132 | [-0.00256, 0.00521] | 0.3978 | 0.7093 | 1/5 |
| nadam | 5 | 0.00695 | [-0.00331, 0.01721] | 0.1331 | 0.3994 | 1/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00169, 95% CI [-0.00970, 0.00632], p 0.5896, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 72.6 | 35.6 | 26.8 | 1.4 | 8.8 | 2.0 | 6.8 | 0.121 | 0.227 | 8.8 |
| b25_plain | 5 | 70.6 | 54.0 | 0.0 | 2.2 | 14.4 | 3.0 | 11.4 | 0.204 | 0.208 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.2591 | 5/5, 250 | 5/5, 276 | 5/5, 276 | both 5: B25 faster 2, slower 3; mean saving 9 units (95% CI -55 to 74); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.2120 | 5/5, 500 | 5/5, 476 | 5/5, 500 | both 5: B25 faster 4, slower 1; mean saving 44 units (95% CI -30 to 119); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.1980 | 5/5, 775 | 5/5, 775 | 5/5, 801 | both 5: B25 faster 2, slower 3; mean saving 30 units (95% CI -184 to 243); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.1908 | 4/5, 838 | 4/5, 888 | 3/5, 900 | both 3: B25 faster 1, slower 2; mean saving -59 units (95% CI -400 to 283); only B25 1, only AdamW 1, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.1888 | 4/5, 900 | 3/5, 826 | 2/5, 876 | both 3: B25 faster 2, slower 1; mean saving 25 units (95% CI -139 to 189); only B25 0, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.2591 | 5/5 / 5/5 | 9 (-55 to 74) | 2 / 3 | no | no |
| 25% | 1% | 0.2617 | 5/5 / 5/5 | 4 (-51 to 60) | 2 / 3 | no | no |
| 25% | 2% | 0.2643 | 5/5 / 5/5 | 4 (-51 to 60) | 2 / 3 | no | no |
| 25% | 5% | 0.2721 | 5/5 / 5/5 | -6 (-65 to 54) | 1 / 4 | no | no |
| 50% | 0% | 0.2120 | 5/5 / 5/5 | 44 (-30 to 119) | 4 / 1 | no | no |
| 50% | 1% | 0.2141 | 5/5 / 5/5 | 34 (-22 to 91) | 3 / 2 | no | no |
| 50% | 2% | 0.2162 | 5/5 / 5/5 | 24 (-34 to 83) | 2 / 3 | no | no |
| 50% | 5% | 0.2226 | 5/5 / 5/5 | 29 (-48 to 107) | 3 / 2 | no | yes |
| 75% | 0% | 0.1980 | 5/5 / 5/5 | 30 (-184 to 243) | 2 / 3 | no | no |
| 75% | 1% | 0.2000 | 5/5 / 5/5 | 5 (-173 to 182) | 2 / 3 | no | no |
| 75% | 2% | 0.2020 | 5/5 / 5/5 | -25 (-148 to 97) | 1 / 4 | no | no |
| 75% | 5% | 0.2079 | 5/5 / 5/5 | 45 (-57 to 146) | 3 / 2 | no | yes |
| 95% | 0% | 0.1908 | 4/5 / 4/5 | -59 (-400 to 283) | 1 / 2 | no | no |
| 95% | 1% | 0.1927 | 4/5 / 5/5 | 43 (-229 to 316) | 3 / 1 | no | no |
| 95% | 2% | 0.1946 | 4/5 / 5/5 | 93 (-246 to 433) | 3 / 1 | no | no |
| 95% | 5% | 0.2003 | 5/5 / 5/5 | 5 (-173 to 182) | 2 / 3 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.2591 ± 0.0084 | 0.2120 ± 0.0096 | 0.1980 ± 0.0042 | 0.1888 ± 0.0035 | 0.2362 ± 0.0049 |
| b25 | 0.2645 ± 0.0126 | 0.2168 ± 0.0156 | 0.1994 ± 0.0058 | 0.1857 ± 0.0072 | 0.2371 ± 0.0027 |
| b25_plain | 0.2628 ± 0.0113 | 0.2180 ± 0.0141 | 0.2007 ± 0.0044 | 0.1874 ± 0.0086 | 0.2377 ± 0.0033 |
| sgd | 0.3148 ± 0.0248 | 0.2690 ± 0.0071 | 0.2426 ± 0.0072 | 0.2285 ± 0.0098 | 0.2957 ± 0.0098 |
| adam | 0.2591 ± 0.0083 | 0.2121 ± 0.0096 | 0.1981 ± 0.0042 | 0.1890 ± 0.0035 | 0.2362 ± 0.0049 |
| rmsprop | 0.2485 ± 0.0046 | 0.2064 ± 0.0073 | 0.1931 ± 0.0070 | 0.1844 ± 0.0049 | 0.2245 ± 0.0043 |
| nadam | 0.2455 ± 0.0071 | 0.2044 ± 0.0065 | 0.1914 ± 0.0066 | 0.1788 ± 0.0032 | 0.2249 ± 0.0030 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00543 | [-0.01369, 0.02455] | 0.4744 |
| at 50% of the budget | 5 | 0.00479 | [-0.00576, 0.01533] | 0.2762 |
| at 75% of the budget | 5 | 0.00141 | [-0.00895, 0.01176] | 0.7250 |
| at 100% of the budget | 5 | -0.00306 | [-0.01181, 0.00568] | 0.3860 |
| mean over the whole budget (area under the curve) | 5 | 0.00097 | [-0.00541, 0.00735] | 0.6953 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.