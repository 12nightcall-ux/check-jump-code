# Comparison: ftt_california

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

Regression task (california): the 'loss' columns are mean squared error, and the columns named 'Pearson r' are Pearson correlations (higher is better).

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val Pearson r | test Pearson r @best-val ckpt | test Pearson r @end | val Spearman @best-val ckpt | test Spearman @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.2183 ± 0.0032 | 0.8863 ± 0.0020 | 0.8805 ± 0.0021 | 0.8774 ± 0.0025 | 0.8894 ± 0.0023 | 0.8863 ± 0.0026 | 19 ± 0 | 0.946 ± 0.010 |
| adam | 5 | 0.2050 ± 0.0040 | 0.8923 ± 0.0023 | 0.8860 ± 0.0009 | 0.8849 ± 0.0039 | 0.8960 ± 0.0021 | 0.8919 ± 0.0014 | 20 ± 0 | 1.000 ± 0.001 |
| adamw | 5 | 0.1925 ± 0.0032 | 0.9004 ± 0.0027 | 0.8960 ± 0.0039 | 0.8936 ± 0.0054 | 0.9061 ± 0.0024 | 0.9036 ± 0.0030 | 20 ± 0 | 1.000 ± 0.000 |
| rmsprop | 5 | 0.2277 ± 0.0047 | 0.8831 ± 0.0020 | 0.8789 ± 0.0024 | 0.8773 ± 0.0045 | 0.8887 ± 0.0017 | 0.8867 ± 0.0024 | 20 ± 0 | 1.000 ± 0.000 |
| nadam | 5 | 0.2107 ± 0.0036 | 0.8913 ± 0.0026 | 0.8876 ± 0.0030 | 0.8840 ± 0.0051 | 0.8956 ± 0.0042 | 0.8934 ± 0.0049 | 20 ± 0 | 1.000 ± 0.000 |
| b25 | 5 | 0.1930 ± 0.0054 | 0.8992 ± 0.0029 | 0.8959 ± 0.0031 | 0.8939 ± 0.0023 | 0.9048 ± 0.0033 | 0.9030 ± 0.0032 | 26 ± 0 | 1.000 ± 0.000 |
| b25_plain | 5 | 0.1930 ± 0.0054 | 0.8992 ± 0.0029 | 0.8959 ± 0.0031 | 0.8939 ± 0.0023 | 0.9048 ± 0.0033 | 0.9030 ± 0.0032 | 26 ± 0 | 1.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00044 | [-0.00712, 0.00801] | 0.16 (4) | 0.8785 | 0.07 | 3/5 |
| val loss at end (exploratory) | 5 | 0.00324 | [-0.01083, 0.01731] | 0.64 (4) | 0.5575 | 0.29 | 1/5 |
| test Pearson r @best-val ckpt (exploratory) | 5 | -0.00008 | [-0.00629, 0.00614] | -0.03 (4) | 0.9742 | -0.02 | 3/5 |
| test Pearson r @end (exploratory) | 5 | 0.00028 | [-0.00746, 0.00803] | 0.10 (4) | 0.9240 | 0.05 | 3/5 |
| test Spearman @best-val ckpt (exploratory; regression tasks) | 5 | -0.00065 | [-0.00535, 0.00405] | -0.38 (4) | 0.7209 | -0.17 | 3/5 |
| wall-clock s (exploratory) | 5 | 6.63437 | [6.34555, 6.92318] | 63.78 (4) | 0.0000 | 28.52 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.02535 | [-0.03497, -0.01573] | 0.0019 | 0.0027 | 5/5 |
| adam | 5 | -0.01203 | [-0.01580, -0.00825] | 0.0009 | 0.0027 | 5/5 |
| rmsprop | 5 | -0.03476 | [-0.03888, -0.03063] | 0.0000 | 0.0001 | 5/5 |
| nadam | 5 | -0.01776 | [-0.02386, -0.01165] | 0.0013 | 0.0027 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00000, 95% CI [0.00000, 0.00000], p 1.0000, B25 better in 0/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 76.0 | 45.4 | 30.6 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 | n/a | 0.0 |
| b25_plain | 5 | 76.0 | 76.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000 | n/a | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.2405 | 5/5, 225 | 5/5, 225 | 5/5, 225 | both 5: B25 faster 2, slower 3; mean saving -5 units (95% CI -102 to 92); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.2145 | 5/5, 475 | 5/5, 450 | 5/5, 450 | both 5: B25 faster 2, slower 2; mean saving -5 units (95% CI -85 to 75); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.2024 | 5/5, 700 | 5/5, 675 | 5/5, 675 | both 5: B25 faster 4, slower 1; mean saving 20 units (95% CI -86 to 126); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.1940 | 2/5, 950 | 3/5, 800 | 3/5, 800 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 2, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.1925 | 2/5, 950 | 3/5, 800 | 3/5, 800 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 2, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.2405 | 5/5 / 5/5 | -5 (-102 to 92) | 2 / 3 | no | no |
| 25% | 1% | 0.2429 | 5/5 / 5/5 | 5 (-110 to 120) | 2 / 3 | no | no |
| 25% | 2% | 0.2453 | 5/5 / 5/5 | -10 (-129 to 109) | 2 / 3 | no | no |
| 25% | 5% | 0.2525 | 5/5 / 5/5 | -15 (-115 to 85) | 2 / 3 | no | no |
| 50% | 0% | 0.2145 | 5/5 / 5/5 | -5 (-85 to 75) | 2 / 2 | no | no |
| 50% | 1% | 0.2166 | 5/5 / 5/5 | -45 (-85 to -5) | 0 / 4 | no | no |
| 50% | 2% | 0.2188 | 5/5 / 5/5 | -15 (-62 to 32) | 2 / 3 | no | no |
| 50% | 5% | 0.2252 | 5/5 / 5/5 | -25 (-107 to 57) | 1 / 2 | no | yes |
| 75% | 0% | 0.2024 | 5/5 / 5/5 | 20 (-86 to 126) | 4 / 1 | no | no |
| 75% | 1% | 0.2044 | 5/5 / 5/5 | 65 (4 to 126) | 4 / 0 | YES | yes |
| 75% | 2% | 0.2065 | 5/5 / 5/5 | 10 (-125 to 145) | 3 / 2 | no | yes |
| 75% | 5% | 0.2125 | 5/5 / 5/5 | 60 (-63 to 183) | 3 / 1 | no | yes |
| 95% | 0% | 0.1940 | 2/5 / 3/5 | n/a | 1 / 0 | no | no |
| 95% | 1% | 0.1959 | 4/5 / 3/5 | 17 (-173 to 206) | 1 / 1 | no | no |
| 95% | 2% | 0.1978 | 5/5 / 3/5 | 50 (-74 to 174) | 2 / 0 | no | no |
| 95% | 5% | 0.2037 | 5/5 / 5/5 | 75 (-1 to 151) | 4 / 0 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.2405 ± 0.0096 | 0.2145 ± 0.0071 | 0.2024 ± 0.0044 | 0.1925 ± 0.0032 | 0.2245 ± 0.0023 |
| b25 | 0.2393 ± 0.0048 | 0.2141 ± 0.0054 | 0.1981 ± 0.0048 | 0.1930 ± 0.0054 | 0.2238 ± 0.0022 |
| b25_plain | 0.2393 ± 0.0048 | 0.2141 ± 0.0054 | 0.1981 ± 0.0048 | 0.1930 ± 0.0054 | 0.2238 ± 0.0022 |
| sgd | 0.2833 ± 0.0128 | 0.2409 ± 0.0101 | 0.2301 ± 0.0050 | 0.2183 ± 0.0032 | 0.2645 ± 0.0069 |
| adam | 0.2525 ± 0.0060 | 0.2233 ± 0.0047 | 0.2125 ± 0.0053 | 0.2050 ± 0.0040 | 0.2340 ± 0.0027 |
| rmsprop | 0.2670 ± 0.0088 | 0.2438 ± 0.0064 | 0.2331 ± 0.0033 | 0.2277 ± 0.0047 | 0.2573 ± 0.0046 |
| nadam | 0.2781 ± 0.0132 | 0.2335 ± 0.0071 | 0.2204 ± 0.0064 | 0.2107 ± 0.0036 | 0.2560 ± 0.0049 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.00119 | [-0.01844, 0.01607] | 0.8577 |
| at 50% of the budget | 5 | -0.00041 | [-0.00528, 0.00447] | 0.8285 |
| at 75% of the budget | 5 | -0.00428 | [-0.01047, 0.00191] | 0.1275 |
| at 100% of the budget | 5 | 0.00044 | [-0.00712, 0.00801] | 0.8785 |
| mean over the whole budget (area under the curve) | 5 | -0.00071 | [-0.00504, 0.00362] | 0.6730 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.