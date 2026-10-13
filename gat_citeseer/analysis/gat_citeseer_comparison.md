# Comparison: gat_citeseer

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 1.6986 ± 0.0150 | 0.4812 ± 0.1394 | 0.4792 ± 0.1351 | 0.4836 ± 0.1359 | 11 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 0.9844 ± 0.0125 | 0.7020 ± 0.0071 | 0.6992 ± 0.0104 | 0.6868 ± 0.0152 | 11 ± 0 | 0.000 ± 0.000 |
| adamw | 5 | 0.9767 ± 0.0286 | 0.6860 ± 0.0097 | 0.6738 ± 0.0109 | 0.4992 ± 0.0264 | 12 ± 0 | 0.164 ± 0.048 |
| rmsprop | 5 | 1.0279 ± 0.0234 | 0.6632 ± 0.0070 | 0.6512 ± 0.0143 | 0.6452 ± 0.0108 | 11 ± 0 | 0.000 ± 0.000 |
| nadam | 5 | 0.9947 ± 0.0152 | 0.6980 ± 0.0071 | 0.6972 ± 0.0119 | 0.6878 ± 0.0130 | 11 ± 0 | 0.000 ± 0.000 |
| b25 | 5 | 0.9953 ± 0.0262 | 0.6800 ± 0.0216 | 0.6646 ± 0.0127 | 0.6350 ± 0.0130 | 13 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 1.0065 ± 0.0300 | 0.6736 ± 0.0107 | 0.6694 ± 0.0178 | 0.6412 ± 0.0172 | 13 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.01850 | [-0.01476, 0.05177] | 1.54 (4) | 0.1974 | 0.69 | 2/5 |
| val loss at end (exploratory) | 5 | -143.88173 | [-320.07336, 32.30989] | -2.27 (4) | 0.0860 | -1.01 | 5/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00920 | [-0.02542, 0.00702] | -1.57 (4) | 0.1905 | -0.70 | 1/5 |
| test acc @end (exploratory) | 5 | 0.13580 | [0.09229, 0.17931] | 8.67 (4) | 0.0010 | 3.88 | 5/5 |
| wall-clock s (exploratory) | 5 | 1.11630 | [0.97426, 1.25835] | 21.82 (4) | 0.0000 | 9.76 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.70332 | [-0.72569, -0.68095] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | 0.01087 | [-0.02342, 0.04516] | 0.4285 | 0.8570 | 2/5 |
| rmsprop | 5 | -0.03260 | [-0.07077, 0.00557] | 0.0767 | 0.2301 | 4/5 |
| nadam | 5 | 0.00052 | [-0.03300, 0.03404] | 0.9678 | 0.9678 | 2/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.01121, 95% CI [-0.03219, 0.00977], p 0.2120, B25 better in 4/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 70.4 | 0.8 | 51.0 | 6.2 | 12.4 | 11.0 | 1.4 | 0.176 | 0.887 | 12.4 |
| b25_plain | 5 | 60.0 | 1.2 | 0.0 | 21.2 | 37.6 | 35.6 | 2.0 | 0.627 | 0.947 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.9767 | 2/5, 175 | 1/5, 101 | 1/5, 101 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.9767 | 2/5, 175 | 1/5, 101 | 1/5, 101 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.9767 | 2/5, 175 | 1/5, 101 | 1/5, 101 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.9767 | 2/5, 175 | 1/5, 101 | 1/5, 101 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.9767 | 2/5, 175 | 1/5, 101 | 1/5, 101 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.9767 | 2/5 / 1/5 | n/a | 1 / 0 | no | no |
| 25% | 1% | 0.9865 | 3/5 / 1/5 | n/a | 1 / 0 | no | no |
| 25% | 2% | 0.9963 | 4/5 / 2/5 | 62 (-97 to 221) | 2 / 0 | no | no |
| 25% | 5% | 1.0256 | 5/5 / 4/5 | 56 (36 to 76) | 4 / 0 | YES | no |
| 50% | 0% | 0.9767 | 2/5 / 1/5 | n/a | 1 / 0 | no | no |
| 50% | 1% | 0.9865 | 3/5 / 1/5 | n/a | 1 / 0 | no | no |
| 50% | 2% | 0.9963 | 4/5 / 2/5 | 62 (-97 to 221) | 2 / 0 | no | no |
| 50% | 5% | 1.0256 | 5/5 / 4/5 | 56 (36 to 76) | 4 / 0 | YES | no |
| 75% | 0% | 0.9767 | 2/5 / 1/5 | n/a | 1 / 0 | no | no |
| 75% | 1% | 0.9865 | 3/5 / 1/5 | n/a | 1 / 0 | no | no |
| 75% | 2% | 0.9963 | 4/5 / 2/5 | 62 (-97 to 221) | 2 / 0 | no | no |
| 75% | 5% | 1.0256 | 5/5 / 4/5 | 56 (36 to 76) | 4 / 0 | YES | no |
| 95% | 0% | 0.9767 | 2/5 / 1/5 | n/a | 1 / 0 | no | no |
| 95% | 1% | 0.9865 | 3/5 / 1/5 | n/a | 1 / 0 | no | no |
| 95% | 2% | 0.9963 | 4/5 / 2/5 | 62 (-97 to 221) | 2 / 0 | no | no |
| 95% | 5% | 1.0256 | 5/5 / 4/5 | 56 (36 to 76) | 4 / 0 | YES | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.9767 ± 0.0286 | 0.9767 ± 0.0286 | 0.9767 ± 0.0286 | 0.9767 ± 0.0286 | 1.0360 ± 0.0247 |
| b25 | 0.9953 ± 0.0262 | 0.9953 ± 0.0262 | 0.9953 ± 0.0262 | 0.9953 ± 0.0262 | 1.0448 ± 0.0232 |
| b25_plain | 1.0065 ± 0.0300 | 1.0065 ± 0.0300 | 1.0065 ± 0.0300 | 1.0065 ± 0.0300 | 1.0515 ± 0.0278 |
| sgd | 1.7780 ± 0.0068 | 1.7659 ± 0.0034 | 1.7445 ± 0.0072 | 1.6986 ± 0.0150 | 1.7591 ± 0.0031 |
| adam | 1.1144 ± 0.0104 | 0.9844 ± 0.0125 | 0.9844 ± 0.0125 | 0.9844 ± 0.0125 | 1.0998 ± 0.0111 |
| rmsprop | 1.0279 ± 0.0234 | 1.0279 ± 0.0234 | 1.0279 ± 0.0234 | 1.0279 ± 0.0234 | 1.0362 ± 0.0225 |
| nadam | 1.1211 ± 0.0154 | 0.9955 ± 0.0152 | 0.9947 ± 0.0152 | 0.9947 ± 0.0152 | 1.1090 ± 0.0133 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.01850 | [-0.01476, 0.05177] | 0.1974 |
| at 50% of the budget | 5 | 0.01850 | [-0.01476, 0.05177] | 0.1974 |
| at 75% of the budget | 5 | 0.01850 | [-0.01476, 0.05177] | 0.1974 |
| at 100% of the budget | 5 | 0.01850 | [-0.01476, 0.05177] | 0.1974 |
| mean over the whole budget (area under the curve) | 5 | 0.00876 | [-0.01719, 0.03471] | 0.4016 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.