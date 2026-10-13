# Comparison: mbert_sst2

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.3216 ± 0.0202 | 0.8688 ± 0.0133 | 0.8252 ± 0.0108 | 0.8096 ± 0.0261 | 619 ± 1 | 1.000 ± 0.000 |
| adam | 5 | 0.2306 ± 0.0043 | 0.9062 ± 0.0058 | 0.8782 ± 0.0053 | 0.8670 ± 0.0122 | 670 ± 1 | 0.999 ± 0.001 |
| adamw | 5 | 0.2322 ± 0.0032 | 0.9068 ± 0.0061 | 0.8748 ± 0.0059 | 0.8679 ± 0.0114 | 678 ± 1 | 0.999 ± 0.001 |
| rmsprop | 5 | 0.2467 ± 0.0291 | 0.9038 ± 0.0145 | 0.8631 ± 0.0273 | 0.8610 ± 0.0264 | 652 ± 3 | 0.990 ± 0.006 |
| nadam | 5 | 0.2360 ± 0.0040 | 0.9070 ± 0.0028 | 0.8807 ± 0.0061 | 0.8775 ± 0.0045 | 682 ± 0 | 1.000 ± 0.000 |
| b25 | 5 | 0.2356 ± 0.0089 | 0.9064 ± 0.0066 | 0.8764 ± 0.0117 | 0.8761 ± 0.0056 | 742 ± 11 | 1.000 ± 0.000 |
| b25_plain | 5 | 0.2354 ± 0.0052 | 0.9048 ± 0.0042 | 0.8704 ± 0.0071 | 0.8697 ± 0.0098 | 721 ± 4 | 1.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00336 | [-0.00637, 0.01309] | 0.96 (4) | 0.3917 | 0.43 | 1/5 |
| val loss at end (exploratory) | 5 | 0.00443 | [-0.02845, 0.03730] | 0.37 (4) | 0.7274 | 0.17 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00161 | [-0.01547, 0.01868] | 0.26 (4) | 0.8069 | 0.12 | 3/5 |
| test acc @end (exploratory) | 5 | 0.00826 | [-0.01181, 0.02833] | 1.14 (4) | 0.3171 | 0.51 | 2/5 |
| wall-clock s (exploratory) | 5 | 64.08155 | [50.74349, 77.41960] | 13.34 (4) | 0.0002 | 5.97 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.08601 | [-0.11519, -0.05683] | 0.0012 | 0.0049 | 5/5 |
| adam | 5 | 0.00497 | [-0.00588, 0.01581] | 0.2725 | 0.8175 | 1/5 |
| rmsprop | 5 | -0.01111 | [-0.04503, 0.02282] | 0.4148 | 0.8295 | 3/5 |
| nadam | 5 | -0.00040 | [-0.01514, 0.01435] | 0.9442 | 0.9442 | 2/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00013, 95% CI [-0.01097, 0.01122], p 0.9765, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 72.4 | 36.4 | 25.6 | 1.8 | 8.6 | 1.8 | 6.8 | 0.119 | 0.209 | 8.6 |
| b25_plain | 5 | 71.0 | 53.6 | 0.0 | 3.8 | 13.6 | 3.8 | 9.8 | 0.192 | 0.279 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.3449 | 5/5, 275 | 5/5, 276 | 5/5, 251 | both 5: B25 faster 0, slower 5; mean saving -31 units (95% CI -65 to 3); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.2832 | 5/5, 475 | 5/5, 600 | 5/5, 476 | both 5: B25 faster 0, slower 5; mean saving -75 units (95% CI -145 to -6); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.2501 | 5/5, 800 | 5/5, 825 | 5/5, 826 | both 5: B25 faster 1, slower 4; mean saving -70 units (95% CI -189 to 49); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.2349 | 4/5, 950 | 2/5, 900 | 3/5, 926 | both 2: B25 faster 1, slower 1; mean saving 37 units (95% CI -756 to 831); only B25 0, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.2322 | 2/5, 950 | 1/5, 925 | 2/5, 925 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 0, only AdamW 1, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.3449 | 5/5 / 5/5 | -31 (-65 to 3) | 0 / 5 | no | no |
| 25% | 1% | 0.3484 | 5/5 / 5/5 | -16 (-63 to 31) | 1 / 4 | no | no |
| 25% | 2% | 0.3518 | 5/5 / 5/5 | -26 (-83 to 32) | 1 / 4 | no | no |
| 25% | 5% | 0.3622 | 5/5 / 5/5 | -31 (-98 to 37) | 1 / 4 | no | no |
| 50% | 0% | 0.2832 | 5/5 / 5/5 | -75 (-145 to -6) | 0 / 5 | no | no |
| 50% | 1% | 0.2860 | 5/5 / 5/5 | -15 (-107 to 77) | 2 / 3 | no | no |
| 50% | 2% | 0.2889 | 5/5 / 5/5 | -40 (-140 to 59) | 1 / 4 | no | no |
| 50% | 5% | 0.2974 | 5/5 / 5/5 | -20 (-88 to 47) | 1 / 4 | no | no |
| 75% | 0% | 0.2501 | 5/5 / 5/5 | -70 (-189 to 49) | 1 / 4 | no | no |
| 75% | 1% | 0.2526 | 5/5 / 5/5 | -80 (-228 to 68) | 2 / 3 | no | no |
| 75% | 2% | 0.2552 | 5/5 / 5/5 | -51 (-204 to 103) | 2 / 3 | no | no |
| 75% | 5% | 0.2627 | 5/5 / 5/5 | -45 (-188 to 97) | 1 / 4 | no | no |
| 95% | 0% | 0.2349 | 4/5 / 2/5 | 37 (-756 to 831) | 1 / 1 | no | no |
| 95% | 1% | 0.2372 | 5/5 / 3/5 | 42 (-193 to 276) | 1 / 2 | no | no |
| 95% | 2% | 0.2396 | 5/5 / 3/5 | 17 (-183 to 216) | 2 / 1 | no | no |
| 95% | 5% | 0.2466 | 5/5 / 5/5 | -70 (-203 to 62) | 2 / 3 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.3449 ± 0.0080 | 0.2832 ± 0.0035 | 0.2501 ± 0.0062 | 0.2322 ± 0.0032 | 0.3085 ± 0.0067 |
| b25 | 0.3614 ± 0.0215 | 0.2919 ± 0.0106 | 0.2611 ± 0.0084 | 0.2356 ± 0.0089 | 0.3229 ± 0.0131 |
| b25_plain | 0.3481 ± 0.0102 | 0.2836 ± 0.0070 | 0.2603 ± 0.0051 | 0.2354 ± 0.0052 | 0.3175 ± 0.0052 |
| sgd | 0.4518 ± 0.0251 | 0.4025 ± 0.0346 | 0.3583 ± 0.0185 | 0.3216 ± 0.0202 | 0.4174 ± 0.0196 |
| adam | 0.3483 ± 0.0118 | 0.2857 ± 0.0048 | 0.2465 ± 0.0067 | 0.2306 ± 0.0043 | 0.3078 ± 0.0056 |
| rmsprop | 0.3724 ± 0.0514 | 0.2949 ± 0.0369 | 0.2719 ± 0.0300 | 0.2467 ± 0.0291 | 0.3372 ± 0.0364 |
| nadam | 0.3690 ± 0.0068 | 0.2986 ± 0.0030 | 0.2589 ± 0.0047 | 0.2360 ± 0.0040 | 0.3233 ± 0.0046 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.01645 | [-0.00967, 0.04257] | 0.1552 |
| at 50% of the budget | 5 | 0.00869 | [-0.00515, 0.02254] | 0.1563 |
| at 75% of the budget | 5 | 0.01100 | [-0.00290, 0.02489] | 0.0930 |
| at 100% of the budget | 5 | 0.00336 | [-0.00637, 0.01309] | 0.3917 |
| mean over the whole budget (area under the curve) | 5 | 0.01438 | [0.00530, 0.02345] | 0.0117 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.