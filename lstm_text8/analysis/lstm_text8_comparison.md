# Comparison: lstm_text8

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 2.3513 ± 0.0066 | 0.3034 ± 0.0013 | 0.3050 ± 0.0021 | 0.3050 ± 0.0021 | 18 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 1.9005 ± 0.0115 | 0.4315 ± 0.0041 | 0.4264 ± 0.0030 | 0.4264 ± 0.0030 | 19 ± 0 | 0.024 ± 0.013 |
| adamw | 5 | 1.5814 ± 0.0105 | 0.5211 ± 0.0033 | 0.5126 ± 0.0021 | 0.5132 ± 0.0022 | 18 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 1.5410 ± 0.0053 | 0.5313 ± 0.0020 | 0.5218 ± 0.0011 | 0.5218 ± 0.0011 | 18 ± 0 | 0.001 ± 0.001 |
| nadam | 5 | 1.8934 ± 0.0055 | 0.4314 ± 0.0036 | 0.4263 ± 0.0032 | 0.4272 ± 0.0047 | 19 ± 0 | 0.004 ± 0.002 |
| b25 | 5 | 1.5797 ± 0.0106 | 0.5210 ± 0.0033 | 0.5127 ± 0.0041 | 0.5135 ± 0.0029 | 20 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 1.5861 ± 0.0090 | 0.5195 ± 0.0030 | 0.5120 ± 0.0035 | 0.5124 ± 0.0032 | 20 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00168 | [-0.00882, 0.00545] | -0.65 (4) | 0.5483 | -0.29 | 3/5 |
| val loss at end (exploratory) | 5 | -0.00169 | [-0.01022, 0.00684] | -0.55 (4) | 0.6112 | -0.25 | 3/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00008 | [-0.00316, 0.00333] | 0.07 (4) | 0.9467 | 0.03 | 3/5 |
| test acc @end (exploratory) | 5 | 0.00030 | [-0.00222, 0.00282] | 0.33 (4) | 0.7568 | 0.15 | 3/5 |
| wall-clock s (exploratory) | 5 | 1.33999 | [1.22024, 1.45975] | 31.07 (4) | 0.0000 | 13.89 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.77159 | [-0.78187, -0.76131] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.32082 | [-0.33065, -0.31098] | 0.0000 | 0.0000 | 5/5 |
| rmsprop | 5 | 0.03867 | [0.01936, 0.05797] | 0.0051 | 0.0051 | 0/5 |
| nadam | 5 | -0.31371 | [-0.33067, -0.29674] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00642, 95% CI [-0.00971, -0.00314], p 0.0056, B25 better in 5/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 74.2 | 66.6 | 2.0 | 2.6 | 3.0 | 2.4 | 0.6 | 0.040 | 0.800 | 3.0 |
| b25_plain | 5 | 74.0 | 67.8 | 0.0 | 2.2 | 4.0 | 3.4 | 0.6 | 0.054 | 0.850 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 1.8218 | 5/5, 275 | 5/5, 276 | 5/5, 276 | both 5: B25 faster 0, slower 5; mean saving -11 units (95% CI -28 to 6); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 1.6859 | 5/5, 525 | 5/5, 501 | 5/5, 526 | both 5: B25 faster 2, slower 3; mean saving 9 units (95% CI -8 to 26); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 1.6205 | 5/5, 775 | 5/5, 776 | 5/5, 800 | both 5: B25 faster 2, slower 3; mean saving -1 units (95% CI -32 to 30); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 1.5904 | 4/5, 950 | 4/5, 938 | 3/5, 976 | both 3: B25 faster 2, slower 1; mean saving 16 units (95% CI -79 to 111); only B25 1, only AdamW 1, neither 0 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 1.5814 | 2/5, 950 | 2/5, 913 | 1/5, 926 | both 2: B25 faster 2, slower 0; mean saving 37 units (95% CI -434 to 509); only B25 0, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 1.8218 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | no |
| 25% | 1% | 1.8400 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | no |
| 25% | 2% | 1.8582 | 5/5 / 5/5 | -16 (-33 to 1) | 0 / 5 | no | yes |
| 25% | 5% | 1.9129 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | yes |
| 50% | 0% | 1.6859 | 5/5 / 5/5 | 9 (-8 to 26) | 2 / 3 | no | no |
| 50% | 1% | 1.7028 | 5/5 / 5/5 | -6 (-20 to 8) | 0 / 5 | no | yes |
| 50% | 2% | 1.7197 | 5/5 / 5/5 | -6 (-20 to 8) | 0 / 5 | no | yes |
| 50% | 5% | 1.7702 | 5/5 / 5/5 | -6 (-20 to 8) | 0 / 5 | no | yes |
| 75% | 0% | 1.6205 | 5/5 / 5/5 | -1 (-32 to 30) | 2 / 3 | no | no |
| 75% | 1% | 1.6367 | 5/5 / 5/5 | -6 (-32 to 20) | 1 / 4 | no | yes |
| 75% | 2% | 1.6529 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | yes |
| 75% | 5% | 1.7015 | 5/5 / 5/5 | -6 (-20 to 8) | 0 / 5 | no | yes |
| 95% | 0% | 1.5904 | 4/5 / 4/5 | 16 (-79 to 111) | 2 / 1 | no | no |
| 95% | 1% | 1.6063 | 5/5 / 5/5 | 4 (-22 to 30) | 2 / 3 | no | yes |
| 95% | 2% | 1.6222 | 5/5 / 5/5 | 4 (-22 to 30) | 2 / 3 | no | yes |
| 95% | 5% | 1.6699 | 5/5 / 5/5 | -6 (-32 to 20) | 1 / 4 | no | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 1.8218 ± 0.0173 | 1.6859 ± 0.0140 | 1.6205 ± 0.0115 | 1.5814 ± 0.0105 | 1.7553 ± 0.0122 |
| b25 | 1.8629 ± 0.0141 | 1.6955 ± 0.0133 | 1.6243 ± 0.0148 | 1.5797 ± 0.0106 | 1.7831 ± 0.0118 |
| b25_plain | 1.8665 ± 0.0161 | 1.7021 ± 0.0135 | 1.6289 ± 0.0103 | 1.5861 ± 0.0090 | 1.7876 ± 0.0115 |
| sgd | 2.8318 ± 0.0063 | 2.6532 ± 0.0187 | 2.4490 ± 0.0127 | 2.3513 ± 0.0066 | 2.6531 ± 0.0094 |
| adam | 2.0265 ± 0.0106 | 1.9585 ± 0.0120 | 1.9259 ± 0.0107 | 1.9005 ± 0.0115 | 2.0040 ± 0.0110 |
| rmsprop | 1.7321 ± 0.0090 | 1.6223 ± 0.0086 | 1.5686 ± 0.0079 | 1.5410 ± 0.0053 | 1.6773 ± 0.0096 |
| nadam | 2.0311 ± 0.0073 | 1.9577 ± 0.0067 | 1.9179 ± 0.0091 | 1.8934 ± 0.0055 | 1.9996 ± 0.0053 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.04112 | [0.03468, 0.04755] | 0.0001 |
| at 50% of the budget | 5 | 0.00955 | [0.00615, 0.01295] | 0.0015 |
| at 75% of the budget | 5 | 0.00378 | [-0.00469, 0.01224] | 0.2832 |
| at 100% of the budget | 5 | -0.00168 | [-0.00882, 0.00545] | 0.5483 |
| mean over the whole budget (area under the curve) | 5 | 0.02782 | [0.02621, 0.02943] | 0.0000 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.