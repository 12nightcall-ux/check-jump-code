# Comparison: bilstm_cola

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | val MCC @best-val ckpt | test MCC @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.6179 ± 0.0000 | 0.6913 ± 0.0000 | 0.6842 ± 0.0000 | 0.6842 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 313 ± 1 | 0.000 ± 0.000 |
| adam | 5 | 0.6119 ± 0.0006 | 0.6913 ± 0.0000 | 0.6840 ± 0.0005 | 0.6819 ± 0.0047 | 0.0000 ± 0.0000 | 0.0004 ± 0.0010 | 285 ± 1 | 0.013 ± 0.005 |
| adamw | 5 | 0.6119 ± 0.0006 | 0.6913 ± 0.0000 | 0.6842 ± 0.0000 | 0.6819 ± 0.0046 | 0.0000 ± 0.0000 | 0.0054 ± 0.0122 | 326 ± 1 | 0.013 ± 0.005 |
| rmsprop | 5 | 0.6122 ± 0.0008 | 0.6913 ± 0.0000 | 0.6840 ± 0.0005 | 0.6842 ± 0.0014 | 0.0000 ± 0.0000 | -0.0046 ± 0.0104 | 280 ± 1 | 0.016 ± 0.005 |
| nadam | 5 | 0.6119 ± 0.0009 | 0.6913 ± 0.0000 | 0.6835 ± 0.0016 | 0.6828 ± 0.0019 | 0.0000 ± 0.0000 | -0.0042 ± 0.0218 | 289 ± 1 | 0.018 ± 0.005 |
| b25 | 5 | 0.6126 ± 0.0012 | 0.6911 ± 0.0004 | 0.6842 ± 0.0008 | 0.6837 ± 0.0006 | -0.0041 ± 0.0093 | 0.0043 ± 0.0248 | 372 ± 5 | 0.010 ± 0.005 |
| b25_plain | 5 | 0.6128 ± 0.0019 | 0.6909 ± 0.0009 | 0.6837 ± 0.0006 | 0.6826 ± 0.0030 | -0.0059 ± 0.0131 | -0.0042 ± 0.0107 | 345 ± 1 | 0.014 ± 0.006 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00070 | [-0.00094, 0.00235] | 1.19 (4) | 0.3005 | 0.53 | 2/5 |
| val loss at end (exploratory) | 5 | -0.00722 | [-0.02120, 0.00676] | -1.43 (4) | 0.2250 | -0.64 | 4/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00000 | [-0.00103, 0.00103] | 0.00 (4) | 1.0000 | 0.00 | 1/5 |
| test acc @end (exploratory) | 5 | 0.00187 | [-0.00417, 0.00791] | 0.86 (4) | 0.4382 | 0.38 | 1/5 |
| test MCC @best-val ckpt (exploratory; CoLA) | 5 | -0.00113 | [-0.03667, 0.03442] | -0.09 (4) | 0.9342 | -0.04 | 1/5 |
| wall-clock s (exploratory) | 5 | 46.04401 | [40.58576, 51.50225] | 23.42 (4) | 0.0000 | 10.47 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.00530 | [-0.00671, -0.00388] | 0.0005 | 0.0020 | 5/5 |
| adam | 5 | 0.00070 | [-0.00094, 0.00234] | 0.3037 | 0.9111 | 2/5 |
| rmsprop | 5 | 0.00046 | [-0.00138, 0.00231] | 0.5241 | 0.9111 | 2/5 |
| nadam | 5 | 0.00070 | [-0.00119, 0.00259] | 0.3640 | 0.9111 | 1/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00021, 95% CI [-0.00269, 0.00226], p 0.8211, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 70.4 | 25.6 | 26.8 | 0.6 | 17.4 | 8.4 | 9.0 | 0.247 | 0.483 | 17.4 |
| b25_plain | 5 | 67.0 | 36.8 | 0.0 | 0.4 | 29.8 | 12.8 | 17.0 | 0.445 | 0.430 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6169 | 5/5, 275 | 5/5, 226 | 5/5, 250 | both 5: B25 faster 2, slower 3; mean saving 5 units (95% CI -90 to 99); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.6160 | 5/5, 500 | 5/5, 451 | 5/5, 525 | both 5: B25 faster 2, slower 3; mean saving -81 units (95% CI -320 to 159); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.6143 | 5/5, 875 | 5/5, 726 | 4/5, 838 | both 5: B25 faster 3, slower 2; mean saving 79 units (95% CI -141 to 300); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.6120 | 3/5, 900 | 2/5, 938 | 1/5, 825 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.6119 | 3/5, 900 | 2/5, 938 | 1/5, 825 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6169 | 5/5 / 5/5 | 5 (-90 to 99) | 2 / 3 | no | no |
| 25% | 1% | 0.6231 | 5/5 / 5/5 | -11 (-27 to 5) | 0 / 5 | no | yes |
| 25% | 2% | 0.6292 | 5/5 / 5/5 | -2 (-3 to -1) | 0 / 5 | no | yes |
| 25% | 5% | 0.6477 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | yes |
| 50% | 0% | 0.6160 | 5/5 / 5/5 | -81 (-320 to 159) | 2 / 3 | no | no |
| 50% | 1% | 0.6221 | 5/5 / 5/5 | -11 (-27 to 5) | 0 / 5 | no | yes |
| 50% | 2% | 0.6283 | 5/5 / 5/5 | -2 (-3 to -1) | 0 / 5 | no | yes |
| 50% | 5% | 0.6468 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | yes |
| 75% | 0% | 0.6143 | 5/5 / 5/5 | 79 (-141 to 300) | 3 / 2 | no | no |
| 75% | 1% | 0.6205 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | yes |
| 75% | 2% | 0.6266 | 5/5 / 5/5 | -7 (-20 to 7) | 0 / 5 | no | yes |
| 75% | 5% | 0.6450 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | yes |
| 95% | 0% | 0.6120 | 3/5 / 2/5 | n/a | 0 / 1 | no | no |
| 95% | 1% | 0.6181 | 5/5 / 5/5 | 4 (-51 to 60) | 2 / 3 | no | no |
| 95% | 2% | 0.6242 | 5/5 / 5/5 | -11 (-27 to 5) | 0 / 5 | no | no |
| 95% | 5% | 0.6426 | 5/5 / 5/5 | -0 (-2 to 1) | 0 / 1 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6169 ± 0.0001 | 0.6160 ± 0.0004 | 0.6143 ± 0.0010 | 0.6119 ± 0.0006 | 0.6162 ± 0.0005 |
| b25 | 0.6169 ± 0.0004 | 0.6159 ± 0.0004 | 0.6149 ± 0.0012 | 0.6126 ± 0.0012 | 0.6171 ± 0.0005 |
| b25_plain | 0.6170 ± 0.0005 | 0.6160 ± 0.0005 | 0.6142 ± 0.0011 | 0.6128 ± 0.0019 | 0.6169 ± 0.0006 |
| sgd | 0.6191 ± 0.0004 | 0.6179 ± 0.0000 | 0.6179 ± 0.0000 | 0.6179 ± 0.0000 | 0.6227 ± 0.0004 |
| adam | 0.6169 ± 0.0001 | 0.6160 ± 0.0004 | 0.6143 ± 0.0010 | 0.6119 ± 0.0006 | 0.6162 ± 0.0005 |
| rmsprop | 0.6165 ± 0.0002 | 0.6156 ± 0.0003 | 0.6138 ± 0.0011 | 0.6122 ± 0.0008 | 0.6154 ± 0.0004 |
| nadam | 0.6168 ± 0.0001 | 0.6157 ± 0.0002 | 0.6137 ± 0.0011 | 0.6119 ± 0.0009 | 0.6160 ± 0.0005 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00003 | [-0.00053, 0.00058] | 0.9047 |
| at 50% of the budget | 5 | -0.00006 | [-0.00062, 0.00050] | 0.7850 |
| at 75% of the budget | 5 | 0.00060 | [-0.00109, 0.00230] | 0.3793 |
| at 100% of the budget | 5 | 0.00070 | [-0.00094, 0.00235] | 0.3005 |
| mean over the whole budget (area under the curve) | 5 | 0.00087 | [0.00019, 0.00156] | 0.0241 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.