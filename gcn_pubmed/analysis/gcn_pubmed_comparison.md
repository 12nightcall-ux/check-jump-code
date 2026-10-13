# Comparison: gcn_pubmed

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.8051 ± 0.0446 | 0.5876 ± 0.0497 | 0.5702 ± 0.0575 | 0.5920 ± 0.0644 | 8 ± 0 | 0.000 ± 0.001 |
| adam | 5 | 0.5428 ± 0.0018 | 0.7972 ± 0.0036 | 0.7942 ± 0.0045 | 0.7930 ± 0.0057 | 8 ± 0 | 0.000 ± 0.000 |
| adamw | 5 | 0.5585 ± 0.0083 | 0.7872 ± 0.0048 | 0.7672 ± 0.0037 | 0.7624 ± 0.0027 | 8 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 0.5625 ± 0.0106 | 0.7840 ± 0.0037 | 0.7658 ± 0.0019 | 0.7628 ± 0.0066 | 8 ± 0 | 0.000 ± 0.000 |
| nadam | 5 | 0.5430 ± 0.0013 | 0.7996 ± 0.0064 | 0.7938 ± 0.0026 | 0.7946 ± 0.0037 | 8 ± 0 | 0.000 ± 0.000 |
| b25 | 5 | 0.5546 ± 0.0095 | 0.7864 ± 0.0050 | 0.7682 ± 0.0083 | 0.7592 ± 0.0058 | 9 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 0.5559 ± 0.0122 | 0.7780 ± 0.0103 | 0.7634 ± 0.0103 | 0.7616 ± 0.0015 | 9 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.00395 | [-0.01735, 0.00944] | -0.82 (4) | 0.4585 | -0.37 | 4/5 |
| val loss at end (exploratory) | 5 | 0.30952 | [0.24249, 0.37656] | 12.82 (4) | 0.0002 | 5.73 | 0/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00100 | [-0.00580, 0.00780] | 0.41 (4) | 0.7040 | 0.18 | 3/5 |
| test acc @end (exploratory) | 5 | -0.00320 | [-0.00812, 0.00172] | -1.81 (4) | 0.1452 | -0.81 | 0/5 |
| wall-clock s (exploratory) | 5 | 0.48729 | [0.43889, 0.53569] | 27.95 (4) | 0.0000 | 12.50 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.25050 | [-0.30949, -0.19152] | 0.0003 | 0.0012 | 5/5 |
| adam | 5 | 0.01176 | [-0.00015, 0.02368] | 0.0519 | 0.1433 | 0/5 |
| rmsprop | 5 | -0.00795 | [-0.02094, 0.00503] | 0.1643 | 0.1643 | 4/5 |
| nadam | 5 | 0.01162 | [0.00018, 0.02305] | 0.0478 | 0.1433 | 0/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00128, 95% CI [-0.01855, 0.01598], p 0.8465, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 70.6 | 3.2 | 50.6 | 0.0 | 16.8 | 16.2 | 0.6 | 0.238 | 0.964 | 16.8 |
| b25_plain | 5 | 59.6 | 4.2 | 0.0 | 2.0 | 53.4 | 47.2 | 6.2 | 0.896 | 0.884 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.5903 | 5/5, 275 | 5/5, 150 | 5/5, 175 | both 5: B25 faster 5, slower 0; mean saving 115 units (95% CI 98 to 132); only B25 0, only AdamW 0, neither 0 | SAVING SUPPORTED |
| 50% | 0.5585 | 3/5, 300 | 3/5, 175 | 3/5, 175 | both 2: B25 faster 2, slower 0; mean saving 125 units (95% CI 125 to 125); only B25 1, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.5585 | 3/5, 300 | 3/5, 175 | 3/5, 175 | both 2: B25 faster 2, slower 0; mean saving 125 units (95% CI 125 to 125); only B25 1, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.5585 | 3/5, 300 | 3/5, 175 | 3/5, 175 | both 2: B25 faster 2, slower 0; mean saving 125 units (95% CI 125 to 125); only B25 1, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.5585 | 3/5, 300 | 3/5, 175 | 3/5, 175 | both 2: B25 faster 2, slower 0; mean saving 125 units (95% CI 125 to 125); only B25 1, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.5903 | 5/5 / 5/5 | 115 (98 to 132) | 5 / 0 | YES | yes |
| 25% | 1% | 0.5962 | 5/5 / 5/5 | 115 (98 to 132) | 5 / 0 | YES | yes |
| 25% | 2% | 0.6021 | 5/5 / 5/5 | 100 (100 to 100) | 5 / 0 | YES | yes |
| 25% | 5% | 0.6198 | 5/5 / 5/5 | 105 (91 to 118) | 5 / 0 | YES | yes |
| 50% | 0% | 0.5585 | 3/5 / 3/5 | 125 (125 to 125) | 2 / 0 | no | no |
| 50% | 1% | 0.5641 | 4/5 / 4/5 | 117 (81 to 152) | 3 / 0 | no | no |
| 50% | 2% | 0.5697 | 4/5 / 5/5 | 131 (111 to 151) | 4 / 0 | YES | no |
| 50% | 5% | 0.5865 | 5/5 / 5/5 | 115 (98 to 132) | 5 / 0 | YES | no |
| 75% | 0% | 0.5585 | 3/5 / 3/5 | 125 (125 to 125) | 2 / 0 | no | no |
| 75% | 1% | 0.5641 | 4/5 / 4/5 | 117 (81 to 152) | 3 / 0 | no | no |
| 75% | 2% | 0.5697 | 4/5 / 5/5 | 131 (111 to 151) | 4 / 0 | YES | no |
| 75% | 5% | 0.5865 | 5/5 / 5/5 | 115 (98 to 132) | 5 / 0 | YES | no |
| 95% | 0% | 0.5585 | 3/5 / 3/5 | 125 (125 to 125) | 2 / 0 | no | no |
| 95% | 1% | 0.5641 | 4/5 / 4/5 | 117 (81 to 152) | 3 / 0 | no | no |
| 95% | 2% | 0.5697 | 4/5 / 5/5 | 131 (111 to 151) | 4 / 0 | YES | no |
| 95% | 5% | 0.5865 | 5/5 / 5/5 | 115 (98 to 132) | 5 / 0 | YES | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.5903 ± 0.0114 | 0.5585 ± 0.0083 | 0.5585 ± 0.0083 | 0.5585 ± 0.0083 | 0.6207 ± 0.0096 |
| b25 | 0.5546 ± 0.0095 | 0.5546 ± 0.0095 | 0.5546 ± 0.0095 | 0.5546 ± 0.0095 | 0.6017 ± 0.0093 |
| b25_plain | 0.5559 ± 0.0122 | 0.5559 ± 0.0122 | 0.5559 ± 0.0122 | 0.5559 ± 0.0122 | 0.6016 ± 0.0101 |
| sgd | 1.0829 ± 0.0054 | 1.0408 ± 0.0176 | 0.9391 ± 0.0552 | 0.8051 ± 0.0446 | 1.0027 ± 0.0277 |
| adam | 0.7487 ± 0.0286 | 0.5915 ± 0.0086 | 0.5545 ± 0.0014 | 0.5428 ± 0.0018 | 0.6679 ± 0.0100 |
| rmsprop | 0.5970 ± 0.0321 | 0.5625 ± 0.0106 | 0.5625 ± 0.0106 | 0.5625 ± 0.0106 | 0.6053 ± 0.0188 |
| nadam | 0.6986 ± 0.0108 | 0.5836 ± 0.0067 | 0.5536 ± 0.0037 | 0.5430 ± 0.0013 | 0.6494 ± 0.0024 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.03568 | [-0.04448, -0.02688] | 0.0004 |
| at 50% of the budget | 5 | -0.00395 | [-0.01735, 0.00944] | 0.4585 |
| at 75% of the budget | 5 | -0.00395 | [-0.01735, 0.00944] | 0.4585 |
| at 100% of the budget | 5 | -0.00395 | [-0.01735, 0.00944] | 0.4585 |
| mean over the whole budget (area under the curve) | 5 | -0.01897 | [-0.02972, -0.00822] | 0.0081 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.