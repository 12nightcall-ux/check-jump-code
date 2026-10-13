# Comparison: bilstm_stsb

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

Regression task (stsb): the 'loss' columns are mean squared error, and the columns named 'Pearson r' are Pearson correlations (higher is better).

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val Pearson r | test Pearson r @best-val ckpt | test Pearson r @end | val Spearman @best-val ckpt | test Spearman @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.8992 ± 0.0081 | 0.7776 ± 0.0019 | 0.7721 ± 0.0040 | 0.7740 ± 0.0036 | 0.7826 ± 0.0024 | 0.7351 ± 0.0037 | 736 ± 1 | 0.941 ± 0.037 |
| adam | 5 | 1.2334 ± 0.0204 | 0.6734 ± 0.0074 | 0.6896 ± 0.0088 | 0.6886 ± 0.0063 | 0.6751 ± 0.0107 | 0.6321 ± 0.0124 | 784 ± 8 | 0.996 ± 0.002 |
| adamw | 5 | 0.9656 ± 0.0096 | 0.7570 ± 0.0033 | 0.7659 ± 0.0054 | 0.7670 ± 0.0020 | 0.7605 ± 0.0030 | 0.7276 ± 0.0094 | 746 ± 2 | 0.995 ± 0.002 |
| rmsprop | 5 | 1.4590 ± 0.0587 | 0.6030 ± 0.0162 | 0.6307 ± 0.0148 | 0.6310 ± 0.0114 | 0.6082 ± 0.0164 | 0.5752 ± 0.0184 | 779 ± 1 | 0.996 ± 0.002 |
| nadam | 5 | 1.2724 ± 0.0271 | 0.6621 ± 0.0103 | 0.6765 ± 0.0136 | 0.6847 ± 0.0085 | 0.6636 ± 0.0131 | 0.6175 ± 0.0131 | 792 ± 2 | 0.996 ± 0.002 |
| b25 | 5 | 0.9713 ± 0.0154 | 0.7563 ± 0.0056 | 0.7634 ± 0.0066 | 0.7656 ± 0.0055 | 0.7603 ± 0.0074 | 0.7269 ± 0.0072 | 781 ± 2 | 0.996 ± 0.001 |
| b25_plain | 5 | 0.9700 ± 0.0371 | 0.7575 ± 0.0114 | 0.7670 ± 0.0054 | 0.7693 ± 0.0032 | 0.7614 ± 0.0127 | 0.7300 ± 0.0082 | 775 ± 1 | 0.996 ± 0.002 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00570 | [-0.01357, 0.02498] | 0.82 (4) | 0.4575 | 0.37 | 1/5 |
| val loss at end (exploratory) | 5 | 0.02105 | [-0.08538, 0.12747] | 0.55 (4) | 0.6122 | 0.25 | 1/5 |
| test Pearson r @best-val ckpt (exploratory) | 5 | -0.00242 | [-0.01293, 0.00810] | -0.64 (4) | 0.5581 | -0.29 | 2/5 |
| test Pearson r @end (exploratory) | 5 | -0.00136 | [-0.00906, 0.00634] | -0.49 (4) | 0.6491 | -0.22 | 3/5 |
| test Spearman @best-val ckpt (exploratory; regression tasks) | 5 | -0.00069 | [-0.01607, 0.01469] | -0.13 (4) | 0.9065 | -0.06 | 2/5 |
| wall-clock s (exploratory) | 5 | 34.72311 | [31.01696, 38.42926] | 26.01 (4) | 0.0000 | 11.63 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | 0.07215 | [0.04623, 0.09807] | 0.0015 | 0.0015 | 0/5 |
| adam | 5 | -0.26202 | [-0.29166, -0.23238] | 0.0000 | 0.0000 | 5/5 |
| rmsprop | 5 | -0.48762 | [-0.54651, -0.42874] | 0.0000 | 0.0000 | 5/5 |
| nadam | 5 | -0.30108 | [-0.32121, -0.28094] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00135, 95% CI [-0.02830, 0.03099], p 0.9056, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 74.0 | 40.8 | 28.2 | 0.8 | 4.2 | 1.4 | 2.8 | 0.057 | 0.333 | 4.2 |
| b25_plain | 5 | 72.4 | 61.6 | 0.0 | 3.6 | 7.2 | 1.6 | 5.6 | 0.099 | 0.222 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 1.3669 | 5/5, 225 | 5/5, 276 | 5/5, 276 | both 5: B25 faster 1, slower 4; mean saving -16 units (95% CI -57 to 26); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 1.0829 | 5/5, 550 | 5/5, 500 | 5/5, 550 | both 5: B25 faster 3, slower 2; mean saving 39 units (95% CI -140 to 219); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 1.0025 | 5/5, 800 | 5/5, 826 | 4/5, 638 | both 5: B25 faster 0, slower 5; mean saving -20 units (95% CI -60 to 19); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.9711 | 4/5, 912 | 2/5, 863 | 3/5, 825 | both 2: B25 faster 0, slower 2; mean saving -26 units (95% CI -26 to -26); only B25 0, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.9656 | 2/5, 950 | 2/5, 926 | 3/5, 901 | both 1: B25 faster 1, slower 0; mean saving n/a units; only B25 1, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 1.3669 | 5/5 / 5/5 | -16 (-57 to 26) | 1 / 4 | no | no |
| 25% | 1% | 1.3805 | 5/5 / 5/5 | -16 (-57 to 26) | 1 / 4 | no | no |
| 25% | 2% | 1.3942 | 5/5 / 5/5 | -11 (-58 to 36) | 2 / 3 | no | no |
| 25% | 5% | 1.4352 | 5/5 / 5/5 | 9 (-8 to 26) | 2 / 3 | no | no |
| 50% | 0% | 1.0829 | 5/5 / 5/5 | 39 (-140 to 219) | 3 / 2 | no | no |
| 50% | 1% | 1.0937 | 5/5 / 5/5 | -1 (-125 to 124) | 2 / 3 | no | no |
| 50% | 2% | 1.1046 | 5/5 / 5/5 | 9 (-139 to 158) | 2 / 3 | no | no |
| 50% | 5% | 1.1370 | 5/5 / 5/5 | 9 (-74 to 93) | 3 / 2 | no | yes |
| 75% | 0% | 1.0025 | 5/5 / 5/5 | -20 (-60 to 19) | 0 / 5 | no | no |
| 75% | 1% | 1.0125 | 5/5 / 5/5 | 14 (-120 to 149) | 1 / 4 | no | no |
| 75% | 2% | 1.0226 | 5/5 / 5/5 | 9 (-90 to 109) | 2 / 3 | no | no |
| 75% | 5% | 1.0526 | 5/5 / 5/5 | -16 (-135 to 104) | 2 / 3 | no | yes |
| 95% | 0% | 0.9711 | 4/5 / 2/5 | -26 (-26 to -26) | 0 / 2 | no | no |
| 95% | 1% | 0.9808 | 5/5 / 3/5 | -67 (-162 to 28) | 0 / 3 | no | no |
| 95% | 2% | 0.9906 | 5/5 / 5/5 | -65 (-137 to 6) | 1 / 4 | no | no |
| 95% | 5% | 1.0197 | 5/5 / 5/5 | -11 (-119 to 98) | 1 / 4 | no | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 1.3669 ± 0.0901 | 1.0829 ± 0.0253 | 1.0025 ± 0.0187 | 0.9656 ± 0.0096 | 1.2838 ± 0.0433 |
| b25 | 1.4004 ± 0.0529 | 1.1095 ± 0.0384 | 1.0071 ± 0.0212 | 0.9713 ± 0.0154 | 1.3118 ± 0.0482 |
| b25_plain | 1.4460 ± 0.1586 | 1.1248 ± 0.0519 | 0.9996 ± 0.0300 | 0.9700 ± 0.0371 | 1.3185 ± 0.0501 |
| sgd | 2.1393 ± 0.0667 | 1.1717 ± 0.1449 | 0.9453 ± 0.0262 | 0.8992 ± 0.0081 | 1.5611 ± 0.0649 |
| adam | 1.6935 ± 0.0806 | 1.4174 ± 0.0875 | 1.3024 ± 0.0387 | 1.2334 ± 0.0204 | 1.5628 ± 0.0457 |
| rmsprop | 2.3164 ± 0.0494 | 2.0755 ± 0.0633 | 1.6852 ± 0.0727 | 1.4590 ± 0.0587 | 2.0085 ± 0.0560 |
| nadam | 1.7623 ± 0.0918 | 1.4210 ± 0.0280 | 1.3373 ± 0.0251 | 1.2724 ± 0.0271 | 1.5927 ± 0.0110 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.03354 | [-0.03553, 0.10261] | 0.2489 |
| at 50% of the budget | 5 | 0.02660 | [-0.04017, 0.09338] | 0.3307 |
| at 75% of the budget | 5 | 0.00453 | [0.00020, 0.00886] | 0.0438 |
| at 100% of the budget | 5 | 0.00570 | [-0.01357, 0.02498] | 0.4575 |
| mean over the whole budget (area under the curve) | 5 | 0.02807 | [0.00561, 0.05052] | 0.0256 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.