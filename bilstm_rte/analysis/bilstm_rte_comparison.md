# Comparison: bilstm_rte

GPU: Tesla T4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.6784 ± 0.0027 | 0.5762 ± 0.0148 | 0.4940 ± 0.0133 | 0.5293 ± 0.0206 | 845 ± 20 | 0.031 ± 0.032 |
| adam | 5 | 0.6772 ± 0.0006 | 0.5646 ± 0.0148 | 0.4851 ± 0.0111 | 0.5478 ± 0.0144 | 839 ± 0 | 0.509 ± 0.004 |
| adamw | 5 | 0.6825 ± 0.0045 | 0.5668 ± 0.0167 | 0.4916 ± 0.0232 | 0.5349 ± 0.0286 | 884 ± 25 | 0.594 ± 0.083 |
| rmsprop | 5 | 0.6777 ± 0.0016 | 0.5711 ± 0.0082 | 0.4851 ± 0.0137 | 0.5486 ± 0.0257 | 835 ± 2 | 0.519 ± 0.014 |
| nadam | 5 | 0.6778 ± 0.0019 | 0.5812 ± 0.0105 | 0.4916 ± 0.0046 | 0.5446 ± 0.0216 | 973 ± 116 | 0.545 ± 0.006 |
| b25 | 5 | 0.6828 ± 0.0019 | 0.5603 ± 0.0150 | 0.4996 ± 0.0092 | 0.5414 ± 0.0072 | 992 ± 4 | 0.608 ± 0.079 |
| b25_plain | 5 | 0.6828 ± 0.0026 | 0.5560 ± 0.0122 | 0.4940 ± 0.0316 | 0.5317 ± 0.0276 | 932 ± 48 | 0.606 ± 0.090 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00029 | [-0.00476, 0.00534] | 0.16 (4) | 0.8819 | 0.07 | 2/5 |
| val loss at end (exploratory) | 5 | 0.05947 | [-0.45871, 0.57765] | 0.32 (4) | 0.7659 | 0.14 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00803 | [-0.02895, 0.04501] | 0.60 (4) | 0.5790 | 0.27 | 3/5 |
| test acc @end (exploratory) | 5 | 0.00643 | [-0.02787, 0.04072] | 0.52 (4) | 0.6304 | 0.23 | 3/5 |
| wall-clock s (exploratory) | 5 | 108.44386 | [74.35648, 142.53123] | 8.83 (4) | 0.0009 | 3.95 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | 0.00435 | [0.00117, 0.00754] | 0.0192 | 0.0473 | 0/5 |
| adam | 5 | 0.00553 | [0.00281, 0.00825] | 0.0049 | 0.0194 | 0/5 |
| rmsprop | 5 | 0.00505 | [0.00117, 0.00892] | 0.0224 | 0.0473 | 0/5 |
| nadam | 5 | 0.00494 | [0.00154, 0.00835] | 0.0158 | 0.0473 | 0/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00007, 95% CI [-0.00310, 0.00295], p 0.9502, B25 better in 2/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 72.8 | 45.2 | 18.0 | 2.0 | 7.6 | 2.2 | 5.4 | 0.104 | 0.289 | 7.6 |
| b25_plain | 5 | 72.6 | 61.4 | 0.0 | 2.8 | 8.4 | 2.4 | 6.0 | 0.116 | 0.286 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6825 | 1/5, 175 | 2/5, 176 | 2/5, 150 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.6825 | 1/5, 175 | 2/5, 176 | 2/5, 150 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.6825 | 1/5, 175 | 2/5, 176 | 2/5, 150 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.6825 | 1/5, 175 | 2/5, 176 | 2/5, 150 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.6825 | 1/5, 175 | 2/5, 176 | 2/5, 150 | both 1: B25 faster 0, slower 1; mean saving n/a units; only B25 1, only AdamW 0, neither 3 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6825 | 1/5 / 2/5 | n/a | 0 / 1 | no | no |
| 25% | 1% | 0.6893 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | no |
| 25% | 2% | 0.6961 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 25% | 5% | 0.7166 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 50% | 0% | 0.6825 | 1/5 / 2/5 | n/a | 0 / 1 | no | no |
| 50% | 1% | 0.6893 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | no |
| 50% | 2% | 0.6961 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 50% | 5% | 0.7166 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 75% | 0% | 0.6825 | 1/5 / 2/5 | n/a | 0 / 1 | no | no |
| 75% | 1% | 0.6893 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | no |
| 75% | 2% | 0.6961 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 75% | 5% | 0.7166 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 95% | 0% | 0.6825 | 1/5 / 2/5 | n/a | 0 / 1 | no | no |
| 95% | 1% | 0.6893 | 5/5 / 5/5 | -11 (-28 to 6) | 0 / 5 | no | no |
| 95% | 2% | 0.6961 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |
| 95% | 5% | 0.7166 | 5/5 / 5/5 | 0 (0 to 0) | 0 / 0 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6825 ± 0.0045 | 0.6825 ± 0.0045 | 0.6825 ± 0.0045 | 0.6825 ± 0.0045 | 0.6834 ± 0.0037 |
| b25 | 0.6828 ± 0.0019 | 0.6828 ± 0.0019 | 0.6828 ± 0.0019 | 0.6828 ± 0.0019 | 0.6840 ± 0.0015 |
| b25_plain | 0.6828 ± 0.0026 | 0.6828 ± 0.0026 | 0.6828 ± 0.0026 | 0.6828 ± 0.0026 | 0.6840 ± 0.0021 |
| sgd | 0.6914 ± 0.0002 | 0.6889 ± 0.0015 | 0.6831 ± 0.0032 | 0.6784 ± 0.0027 | 0.6869 ± 0.0016 |
| adam | 0.6860 ± 0.0009 | 0.6772 ± 0.0006 | 0.6772 ± 0.0006 | 0.6772 ± 0.0006 | 0.6810 ± 0.0005 |
| rmsprop | 0.6820 ± 0.0025 | 0.6777 ± 0.0016 | 0.6777 ± 0.0016 | 0.6777 ± 0.0016 | 0.6805 ± 0.0014 |
| nadam | 0.6846 ± 0.0014 | 0.6778 ± 0.0019 | 0.6778 ± 0.0019 | 0.6778 ± 0.0019 | 0.6810 ± 0.0015 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00029 | [-0.00476, 0.00534] | 0.8819 |
| at 50% of the budget | 5 | 0.00029 | [-0.00476, 0.00534] | 0.8819 |
| at 75% of the budget | 5 | 0.00029 | [-0.00476, 0.00534] | 0.8819 |
| at 100% of the budget | 5 | 0.00029 | [-0.00476, 0.00534] | 0.8819 |
| mean over the whole budget (area under the curve) | 5 | 0.00057 | [-0.00378, 0.00492] | 0.7341 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.