# Comparison: mlp_covertype

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.5685 ± 0.0033 | 0.7572 ± 0.0031 | 0.7563 ± 0.0025 | 0.7562 ± 0.0025 | 16 ± 0 | 0.010 ± 0.004 |
| adam | 5 | 0.5159 ± 0.0063 | 0.7816 ± 0.0042 | 0.7810 ± 0.0031 | 0.7779 ± 0.0020 | 16 ± 0 | 0.122 ± 0.020 |
| adamw | 5 | 0.4505 ± 0.0059 | 0.8103 ± 0.0029 | 0.8106 ± 0.0025 | 0.8084 ± 0.0034 | 16 ± 0 | 0.154 ± 0.026 |
| rmsprop | 5 | 0.4510 ± 0.0101 | 0.8091 ± 0.0052 | 0.8080 ± 0.0065 | 0.8084 ± 0.0058 | 16 ± 0 | 0.022 ± 0.006 |
| nadam | 5 | 0.4989 ± 0.0037 | 0.7905 ± 0.0041 | 0.7905 ± 0.0020 | 0.7900 ± 0.0014 | 16 ± 0 | 0.138 ± 0.010 |
| b25 | 5 | 0.4505 ± 0.0056 | 0.8107 ± 0.0053 | 0.8104 ± 0.0051 | 0.8099 ± 0.0039 | 17 ± 0 | 0.163 ± 0.024 |
| b25_plain | 5 | 0.4500 ± 0.0055 | 0.8093 ± 0.0052 | 0.8101 ± 0.0051 | 0.8092 ± 0.0040 | 17 ± 0 | 0.163 ± 0.024 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00005 | [-0.00412, 0.00422] | 0.03 (4) | 0.9755 | 0.01 | 2/5 |
| val loss at end (exploratory) | 5 | -0.00035 | [-0.01324, 0.01254] | -0.07 (4) | 0.9439 | -0.03 | 2/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00021 | [-0.00429, 0.00386] | -0.15 (4) | 0.8915 | -0.06 | 3/5 |
| test acc @end (exploratory) | 5 | 0.00156 | [-0.00426, 0.00738] | 0.74 (4) | 0.4984 | 0.33 | 2/5 |
| wall-clock s (exploratory) | 5 | 0.87495 | [0.82373, 0.92618] | 47.42 (4) | 0.0000 | 21.21 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.11795 | [-0.12776, -0.10815] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.06538 | [-0.07341, -0.05734] | 0.0000 | 0.0000 | 5/5 |
| rmsprop | 5 | -0.00050 | [-0.00846, 0.00745] | 0.8688 | 0.8688 | 2/5 |
| nadam | 5 | -0.04839 | [-0.05195, -0.04483] | 0.0000 | 0.0000 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00054, 95% CI [-0.00097, 0.00206], p 0.3739, B25 better in 0/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 75.8 | 51.4 | 24.2 | 0.2 | 0.0 | 0.0 | 0.0 | 0.000 | n/a | 0.0 |
| b25_plain | 5 | 75.8 | 75.4 | 0.0 | 0.2 | 0.2 | 0.2 | 0.0 | 0.003 | 1.000 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.5870 | 5/5, 250 | 5/5, 250 | 5/5, 250 | both 5: B25 faster 2, slower 1; mean saving 5 units (95% CI -41 to 51); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.5204 | 5/5, 525 | 5/5, 525 | 5/5, 525 | both 5: B25 faster 1, slower 1; mean saving 5 units (95% CI -72 to 82); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.4841 | 5/5, 775 | 5/5, 775 | 5/5, 775 | both 5: B25 faster 2, slower 1; mean saving -5 units (95% CI -56 to 46); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 95% | 0.4559 | 4/5, 950 | 4/5, 950 | 4/5, 938 | both 4: B25 faster 1, slower 2; mean saving -12 units (95% CI -64 to 39); only B25 0, only AdamW 0, neither 1 | saving NOT supported |
| 100% | 0.4505 | 2/5, 975 | 3/5, 950 | 4/5, 963 | both 2: B25 faster 1, slower 0; mean saving 25 units (95% CI -293 to 343); only B25 1, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.5870 | 5/5 / 5/5 | 5 (-41 to 51) | 2 / 1 | no | no |
| 25% | 1% | 0.5929 | 5/5 / 5/5 | 10 (-7 to 27) | 2 / 0 | no | no |
| 25% | 2% | 0.5988 | 5/5 / 5/5 | 5 (-9 to 19) | 1 / 0 | no | yes |
| 25% | 5% | 0.6164 | 5/5 / 5/5 | -5 (-19 to 9) | 0 / 1 | no | yes |
| 50% | 0% | 0.5204 | 5/5 / 5/5 | 5 (-72 to 82) | 1 / 1 | no | no |
| 50% | 1% | 0.5256 | 5/5 / 5/5 | -5 (-45 to 35) | 2 / 2 | no | no |
| 50% | 2% | 0.5308 | 5/5 / 5/5 | 10 (-18 to 38) | 1 / 0 | no | yes |
| 50% | 5% | 0.5464 | 5/5 / 5/5 | 0 (-44 to 44) | 1 / 1 | no | yes |
| 75% | 0% | 0.4841 | 5/5 / 5/5 | -5 (-56 to 46) | 2 / 1 | no | no |
| 75% | 1% | 0.4890 | 5/5 / 5/5 | -10 (-38 to 18) | 0 / 1 | no | yes |
| 75% | 2% | 0.4938 | 5/5 / 5/5 | 0 (-49 to 49) | 2 / 2 | no | yes |
| 75% | 5% | 0.5083 | 5/5 / 5/5 | 0 (-44 to 44) | 1 / 1 | no | yes |
| 95% | 0% | 0.4559 | 4/5 / 4/5 | -12 (-64 to 39) | 1 / 2 | no | no |
| 95% | 1% | 0.4605 | 5/5 / 5/5 | 25 (-51 to 101) | 2 / 1 | no | no |
| 95% | 2% | 0.4650 | 5/5 / 5/5 | 0 (-49 to 49) | 2 / 2 | no | yes |
| 95% | 5% | 0.4787 | 5/5 / 5/5 | -15 (-43 to 13) | 0 / 2 | no | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.5870 ± 0.0046 | 0.5204 ± 0.0064 | 0.4841 ± 0.0062 | 0.4505 ± 0.0059 | 0.5392 ± 0.0041 |
| b25 | 0.5859 ± 0.0081 | 0.5219 ± 0.0054 | 0.4847 ± 0.0029 | 0.4505 ± 0.0056 | 0.5387 ± 0.0038 |
| b25_plain | 0.5859 ± 0.0081 | 0.5237 ± 0.0051 | 0.4839 ± 0.0033 | 0.4500 ± 0.0055 | 0.5394 ± 0.0043 |
| sgd | 0.7388 ± 0.0074 | 0.6445 ± 0.0077 | 0.5952 ± 0.0036 | 0.5685 ± 0.0033 | 0.7258 ± 0.0088 |
| adam | 0.6091 ± 0.0063 | 0.5622 ± 0.0049 | 0.5352 ± 0.0052 | 0.5159 ± 0.0063 | 0.5771 ± 0.0025 |
| rmsprop | 0.5717 ± 0.0066 | 0.5201 ± 0.0091 | 0.4812 ± 0.0051 | 0.4510 ± 0.0101 | 0.5326 ± 0.0033 |
| nadam | 0.5975 ± 0.0061 | 0.5518 ± 0.0081 | 0.5201 ± 0.0028 | 0.4989 ± 0.0037 | 0.5667 ± 0.0020 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.00116 | [-0.00657, 0.00425] | 0.5835 |
| at 50% of the budget | 5 | 0.00151 | [-0.00505, 0.00806] | 0.5581 |
| at 75% of the budget | 5 | 0.00059 | [-0.00414, 0.00531] | 0.7478 |
| at 100% of the budget | 5 | 0.00005 | [-0.00412, 0.00422] | 0.9755 |
| mean over the whole budget (area under the curve) | 5 | -0.00047 | [-0.00225, 0.00132] | 0.5076 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.