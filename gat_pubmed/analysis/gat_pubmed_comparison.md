# Comparison: gat_pubmed

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.6289 ± 0.0077 | 0.7680 ± 0.0140 | 0.7510 ± 0.0078 | 0.7418 ± 0.0140 | 11 ± 0 | 0.008 ± 0.009 |
| adam | 5 | 0.5544 ± 0.0271 | 0.7920 ± 0.0081 | 0.7742 ± 0.0069 | 0.6814 ± 0.0693 | 12 ± 0 | 0.234 ± 0.214 |
| adamw | 5 | 0.5783 ± 0.0208 | 0.7704 ± 0.0128 | 0.7522 ± 0.0068 | 0.7314 ± 0.0270 | 12 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 0.5854 ± 0.0220 | 0.7716 ± 0.0116 | 0.7564 ± 0.0083 | 0.7444 ± 0.0565 | 12 ± 0 | 0.000 ± 0.000 |
| nadam | 5 | 0.5741 ± 0.0173 | 0.7748 ± 0.0070 | 0.7748 ± 0.0116 | 0.7468 ± 0.0201 | 12 ± 0 | 0.000 ± 0.000 |
| b25 | 5 | 0.5886 ± 0.0139 | 0.7744 ± 0.0082 | 0.7496 ± 0.0027 | 0.7500 ± 0.0025 | 14 ± 0 | 0.000 ± 0.000 |
| b25_plain | 5 | 0.5790 ± 0.0234 | 0.7744 ± 0.0112 | 0.7538 ± 0.0075 | 0.7516 ± 0.0104 | 13 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.01036 | [-0.00038, 0.02110] | 2.68 (4) | 0.0553 | 1.20 | 1/5 |
| val loss at end (exploratory) | 5 | -1.75311 | [-2.10186, -1.40436] | -13.96 (4) | 0.0002 | -6.24 | 5/5 |
| test acc @best-val ckpt (exploratory) | 5 | -0.00260 | [-0.00987, 0.00467] | -0.99 (4) | 0.3771 | -0.44 | 2/5 |
| test acc @end (exploratory) | 5 | 0.01860 | [-0.01392, 0.05112] | 1.59 (4) | 0.1874 | 0.71 | 4/5 |
| wall-clock s (exploratory) | 5 | 2.29084 | [2.18085, 2.40082] | 57.83 (4) | 0.0000 | 25.86 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.04027 | [-0.06163, -0.01892] | 0.0064 | 0.0254 | 5/5 |
| adam | 5 | 0.03425 | [0.00679, 0.06171] | 0.0257 | 0.0515 | 0/5 |
| rmsprop | 5 | 0.00318 | [-0.01750, 0.02386] | 0.6915 | 0.6915 | 3/5 |
| nadam | 5 | 0.01457 | [0.00672, 0.02243] | 0.0067 | 0.0254 | 0/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff 0.00960, 95% CI [-0.00402, 0.02321], p 0.1221, B25 better in 0/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 63.2 | 0.0 | 21.6 | 0.0 | 41.6 | 41.6 | 0.0 | 0.658 | 1.000 | 41.6 |
| b25_plain | 5 | 59.0 | 0.0 | 0.0 | 0.0 | 59.0 | 59.0 | 0.0 | 1.000 | 1.000 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.5783 | 3/5, 125 | 2/5, 988 | 2/5, 726 | both 2: B25 faster 0, slower 2; mean saving -875 units (95% CI -878 to -872); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.5783 | 3/5, 125 | 2/5, 988 | 2/5, 726 | both 2: B25 faster 0, slower 2; mean saving -875 units (95% CI -878 to -872); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.5783 | 3/5, 125 | 2/5, 988 | 2/5, 726 | both 2: B25 faster 0, slower 2; mean saving -875 units (95% CI -878 to -872); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.5783 | 3/5, 125 | 2/5, 988 | 2/5, 726 | both 2: B25 faster 0, slower 2; mean saving -875 units (95% CI -878 to -872); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.5783 | 3/5, 125 | 2/5, 988 | 2/5, 726 | both 2: B25 faster 0, slower 2; mean saving -875 units (95% CI -878 to -872); only B25 0, only AdamW 1, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.5783 | 3/5 / 2/5 | -875 (-878 to -872) | 0 / 2 | no | no |
| 25% | 1% | 0.5840 | 3/5 / 3/5 | -850 (-912 to -789) | 0 / 3 | no | no |
| 25% | 2% | 0.5898 | 3/5 / 3/5 | -792 (-887 to -697) | 0 / 3 | no | no |
| 25% | 5% | 0.6072 | 5/5 / 5/5 | -726 (-816 to -635) | 0 / 5 | no | no |
| 50% | 0% | 0.5783 | 3/5 / 2/5 | -875 (-878 to -872) | 0 / 2 | no | no |
| 50% | 1% | 0.5840 | 3/5 / 3/5 | -850 (-912 to -789) | 0 / 3 | no | no |
| 50% | 2% | 0.5898 | 3/5 / 3/5 | -792 (-887 to -697) | 0 / 3 | no | no |
| 50% | 5% | 0.6072 | 5/5 / 5/5 | -726 (-816 to -635) | 0 / 5 | no | no |
| 75% | 0% | 0.5783 | 3/5 / 2/5 | -875 (-878 to -872) | 0 / 2 | no | no |
| 75% | 1% | 0.5840 | 3/5 / 3/5 | -850 (-912 to -789) | 0 / 3 | no | no |
| 75% | 2% | 0.5898 | 3/5 / 3/5 | -792 (-887 to -697) | 0 / 3 | no | no |
| 75% | 5% | 0.6072 | 5/5 / 5/5 | -726 (-816 to -635) | 0 / 5 | no | no |
| 95% | 0% | 0.5783 | 3/5 / 2/5 | -875 (-878 to -872) | 0 / 2 | no | no |
| 95% | 1% | 0.5840 | 3/5 / 3/5 | -850 (-912 to -789) | 0 / 3 | no | no |
| 95% | 2% | 0.5898 | 3/5 / 3/5 | -792 (-887 to -697) | 0 / 3 | no | no |
| 95% | 5% | 0.6072 | 5/5 / 5/5 | -726 (-816 to -635) | 0 / 5 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.5783 ± 0.0208 | 0.5783 ± 0.0208 | 0.5783 ± 0.0208 | 0.5783 ± 0.0208 | 0.5953 ± 0.0189 |
| b25 | 0.9448 ± 0.0057 | 0.7469 ± 0.0181 | 0.6231 ± 0.0094 | 0.5886 ± 0.0139 | 0.7859 ± 0.0044 |
| b25_plain | 0.9338 ± 0.0026 | 0.7023 ± 0.0065 | 0.5888 ± 0.0143 | 0.5790 ± 0.0234 | 0.7600 ± 0.0075 |
| sgd | 1.0678 ± 0.0087 | 0.9799 ± 0.0100 | 0.7771 ± 0.0186 | 0.6289 ± 0.0077 | 0.9190 ± 0.0019 |
| adam | 0.5544 ± 0.0271 | 0.5544 ± 0.0271 | 0.5544 ± 0.0271 | 0.5544 ± 0.0271 | 0.5815 ± 0.0238 |
| rmsprop | 0.5854 ± 0.0220 | 0.5854 ± 0.0220 | 0.5854 ± 0.0220 | 0.5854 ± 0.0220 | 0.5867 ± 0.0220 |
| nadam | 0.6080 ± 0.0131 | 0.5745 ± 0.0171 | 0.5741 ± 0.0173 | 0.5741 ± 0.0173 | 0.6256 ± 0.0126 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.36655 | [0.34153, 0.39157] | 0.0000 |
| at 50% of the budget | 5 | 0.16859 | [0.12866, 0.20853] | 0.0003 |
| at 75% of the budget | 5 | 0.04487 | [0.02327, 0.06647] | 0.0045 |
| at 100% of the budget | 5 | 0.01036 | [-0.00038, 0.02110] | 0.0553 |
| mean over the whole budget (area under the curve) | 5 | 0.19059 | [0.16925, 0.21193] | 0.0000 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.