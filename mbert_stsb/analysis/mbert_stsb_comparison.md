# Comparison: mbert_stsb

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

Regression task (stsb): the 'loss' columns are mean squared error, and the columns named 'Pearson r' are Pearson correlations (higher is better).

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val Pearson r | test Pearson r @best-val ckpt | test Pearson r @end | val Spearman @best-val ckpt | test Spearman @best-val ckpt | wall s | clip rate |
|---|---|---|---|---|---|---|---|---|---|
| sgd | 5 | 0.4925 ± 0.0032 | 0.8847 ± 0.0009 | 0.8682 ± 0.0019 | 0.8671 ± 0.0021 | 0.8823 ± 0.0010 | 0.8406 ± 0.0031 | 821 ± 0 | 1.000 ± 0.000 |
| adam | 5 | 0.4575 ± 0.0082 | 0.8934 ± 0.0015 | 0.8739 ± 0.0040 | 0.8726 ± 0.0051 | 0.8900 ± 0.0014 | 0.8497 ± 0.0040 | 851 ± 0 | 1.000 ± 0.000 |
| adamw | 5 | 0.4607 ± 0.0151 | 0.8922 ± 0.0033 | 0.8717 ± 0.0035 | 0.8723 ± 0.0062 | 0.8890 ± 0.0035 | 0.8462 ± 0.0050 | 857 ± 0 | 1.000 ± 0.000 |
| rmsprop | 5 | 0.5087 ± 0.0287 | 0.8806 ± 0.0064 | 0.8614 ± 0.0095 | 0.8612 ± 0.0109 | 0.8781 ± 0.0052 | 0.8372 ± 0.0114 | 833 ± 3 | 0.999 ± 0.003 |
| nadam | 5 | 0.4576 ± 0.0104 | 0.8930 ± 0.0027 | 0.8757 ± 0.0041 | 0.8756 ± 0.0021 | 0.8897 ± 0.0028 | 0.8524 ± 0.0057 | 849 ± 15 | 1.000 ± 0.000 |
| b25 | 5 | 0.4638 ± 0.0131 | 0.8916 ± 0.0032 | 0.8691 ± 0.0039 | 0.8691 ± 0.0034 | 0.8880 ± 0.0028 | 0.8447 ± 0.0044 | 922 ± 1 | 1.000 ± 0.000 |
| b25_plain | 5 | 0.4689 ± 0.0062 | 0.8901 ± 0.0014 | 0.8682 ± 0.0047 | 0.8677 ± 0.0064 | 0.8871 ± 0.0012 | 0.8438 ± 0.0058 | 920 ± 0 | 1.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | 0.00310 | [-0.01476, 0.02095] | 0.48 (4) | 0.6555 | 0.22 | 3/5 |
| val loss at end (exploratory) | 5 | -0.00378 | [-0.01949, 0.01193] | -0.67 (4) | 0.5407 | -0.30 | 4/5 |
| test Pearson r @best-val ckpt (exploratory) | 5 | -0.00253 | [-0.00416, -0.00090] | -4.31 (4) | 0.0125 | -1.93 | 0/5 |
| test Pearson r @end (exploratory) | 5 | -0.00320 | [-0.00812, 0.00172] | -1.80 (4) | 0.1456 | -0.81 | 1/5 |
| test Spearman @best-val ckpt (exploratory; regression tasks) | 5 | -0.00154 | [-0.00265, -0.00043] | -3.84 (4) | 0.0185 | -1.72 | 1/5 |
| wall-clock s (exploratory) | 5 | 64.92490 | [63.68221, 66.16759] | 145.06 (4) | 0.0000 | 64.87 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.02869 | [-0.04420, -0.01318] | 0.0068 | 0.0272 | 5/5 |
| adam | 5 | 0.00633 | [-0.00623, 0.01889] | 0.2341 | 0.4661 | 1/5 |
| rmsprop | 5 | -0.04492 | [-0.09272, 0.00288] | 0.0595 | 0.1784 | 5/5 |
| nadam | 5 | 0.00619 | [-0.00605, 0.01843] | 0.2330 | 0.4661 | 1/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.00505, 95% CI [-0.01798, 0.00787], p 0.3385, B25 better in 3/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 75.4 | 41.6 | 32.6 | 0.6 | 0.6 | 0.2 | 0.4 | 0.008 | 0.333 | 0.6 |
| b25_plain | 5 | 74.4 | 70.2 | 0.0 | 2.2 | 2.0 | 0.0 | 2.0 | 0.027 | 0.000 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.5131 | 5/5, 250 | 5/5, 175 | 5/5, 175 | both 5: B25 faster 1, slower 2; mean saving -5 units (95% CI -82 to 72); only B25 0, only AdamW 0, neither 0 | saving NOT supported |
| 50% | 0.4803 | 4/5, 425 | 5/5, 375 | 5/5, 626 | both 4: B25 faster 4, slower 0; mean saving 94 units (95% CI -16 to 203); only B25 1, only AdamW 0, neither 0 | saving NOT supported |
| 75% | 0.4694 | 4/5, 588 | 3/5, 550 | 2/5, 650 | both 3: B25 faster 1, slower 2; mean saving -108 units (95% CI -651 to 434); only B25 0, only AdamW 1, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.4619 | 4/5, 788 | 2/5, 762 | 1/5, 725 | both 2: B25 faster 1, slower 1; mean saving 25 units (95% CI -2199 to 2249); only B25 0, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.4607 | 4/5, 800 | 2/5, 762 | 1/5, 725 | both 2: B25 faster 1, slower 1; mean saving 62 units (95% CI -2638 to 2763); only B25 0, only AdamW 2, neither 1 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.5131 | 5/5 / 5/5 | -5 (-82 to 72) | 1 / 2 | no | no |
| 25% | 1% | 0.5182 | 5/5 / 5/5 | -50 (-201 to 101) | 1 / 3 | no | no |
| 25% | 2% | 0.5233 | 5/5 / 5/5 | -50 (-201 to 101) | 1 / 3 | no | no |
| 25% | 5% | 0.5387 | 5/5 / 5/5 | -25 (-137 to 87) | 1 / 3 | no | no |
| 50% | 0% | 0.4803 | 4/5 / 5/5 | 94 (-16 to 203) | 4 / 0 | no | no |
| 50% | 1% | 0.4851 | 4/5 / 5/5 | 62 (-32 to 157) | 3 / 0 | no | no |
| 50% | 2% | 0.4899 | 5/5 / 5/5 | 55 (-53 to 163) | 3 / 1 | no | no |
| 50% | 5% | 0.5043 | 5/5 / 5/5 | -10 (-110 to 89) | 2 / 2 | no | yes |
| 75% | 0% | 0.4694 | 4/5 / 3/5 | -108 (-651 to 434) | 1 / 2 | no | no |
| 75% | 1% | 0.4741 | 4/5 / 3/5 | 133 (-280 to 547) | 2 / 1 | no | no |
| 75% | 2% | 0.4788 | 4/5 / 5/5 | 94 (-16 to 203) | 4 / 0 | no | yes |
| 75% | 5% | 0.4929 | 5/5 / 5/5 | 30 (-85 to 144) | 3 / 2 | no | yes |
| 95% | 0% | 0.4619 | 4/5 / 2/5 | 25 (-2199 to 2249) | 1 / 1 | no | no |
| 95% | 1% | 0.4665 | 4/5 / 2/5 | -162 (-321 to -4) | 0 / 2 | no | no |
| 95% | 2% | 0.4711 | 4/5 / 3/5 | 75 (-236 to 386) | 2 / 1 | no | no |
| 95% | 5% | 0.4850 | 4/5 / 5/5 | 62 (-32 to 157) | 3 / 0 | no | yes |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.5131 ± 0.0298 | 0.4803 ± 0.0171 | 0.4694 ± 0.0210 | 0.4607 ± 0.0151 | 0.5307 ± 0.0345 |
| b25 | 0.5213 ± 0.0298 | 0.4846 ± 0.0255 | 0.4729 ± 0.0131 | 0.4638 ± 0.0131 | 0.5169 ± 0.0251 |
| b25_plain | 0.5260 ± 0.0346 | 0.4878 ± 0.0121 | 0.4729 ± 0.0104 | 0.4689 ± 0.0062 | 0.5191 ± 0.0214 |
| sgd | 0.5505 ± 0.0107 | 0.5025 ± 0.0037 | 0.4926 ± 0.0033 | 0.4925 ± 0.0032 | 0.5872 ± 0.0073 |
| adam | 0.5141 ± 0.0268 | 0.4806 ± 0.0181 | 0.4637 ± 0.0111 | 0.4575 ± 0.0082 | 0.5290 ± 0.0308 |
| rmsprop | 0.5813 ± 0.0592 | 0.5349 ± 0.0391 | 0.5177 ± 0.0310 | 0.5087 ± 0.0287 | 0.6619 ± 0.1204 |
| nadam | 0.5037 ± 0.0124 | 0.4731 ± 0.0098 | 0.4661 ± 0.0091 | 0.4576 ± 0.0104 | 0.5032 ± 0.0084 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | 0.00829 | [-0.00995, 0.02654] | 0.2755 |
| at 50% of the budget | 5 | 0.00432 | [-0.01085, 0.01949] | 0.4732 |
| at 75% of the budget | 5 | 0.00353 | [-0.01176, 0.01882] | 0.5562 |
| at 100% of the budget | 5 | 0.00310 | [-0.01476, 0.02095] | 0.6555 |
| mean over the whole budget (area under the curve) | 5 | -0.01377 | [-0.02921, 0.00168] | 0.0686 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.