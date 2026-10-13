# Comparison: gat_cora

GPU: NVIDIA L4 | optimizer file 0b63021273376f1e | search space 07d3598d8c8b9eaa

## Validity checks

- PASS: same experiment, budget, dropout mode, clip, GPU, hyper-parameters per arm; no run over budget; every B25 run has full undo coverage

## Table A - per arm (mean ± sd over seeds)

| arm | seeds | best val loss | best val acc | test acc @best-val ckpt | test acc @end | wall s | clip rate |
|---|---|---|---|---|---|---|---|
| sgd | 5 | 1.6107 ± 0.0320 | 0.7196 ± 0.0347 | 0.7238 ± 0.0299 | 0.7238 ± 0.0299 | 11 ± 0 | 0.000 ± 0.000 |
| adam | 5 | 0.7443 ± 0.0216 | 0.7936 ± 0.0026 | 0.8010 ± 0.0054 | 0.7410 ± 0.0258 | 11 ± 0 | 0.000 ± 0.000 |
| adamw | 5 | 0.6933 ± 0.0474 | 0.7844 ± 0.0155 | 0.7898 ± 0.0158 | 0.7426 ± 0.0298 | 11 ± 0 | 0.000 ± 0.000 |
| rmsprop | 5 | 0.7360 ± 0.0468 | 0.7716 ± 0.0057 | 0.7776 ± 0.0158 | 0.6836 ± 0.0208 | 11 ± 0 | 0.000 ± 0.000 |
| nadam | 5 | 0.7554 ± 0.0228 | 0.7900 ± 0.0073 | 0.7944 ± 0.0033 | 0.7436 ± 0.0226 | 11 ± 0 | 0.000 ± 0.000 |
| b25 | 5 | 0.6740 ± 0.0522 | 0.7876 ± 0.0115 | 0.7952 ± 0.0220 | 0.7466 ± 0.0240 | 13 ± 1 | 0.000 ± 0.000 |
| b25_plain | 5 | 0.6886 ± 0.0416 | 0.7876 ± 0.0142 | 0.7892 ± 0.0152 | 0.7476 ± 0.0237 | 12 ± 0 | 0.000 ± 0.000 |

## Table B - PRIMARY: B25 minus AdamW, paired by seed (negative = B25 better for losses)

| metric | n | mean diff | 95% CI | t (df) | p (two-sided, unadjusted) | d_z | B25 better in |
|---|---|---|---|---|---|---|---|
| best val loss (PRIMARY) | 5 | -0.01927 | [-0.04377, 0.00524] | -2.18 (4) | 0.0944 | -0.98 | 4/5 |
| val loss at end (exploratory) | 5 | -0.01174 | [-0.06024, 0.03676] | -0.67 (4) | 0.5385 | -0.30 | 3/5 |
| test acc @best-val ckpt (exploratory) | 5 | 0.00540 | [-0.01498, 0.02578] | 0.74 (4) | 0.5027 | 0.33 | 3/5 |
| test acc @end (exploratory) | 5 | 0.00400 | [-0.00424, 0.01224] | 1.35 (4) | 0.2488 | 0.60 | 3/5 |
| wall-clock s (exploratory) | 5 | 1.52224 | [0.87126, 2.17321] | 6.49 (4) | 0.0029 | 2.90 | 0/5 |

## Table C - SECONDARY: B25 minus each other baseline, best val loss (Holm-adjusted over these comparisons)

| baseline | n | mean diff | 95% CI | p raw | p Holm | B25 better in |
|---|---|---|---|---|---|---|
| sgd | 5 | -0.93670 | [-0.97882, -0.89459] | 0.0000 | 0.0000 | 5/5 |
| adam | 5 | -0.07026 | [-0.11150, -0.02901] | 0.0091 | 0.0252 | 5/5 |
| rmsprop | 5 | -0.06192 | [-0.09937, -0.02448] | 0.0101 | 0.0252 | 5/5 |
| nadam | 5 | -0.08136 | [-0.12804, -0.03468] | 0.0084 | 0.0252 | 5/5 |

## Ablation (exploratory): B25 minus B25-plain (momentum refill + gate), best val loss

mean diff -0.01462, 95% CI [-0.03218, 0.00294], p 0.0819, B25 better in 4/5 seeds

## Table E - B25 jump behaviour (counts per run, mean over seeds)

accepted = passed the same-batch test; kept = survived the fresh-batch check; undone = reverted by it. attempts = uphill + gate + failed + accepted.

| arm | seeds | attempts | skipped uphill | skipped by gate | probed, failed | accepted | kept | undone | accepted/attempts | kept/accepted | momentum refills |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b25 | 5 | 71.0 | 0.8 | 53.4 | 5.2 | 11.6 | 10.6 | 1.0 | 0.163 | 0.914 | 11.6 |
| b25_plain | 5 | 59.8 | 1.2 | 0.0 | 19.2 | 39.4 | 36.6 | 2.8 | 0.659 | 0.929 | 0.0 |

## Table F - billed compute to reach the SAME loss (target = AdamW's mean best-so-far loss at 25/50/75/95/100% of the budget)

Billed units include B25's probes and undo checks. A target counts as 'reached' at the first checkpoint whose validation loss is at or below it.

| target (AdamW at x% of budget) | loss | adamw: reached, median units | b25: reached, median units | b25_plain: reached, median units | B25 vs AdamW (common seeds) | verdict |
|---|---|---|---|---|---|---|
| 25% | 0.6933 | 3/5, 200 | 3/5, 125 | 3/5, 101 | both 3: B25 faster 3, slower 0; mean saving 83 units (95% CI 47 to 118); only B25 0, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 50% | 0.6933 | 3/5, 200 | 3/5, 125 | 3/5, 101 | both 3: B25 faster 3, slower 0; mean saving 83 units (95% CI 47 to 118); only B25 0, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 75% | 0.6933 | 3/5, 200 | 3/5, 125 | 3/5, 101 | both 3: B25 faster 3, slower 0; mean saving 83 units (95% CI 47 to 118); only B25 0, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 95% | 0.6933 | 3/5, 200 | 3/5, 125 | 3/5, 101 | both 3: B25 faster 3, slower 0; mean saving 83 units (95% CI 47 to 118); only B25 0, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |
| 100% | 0.6933 | 3/5, 200 | 3/5, 125 | 3/5, 101 | both 3: B25 faster 3, slower 0; mean saving 83 units (95% CI 47 to 118); only B25 0, only AdamW 0, neither 2 | insufficient (B25 and AdamW both reached it in fewer than 4 common seeds) |

## Table F2 - near-miss saving (exploratory, Addenda 2 and 2b): billed units to reach a loss within eps of AdamW's target

eps = 0 is the strict rule of Table F. SYMMETRIC columns: BOTH arms are asked to reach the same looser target (1 + eps) x AdamW's loss; this is the valid test (an optimizer identical to AdamW shows no saving). ASYMMETRIC columns (only B25 gets the looser target) are BIASED: even an exact copy of AdamW shows 'yes' for eps > 0 - they are printed for transparency only and must not be quoted as evidence.

| target (AdamW at x% of budget) | eps | looser target loss | seeds reached, AdamW / B25 | SYMMETRIC saving, units (95% CI) | B25 faster / slower | SYMMETRIC near-iso saving | asymmetric (biased) |
|---|---|---|---|---|---|---|---|
| 25% | 0% | 0.6933 | 3/5 / 3/5 | 83 (47 to 118) | 3 / 0 | no | no |
| 25% | 1% | 0.7002 | 3/5 / 3/5 | 74 (13 to 136) | 3 / 0 | no | no |
| 25% | 2% | 0.7072 | 3/5 / 4/5 | 66 (31 to 102) | 3 / 0 | no | no |
| 25% | 5% | 0.7280 | 3/5 / 4/5 | 74 (74 to 74) | 3 / 0 | no | no |
| 50% | 0% | 0.6933 | 3/5 / 3/5 | 83 (47 to 118) | 3 / 0 | no | no |
| 50% | 1% | 0.7002 | 3/5 / 3/5 | 74 (13 to 136) | 3 / 0 | no | no |
| 50% | 2% | 0.7072 | 3/5 / 4/5 | 66 (31 to 102) | 3 / 0 | no | no |
| 50% | 5% | 0.7280 | 3/5 / 4/5 | 74 (74 to 74) | 3 / 0 | no | no |
| 75% | 0% | 0.6933 | 3/5 / 3/5 | 83 (47 to 118) | 3 / 0 | no | no |
| 75% | 1% | 0.7002 | 3/5 / 3/5 | 74 (13 to 136) | 3 / 0 | no | no |
| 75% | 2% | 0.7072 | 3/5 / 4/5 | 66 (31 to 102) | 3 / 0 | no | no |
| 75% | 5% | 0.7280 | 3/5 / 4/5 | 74 (74 to 74) | 3 / 0 | no | no |
| 95% | 0% | 0.6933 | 3/5 / 3/5 | 83 (47 to 118) | 3 / 0 | no | no |
| 95% | 1% | 0.7002 | 3/5 / 3/5 | 74 (13 to 136) | 3 / 0 | no | no |
| 95% | 2% | 0.7072 | 3/5 / 4/5 | 66 (31 to 102) | 3 / 0 | no | no |
| 95% | 5% | 0.7280 | 3/5 / 4/5 | 74 (74 to 74) | 3 / 0 | no | no |

## Table G - best-so-far validation loss at the same billed compute (exploratory; lower is better)

| arm | 25% (250 units) | 50% (500 units) | 75% (750 units) | 100% (1000 units) | mean over a 25-unit grid |
|---|---|---|---|---|---|
| adamw | 0.6933 ± 0.0474 | 0.6933 ± 0.0474 | 0.6933 ± 0.0474 | 0.6933 ± 0.0474 | 0.7866 ± 0.0431 |
| b25 | 0.6740 ± 0.0522 | 0.6740 ± 0.0522 | 0.6740 ± 0.0522 | 0.6740 ± 0.0522 | 0.7583 ± 0.0489 |
| b25_plain | 0.6886 ± 0.0416 | 0.6886 ± 0.0416 | 0.6886 ± 0.0416 | 0.6886 ± 0.0416 | 0.7681 ± 0.0389 |
| sgd | 1.9252 ± 0.0094 | 1.8867 ± 0.0077 | 1.7945 ± 0.0204 | 1.6107 ± 0.0320 | 1.8497 ± 0.0096 |
| adam | 0.8532 ± 0.0256 | 0.7443 ± 0.0216 | 0.7443 ± 0.0216 | 0.7443 ± 0.0216 | 0.8952 ± 0.0204 |
| rmsprop | 0.7360 ± 0.0468 | 0.7360 ± 0.0468 | 0.7360 ± 0.0468 | 0.7360 ± 0.0468 | 0.7566 ± 0.0437 |
| nadam | 0.8795 ± 0.0204 | 0.7554 ± 0.0228 | 0.7554 ± 0.0228 | 0.7554 ± 0.0228 | 0.9083 ± 0.0194 |

| B25 minus AdamW (paired by seed) | n | mean diff | 95% CI | p (unadjusted) |
|---|---|---|---|---|
| at 25% of the budget | 5 | -0.01927 | [-0.04377, 0.00524] | 0.0944 |
| at 50% of the budget | 5 | -0.01927 | [-0.04377, 0.00524] | 0.0944 |
| at 75% of the budget | 5 | -0.01927 | [-0.04377, 0.00524] | 0.0944 |
| at 100% of the budget | 5 | -0.01927 | [-0.04377, 0.00524] | 0.0944 |
| mean over the whole budget (area under the curve) | 5 | -0.02834 | [-0.05080, -0.00588] | 0.0248 |

## Reading guide

- n = 5 seeds: the paired t-test has 4 degrees of freedom, so it is low-powered. Report the effect size and the CI, not only p.
- Only the PRIMARY row is a pre-registered confirmatory test. Table C is Holm-adjusted. Everything marked exploratory is descriptive.
- Lower is better for losses. Test accuracy is reported at the end of training and at the validation-selected checkpoint; selection never uses the test set.
- Billing assumption: a forward-only pass costs 0.3 of a full gradient evaluation, an undo check 0.6. Wall-clock is reported next to billed units.