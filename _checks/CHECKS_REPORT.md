# B25 checks report (Tue Oct  6 23:53:03 2026)

## M - fixed learning rates (lstm_warpeace; each arm keeps its original tuned beta1, beta2 and weight decay; only the learning rate changes)

| lr | AdamW best val loss | B25 best val loss | B25 - AdamW (paired by seed) | B25 jumps per run: accepted / kept |
|---|---|---|---|---|
| 0.0003 | 2.0732 +- 0.0103 (n=5) | 2.0356 +- 0.0142 (n=5) | -0.0377 (CI -0.0548 to -0.0205, p=0.0036, better in 5/5) | 32.6 / 29.0 |
| 0.001 | 1.7969 +- 0.0132 (n=5) | 1.7708 +- 0.0145 (n=5) | -0.0261 (CI -0.0379 to -0.0144, p=0.0034, better in 5/5) | 14.8 / 13.2 |
| 0.003 | 1.6764 +- 0.0121 (n=5) | 1.6364 +- 0.0094 (n=5) | -0.0399 (CI -0.0459 to -0.0340, p=0.0000, better in 5/5) | 5.2 / 4.8 |
| 0.01 | 1.5950 +- 0.0187 (n=5) | 1.5668 +- 0.0115 (n=5) | -0.0282 (CI -0.0397 to -0.0167, p=0.0025, better in 5/5) | 0.0 / 0.0 |

**Effective-learning-rate comparison (paired by seed; negative = the first is better):**

- b25 at 0.003 minus adamw at 0.01: +0.0414 (CI +0.0169 to +0.0660, p=0.0094, better in 0/5)
- b25 at 0.01 minus adamw at 0.01: -0.0282 (CI -0.0397 to -0.0167, p=0.0025, better in 5/5)
- b25 at 0.003 minus adamw at 0.003: -0.0399 (CI -0.0459 to -0.0340, p=0.0000, better in 5/5)
- b25 at 0.001 minus adamw at 0.003: +0.0944 (CI +0.0818 to +0.1070, p=0.0000, better in 0/5)

**Control (lr 3e-3 must reproduce the original 5-seed runs):**

- adamw: new minus original best val loss per seed: max |diff| 0.000000 over 5 seeds
- b25: new minus original best val loss per seed: max |diff| 0.000000 over 5 seeds

**Components (B25 at its tuned lr, 5 seeds):**

| variant | best val loss | minus AdamW | minus B25 | accepted / kept per run |
|---|---|---|---|---|
| b25_nogate | 1.6359 +- 0.0077 (n=5) | -0.0405 (CI -0.0487 to -0.0322, p=0.0002, better in 5/5) | -0.0005 (CI -0.0043 to +0.0033, p=0.7262, better in 3/5) | 5.8 / 5.4 |
| b25_norefill | 1.6518 +- 0.0124 (n=5) | -0.0245 (CI -0.0322 to -0.0169, p=0.0009, better in 5/5) | +0.0154 (CI +0.0047 to +0.0261, p=0.0162, better in 0/5) | 7.2 / 6.6 |
| b25_randdir | 1.6593 +- 0.0165 (n=5) | -0.0171 (CI -0.0304 to -0.0037, p=0.0239, better in 5/5) | +0.0229 (CI +0.0119 to +0.0339, p=0.0045, better in 0/5) | 16.2 / 7.0 |

## R - why do jumps survive? (lstm_warpeace; AdamW and B25 at their original tuned settings, 1000 units, 5 seeds, one fixed subset per size)

revisit = fewer training windows at batch 50 (more passes); batch = smaller batches with the training set cut so that about ONE pass is made.

| kind | train windows | batch | targets per batch | passes in 1000 steps | AdamW loss | B25 loss | B25 - AdamW | accepted | kept | kept share | saving at 50% / 75% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| original | all | 50 | 2500 | 0.97 | 1.6764 +- 0.0121 (n=5) | 1.6364 +- 0.0094 (n=5) | -0.0399 (CI -0.0459 to -0.0340, p=0.0000, better in 5/5) | 5.2 | 4.8 | 0.94 | +85 units (CI +49 to +120; 5 seeds) ; +125 units (CI +103 to +146; 5 seeds) |
| batch | 25000 | 25 | 1250 | 1.0 | 1.7434 +- 0.0123 (n=5) | 1.7066 +- 0.0143 (n=5) | -0.0368 (CI -0.0426 to -0.0310, p=0.0001, better in 5/5) | 4.2 | 3.6 | 0.88 | +64 units (CI +37 to +92; 5 seeds) ; +114 units (CI +87 to +142; 5 seeds) |
| batch | 10000 | 10 | 500 | 1.0 | 1.8458 +- 0.0223 (n=5) | 1.8098 +- 0.0137 (n=5) | -0.0361 (CI -0.0491 to -0.0231, p=0.0015, better in 5/5) | 4.2 | 2.6 | 0.62 | +54 units (CI +8 to +100; 5 seeds) ; +99 units (CI +50 to +148; 5 seeds) |
| batch | 5000 | 5 | 250 | 1.0 | 1.9313 +- 0.0166 (n=5) | 1.9077 +- 0.0177 (n=5) | -0.0236 (CI -0.0317 to -0.0156, p=0.0012, better in 5/5) | 2.8 | 1.4 | 0.58 | +49 units (CI +27 to +71; 5 seeds) ; +69 units (CI +55 to +83; 5 seeds) |
| revisit | 12833 | 50 | 2500 | 3.9 | 1.6760 +- 0.0133 (n=5) | 1.6388 +- 0.0163 (n=5) | -0.0372 (CI -0.0430 to -0.0314, p=0.0001, better in 5/5) | 6.4 | 5.0 | 0.81 | +65 units (CI +48 to +82; 5 seeds) ; +120 units (CI +94 to +146; 5 seeds) |
| revisit | 2567 | 50 | 2500 | 19.5 | 1.7267 +- 0.0059 (n=5) | 1.6994 +- 0.0043 (n=5) | -0.0274 (CI -0.0340 to -0.0207, p=0.0003, better in 5/5) | 5.2 | 4.8 | 0.92 | +85 units (CI +68 to +101; 5 seeds) ; +140 units (CI +112 to +167; 5 seeds) |
| revisit | 513 | 50 | 2500 | 97.5 | 2.0101 +- 0.0023 (n=5) | 1.9962 +- 0.0105 (n=5) | -0.0139 (CI -0.0254 to -0.0023, p=0.0293, better in 5/5) | 4.6 | 4.6 | 1.00 | n/a (fewer than 3 common seeds reach it) ; n/a (fewer than 3 common seeds reach it) |

## D - per-attempt jump diagnostics (B25 with the attempt log on; the log is passive: it adds no gradient evaluation)

The attempt log is written after the gate, so gate skips are NOT in these rows; they are in b25_stats (Table E).

| experiment | arm | runs | logged attempts | skipped uphill | accepted | kept | undone | kept share | AUROC: consecutive-gradient cosine separates kept from undone | accepted share, early / mid / late third |
|---|---|---|---|---|---|---|---|---|---|---|
| lstm_warpeace | b25 | 3 | 184 | 157 | 14 | 14 | 0 | 1.00 | n/a | 0.22 / 0.00 / 0.00 |
| lstm_warpeace | b25_plain | 3 | 217 | 182 | 22 | 21 | 1 | 0.95 | 0.571 | 0.28 / 0.00 / 0.00 |
| lstm_text8 | b25 | 3 | 219 | 204 | 8 | 7 | 1 | 0.88 | 0.714 | 0.10 / 0.00 / 0.00 |
| lstm_text8 | b25_plain | 3 | 221 | 200 | 13 | 11 | 2 | 0.85 | 0.500 | 0.16 / 0.00 / 0.00 |
| lstm_ptbword | b25 | 3 | 131 | 119 | 4 | 1 | 3 | 0.25 | 0.000 | 0.00 / 0.08 / 0.00 |
| lstm_ptbword | b25_plain | 3 | 215 | 167 | 26 | 12 | 14 | 0.46 | 0.613 | 0.01 / 0.23 / 0.11 |
| mlp_covertype | b25 | 3 | 159 | 159 | 0 | 0 | 0 | n/a | n/a | 0.00 / 0.00 / 0.00 |
| mlp_covertype | b25_plain | 3 | 228 | 228 | 0 | 0 | 0 | n/a | n/a | 0.00 / 0.00 / 0.00 |
| bilstm_cola | b25 | 3 | 135 | 85 | 49 | 27 | 22 | 0.55 | 0.588 | 0.39 / 0.42 / 0.21 |
| bilstm_cola | b25_plain | 3 | 201 | 112 | 85 | 36 | 49 | 0.42 | 0.497 | 0.46 / 0.40 / 0.39 |

**Peak GPU memory (MB), AdamW against B25 (the attempt log is on in the B25 runs, so this is an upper bound):**

| experiment | AdamW | B25 | ratio |
|---|---|---|---|
| lstm_warpeace | 206 | 217 | 1.05 |
| lstm_text8 | 205 | 216 | 1.05 |
| lstm_ptbword | 319 | 608 | 1.90 |
| mlp_covertype | 167 | 170 | 1.02 |
| bilstm_cola | 1803 | 5071 | 2.81 |

## W - re-tuning in a WIDE search box (lr 1e-5 to 1e-1, beta1 0 to 0.9995, beta2 0.5 to 0.99999; same weight-decay list; 20 trials; the Optuna sampler and default first trial are the original ones)

### lstm_warpeace_wide

| arm | seeds | best val loss | tuned setting | on a box edge (wide box)? |
|---|---|---|---|---|
| adamw | 10 | 1.4818 +- 0.0038 (n=10) | {"lr": 0.006769452496737681, "betas": [0.0, 0.6841670488775247], "weight_decay": 5e-05} | beta1-bottom |

- adamw: wide box minus original box, same seeds: -0.1950 (CI -0.2111 to -0.1789, p=0.0000, better in 5/5)

### lstm_warpeace_wide2

| arm | seeds | best val loss | tuned setting | on a box edge (wide box)? |
|---|---|---|---|---|
| adamw | 10 | 1.4835 +- 0.0033 (n=10) | {"lr": 0.0086514768907595, "betas": [0.05419800687127287, 0.6568891512951769], "weight_decay": 0.0005} | beta1-bottom |

- adamw: wide box minus original box, same seeds: -0.1948 (CI -0.2122 to -0.1773, p=0.0000, better in 5/5)

