# May 10 add_bo / flat_encoding comparison

## Run scope

Data: `May_10_add/data_corrected.xlsx`, target `intensity`, `ctrl == 0`, HRP filter `0.0001` for `add_bo.py`.

Both runs used the same lighter candidate-generation settings as the earlier May 9 diagnostic so that the comparison stays practical but still uses `q=32`:

```text
q=32, num_top_subspaces=20, num_restarts=5, raw_samples=128,
acq_maxiter=50, sobol_max_samples=128, screen_mc_samples=16,
refine_mc_samples=32
```

Notes:

- `add_bo.py` was run directly and wrote structured diagnostics to `May_10_add/add_bo_compare_diag/`.
- `flat_encoding.py` does not expose all of the above search knobs on its CLI, so I ran its existing `FlatBO.ask(...)` interface with the same settings and captured stdout to `May_10_add/flat_encoding_run.log`.
- Both runs emitted GPyTorch Cholesky jitter warnings during qLogNEI evaluation / optimization. The warnings are preserved in the log files.

Artifacts:

- `May_10_add/add_bo_run.log`
- `May_10_add/add_bo_compare_diag/add_bo_diagnostics.json`
- `May_10_add/add_bo_compare_diag/add_bo_diagnostics.csv`
- `May_10_add/add_bo_compare_diag/old_mixed_hamming_candidates.csv`
- `May_10_add/add_bo_compare_diag/new_additive_set_candidates.csv`
- `May_10_add/flat_encoding_run.log`
- `May_10_add/flat_encoding_candidates.csv`

## Data geometry

The current May 10 data geometry is the same as the synchronized May 9 run:

```text
rows after filters:        461
unique encoded rows:       446
duplicates merged:         15
active additive count:     0 -> 32 rows
                          1 -> 198 rows
                          2 -> 231 rows
                          3/4 -> 0 rows
```

So every `k=3` or `k=4` suggestion is still extrapolation beyond the observed recipe cardinalities.

## Controlled add_bo diagnostics

`add_bo.py` again shows the same qualitative pattern as before: the additive-set kernel fits better and predicts held-out points better than the controlled mixed-Hamming baseline.

| metric | old mixed Hamming | new additive-set |
|---|---:|---:|
| fit seconds | 6.0 | 6.2 |
| marginal log likelihood | -0.596 | -0.219 |
| train RMSE | 2908 | 1975 |
| train MAE | 2127 | 1370 |
| train R2 | 0.9877 | 0.9943 |
| train z-score std | 0.616 | 0.499 |
| noise, standardized | 0.0246 | 0.0159 |
| min eigenvalue of `K + noise I` | 0.0246 | 0.0159 |
| max eigenvalue of `K + noise I` | 853.1 | 910.9 |
| condition number | 3.46e4 | 5.75e4 |
| LOO RMSE, model space | 0.331 | 0.272 |
| LOO z-score std | 1.036 | 0.988 |
| frac `|z| > 2` | 6.05% | 5.83% |
| ACQ value std across pool/repeats | 31.89 | 14.18 |
| mean per-point MC std | 2.84 | 1.75 |
| mean Spearman rank correlation | 0.9996 | 0.9977 |
| top-25 Jaccard stability | 0.969 | 0.782 |

Interpretation:

- The additive-set kernel still wins on MLL, train fit, and fixed-hyperparameter LOO RMSE.
- Its LOO calibration remains close to ideal.
- It is less numerically clean than the old mixed kernel because it fits with lower inferred noise and ends up with the worse condition number.
- Its acquisition frontier is still flatter near the top: lower absolute MC variability, but lower top-25 set stability.

Learned additive-set kernel parameters:

```text
scale_ess      = 2.046
scale_set      = 0.297
scale_main     = 0.0768
scale_pair     = 0.000153
scale_residual = 0.00483
ess lengthscale = [3.34, 1.14]
conc lengthscale = 0.180
```

So the fitted kernel is still dominated by the essentials / environment term, with active-set similarity as a meaningful but secondary effect, and with very little evidence for a large reusable pairwise contribution in the current data.

## flat_encoding actual run

The actual `flat_encoding.py` mixed-GP run gave:

| metric | flat_encoding actual |
|---|---:|
| training points | 446 |
| noise, standardized | 0.0118 |
| min eigenvalue of `K + noise I` | 0.0118 |
| max eigenvalue of `K + noise I` | 541 |
| condition number | 4.58e4 |
| avg abs residual, original y | 1.26e3 |
| max abs residual, original y | 9.03e3 |
| LOO z-score mean | +0.01 |
| LOO z-score std | 0.98 |
| frac `|z| > 2` | 3.8% |
| frac `|z| > 3` | 0.4% |
| prescreen best / worst top-20 | 10.3976 / 10.0167 |
| mixed optimize time | 1640.2 s |
| joint acq value | 11.1090 |
| total runtime | 2522.4 s |

This actual flat run still looks reasonably calibrated and numerically usable, but it lands at a different hyperparameter solution from the controlled `add_bo.py` old mixed model. The most visible differences are:

- lower inferred noise than the controlled old mixed fit
- worse condition number than the controlled old mixed fit
- a noticeably different candidate portfolio

That difference is expected because `flat_encoding.py` fits through `fit_model.py` with its own defaults, whereas `add_bo.py` uses its own controlled fitting and diagnostic flow.

## Candidate distribution

All three runs still collapse to all-`k=4` batches under the lighter `q=32` run.

### flat_encoding actual

```text
active_count_dist: {4: 32}
top additives: imidazole 32, cmc 31, bsa 21, cacl2 21, feso4 15, pva 6, sucrose 2
top pairs: cmc+imidazole 31, cmc+cacl2 21, bsa+imidazole 21,
           imidazole+cacl2 21, cmc+bsa 20, imidazole+feso4 15
pred_mean mean/min/max: 171774 / 137475 / 192597
pred_sigma mean/min/max: 23800 / 15433 / 30544
```

### add_bo old mixed Hamming

```text
active_count_dist: {4: 32}
top additives: bsa 32, imidazole 32, cmc 21, sucrose 18, pva 10,
               edta 6, peg200k 3, feso4 3, peg20k 2, mgso4 1
top pairs: bsa+imidazole 32, cmc+bsa 21, cmc+imidazole 21,
           bsa+sucrose 18, imidazole+sucrose 18, bsa+pva 10
pred_mean mean/min/max: 192719 / 156209 / 224038
pred_sigma mean/min/max: 25144 / 15935 / 34989
```

### add_bo new additive-set

```text
active_count_dist: {4: 32}
top additives: imidazole 32, cacl2 32, bsa 22, cmc 20,
               edta 6, feso4 5, peg20k 5, sucrose 4, mgso4 2
top pairs: imidazole+cacl2 32, bsa+imidazole 22, bsa+cacl2 22,
           cmc+imidazole 20, cmc+cacl2 20, cmc+bsa 10
pred_mean mean/min/max: 188756 / 170982 / 203301
pred_sigma mean/min/max: 13256 / 10104 / 16225
```

Interpretation:

- `flat_encoding.py` actual still prefers the old familiar `imidazole + cmc` core, with substantial `cacl2`, `bsa`, and `feso4` support.
- The controlled `add_bo.py` old mixed model no longer matches the actual flat run very closely. It now shifts toward `bsa + imidazole` plus more `sucrose` / `pva` / `edta`, and largely drops the strong `cacl2` emphasis seen in the actual flat run.
- The new additive-set kernel still concentrates around `imidazole + cacl2 + bsa`, with `cmc` as the main fourth component and smaller side exploration into `edta`, `feso4`, `peg20k`, `sucrose`, and `mgso4`.
- The new kernel produces a substantially tighter candidate uncertainty band than either mixed-Hamming run.

## Top-subspace selection

The top-subspace story also remains the same:

- controlled old mixed: top-20 screened subspaces were all `k=4`
- additive-set kernel: top-20 screened subspaces were `19 x k=4` and `1 x k=3`
- final optimized batches: all `k=4`

So the kernel redesign improved the surrogate, but it still did not remove the high-cardinality pull of qLogNEI on this sparse dataset.

## Overall assessment

The May 10 rerun supports the same high-level conclusions as the May 9 report:

1. The additive-set kernel remains the better controlled surrogate. It improves MLL, in-sample fit, and fixed-hyperparameter LOO RMSE while keeping LOO calibration close to the target.
2. The actual `flat_encoding.py` mixed GP remains usable and reasonably calibrated, but its candidate portfolio is both more uncertain and more sensitive to the mixed-kernel fit path.
3. All workflows still concentrate on `k=4` candidates because the data contain only `k=0/1/2` observations and qLogNEI still rewards value plus uncertainty in high-cardinality extrapolation regions.

Recommended next step is unchanged: keep the additive-set kernel direction, but add a candidate-generation policy layer if you want a more balanced experimental portfolio, for example:

- cardinality quotas across `k=1/2/3/4`
- a soft complexity penalty for larger active sets
- a cost-aware or constrained acquisition objective

