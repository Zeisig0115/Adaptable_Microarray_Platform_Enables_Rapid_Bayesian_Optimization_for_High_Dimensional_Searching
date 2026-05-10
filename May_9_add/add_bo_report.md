# May 9 additive BO comparison report

## Run scope

Data: `May_9_add/data_corrected.xlsx`, sheet `Corrected Data`, target `intensity`, `ctrl == 0`, HRP = 0.0001.

I first noticed that `flat_encoding.py` had been updated to `CONC_LO = 0.001`, `CONC_HI = 2.0`, while the first version of `add_bo.py` still used the old `0.005..1.0` bounds. I synchronized `add_bo.py` to `0.001..2.0` before producing the final diagnostics below. After synchronization both pipelines use 461 filtered rows, 446 unique encoded rows, and merge 15 duplicate encoded points.

Candidate generation used a lighter but still `q=32` setting because full mixed optimization is very expensive:

```text
q=32, num_top_subspaces=20, num_restarts=5, raw_samples=128,
acq_maxiter=50, sobol_max_samples=128, screen_mc_samples=16,
refine_mc_samples=32
```

## Data geometry

The additive data remain sparse:

```text
0 active additives: 32 rows
1 active additive:  198 rows
2 active additives: 231 rows
3/4 active additives: 0 rows
```

So every `k=3` or `k=4` suggestion is structural extrapolation from no/singular/pairwise observations. The synchronized encoding no longer clips any observed additive concentration above the upper bound.

## GP fit and calibration

Controlled diagnostics from `add_bo.py`, old MixedSingleTaskGP vs new additive-set kernel:

| metric | old mixed Hamming | new additive-set |
|---|---:|---:|
| marginal log likelihood | -0.596 | -0.219 |
| train RMSE | 2908 | 1975 |
| train MAE | 2127 | 1370 |
| train R2 | 0.9877 | 0.9943 |
| train z-score std | 0.616 | 0.499 |
| train frac \|z\| > 2 | 1.35% | 0.45% |
| noise, standardized | 0.0246 | 0.0159 |

The new kernel fits the training data more tightly and achieves a substantially better MLL under the same `fit_maxiter=75`. Its posterior z-scores are more under-dispersed on training posterior diagnostics, meaning it is not underestimating uncertainty there; if anything, both models are conservative on in-sample posterior calibration, with the new kernel more conservative.

## Matrix diagnostics

| metric | old mixed Hamming | new additive-set |
|---|---:|---:|
| min eigenvalue of K + noise I | 0.0246 | 0.0159 |
| max eigenvalue of K + noise I | 853.1 | 910.9 |
| condition number | 3.46e4 | 5.75e4 |

The new kernel has a worse condition number, mainly because it fits with lower inferred noise. Both are numerically usable in the direct matrix diagnostics, but qLogNEI optimization emitted GPyTorch Cholesky jitter warnings for both workflows. The new-kernel candidate generation also had one SciPy line-search retry. This says the new prior is more expressive but not numerically cleaner.

The actual `flat_encoding.py` run printed a similar but not identical old-kernel fit: noise 0.0118, min eigenvalue 0.0118, max eigenvalue 541, condition number 4.58e4. The difference is expected because `flat_encoding.py` uses its own default `fit_gpytorch_mll` settings, while the controlled `add_bo.py` comparison fixed `fit_maxiter=75`.

## LOO diagnostics

Fixed-hyperparameter fast LOO:

| metric | old mixed Hamming | new additive-set |
|---|---:|---:|
| LOO RMSE, standardized model space | 0.331 | 0.272 |
| LOO z-score std | 1.036 | 0.988 |
| LOO frac \|z\| > 2 | 6.05% | 5.83% |

The new kernel improves LOO RMSE while keeping LOO calibration near the target `z_std ~= 1`. This is the strongest diagnostic in favor of the new prior: it is not merely memorizing training points; under fixed hyperparameters it predicts held-out points better with comparable calibration.

The actual `flat_encoding.py` run reported LOO z-score std 0.98, frac `|z| > 2` = 3.8%, frac `|z| > 3` = 0.4%. That is also acceptable calibration, but it does not beat the new kernel on controlled LOO RMSE.

## Acquisition function stability

ACQ stability was measured on a fixed random candidate pool with repeated qLogNEI samplers:

| metric | old mixed Hamming | new additive-set |
|---|---:|---:|
| ACQ value std across pool/repeats | 31.89 | 12.88 |
| mean per-point MC std | 2.84 | 1.60 |
| mean Spearman rank correlation | 0.9996 | 0.9982 |
| top-25 Jaccard stability | 0.969 | 0.817 |

Both acquisition functions are rank-stable globally. The old kernel has a more stable top-25 set under repeated MC sampling, while the new kernel has lower absolute MC variability but a less stable very-top set. My interpretation: the new kernel creates a flatter or more competitive high-acquisition frontier, so the top few candidates are easier to reshuffle even though global ranking is still stable.

## Candidate distribution

Both pipelines converged to all `k=4` candidates under the lighter `q=32` candidate-generation run.

`flat_encoding.py` old mixed baseline:

```text
active_count_dist: {4: 32}
top additives: imidazole 32, cmc 31, bsa 21, cacl2 21, feso4 15, pva 6, sucrose 2
top pairs: cmc+imidazole 31, cmc+cacl2 21, bsa+imidazole 21,
           imidazole+cacl2 21, cmc+bsa 20, imidazole+feso4 15
pred_mean mean/min/max: 171774 / 137475 / 192597
pred_sigma mean/min/max: 23800 / 15433 / 30544
```

New additive-set kernel:

```text
active_count_dist: {4: 32}
top additives: imidazole 32, cacl2 32, bsa 23, cmc 15, sucrose 6,
               edta 6, peg20k 6, feso4 4, mgso4 3, peg200k 1
top pairs: imidazole+cacl2 32, bsa+imidazole 23, bsa+cacl2 23,
           cmc+imidazole 15, cmc+cacl2 15, cmc+bsa 8
pred_mean mean/min/max: 185639 / 167006 / 203300
pred_sigma mean/min/max: 13929 / 10169 / 17931
```

The new kernel gives a more concentrated core around `imidazole + cacl2 + bsa`, with `cmc` less dominant than in the old mixed baseline and more secondary exploration into `sucrose`, `edta`, `peg20k`, and `mgso4`. It also predicts higher means and lower sigma for its chosen candidates. This is coherent with the new hierarchical prior: candidates are built from additives and pairs that looked reusable, rather than from a broad Hamming/RBF extrapolation with larger uncertainty.

## Top-subspace selection

The new additive-set run screened all 9109 subspaces and selected top 20 subspaces with this cardinality distribution:

```text
k=4: 19 subspaces
k=3: 1 subspace
```

The final optimized batch was all `k=4`. The actual `flat_encoding.py` run also returned all `k=4`. Therefore the new prior alone did not solve the high-cardinality pull under qLogNEI. It changed which four-additive combinations were preferred and reduced candidate uncertainty, but the acquisition still strongly favors `k=4` under `k_max=4`.

## Interpretation

The new kernel improves fit, MLL, and LOO RMSE, with LOO calibration close to ideal. That supports the belief-driven prior direction.

However, it does not by itself solve the `k=4` candidate concentration. This is partly because the data contain no `k=3/4` observations and qLogNEI rewards high predicted value plus remaining uncertainty in extrapolated regions. The prior now makes sparse active-set similarity more defensible, but the acquisition/search policy still needs a cardinality-aware control if we want a balanced candidate portfolio.

Recommended next step: keep the additive-set kernel idea, but add a candidate-generation policy layer such as quota by cardinality, e.g. reserve candidates across `k=1/2/3/4`, or add a soft complexity penalty / cost-aware objective for larger active sets. Kernel redesign improved the surrogate; cardinality control should handle the experimental design risk.
