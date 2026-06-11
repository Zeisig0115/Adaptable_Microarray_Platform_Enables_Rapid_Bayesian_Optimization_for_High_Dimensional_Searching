# May 09 add_bo multiseed GP/LOOCV diagnostics

## Scope

- Data: `logs/May_09_full_log/data_corrected.xlsx`, sheet `Corrected Data`.
- Filters: `HRP == 0.0001`, `ctrl == 0`, target `intensity`.
- Encoded data: 461 filtered rows, 446 unique encoded rows, 15 merged duplicate encoded points, 46 encoded dimensions, 22 additives.
- Active-additive geometry after filtering: 0 active = 32 rows, 1 active = 198 rows, 2 active = 231 rows; no observed 3/4-additive recipes.
- Seeds: 0 through 9. `fit_maxiter=75`; candidate generation and acquisition-stability diagnostics were skipped.
- Diagnostics: training posterior fit, kernel/noise matrix eigenspectrum, fixed-hyperparameter LOO in standardized model space, and kernel parameter summaries.

## Kernel implementations inspected

- `old_mixed_hamming`: BoTorch `MixedSingleTaskGP`; installed source forms `K_cont_1 + K_cat_1 + K_cont_2 * K_cat_2`, where `CategoricalKernel` uses an exponential Hamming-style distance averaged over categorical dimensions.
- `baseline_kernel`: project `BaselineKernel`; product of custom continuous RBF over essentials/additive concentrations and isotropic exponential Hamming over binary additive indicators.
- `new_additive_set`: project `AdditiveSetKernel`; essential RBF times a weighted sum of essential, active-set Jaccard, shared additive main, and shared pair terms.

## Main numeric summary across seeds

| model | MLL mean +/- sd | LOO RMSE mean +/- sd | LOO z std mean +/- sd | train RMSE mean +/- sd | noise mean +/- sd | condition number mean +/- sd |
|---|---:|---:|---:|---:|---:|---:|
| old_mixed_hamming | -0.5959 +/- 0 | 0.3314 +/- 0 | 1.036 +/- 0 | 2908 +/- 0 | 0.02464 +/- 0 | 3.462e+04 +/- 0 |
| baseline_kernel | -0.4681 +/- 0 | 0.4677 +/- 0 | 1.053 +/- 0 | 1823 +/- 0 | 0.01442 +/- 0 | 878.7 +/- 0 |
| new_additive_set | -0.1641 +/- 0 | 0.2713 +/- 0 | 0.9879 +/- 0 | 1584 +/- 0 | 0.01091 +/- 0 | 1.583e+04 +/- 0 |

## Per-seed best-model counts

- Best `mll`: new_additive_set: 10.
- Best `train_rmse`: new_additive_set: 10.
- Best `train_mae`: new_additive_set: 10.
- Best `condition_number`: baseline_kernel: 10.
- Best `loo_rmse_model_space`: new_additive_set: 10.
- Best `loo_mae_model_space`: new_additive_set: 10.
- Best `abs_loo_z_std_minus_1`: new_additive_set: 10.

## Files

- Combined seed/model rows: `logs/May_09_full_log/add_bo_multiseed_gp_loocv/all_seed_model_diagnostics.csv`
- Metric summary: `logs/May_09_full_log/add_bo_multiseed_gp_loocv/metric_summary_by_model.csv`
- Per-seed winners: `logs/May_09_full_log/add_bo_multiseed_gp_loocv/per_seed_best_model.csv`
- Kernel parameter summary: `logs/May_09_full_log/add_bo_multiseed_gp_loocv/kernel_parameter_summary_by_model.csv`
- Data summary by seed: `logs/May_09_full_log/add_bo_multiseed_gp_loocv/data_summary_by_seed.csv`

## Observed cautions

- This is a GP fit / fixed-hyperparameter LOO diagnostic run, not a candidate-generation reproduction.
- LOO RMSE is in standardized model space, matching `add_bo.py` implementation.
- No stderr warnings were emitted in the 10 runs, and all LOO Cholesky jitter values were 0.
- The data contain no observed 3- or 4-additive recipes, so these diagnostics do not validate extrapolation to higher-cardinality candidate batches.
