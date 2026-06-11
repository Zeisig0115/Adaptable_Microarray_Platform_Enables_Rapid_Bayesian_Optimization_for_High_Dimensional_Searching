# Multi-seed kernel GP-fit / LOOCV / numerical comparison

Data: `logs/May_09_full_log/data_corrected.xlsx`, sheet `Corrected Data`, target `intensity`, HRP=1e-4, ctrl==0.

Seeds (12): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 123, 2024]. Candidate generation and acquisition-stability MC diagnostics disabled (`--skip_candidates --no-acq_stability_diagnostics`).

Data geometry (identical across all seeds): 461 filtered rows, 446 unique encoded rows, 15 duplicates merged, active-additive-count distribution {'1': 198, '2': 231, '0': 32} (0/1/2 additives only -> essentials + singular + pairwise).

**Across-seed range == 0 for every metric below** (each value is bit-identical across all 12 seeds; the exact-GP MLL fit starts from fixed hyperparameter initialization and uses deterministic L-BFGS-B, so LOOCV + numerical diagnostics do not depend on the seed).

| group | metric | Old Hamming (MixedSingleTaskGP) | AdditiveSetKernel | BaselineKernel |
|---|---|---:|---:|---:|
| GP fit | mll | -0.5959 | -0.1641 | -0.4681 |
| Train fit | train_rmse | 2907.6848 | 1583.7282 | 1822.6137 |
| Train fit | train_mae | 2126.9649 | 1098.3015 | 1179.6882 |
| Train fit | train_r2 | 0.9877 | 0.9964 | 0.9952 |
| Train fit | train_z_mean | 0.0002194 | 0.0001866 | -0.0014 |
| Train fit | train_z_std | 0.6158 | 0.4960 | 0.4961 |
| Train fit | train_frac_abs_z_gt_2 | 0.0135 | 0.0067 | 0.0045 |
| Train fit | train_frac_abs_z_gt_3 | 0.0000 | 0.0000 | 0.0022 |
| Train fit | target_std | 2.627e+04 | 2.627e+04 | 2.627e+04 |
| Numerical / matrix | kernel_diag_min | 2.8783 | 0.5843 | 0.3747 |
| Numerical / matrix | kernel_diag_max | 2.8783 | 2.6277 | 0.3747 |
| Numerical / matrix | noise_standardized | 0.0246 | 0.0109 | 0.0144 |
| Numerical / matrix | eig_min | 0.0246 | 0.0109 | 0.0144 |
| Numerical / matrix | eig_max | 853.0818 | 172.6051 | 12.6709 |
| Numerical / matrix | condition_number | 3.462e+04 | 1.583e+04 | 878.7135 |
| LOOCV | loo_rmse_model_space | 0.3314 | 0.2713 | 0.4677 |
| LOOCV | loo_mae_model_space | 0.2308 | 0.1941 | 0.3194 |
| LOOCV | loo_z_mean | 0.0192 | 0.0054 | 0.0261 |
| LOOCV | loo_z_std | 1.0359 | 0.9879 | 1.0526 |
| LOOCV | loo_frac_abs_z_gt_2 | 0.0605 | 0.0583 | 0.0628 |
| LOOCV | loo_frac_abs_z_gt_3 | 0.0067 | 0.0022 | 0.0112 |
| LOOCV | loo_cholesky_jitter | 0.0000 | 0.0000 | 0.0000 |

fit_seconds (wall-clock, varies, not a modeling quantity): Old Hamming (MixedSingleTaskGP) mean=7.86s, AdditiveSetKernel mean=8.84s, BaselineKernel mean=4.58s