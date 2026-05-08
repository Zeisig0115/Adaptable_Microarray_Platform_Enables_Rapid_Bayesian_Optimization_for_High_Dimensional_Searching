# Apr 29 LHS/BO1/BO2 GP Validation Report

This report uses extracted replicate-level objective tables in `Apr_29_full_log`.

Important design fact: BO1 and BO2 contain the same 32 conditions for each HRP; BO2 is therefore a repeat / temporal validation of BO1 conditions, not a new spatial BO round.

Log-space physical bounds used here: `[log10(0.001), 0]` for both TMB and H2O2.

## Run-Level Data Summary

| HRP | source_run | n_conditions | condition_mean_AUC_mean | condition_mean_AUC_max | condition_sd_median | condition_sd_max |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | LHS | 32 | 1.83e+03 | 6.99e+03 | 68.3 | 806 |
| 1 | BO1 | 32 | 1.85e+03 | 6.57e+03 | 64 | 1.59e+03 |
| 1 | BO2 | 32 | 851 | 1.07e+04 | 72.9 | 758 |
| 0.01 | LHS | 32 | 1.89e+03 | 7.49e+03 | 78.5 | 1.44e+03 |
| 0.01 | BO1 | 32 | 1.69e+03 | 8.72e+03 | 55 | 1.21e+03 |
| 0.01 | BO2 | 32 | 974 | 6.96e+03 | 103 | 1.43e+03 |
| 0.0001 | LHS | 32 | 1.42e+03 | 2.37e+03 | 55.1 | 181 |
| 0.0001 | BO1 | 32 | 101 | 469 | 39 | 106 |
| 0.0001 | BO2 | 32 | -22.9 | 439 | 41.4 | 105 |

## Matrix and In-Sample Diagnostics

| HRP | train_combo | model | n_train | n_conditions | matrix_condition | noise_stdzd_median | noise_stdzd_max | train_avg_abs_residual | train_high_low_residual_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LHS_BO1 | replicate_inferred | 256 | 64 | 149 | 0.0589 | 0.0589 | 184 | 5.22 |
| 1 | LHS_BO1 | fixed_sem_raw | 64 | 64 | 31.6 | 0.000355 | 0.212 | 37.9 | 247 |
| 1 | LHS_BO1 | fixed_sem_shrunk | 64 | 64 | 28.1 | 0.00798 | 0.114 | 30.9 | 5.5 |
| 1 | LHS_BO1_BO2 | replicate_inferred | 384 | 64 | 38.2 | 0.651 | 0.651 | 821 | 1.04 |
| 1 | LHS_BO1_BO2 | fixed_sem_raw | 64 | 64 | 10.1 | 0.00117 | 2.35 | 148 | 48.7 |
| 1 | LHS_BO1_BO2 | fixed_sem_shrunk | 64 | 64 | 7.52 | 0.116 | 1.29 | 197 | 2.93 |
| 0.01 | LHS_BO1 | replicate_inferred | 256 | 64 | 274 | 0.0353 | 0.0353 | 179 | 3.86 |
| 0.01 | LHS_BO1 | fixed_sem_raw | 64 | 64 | 11.9 | 0.000214 | 0.109 | 30 | 227 |
| 0.01 | LHS_BO1 | fixed_sem_shrunk | 64 | 64 | 11.7 | 0.00465 | 0.0591 | 23.6 | 5.38 |
| 0.01 | LHS_BO1_BO2 | replicate_inferred | 384 | 64 | 41.2 | 0.457 | 0.457 | 667 | 1.24 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_raw | 64 | 64 | 12 | 0.00119 | 1.05 | 125 | 21.6 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_shrunk | 64 | 64 | 9.19 | 0.0685 | 0.594 | 153 | 2.65 |
| 0.0001 | LHS_BO1 | replicate_inferred | 256 | 64 | 1.43e+03 | 0.00635 | 0.00635 | 38.3 | 1.21 |
| 0.0001 | LHS_BO1 | fixed_sem_raw | 64 | 64 | 4.82 | 0.00103 | 0.0155 | 1.19 | 1.73 |
| 0.0001 | LHS_BO1 | fixed_sem_shrunk | 64 | 64 | 4.82 | 0.0013 | 0.00856 | 1.21 | 1.3 |
| 0.0001 | LHS_BO1_BO2 | replicate_inferred | 384 | 64 | 1.11e+03 | 0.0151 | 0.0151 | 63.9 | 0.618 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_raw | 64 | 64 | 4.81 | 0.0016 | 0.0144 | 1.54 | 0.886 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_shrunk | 64 | 64 | 4.81 | 0.00192 | 0.0083 | 1.69 | 0.943 |

## Condition LOO Calibration

| HRP | train_combo | model | loo_z_std | loo_abs_z_gt_2_pct | loo_abs_z_gt_3_pct | loo_mae | loo_max_abs_error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LHS_BO1 | replicate_inferred | 1.03 | 6.25 | 0 | 1.3e+03 | 4.09e+03 |
| 1 | LHS_BO1 | fixed_sem_raw | 1.05 | 6.25 | 0 | 1.29e+03 | 4.23e+03 |
| 1 | LHS_BO1 | fixed_sem_shrunk | 1.05 | 6.25 | 0 | 1.3e+03 | 4.17e+03 |
| 1 | LHS_BO1_BO2 | replicate_inferred | 1.4 | 4.69 | 3.12 | 996 | 5.18e+03 |
| 1 | LHS_BO1_BO2 | fixed_sem_raw | 1.14 | 4.69 | 1.56 | 924 | 5.66e+03 |
| 1 | LHS_BO1_BO2 | fixed_sem_shrunk | 1.2 | 6.25 | 1.56 | 947 | 5.55e+03 |
| 0.01 | LHS_BO1 | replicate_inferred | 1.24 | 12.5 | 6.25 | 1.57e+03 | 7.02e+03 |
| 0.01 | LHS_BO1 | fixed_sem_raw | 1.22 | 12.5 | 6.25 | 1.55e+03 | 7.04e+03 |
| 0.01 | LHS_BO1 | fixed_sem_shrunk | 1.22 | 12.5 | 6.25 | 1.56e+03 | 7.03e+03 |
| 0.01 | LHS_BO1_BO2 | replicate_inferred | 1.08 | 10.9 | 3.12 | 1.22e+03 | 6.49e+03 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_raw | 1.23 | 9.38 | 4.69 | 1.17e+03 | 6.51e+03 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_shrunk | 1.21 | 9.38 | 3.12 | 1.17e+03 | 6.43e+03 |
| 0.0001 | LHS_BO1 | replicate_inferred | 1.22 | 6.25 | 1.56 | 727 | 2.37e+03 |
| 0.0001 | LHS_BO1 | fixed_sem_raw | 1.21 | 6.25 | 1.56 | 727 | 2.37e+03 |
| 0.0001 | LHS_BO1 | fixed_sem_shrunk | 1.21 | 6.25 | 1.56 | 727 | 2.37e+03 |
| 0.0001 | LHS_BO1_BO2 | replicate_inferred | 1.28 | 7.81 | 1.56 | 763 | 2.39e+03 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_raw | 1.21 | 6.25 | 1.56 | 764 | 2.39e+03 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_shrunk | 1.21 | 6.25 | 1.56 | 764 | 2.39e+03 |

## BO2 Holdout Validation for LHS+BO1 Models

| HRP | model | bo2_holdout_bias_actual_minus_pred | bo2_holdout_mae | bo2_holdout_z_std | bo2_holdout_abs_z_gt_2_pct | bo2_holdout_abs_z_gt_3_pct |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | replicate_inferred | -998 | 2.18e+03 | 12.7 | 68.8 | 56.2 |
| 1 | fixed_sem_raw | -940 | 2.13e+03 | 14.7 | 87.5 | 78.1 |
| 1 | fixed_sem_shrunk | -968 | 2.15e+03 | 12.2 | 68.8 | 59.4 |
| 0.01 | replicate_inferred | -717 | 1.75e+03 | 12.2 | 56.2 | 37.5 |
| 0.01 | fixed_sem_raw | -676 | 1.71e+03 | 9.73 | 84.4 | 81.2 |
| 0.01 | fixed_sem_shrunk | -696 | 1.73e+03 | 8.58 | 62.5 | 46.9 |
| 0.0001 | replicate_inferred | -126 | 140 | 2.91 | 78.1 | 65.6 |
| 0.0001 | fixed_sem_raw | -125 | 140 | 4.66 | 71.9 | 71.9 |
| 0.0001 | fixed_sem_shrunk | -125 | 140 | 3.19 | 75 | 71.9 |

## Acquisition Stability and Candidate Distribution

| HRP | train_combo | model | acq_success | acq_elapsed_sec | acq_n_warnings | candidate_pred_max | candidate_latent_std_mean | candidate_nearest_train_logdist_median | candidate_pairwise_logdist_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | LHS_BO1 | replicate_inferred | True | 1.63 | 2 | 6.2e+03 | 1.7e+03 | 0.17 | 1.6 |
| 1 | LHS_BO1 | fixed_sem_raw | True | 42.9 | 0 | 6.35e+03 | 1.54e+03 | 0.0903 | 1.69 |
| 1 | LHS_BO1 | fixed_sem_shrunk | True | 42.9 | 0 | 6.49e+03 | 1.55e+03 | 0.125 | 1.83 |
| 1 | LHS_BO1_BO2 | replicate_inferred | True | 52.4 | 1 | 5.67e+03 | 1.64e+03 | 0.133 | 1.74 |
| 1 | LHS_BO1_BO2 | fixed_sem_raw | True | 78 | 1 | 6.23e+03 | 1.16e+03 | 0.166 | 1.71 |
| 1 | LHS_BO1_BO2 | fixed_sem_shrunk | True | 38 | 0 | 5.85e+03 | 1.2e+03 | 0.167 | 1.76 |
| 0.01 | LHS_BO1 | replicate_inferred | True | 46.4 | 7 | 8.07e+03 | 1.94e+03 | 0.144 | 1.74 |
| 0.01 | LHS_BO1 | fixed_sem_raw | True | 39.4 | 0 | 7.88e+03 | 1.99e+03 | 0.181 | 1.6 |
| 0.01 | LHS_BO1 | fixed_sem_shrunk | True | 38.9 | 0 | 7.93e+03 | 1.99e+03 | 0.178 | 1.6 |
| 0.01 | LHS_BO1_BO2 | replicate_inferred | True | 53 | 1 | 6.15e+03 | 1.87e+03 | 0.233 | 1.63 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_raw | True | 59.3 | 2 | 7.25e+03 | 1.41e+03 | 0.175 | 1.6 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_shrunk | True | 36.4 | 0 | 6.62e+03 | 1.45e+03 | 0.167 | 1.74 |
| 0.0001 | LHS_BO1 | replicate_inferred | True | 1.09 | 9 | 1.3e+03 | 706 | 0.186 | 1.51 |
| 0.0001 | LHS_BO1 | fixed_sem_raw | True | 43.1 | 2 | 1.66e+03 | 665 | 0.0842 | 1.57 |
| 0.0001 | LHS_BO1 | fixed_sem_shrunk | True | 22.3 | 0 | 1.66e+03 | 666 | 0.0772 | 1.43 |
| 0.0001 | LHS_BO1_BO2 | replicate_inferred | True | 34.2 | 2 | 1.68e+03 | 638 | 0.064 | 1.4 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_raw | True | 19.9 | 0 | 1.65e+03 | 692 | 0.0721 | 1.51 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_shrunk | True | 21.4 | 0 | 1.66e+03 | 693 | 0.0805 | 1.42 |

## Top Generated Candidates

| HRP | train_combo | model | TMB | H2O2 | pred_mean | latent_std | nearest_train_logdist |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0001 | LHS_BO1 | fixed_sem_raw | 0.204 | 0.359 | 1.66e+03 | 522 | 0.0273 |
| 0.0001 | LHS_BO1 | fixed_sem_raw | 0.163 | 0.00455 | 1.54e+03 | 560 | 0.0339 |
| 0.0001 | LHS_BO1 | fixed_sem_raw | 0.142 | 0.00496 | 1.51e+03 | 574 | 0.0369 |
| 0.0001 | LHS_BO1 | fixed_sem_shrunk | 0.204 | 0.359 | 1.66e+03 | 520 | 0.027 |
| 0.0001 | LHS_BO1 | fixed_sem_shrunk | 0.152 | 0.00515 | 1.52e+03 | 567 | 0.0353 |
| 0.0001 | LHS_BO1 | fixed_sem_shrunk | 0.164 | 0.00453 | 1.52e+03 | 572 | 0.0364 |
| 0.0001 | LHS_BO1 | replicate_inferred | 0.157 | 0.00411 | 1.3e+03 | 654 | 0.0639 |
| 0.0001 | LHS_BO1 | replicate_inferred | 0.132 | 0.0376 | 1.29e+03 | 585 | 0.0399 |
| 0.0001 | LHS_BO1 | replicate_inferred | 0.0236 | 0.206 | 1.21e+03 | 641 | 0.058 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_raw | 0.204 | 0.359 | 1.65e+03 | 542 | 0.0271 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_raw | 0.164 | 0.00456 | 1.52e+03 | 591 | 0.0355 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_raw | 0.141 | 0.00481 | 1.51e+03 | 593 | 0.0359 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.204 | 0.359 | 1.66e+03 | 537 | 0.0264 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.156 | 0.00438 | 1.51e+03 | 591 | 0.0356 |
| 0.0001 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.141 | 0.00487 | 1.51e+03 | 592 | 0.0357 |
| 0.0001 | LHS_BO1_BO2 | replicate_inferred | 0.203 | 0.359 | 1.68e+03 | 497 | 0.025 |
| 0.0001 | LHS_BO1_BO2 | replicate_inferred | 0.155 | 0.00441 | 1.54e+03 | 544 | 0.0328 |
| 0.0001 | LHS_BO1_BO2 | replicate_inferred | 0.143 | 0.00497 | 1.52e+03 | 551 | 0.0341 |
| 0.01 | LHS_BO1 | fixed_sem_raw | 0.0017 | 0.997 | 7.88e+03 | 958 | 0.00721 |
| 0.01 | LHS_BO1 | fixed_sem_raw | 0.00165 | 1 | 7.87e+03 | 963 | 0.00731 |
| 0.01 | LHS_BO1 | fixed_sem_raw | 0.00167 | 0.974 | 7.85e+03 | 970 | 0.00745 |
| 0.01 | LHS_BO1 | fixed_sem_shrunk | 0.0017 | 0.997 | 7.93e+03 | 942 | 0.00719 |
| 0.01 | LHS_BO1 | fixed_sem_shrunk | 0.00165 | 1 | 7.9e+03 | 960 | 0.00752 |
| 0.01 | LHS_BO1 | fixed_sem_shrunk | 0.00168 | 0.974 | 7.89e+03 | 960 | 0.00752 |
| 0.01 | LHS_BO1 | replicate_inferred | 0.954 | 0.525 | 8.07e+03 | 909 | 0.00675 |
| 0.01 | LHS_BO1 | replicate_inferred | 0.931 | 0.518 | 8.02e+03 | 944 | 0.00738 |
| 0.01 | LHS_BO1 | replicate_inferred | 0.956 | 0.51 | 7.95e+03 | 994 | 0.0083 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_raw | 0.932 | 0.0166 | 7.25e+03 | 324 | 0.00134 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_raw | 0.923 | 0.0166 | 7.16e+03 | 444 | 0.00285 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_raw | 0.928 | 0.0165 | 7.11e+03 | 503 | 0.00379 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.93 | 0.0165 | 6.62e+03 | 642 | 0.00441 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.92 | 0.0166 | 6.6e+03 | 653 | 0.00469 |
| 0.01 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.00624 | 0.00478 | 4.98e+03 | 922 | 0.0106 |
| 0.01 | LHS_BO1_BO2 | replicate_inferred | 0.914 | 0.0165 | 6.15e+03 | 1.06e+03 | 0.00764 |
| 0.01 | LHS_BO1_BO2 | replicate_inferred | 0.934 | 0.0163 | 6.07e+03 | 1.09e+03 | 0.00867 |
| 0.01 | LHS_BO1_BO2 | replicate_inferred | 0.9 | 0.0161 | 5.52e+03 | 1.4e+03 | 0.0194 |
| 1 | LHS_BO1 | fixed_sem_raw | 0.863 | 0.501 | 6.35e+03 | 822 | 0.00871 |
| 1 | LHS_BO1 | fixed_sem_raw | 0.897 | 0.492 | 6.12e+03 | 1.01e+03 | 0.0178 |
| 1 | LHS_BO1 | fixed_sem_raw | 0.87 | 0.483 | 5.97e+03 | 1.13e+03 | 0.022 |
| 1 | LHS_BO1 | fixed_sem_shrunk | 0.867 | 0.5 | 6.49e+03 | 779 | 0.00786 |
| 1 | LHS_BO1 | fixed_sem_shrunk | 0.889 | 0.487 | 6.18e+03 | 1.06e+03 | 0.0203 |
| 1 | LHS_BO1 | fixed_sem_shrunk | 0.937 | 0.462 | 5.94e+03 | 1.13e+03 | 0.0292 |
| 1 | LHS_BO1 | replicate_inferred | 0.896 | 0.482 | 6.2e+03 | 1.13e+03 | 0.0257 |
| 1 | LHS_BO1 | replicate_inferred | 0.024 | 0.00141 | 3.83e+03 | 1.46e+03 | 0.0503 |
| 1 | LHS_BO1 | replicate_inferred | 0.503 | 0.185 | 2.67e+03 | 1.69e+03 | 0.12 |
| 1 | LHS_BO1_BO2 | fixed_sem_raw | 0.876 | 0.505 | 6.23e+03 | 469 | 0.00262 |
| 1 | LHS_BO1_BO2 | fixed_sem_raw | 0.872 | 0.512 | 6.18e+03 | 490 | 0.00314 |
| 1 | LHS_BO1_BO2 | fixed_sem_raw | 0.872 | 0.499 | 6e+03 | 599 | 0.008 |
| 1 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.871 | 0.511 | 5.85e+03 | 565 | 0.00313 |
| 1 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.877 | 0.503 | 5.82e+03 | 578 | 0.00438 |
| 1 | LHS_BO1_BO2 | fixed_sem_shrunk | 0.864 | 0.5 | 5.64e+03 | 684 | 0.00863 |
| 1 | LHS_BO1_BO2 | replicate_inferred | 0.867 | 0.503 | 5.67e+03 | 882 | 0.00616 |
| 1 | LHS_BO1_BO2 | replicate_inferred | 0.881 | 0.489 | 5.54e+03 | 949 | 0.0166 |
| 1 | LHS_BO1_BO2 | replicate_inferred | 0.865 | 0.521 | 5.43e+03 | 933 | 0.0117 |

## Interpretation

- Fixed-noise GP improves matrix conditioning because repeated observations are collapsed to condition means and the likelihood uses condition-specific diagonal noise.
- Raw SEM is unstable with only four replicates for LHS-only conditions; the shrunk SEM variant is the more defensible fixed-noise baseline.
- BO2 holdout errors mainly measure run-to-run / temporal drift at the BO1-selected conditions. Fixed noise cannot fully correct this drift because it is not a batch-effect model.
- If future BO will use candidates generated from Apr29-like data, use the shrunk fixed-noise GP or add a learned extra global noise term on top of SEM; do not rely on raw SEM alone.

## Detailed Conclusions

### 1. What Apr 29 validates, and what it does not

BO1 and BO2 have exactly the same 32 conditions at each HRP level. Therefore, `LHS+BO1 -> BO2` is not a test of whether the model can extrapolate to a new batch of BO candidates. It is a test of whether the model trained on LHS plus the first measurement of the BO-selected conditions can predict the second measurement of those same BO-selected conditions.

This matters because the BO2 mean response is substantially lower than BO1 for HRP=1 and HRP=0.01, and the HRP=0.0001 BO responses are far below the LHS responses. A GP over only `(TMB, H2O2)` has no variable that can represent this run/batch drift. Fixed observation noise can soften the likelihood, but it cannot learn a systematic run effect.

### 2. LHS+BO1 fits are internally well-behaved, but BO2 holdout exposes drift

For `LHS+BO1`, condition-level LOO looks acceptable for HRP=1: z-score std is about 1.03-1.05 and no `|z|>3` points. HRP=0.01 and HRP=0.0001 are less well calibrated, with HRP=0.01 showing `|z|>3 = 6.25%` across all three model variants.

However, BO2 holdout calibration is very poor. For HRP=1, the BO2 actual mean is lower than prediction by about 940-998 AUC on average, and the BO2 z-score std is 12.2-14.7. For HRP=0.01, the bias is about -676 to -717 AUC and z-score std is 8.58-12.2. For HRP=0.0001, the absolute MAE is smaller, about 140 AUC, but the z-score std remains 2.91-4.66 because the estimated experimental noise is small.

Interpretation: the model family is not just missing heteroscedasticity; the Apr 29 sequence contains run-level / temporal nonstationarity that is outside the feature space.

### 3. Fixed-noise GP improves numerical conditioning

The largest numerical improvement is matrix conditioning. For `LHS+BO1`, current replicate-level GP has condition numbers of 149, 274, and 1430 for HRP=1, 0.01, and 0.0001. The shrunk fixed-noise GP reduces these to 28.1, 11.7, and 4.82.

For `LHS+BO1+BO2`, replicate-level GP condition numbers are 38.2, 41.2, and 1110. The shrunk fixed-noise GP reduces them to 7.52, 9.19, and 4.81. This is the strongest empirical argument for using condition-level fixed-noise modeling: it removes duplicate-X degeneracy and provides a sensible diagonal noise structure.

### 4. Raw SEM overfits low-noise conditions; shrinkage is safer

Raw SEM often gives very small standardized noise at low-variance conditions. For example, in `LHS+BO1`, fixed raw median standardized noise is `0.000355` for HRP=1 and `0.000214` for HRP=0.01, while max noise is two to three orders of magnitude larger. This creates extreme in-sample high/low residual ratios: 247 for HRP=1 and 227 for HRP=0.01.

Shrunk SEM keeps the heteroscedastic structure but avoids treating four-replicate variance estimates as exact. Its high/low residual ratios are much more reasonable: 5.5 for HRP=1 and 5.38 for HRP=0.01 in `LHS+BO1`, and 2.93 / 2.65 in `LHS+BO1+BO2`.

### 5. Candidate behavior changes most for HRP=0.01

For HRP=1, all model variants point to the same region: high TMB around 0.86-0.90 and H2O2 around 0.48-0.51. Adding BO2 lowers current replicate-GP predicted max from about 6200 to 5670; fixed shrunk is about 5850.

For HRP=0.01, `LHS+BO1` shows a major disagreement. Current replicate GP recommends high TMB / mid-high H2O2 around `(0.95, 0.52)`, while fixed-noise models recommend very low TMB / high H2O2 around `(0.0017, 1.0)`. After adding BO2, all models shift toward high TMB / low H2O2 around `(0.92-0.93, 0.0165)`. This means the BO2 repeat data materially changes the inferred response surface for HRP=0.01.

For HRP=0.0001, all variants suggest regions near TMB 0.14-0.20 and either very low H2O2 or H2O2 around 0.36. But this is also the layer with the strongest LHS-vs-BO distribution shift: LHS condition mean AUC averages 1423, while BO1 averages 101 and BO2 averages -22.9. Any recommendation at this HRP should be treated as fragile unless the run-level discrepancy is explained experimentally.

### 6. Acquisition optimization stability

All acquisition optimizations succeeded. Fixed shrunk had the cleanest warning profile: zero acquisition warnings in every `LHS+BO1` run and every `LHS+BO1+BO2` run. Current replicate GP produced warnings in several cases, especially HRP=0.01 `LHS+BO1` and HRP=0.0001 `LHS+BO1`.

The warning pattern supports the same conclusion as the matrix diagnostics: fixed-noise, especially shrunk fixed-noise, is numerically more stable for this dataset.

### 7. Recommendation

For Apr 29-like data, the best practical baseline is `fixed_sem_shrunk`, not raw fixed SEM and not replicate-level inferred global noise. It improves conditioning and acquisition stability while avoiding the worst raw-SEM overconfidence.

That said, the main remaining model misspecification is not solved by fixed noise: Apr 29 shows strong run/batch drift, especially between BO1 and BO2 and between LHS and BO rounds at HRP=0.0001. A more defensible next model would include either a run/batch covariate, a hierarchical run effect, or at minimum a learned extra global noise term on top of shrunk SEM.
