# Per-additive alpha experiment

## Scope

- New kernel: `PerAdditiveAlphaKernel`, with `outputscale * k_ess * (c0 + sum_j alpha_j b_j b'_j k_conc_j)`.
- This changes only the single current-baseline main_scale into 22 alpha_j values.
- Candidate generation and acquisition-stability diagnostics were skipped.
- LOO diagnostics are fixed-hyperparameter LOO in standardized model space.
- The outputscale, c0, and alpha_j terms are partially scale-redundant.
- Comparison summary source: `logs/Jun_11_full_log/add_bo_additive_block_gp_loocv/comparison_metric_summary.csv`.

## Data

- Filtered rows: 461.
- Unique encoded rows: 446.
- Duplicates merged: 15.
- Encoded dimensions: 46.
- Additives: 22.
- Active-count distribution: {1: 198, 2: 231, 0: 32}.

## Per-additive alpha result

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| per_add_alpha_kernel | -0.261321 | 0.28758 | 0.980426 | 6448.19 | 0.0715065 | 3283.72 |

## Comparison

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| old_mixed_hamming | -0.595856 | 0.331417 | 1.03589 | 2907.68 | 0.0246446 | 34615.4 |
| new_additive_set | -0.164145 | 0.271278 | 0.987908 | 1583.73 | 0.0109057 | 15827.1 |
| baseline_kernel | -0.4681 | 0.46766 | 1.05255 | 1822.61 | 0.0144198 | 878.714 |
| per_add_theta_kernel | -0.463942 | 0.454045 | 1.01812 | 1970.94 | 0.0173518 | 1005.07 |
| additive_block_kernel | -0.313154 | 0.289985 | 0.980001 | 6474.5 | 0.0725509 | 3938.07 |
| per_add_alpha_kernel | -0.261321 | 0.28758 | 0.980426 | 6448.19 | 0.0715065 | 3283.72 |

## Cautions

- This run checks GP fit and fixed-hyperparameter LOO only.
- It does not validate acquisition behavior or 3/4-additive extrapolation.
- The 22 alpha_j values add flexibility over data containing only 0/1/2-active-additive observations.
