# Additive-block kernel experiment

## Scope

- New kernel: `AdditiveBlockKernel`, with `outputscale * k_ess * (c0 + main_scale * sum_j b_j b'_j k_conc_j)`.
- Candidate generation and acquisition-stability diagnostics were skipped.
- LOO diagnostics are fixed-hyperparameter LOO in standardized model space.
- The outputscale, c0, and main_scale terms are partially scale-redundant.
- Historical comparison source: `logs/May_09_full_log/add_bo_per_add_theta_gp_loocv/comparison_all_seed_diagnostics.csv`.

## Data

- Filtered rows: 461.
- Unique encoded rows: 446.
- Duplicates merged: 15.
- Encoded dimensions: 46.
- Additives: 22.
- Active-count distribution: {1: 198, 2: 231, 0: 32}.

## Additive-block result

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| additive_block_kernel | -0.313154 | 0.289985 | 0.980001 | 6474.5 | 0.0725509 | 3938.07 |

## Comparison

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| old_mixed_hamming | -0.595856 | 0.331417 | 1.03589 | 2907.68 | 0.0246446 | 34615.4 |
| new_additive_set | -0.164145 | 0.271278 | 0.987908 | 1583.73 | 0.0109057 | 15827.1 |
| baseline_kernel | -0.4681 | 0.46766 | 1.05255 | 1822.61 | 0.0144198 | 878.714 |
| per_add_theta_kernel | -0.463942 | 0.454045 | 1.01812 | 1970.94 | 0.0173518 | 1005.07 |
| additive_block_kernel | -0.313154 | 0.289985 | 0.980001 | 6474.5 | 0.0725509 | 3938.07 |

## Cautions

- This run checks GP fit and fixed-hyperparameter LOO only.
- It does not validate acquisition behavior or 3/4-additive extrapolation.
- The kernel diagonal changes with active additive count through the main-effect sum.
