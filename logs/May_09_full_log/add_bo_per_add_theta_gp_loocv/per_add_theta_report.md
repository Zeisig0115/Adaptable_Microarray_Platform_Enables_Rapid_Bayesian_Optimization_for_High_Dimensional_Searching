# Per-additive Hamming theta experiment

## Scope

- New kernel: `PerAdditiveThetaKernel`, a `BaselineKernel` variant with `exp(-sum_i theta_i * mismatch_i)` instead of `exp(-theta * sum_i mismatch_i)`.
- Priors and initial values otherwise match the scalar-theta baseline: outputscale 1.0, essential lengthscales 0.35, concentration lengthscales 0.3, theta/theta_i 0.5, LogNormal prior sigma 0.75.
- Candidate generation and acquisition-stability diagnostics were skipped; this is a GP fit / fixed-hyperparameter LOO diagnostic experiment.
- Previous comparison source: `logs/May_09_full_log/add_bo_multiseed_gp_loocv/all_seed_model_diagnostics.csv`.

## Data

- Filtered rows: 461.
- Unique encoded rows: 446.
- Duplicates merged: 15.
- Encoded dimensions: 46; additives: 22.
- Active-count distribution: {1: 198, 2: 231, 0: 32}.

## Per-additive theta result

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| per_add_theta_kernel | -0.463942 | 0.454045 | 1.01812 | 1970.94 | 0.0173518 | 1005.07 |

## Comparison with previous kernels

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| old_mixed_hamming | -0.595856 | 0.331417 | 1.03589 | 2907.68 | 0.0246446 | 34615.4 |
| new_additive_set | -0.164145 | 0.271278 | 0.987908 | 1583.73 | 0.0109057 | 15827.1 |
| baseline_kernel | -0.4681 | 0.46766 | 1.05255 | 1822.61 | 0.0144198 | 878.714 |
| per_add_theta_kernel | -0.463942 | 0.454045 | 1.01812 | 1970.94 | 0.0173518 | 1005.07 |

## Cautions

- The added theta_i parameters increase categorical flexibility from 1 to 22 parameters; this is weakly supported by a dataset containing only 0/1/2-active-additive observations.
- LOO RMSE is in standardized model space, matching `add_bo.py`.
- These diagnostics do not validate extrapolation to 3/4-additive recipes.
