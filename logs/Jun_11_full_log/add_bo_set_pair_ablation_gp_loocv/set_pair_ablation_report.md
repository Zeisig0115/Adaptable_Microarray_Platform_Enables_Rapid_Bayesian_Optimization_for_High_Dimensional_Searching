# Set/pair ablation over the per-additive-amplitude baseline

## Scope

- Base kernel: `per_add_alpha` = `sigma_f^2 * k_ess * (c0 + sum_j alpha_j b_j b'_j k_conc_j)`.
- 1a `per_add_alpha_set`: base + `s_set * Tanimoto(b,b')`.
- 1b `per_add_alpha_pair`: base + `s_pair * k_pair`, `k_pair = sum_{i<j} m_i m_j` (closed form).
- `per_add_alpha_set_pair`: base + both terms (folded-equivalent of AdditiveSetKernel).
- Set/pair terms reuse AdditiveSetKernel's exact closed forms and scale priors; each ablation adds one global scale.
- GP fit + fixed-hyperparameter LOO in standardized model space only; candidates / acquisition stability skipped.
- Diagnostics are deterministic w.r.t. seed (MAP fit from fixed init); seeds reproduce the format, not independent samples.

## Data

- Filtered rows: 461.
- Unique encoded rows: 446.
- Duplicates merged: 15.
- Encoded dimensions: 46.
- Additives: 22.
- Active-count distribution: {'1': 198, '2': 231, '0': 32}.

## Decomposition result

| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |
|---|---:|---:|---:|---:|---:|---:|
| baseline_kernel | -0.313154 | 0.289985 | 0.980001 | 6474.5 | 0.0725509 | 3938.07 |
| per_add_alpha | -0.261321 | 0.28758 | 0.980426 | 6448.19 | 0.0715065 | 3283.72 |
| per_add_alpha_set | -0.108224 | 0.270971 | 1.0041 | 1760.76 | 0.0127609 | 12282.1 |
| per_add_alpha_pair | -0.175803 | 0.299386 | 1.00772 | 2836.48 | 0.0250273 | 10300.3 |
| per_add_alpha_set_pair | -0.108604 | 0.270895 | 1.00501 | 1858.47 | 0.0137729 | 11888.2 |
| new_additive_set | -0.164145 | 0.271278 | 0.987908 | 1583.73 | 0.0109057 | 15827.1 |

## Historical reference (prior Jun_11 logs, separate runs)

| model | MLL | LOO RMSE | noise | condition number |
|---|---:|---:|---:|---:|
| old_mixed_hamming | -0.595856 | 0.331417 | 0.0246446 | 34615.4 |
| additive_block_kernel (== current BaselineKernel) | -0.313154 | 0.289985 | 0.0725509 | 3938.07 |
| per_add_alpha_kernel | -0.261321 | 0.28758 | 0.0715065 | 3283.72 |
| new_additive_set | -0.164145 | 0.271278 | 0.0109057 | 15827.1 |

## Cautions

- This run checks GP fit and fixed-hyperparameter LOO only; it does not validate acquisition behavior or 3/4-additive extrapolation.
- The pair term is identically 0 unless two recipes share >=2 active additives; with only 0/1/2-active data its off-diagonal support is nearly empty.
- The set/pair terms raise the covariance condition number relative to the per_add_alpha base.
- Seeds run: 10.