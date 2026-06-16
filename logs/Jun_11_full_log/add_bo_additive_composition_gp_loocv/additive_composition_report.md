# Essentials/additive additive-composition experiment (Exp1)

## Scope

- Records EXPERIMENT 1: replace the BaselineKernel multiplicative essentials
  gate with an ADDITIVE essentials/additive split.
- `baseline_kernel` (multiplicative gate): `of * k_ess * (c0 + s_main * k_main)`.
- `add_composition` (Exp1): `of * (k_ess + s_add * (c0 + s_main * k_main))`.
  Subclasses BaselineKernel; adds ONE scalar `s_add`; identical priors/lengthscales.
- `per_add_alpha` (current production default) and `baseline_set` (Exp2) are run
  and tabulated for reference only; Exp2 is handled in a separate discussion.
- GP fit + fixed-hyperparameter LOO in standardized model space only; candidate
  generation and acquisition stability are intentionally skipped.
- Diagnostics are deterministic w.r.t. seed (MAP fit from fixed init); seeds
  reproduce the format, not independent samples.
- `k_main = sum_j b_j b'_j exp(-0.5((c_j-c'_j)/ell_conc_j)^2)`; `of`, `c0`,
  `s_main`, `s_add` are partially scale-redundant (kept explicit).

## Data

- Filtered rows: 461.
- Unique encoded rows: 446.
- Duplicates merged: 15.
- Encoded dimensions: 46.
- Additives: 22.
- Active-count distribution: {1: 198, 2: 231, 0: 32}.

## Result

| model | n_params | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number | kernel diag max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_kernel | 27 | -0.313154 | 0.289985 | 0.980001 | 6474.5 | 0.0725509 | 3938.07 | 0.970386 |
| per_add_alpha | 49 | -0.108224 | 0.270971 | 1.0041 | 1760.76 | 0.0127609 | 12282.1 | 1.3174 |
| add_composition | 28 | -0.310216 | 0.29121 | 0.981494 | 6480.99 | 0.0725729 | 6284.86 | 1.36833 |
| baseline_set | 28 | -0.15097 | 0.272504 | 1.0023 | 1696.57 | 0.0120639 | 16251.1 | 0.973631 |

## Fitted scalar hyperparameters (seed 0)

- baseline_kernel : of=1.103, c0=0.6139, s_main=0.133.
- add_composition : of=0.9784, c0=0.182, s_main=0.2338, s_add=0.6134.
- baseline_set    : of=0.9835, c0=0.4337, s_main=0.1172, s_set=0.322.

## Interpretation (Exp1)

- The additive split does NOT improve the measured fit: `add_composition` matches
  `baseline_kernel` on MLL and is marginally worse on LOO RMSE, with essentially
  unchanged noise and train RMSE. The learned `s_add` is O(1), i.e. the optimizer
  finds no reason to prefer the additive form here.
- First-principles reason: in this DOE the essentials are swept ONLY when all
  additives are off (32 rows), and all additive chemistry was measured at a single
  fixed essentials point (tmb=1.0, h2o2=0.01). The two data slices are disjoint, so
  within each slice the multiplicative gate and the additive split are nearly
  equivalent; they differ only in the UNOBSERVED joint region (additives present
  AND essentials != (1.0, 0.01)), which LOO/MLL on this dataset cannot see.
- Consequence: this dataset cannot adjudicate the essentials x additive coupling
  form. The 'no measured gain' here does not by itself reject additive composition
  for BO extrapolation honesty; it only shows the choice is unidentified from data.

## Historical reference (prior Jun_11 logs, separate runs)

| model | MLL | LOO RMSE | noise | condition number |
|---|---:|---:|---:|---:|
| old_mixed_hamming | -0.595856 | 0.331417 | 0.0246446 | 34615.4 |
| additive_block_kernel (== current BaselineKernel) | -0.313154 | 0.289985 | 0.0725509 | 3938.07 |
| per_add_alpha_kernel (+set, default) | -0.108224 | 0.270971 | 0.0127609 | 12282.1 |
| new_additive_set | -0.164145 | 0.271278 | 0.0109057 | 15827.1 |

## Cautions

- This run checks GP fit and fixed-hyperparameter LOO only; it does not validate
  acquisition behavior or 3/4-additive extrapolation.
- `baseline_set` (Exp2) numbers are recorded here for reference; their
  interpretation (why a single Tanimoto set term helps so much) is intentionally
  deferred to a separate discussion.
- Seeds run: 3.