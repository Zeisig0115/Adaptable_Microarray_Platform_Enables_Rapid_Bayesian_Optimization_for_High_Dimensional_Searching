"""Diagnostic comparison of Matern smoothness (nu) for the fixed_noise_ess_bo route.

This is a DIAGNOSTIC ONLY. It does not modify the committed pipeline, and it does
not affect the candidates already sent to the lab (those were generated with the
shrunk noise model and nu=0.5). It only informs the nu choice for future rounds.

Method: reuse the pipeline's own functions verbatim (data load, condition table,
fixed-noise train tensors, MAP fit, and condition-level grouped LOO). The ONLY thing
varied across runs is the Matern nu. This is done by overriding the module-global
name `matern_with_hvarfner_prior` that fixed_noise_ess_bo.fit_model looks up, so every
other line (preprocessing, prior, fixed Yvar = sem_var_raw, LOO z-denominator) stays
byte-for-byte identical between nu settings.

Metric:
  - LOO z_std closest to 1.0  -> best-calibrated (z_std>1 overconfident, <1 underconfident)
  - LOO MAE / max|err|        -> accuracy on original AUC scale (compare within an HRP only)
  - %|z|>2 near 5%, %|z|>3 near 0.3% -> tail calibration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import fit_model as fm            # noqa: E402
import fixed_noise_ess_bo as fn   # noqa: E402

torch.set_default_dtype(torch.double)

DATA_DIR = REPO / "logs" / "Jun_03_full_log"
HRPS = ["1", "0.01", "0.0001"]
NUS = [0.5, 1.5, 2.5]
TARGET = "AUC"
MODEL = "fixed_sem_raw"
DEVICE = torch.device("cpu")
SEED = 42


def patch_nu(target_nu: float) -> None:
    # fixed_noise_ess_bo.fit_model calls matern_with_hvarfner_prior(d, nu=0.5);
    # override that name so the forced nu wins while everything else is unchanged.
    # The override must accept (and ignore) the nu kwarg the pipeline passes.
    fn.matern_with_hvarfner_prior = lambda d, nu=0.5: fm.matern_with_hvarfner_prior(d, nu=target_nu)


def main() -> None:
    bounds = fn.make_bounds(DEVICE)
    rows = []
    for hrp in HRPS:
        df = pd.read_excel(DATA_DIR / f"6_3_LHS_HRP_{hrp}_res.xlsx")
        cond = fn.prepare_condition_table(df, TARGET, 0.0)  # alpha irrelevant: raw uses sem_var_raw
        print(f"\n================  HRP = {hrp}   (n_conditions={cond.shape[0]}, reps=4)  ================")
        print(f"{'nu':>5} | {'ls_TMB':>7} {'ls_H2O2':>7} | {'z_mean':>7} {'z_std':>6} "
              f"| {'|z|>2%':>6} {'|z|>3%':>6} | {'MAE':>8} {'max|err|':>9}")
        print("-" * 84)
        for nu in NUS:
            fn.set_seeds(SEED)
            patch_nu(nu)
            tx, ty, tyv, _ = fn.model_train_data(df, cond, TARGET, MODEL, DEVICE)
            model = fn.fit_model(tx, ty, bounds, tyv)
            ls_t, ls_h = fn.lengthscales(model)
            s = fn.condition_loo(df, cond, TARGET, MODEL, bounds, DEVICE)
            rows.append({"HRP": hrp, "nu": nu, "ls_TMB": ls_t, "ls_H2O2": ls_h, **s})
            print(f"{nu:>5.1f} | {ls_t:>7.3f} {ls_h:>7.3f} | "
                  f"{s['condition_loo_z_mean']:>7.3f} {s['condition_loo_z_std']:>6.3f} | "
                  f"{s['condition_loo_abs_z_gt_2_pct']:>6.2f} {s['condition_loo_abs_z_gt_3_pct']:>6.2f} | "
                  f"{s['condition_loo_mae']:>8.1f} {s['condition_loo_max_abs_error']:>9.1f}")

    out = pd.DataFrame(rows)
    out_path = Path(__file__).with_name("nu_compare_loo.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
