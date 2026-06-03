# -*- coding: utf-8 -*-
"""
Shrinkage / fixed-noise yvar comparison for Jun_03 ESS BO (read-only diagnostic).

User concern: the condition-mean fixed-noise GP may over-trust a per-condition
observation variance that is estimated from only n=4 replicates (an occasionally
tiny SEM becomes a hard constraint -> over-confident extrapolation).

This script compares several ways of setting train_Yvar for the condition-mean GP,
scored by condition-level leave-one-out (LOO) calibration:
  - z_std should be ~1.0   (>1 => over-confident, <1 => under-confident)
  - |z|>2 should be ~5%
LOO z-denominator is fixed to the raw per-condition SEM^2 (same convention as
fixed_noise_ess_bo.condition_loo), so only the *training* yvar varies across schemes.

It does NOT modify the first-round production artifacts. Outputs go to
logs/Jun_03_full_log/shrinkage_compare/.

Schemes (original AUC^2 scale; Standardize(m=1) rescales internally):
  raw_a0           : per-condition SEM^2 with the existing 5%-of-median micro-floor
                     (== what model 'fixed_sem_raw' uses, alpha=0)
  floor_p25/50/75  : max(SEM^2_unclipped, percentile_q) -- floor only: lifts only the
                     suspiciously small variances, leaves large (high-noise) ones intact
  shrunk_a0.25_med : 0.75*SEM^2_raw + 0.25*pooled, pooled = median(var)/median(count) (robust)
  shrunk_a0.5_mean : current production (alpha=0.5, pooled = mean(var, w=count-1)/median(count))
"""
from __future__ import annotations
import sys
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\PyCharm\Blueness")
warnings.filterwarnings("ignore")
import fixed_noise_ess_bo as fn  # noqa: E402

DAY = Path(r"C:\PyCharm\Blueness\logs\Jun_03_full_log")
OUT = DAY / "shrinkage_compare"
OUT.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cpu")
HRPS = ["1", "0.01", "0.0001"]
PROD_ALPHA = 0.5  # production shrink_alpha, used to reproduce the reference scheme


def build_schemes(cond: pd.DataFrame) -> dict[str, np.ndarray]:
    var = cond["var"].to_numpy(float)
    count = cond["count"].to_numpy(float)
    raw_unclipped = var / count
    sem_var_raw = cond["sem_var_raw"].to_numpy(float)        # clipped at 5%-of-median (production raw)
    sem_var_shrunk = cond["sem_var_shrunk"].to_numpy(float)  # production (alpha=0.5, mean pool)

    schemes: dict[str, np.ndarray] = {}
    schemes["raw_a0"] = sem_var_raw
    for q in (25, 50, 75):
        floor = float(np.percentile(raw_unclipped, q))
        schemes[f"floor_p{q}"] = np.maximum(raw_unclipped, floor)
    pooled_med = float(np.median(var) / np.median(count))
    schemes["shrunk_a0.25_med"] = 0.75 * sem_var_raw + 0.25 * pooled_med
    schemes["shrunk_a0.5_mean"] = sem_var_shrunk
    return schemes


def loo_calibration(cond: pd.DataFrame, yvar: np.ndarray) -> dict[str, float]:
    n = len(cond)
    y = cond["mean"].to_numpy(float)
    sem_var_raw = cond["sem_var_raw"].to_numpy(float)  # fixed z-denominator term
    bounds = fn.make_bounds(DEV)
    idx = np.arange(n)
    z = np.zeros(n)
    ae = np.zeros(n)
    for i in range(n):
        keep = idx != i
        tr = cond.iloc[idx[keep]]
        model = fn.fit_model(
            fn.tensor_x(tr, DEV),
            fn.tensor_y(tr["mean"], DEV),
            bounds,
            fn.tensor_y(yvar[keep], DEV),
        )
        with torch.no_grad():
            post = model.posterior(fn.tensor_x(cond.iloc[[i]], DEV), observation_noise=False)
        mu = float(post.mean.item())
        lv = float(post.variance.item())
        denom = math.sqrt(max(lv + sem_var_raw[i], 1e-12))
        z[i] = (y[i] - mu) / denom
        ae[i] = abs(y[i] - mu)
    return {
        "z_mean": float(z.mean()),
        "z_std": float(z.std(ddof=0)),
        "abs_z_gt2_pct": float((np.abs(z) > 2).mean() * 100),
        "abs_z_gt3_pct": float((np.abs(z) > 3).mean() * 100),
        "mae": float(ae.mean()),
    }


def full_fit_lengthscale(cond: pd.DataFrame, yvar: np.ndarray) -> tuple[float, float]:
    model = fn.fit_model(
        fn.tensor_x(cond, DEV),
        fn.tensor_y(cond["mean"], DEV),
        fn.make_bounds(DEV),
        fn.tensor_y(yvar, DEV),
    )
    return fn.lengthscales(model)


def main() -> None:
    rows: list[dict] = []
    for hrp in HRPS:
        fn.set_seeds(42)
        res = pd.read_excel(DAY / f"6_3_LHS_HRP_{hrp}_res.xlsx")
        cond = fn.prepare_condition_table(res, "AUC", PROD_ALPHA)
        span = float(cond["mean"].max() - cond["mean"].min())
        print(f"\n===== HRP={hrp}  n_cond={len(cond)}  signal_span={span:.0f} =====", flush=True)
        for name, yvar in build_schemes(cond).items():
            ls = full_fit_lengthscale(cond, yvar)
            cal = loo_calibration(cond, yvar)
            sem_sd = np.sqrt(yvar)
            rows.append({
                "HRP": hrp, "scheme": name,
                "sem_sd_min": float(sem_sd.min()),
                "sem_sd_med": float(np.median(sem_sd)),
                "sem_sd_max": float(sem_sd.max()),
                "ls_TMB": ls[0], "ls_H2O2": ls[1],
                **cal,
            })
            print(f"  {name:17s} sem_sd[min/med/max]={sem_sd.min():6.1f}/{np.median(sem_sd):6.1f}/{sem_sd.max():6.1f}"
                  f"  z_std={cal['z_std']:.2f}  |z|>2={cal['abs_z_gt2_pct']:4.1f}%  |z|>3={cal['abs_z_gt3_pct']:4.1f}%"
                  f"  MAE={cal['mae']:6.1f}  ls=({ls[0]:.2f},{ls[1]:.2f})", flush=True)

    df = pd.DataFrame(rows)
    out_csv = OUT / "yvar_calibration_compare.csv"
    df.to_csv(out_csv, index=False)

    print("\n##### z_std by scheme (target ~1.00) #####")
    print(df.pivot(index="scheme", columns="HRP", values="z_std").round(2).to_string())
    print("\n##### |z|>2 percent by scheme (target ~5%) #####")
    print(df.pivot(index="scheme", columns="HRP", values="abs_z_gt2_pct").round(1).to_string())
    print(f"\nSaved -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
