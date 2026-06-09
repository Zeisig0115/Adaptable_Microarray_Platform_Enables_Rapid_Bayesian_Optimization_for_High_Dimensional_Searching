"""Leave-one-condition-out (LOCO) kernel-smoothness comparison for the essentials
fixed-noise GP, run on the LHS data.

Goal
----
Decide the Matern smoothness (nu in {0.5, 1.5, 2.5}) vs RBF for the essentials
fixed-noise workflow, per HRP level, using a replicate-aware, leakage-free
diagnostic instead of in-sample marginal likelihood.

Methodology (mirrors `fixed_noise_ess_bo.condition_loo` for model='fixed_sem_shrunk')
------------------------------------------------------------------------------------
- Group replicates into conditions via `prepare_condition_table` (shrink_alpha=0.5).
- For each condition i: drop it from the TRAINING condition table, refit the GP on
  the remaining n-1 condition means with FixedNoise = sem_var_shrunk, then predict
  the latent mean/var at the held-out condition's (TMB, H2O2). No replicate of the
  held-out condition is ever in the training set -> no leakage.
- Calibration uses the held-out condition's RAW SEM variance as observation noise:
  denom^2 = latent_var + sem_var_raw   (identical to condition_loo).
- All four kernels share the SAME Hvarfner dimension-scaled LogNormal lengthscale
  prior and the same 0.025 lengthscale floor; only the kernel class / smoothness
  differs. This matches botorch.get_covar_module_with_dim_scaled_prior, where RBF
  and Matern differ only in the base-kernel class.

Metrics per (HRP, kernel)
-------------------------
- z_mean (bias, ~0 ideal), z_std (calibration, ~1 ideal; >1 over-confident,
  <1 under-confident), %|z|>2, %|z|>3
- MAE, RMSE in AUC units
- mean NLPD (Gaussian negative log predictive density; proper scoring rule,
  lower=better) as the primary ranking metric, with a PAIRED comparison vs the
  best kernel (same folds) to judge significance
- mean fitted ARD lengthscale (normalized space) for interpretability / steepness
- a BO-relevant view: the same metrics restricted to the top-25% (high-AUC) conditions

This script is read-only: it does not modify any project file or data. Reproducible
(fixed seed). Delete it freely; it is a diagnostic, not part of the workflow.
"""
from __future__ import annotations

import math
import warnings
from math import log, pi, sqrt

import numpy as np
import pandas as pd
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

import fixed_noise_ess_bo as M

torch.set_default_dtype(torch.double)
warnings.filterwarnings("ignore")

SQRT2 = sqrt(2.0)
SQRT3 = sqrt(3.0)
HRPS = ["1", "0.01", "0.0001"]
DEVICE = torch.device("cpu")
TOP_FRAC = 0.25  # high-AUC subset for the BO-relevant view


def make_kernel(kind: str, d: int):
    """Hvarfner dimension-scaled prior + 0.025 floor for every kernel; only the
    smoothness (kernel class / nu) varies. Mirrors matern_with_hvarfner_prior and
    botorch.get_covar_module_with_dim_scaled_prior."""
    ls_prior = LogNormalPrior(loc=SQRT2 + log(d) * 0.5, scale=SQRT3)
    constraint = GreaterThan(2.5e-2, transform=None, initial_value=ls_prior.mode)
    if kind == "rbf":
        return RBFKernel(
            ard_num_dims=d, lengthscale_prior=ls_prior, lengthscale_constraint=constraint
        )
    nu = {"matern05": 0.5, "matern15": 1.5, "matern25": 2.5}[kind]
    return MaternKernel(
        nu=nu, ard_num_dims=d, lengthscale_prior=ls_prior, lengthscale_constraint=constraint
    )


def fit_with_kernel(train_x, train_y, bounds, train_yvar, kernel):
    """Identical to fixed_noise_ess_bo.fit_model except the kernel is injected."""
    d = train_x.shape[-1]
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        covar_module=kernel,
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def loco_for_kernel(cond: pd.DataFrame, bounds, kind: str):
    """Returns a per-condition DataFrame of diagnostics for one kernel, aligned to
    the condition-table index, plus the count of failed folds."""
    d = len(M.ESSENTIALS)
    recs: dict[int, dict] = {}
    fails = 0
    for i, row in cond.iterrows():
        train_cond = cond.drop(index=i).reset_index(drop=True)
        train_x = M.tensor_x(train_cond, DEVICE)
        train_y = M.tensor_y(train_cond["mean"], DEVICE)
        train_yvar = M.tensor_y(train_cond["sem_var_shrunk"], DEVICE)
        try:
            model = fit_with_kernel(train_x, train_y, bounds, train_yvar, make_kernel(kind, d))
            query_x = M.tensor_x(pd.DataFrame([row]), DEVICE)
            with torch.no_grad():
                post = model.posterior(query_x, observation_noise=False)
            mu = float(post.mean.item())
            latent_var = float(post.variance.item())
            ls = model.covar_module.lengthscale.detach().reshape(-1).cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print(f"    [fold {i}] FAILED ({kind}): {exc}")
            continue
        sem_var = float(row["sem_var_raw"])
        denom2 = max(latent_var + sem_var, 1e-12)
        err = float(row["mean"]) - mu
        z = err / sqrt(denom2)
        nlpd = 0.5 * log(2.0 * pi * denom2) + 0.5 * err * err / denom2
        recs[i] = {
            "mean": float(row["mean"]),
            "z": z,
            "abs_err": abs(err),
            "sq_err": err * err,
            "nlpd": nlpd,
            "ls0": float(ls[0]),
            "ls1": float(ls[1]) if ls.size > 1 else float("nan"),
        }
    return pd.DataFrame.from_dict(recs, orient="index"), fails


def summarize(d: pd.DataFrame) -> dict:
    z = d["z"].to_numpy()
    return {
        "z_mean": float(z.mean()),
        "z_std": float(z.std(ddof=0)),
        "pct_gt2": float((np.abs(z) > 2.0).mean() * 100.0),
        "pct_gt3": float((np.abs(z) > 3.0).mean() * 100.0),
        "mae": float(d["abs_err"].mean()),
        "rmse": float(np.sqrt(d["sq_err"].mean())),
        "nlpd": float(d["nlpd"].mean()),
        "nlpd_se": float(d["nlpd"].std(ddof=1) / sqrt(len(d))),
        "ls0": float(d["ls0"].mean()),
        "ls1": float(d["ls1"].mean()),
        "n": int(len(d)),
    }


KINDS = ["matern05", "matern15", "matern25", "rbf"]
LABEL = {"matern05": "Matern 1/2", "matern15": "Matern 3/2", "matern25": "Matern 5/2", "rbf": "RBF"}


def main() -> None:
    M.set_seeds(0)
    bounds = M.make_bounds(DEVICE)
    print("LOCO kernel comparison | shrink_alpha=0.5 | target=AUC | seed=0")
    print("calibration denom^2 = latent_var + sem_var_raw (mirrors condition_loo)\n")

    for hrp in HRPS:
        path = M.DEFAULT_MAY05_LOG_DIR / f"6_3_LHS_HRP_{hrp}_res.xlsx"
        df = pd.read_excel(path)
        cond = M.prepare_condition_table(df, "AUC", shrink_alpha=0.5)
        n_cond = cond.shape[0]
        thr = cond["mean"].quantile(1.0 - TOP_FRAC)
        top_idx = set(cond.index[cond["mean"] >= thr])

        per_kernel: dict[str, pd.DataFrame] = {}
        summ_all: dict[str, dict] = {}
        summ_top: dict[str, dict] = {}
        for kind in KINDS:
            dfk, fails = loco_for_kernel(cond, bounds, kind)
            per_kernel[kind] = dfk
            summ_all[kind] = summarize(dfk)
            summ_top[kind] = summarize(dfk.loc[dfk.index.intersection(top_idx)])
            if fails:
                print(f"    {LABEL[kind]}: {fails} failed folds")

        print(f"\n================  HRP = {hrp}   (n_cond={n_cond})  ================")
        hdr = f"{'kernel':<11}{'z_mean':>8}{'z_std':>8}{'%|z|>2':>8}{'%|z|>3':>8}{'MAE':>9}{'RMSE':>9}{'NLPD':>9}{'SE':>7}{'ls0':>7}{'ls1':>7}"
        print(hdr)
        for kind in KINDS:
            s = summ_all[kind]
            print(f"{LABEL[kind]:<11}{s['z_mean']:>8.3f}{s['z_std']:>8.3f}"
                  f"{s['pct_gt2']:>8.1f}{s['pct_gt3']:>8.1f}{s['mae']:>9.1f}{s['rmse']:>9.1f}"
                  f"{s['nlpd']:>9.3f}{s['nlpd_se']:>7.3f}{s['ls0']:>7.3f}{s['ls1']:>7.3f}")

        # NLPD ranking + paired significance vs the best kernel (same folds).
        best = min(KINDS, key=lambda k: summ_all[k]["nlpd"])
        order = sorted(KINDS, key=lambda k: summ_all[k]["nlpd"])
        print(f"  NLPD ranking (low=better): " + " < ".join(LABEL[k] for k in order))
        print(f"  paired NLPD vs best ({LABEL[best]}):")
        for kind in KINDS:
            if kind == best:
                continue
            j = per_kernel[kind].index.intersection(per_kernel[best].index)
            diff = per_kernel[kind].loc[j, "nlpd"].to_numpy() - per_kernel[best].loc[j, "nlpd"].to_numpy()
            md = float(diff.mean())
            se = float(diff.std(ddof=1) / sqrt(len(diff)))
            ratio = md / se if se > 0 else float("nan")
            verdict = "significant" if ratio > 2 else ("marginal" if ratio > 1 else "n.s.")
            print(f"    {LABEL[kind]:<11} dNLPD={md:+.3f} +/- {se:.3f}  (mean/SE={ratio:+.2f}, {verdict})")

        # BO-relevant view: high-AUC conditions only.
        n_top = summ_top[KINDS[0]]["n"]
        print(f"  -- top {int(TOP_FRAC*100)}% high-AUC conditions (n={n_top}) --")
        top_order = sorted(KINDS, key=lambda k: summ_top[k]["nlpd"])
        for kind in KINDS:
            s = summ_top[kind]
            print(f"    {LABEL[kind]:<11} MAE={s['mae']:>8.1f}  RMSE={s['rmse']:>8.1f}  "
                  f"NLPD={s['nlpd']:>7.3f}  z_std={s['z_std']:.3f}")
        print(f"    top-subset NLPD ranking: " + " < ".join(LABEL[k] for k in top_order))


if __name__ == "__main__":
    main()
