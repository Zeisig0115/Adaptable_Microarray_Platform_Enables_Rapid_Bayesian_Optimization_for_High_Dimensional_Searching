"""Does the Hvarfner lengthscale prior actually help? A LOCO ablation on LHS data.

Context
-------
In the essentials fixed-noise workflow the noise is supplied (FixedNoiseGaussian-
Likelihood), so the Hvarfner noise prior LogNormal(-4,1) is never used. The only
Hvarfner prior that is live is the dimension-scaled LENGTHSCALE prior
    LogNormalPrior(loc = sqrt(2) + 0.5*log(d), scale = sqrt(3)).
This script isolates that prior's contribution.

Design (clean ablation)
------------------------
Same leave-one-condition-out (LOCO) protocol as loo_kernel_comparison.py / the
project's condition_loo. For each (HRP, kernel in {Matern 1/2, Matern 3/2}) we
compare three lengthscale treatments, holding the kernel class, the 0.025 floor,
the (no-)outputscale design, and the optimizer INIT all identical -- so the only
thing that changes is the regularization term added to the marginal likelihood:
    hvarfner : LogNormalPrior(sqrt(2)+0.5 log d, sqrt(3))   (current model)
    mle      : no lengthscale prior  -> pure MLL fit (MAP -> MLE)
    gamma    : GammaPrior(3.0, 6.0)  (BoTorch legacy lengthscale prior)

All three start optimization from the SAME init (the Hvarfner prior mode ~0.29),
so any difference is attributable to the prior term, not to initialization.

Metrics per (HRP, kernel, variant)
----------------------------------
- held-out: z_std (~1 ideal), %|z|>2, MAE, RMSE, mean NLPD (lower=better) +- SE
- fitted lengthscale behaviour (pooled over folds x 2 dims): median, min, max,
  and the count of folds pinned at the 0.025 FLOOR (overfit/collapse) or pushed
  to a near-flat lengthscale >= 3.0 (underfit). These expose WHY the prior helps.
- paired NLPD difference vs the hvarfner variant on the same folds (mean/SE).

Read-only, reproducible (seed=0). Delete freely; it is a diagnostic.
"""
from __future__ import annotations

from math import log, pi, sqrt

import numpy as np
import pandas as pd
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior, LogNormalPrior

import warnings

import fixed_noise_ess_bo as M

torch.set_default_dtype(torch.double)
warnings.filterwarnings("ignore")

SQRT2 = sqrt(2.0)
SQRT3 = sqrt(3.0)
HRPS = ["1", "0.01", "0.0001"]
DEVICE = torch.device("cpu")
D = len(M.ESSENTIALS)
# Common optimizer init for ALL variants = Hvarfner prior mode (~0.29). Isolates
# the prior term from initialization effects.
COMMON_INIT = LogNormalPrior(loc=SQRT2 + log(D) * 0.5, scale=SQRT3).mode
FLOOR = 2.5e-2
FLOOR_TOL = 2.6e-2   # "pinned at floor" if fitted ls <= this
LONG_TOL = 3.0       # "near-flat / underfit" if fitted ls >= this

KERNELS = {"matern05": ("Matern 1/2", 0.5), "matern15": ("Matern 3/2", 1.5)}
VARIANTS = ["hvarfner", "mle", "gamma"]


def make_kernel(kind: str, prior_kind: str, d: int):
    if prior_kind == "hvarfner":
        ls_prior = LogNormalPrior(loc=SQRT2 + log(d) * 0.5, scale=SQRT3)
    elif prior_kind == "gamma":
        ls_prior = GammaPrior(3.0, 6.0)
    elif prior_kind == "mle":
        ls_prior = None
    else:
        raise ValueError(prior_kind)
    constraint = GreaterThan(FLOOR, transform=None, initial_value=COMMON_INIT)
    return MaternKernel(
        nu=KERNELS[kind][1],
        ard_num_dims=d,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=constraint,
    )


def fit_with_kernel(train_x, train_y, bounds, train_yvar, kernel):
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


def loco(cond: pd.DataFrame, bounds, kind: str, prior_kind: str):
    recs: dict[int, dict] = {}
    fails = 0
    for i, row in cond.iterrows():
        train_cond = cond.drop(index=i).reset_index(drop=True)
        tx = M.tensor_x(train_cond, DEVICE)
        ty = M.tensor_y(train_cond["mean"], DEVICE)
        tv = M.tensor_y(train_cond["sem_var_shrunk"], DEVICE)
        try:
            model = fit_with_kernel(tx, ty, bounds, tv, make_kernel(kind, prior_kind, D))
            qx = M.tensor_x(pd.DataFrame([row]), DEVICE)
            with torch.no_grad():
                post = model.posterior(qx, observation_noise=False)
            mu = float(post.mean.item())
            lv = float(post.variance.item())
            ls = model.covar_module.lengthscale.detach().reshape(-1).cpu().numpy()
        except Exception as exc:  # noqa: BLE001
            fails += 1
            print(f"      [fold {i}] FAILED ({kind}/{prior_kind}): {exc}")
            continue
        sem_var = float(row["sem_var_raw"])
        denom2 = max(lv + sem_var, 1e-12)
        err = float(row["mean"]) - mu
        recs[i] = {
            "z": err / sqrt(denom2),
            "abs_err": abs(err),
            "sq_err": err * err,
            "nlpd": 0.5 * log(2.0 * pi * denom2) + 0.5 * err * err / denom2,
            "ls_lo": float(np.min(ls)),
            "ls_hi": float(np.max(ls)),
        }
    return pd.DataFrame.from_dict(recs, orient="index"), fails


def summarize(d: pd.DataFrame) -> dict:
    z = d["z"].to_numpy()
    ls_all = np.concatenate([d["ls_lo"].to_numpy(), d["ls_hi"].to_numpy()])
    return {
        "z_std": float(z.std(ddof=0)),
        "pct_gt2": float((np.abs(z) > 2.0).mean() * 100.0),
        "mae": float(d["abs_err"].mean()),
        "rmse": float(np.sqrt(d["sq_err"].mean())),
        "nlpd": float(d["nlpd"].mean()),
        "nlpd_se": float(d["nlpd"].std(ddof=1) / sqrt(len(d))),
        "ls_med": float(np.median(ls_all)),
        "ls_min": float(ls_all.min()),
        "ls_max": float(ls_all.max()),
        "n_floor": int((d["ls_lo"].to_numpy() <= FLOOR_TOL).sum()),
        "n_long": int((d["ls_hi"].to_numpy() >= LONG_TOL).sum()),
        "n": int(len(d)),
    }


def main() -> None:
    M.set_seeds(0)
    bounds = M.make_bounds(DEVICE)
    print("LOCO Hvarfner-prior ablation | shrink_alpha=0.5 | target=AUC | seed=0")
    print(f"all variants share kernel class, 0.025 floor, no outputscale, init={float(COMMON_INIT):.3f}")
    print("only the lengthscale prior term differs.\n")

    for hrp in HRPS:
        df = pd.read_excel(M.DEFAULT_MAY05_LOG_DIR / f"6_3_LHS_HRP_{hrp}_res.xlsx")
        cond = M.prepare_condition_table(df, "AUC", shrink_alpha=0.5)
        for kind in KERNELS:
            per: dict[str, pd.DataFrame] = {}
            summ: dict[str, dict] = {}
            for v in VARIANTS:
                dfk, fails = loco(cond, bounds, kind, v)
                per[v] = dfk
                summ[v] = summarize(dfk)
                if fails:
                    print(f"    {kind}/{v}: {fails} failed folds")
            print(f"\n===== HRP={hrp} | {KERNELS[kind][0]} (n_cond={cond.shape[0]}) =====")
            hdr = (f"{'variant':<9}{'z_std':>7}{'%|z|>2':>8}{'MAE':>8}{'RMSE':>8}"
                   f"{'NLPD':>8}{'SE':>6}{'ls_med':>8}{'ls_min':>8}{'ls_max':>9}{'#floor':>7}{'#long':>6}")
            print(hdr)
            for v in VARIANTS:
                s = summ[v]
                print(f"{v:<9}{s['z_std']:>7.3f}{s['pct_gt2']:>8.1f}{s['mae']:>8.1f}{s['rmse']:>8.1f}"
                      f"{s['nlpd']:>8.3f}{s['nlpd_se']:>6.3f}{s['ls_med']:>8.3f}{s['ls_min']:>8.3f}"
                      f"{s['ls_max']:>9.3f}{s['n_floor']:>7d}{s['n_long']:>6d}")
            # paired NLPD vs hvarfner (same folds)
            base = per["hvarfner"]
            for v in ("mle", "gamma"):
                j = per[v].index.intersection(base.index)
                diff = per[v].loc[j, "nlpd"].to_numpy() - base.loc[j, "nlpd"].to_numpy()
                md = float(diff.mean())
                se = float(diff.std(ddof=1) / sqrt(len(diff))) if len(diff) > 1 else float("nan")
                ratio = md / se if se and se > 0 else float("nan")
                tag = "worse than hvarfner" if md > 0 else "better than hvarfner"
                sig = "significant" if abs(ratio) > 2 else ("marginal" if abs(ratio) > 1 else "n.s.")
                print(f"  {v} vs hvarfner: dNLPD={md:+.3f} +/- {se:.3f} (mean/SE={ratio:+.2f}, {sig}; +=worse)")


if __name__ == "__main__":
    main()
