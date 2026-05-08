from __future__ import annotations

import argparse
import json
import math
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

from fit_model import matern_with_hvarfner_prior


torch.set_default_dtype(torch.double)

ESSENTIALS = ["TMB", "H2O2"]
PHYSICAL_BOUNDS = {"TMB": (0.005, 1.0), "H2O2": (0.005, 1.0)}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_bounds(device: torch.device) -> torch.Tensor:
    lo = [math.log10(PHYSICAL_BOUNDS[e][0]) for e in ESSENTIALS]
    hi = [math.log10(PHYSICAL_BOUNDS[e][1]) for e in ESSENTIALS]
    return torch.tensor([lo, hi], dtype=torch.double, device=device)


def tensor_x(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    z = np.log10(df[ESSENTIALS].to_numpy(dtype=float))
    return torch.tensor(z, dtype=torch.double, device=device)


def tensor_y(values: np.ndarray | pd.Series, device: torch.device) -> torch.Tensor:
    y = np.asarray(values, dtype=float).reshape(-1, 1)
    return torch.tensor(y, dtype=torch.double, device=device)


def prepare_condition_table(df: pd.DataFrame, target: str, shrink_alpha: float) -> pd.DataFrame:
    grouped = (
        df.groupby(ESSENTIALS, as_index=False)[target]
        .agg(mean="mean", median="median", sd=lambda s: float(s.std(ddof=1)), count="count")
        .sort_values(ESSENTIALS)
        .reset_index(drop=True)
    )
    grouped["var"] = grouped["sd"].pow(2)
    grouped["sem_var_raw"] = grouped["var"] / grouped["count"]

    median_sem_var = float(np.nanmedian(grouped["sem_var_raw"]))
    floor = max(median_sem_var * 0.05, 1e-9)
    grouped["sem_var_raw"] = grouped["sem_var_raw"].clip(lower=floor)

    pooled_var = float(np.average(grouped["var"], weights=grouped["count"] - 1))
    pooled_sem_var = max(pooled_var / float(grouped["count"].median()), floor)
    grouped["sem_var_shrunk"] = (
        (1.0 - shrink_alpha) * grouped["sem_var_raw"] + shrink_alpha * pooled_sem_var
    )
    grouped["sem_sd_raw"] = np.sqrt(grouped["sem_var_raw"])
    grouped["sem_sd_shrunk"] = np.sqrt(grouped["sem_var_shrunk"])
    return grouped


def fit_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    bounds: torch.Tensor,
    train_yvar: torch.Tensor | None,
) -> SingleTaskGP:
    d = train_x.shape[-1]
    model = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        train_Yvar=train_yvar,
        covar_module=matern_with_hvarfner_prior(d, nu=0.5),
        input_transform=Normalize(d=d, bounds=bounds),
        outcome_transform=Standardize(m=1),
    ).to(train_x)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


def model_train_data(
    df: pd.DataFrame,
    cond: pd.DataFrame,
    target: str,
    model_name: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, pd.DataFrame]:
    if model_name == "replicate_inferred":
        return tensor_x(df, device), tensor_y(df[target], device), None, df.copy()

    if model_name == "fixed_sem_raw":
        yvar_col = "sem_var_raw"
    elif model_name == "fixed_sem_shrunk":
        yvar_col = "sem_var_shrunk"
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    train_x = tensor_x(cond, device)
    train_y = tensor_y(cond["mean"], device)
    train_yvar = tensor_y(cond[yvar_col], device)
    return train_x, train_y, train_yvar, cond.copy()


def posterior_frame(model: SingleTaskGP, query_x: torch.Tensor) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        latent = model.posterior(query_x, observation_noise=False)
        noisy = model.posterior(query_x, observation_noise=True)
    phys = 10.0 ** query_x.detach().cpu().numpy()
    return pd.DataFrame(
        {
            "TMB": phys[:, 0],
            "H2O2": phys[:, 1],
            "latent_mean": latent.mean.squeeze(-1).detach().cpu().numpy(),
            "latent_std": latent.variance.sqrt().squeeze(-1).detach().cpu().numpy(),
            "default_observation_std": noisy.variance.sqrt().squeeze(-1).detach().cpu().numpy(),
        }
    )


def covariance_frame(model: SingleTaskGP, query_x: torch.Tensor) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(query_x, observation_noise=False)
        cov = posterior.distribution.covariance_matrix.detach().cpu().numpy()
    return pd.DataFrame(cov)


def generate_candidates(
    model: SingleTaskGP,
    bounds: torch.Tensor,
    baseline_x: torch.Tensor,
    q: int,
    seed: int,
    num_restarts: int,
    raw_samples: int,
) -> torch.Tensor:
    set_seeds(seed)
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    acqf = qLogNoisyExpectedImprovement(
        model,
        X_baseline=baseline_x,
        sampler=sampler,
        cache_root=False,
        prune_baseline=True,
    )
    candidates, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=False,
    )
    return candidates.detach()


def make_grid(bounds: torch.Tensor, grid_size: int, device: torch.device) -> torch.Tensor:
    x1 = torch.linspace(bounds[0, 0], bounds[1, 0], grid_size, device=device)
    x2 = torch.linspace(bounds[0, 1], bounds[1, 1], grid_size, device=device)
    mesh = torch.cartesian_prod(x1, x2)
    return mesh.to(dtype=torch.double, device=device)


def lengthscales(model: SingleTaskGP) -> tuple[float | None, float | None]:
    covar = model.covar_module
    ls = getattr(covar, "lengthscale", None)
    if ls is None:
        return None, None
    vals = ls.detach().cpu().reshape(-1).numpy()
    if len(vals) < 2:
        return float(vals[0]), None
    return float(vals[0]), float(vals[1])


def condition_loo(
    df: pd.DataFrame,
    cond: pd.DataFrame,
    target: str,
    model_name: str,
    bounds: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    z_scores: list[float] = []
    abs_errors: list[float] = []

    for i, row in cond.iterrows():
        is_holdout = np.isclose(df["TMB"], row["TMB"]) & np.isclose(df["H2O2"], row["H2O2"])
        if model_name == "replicate_inferred":
            train_df = df.loc[~is_holdout].copy()
            train_x, train_y, train_yvar, _ = model_train_data(
                train_df, cond, target, model_name, device
            )
        else:
            train_cond = cond.drop(index=i).reset_index(drop=True)
            train_x, train_y, train_yvar, _ = model_train_data(
                df, train_cond, target, model_name, device
            )

        model = fit_model(train_x, train_y, bounds, train_yvar)
        query_x = tensor_x(pd.DataFrame([row]), device)
        with torch.no_grad():
            post = model.posterior(query_x, observation_noise=False)
        mu = float(post.mean.item())
        latent_var = float(post.variance.item())
        sem_var = float(row["sem_var_raw"])
        denom = math.sqrt(max(latent_var + sem_var, 1e-12))
        z_scores.append((float(row["mean"]) - mu) / denom)
        abs_errors.append(abs(float(row["mean"]) - mu))

    z = np.asarray(z_scores)
    ae = np.asarray(abs_errors)
    return {
        "condition_loo_z_mean": float(z.mean()),
        "condition_loo_z_std": float(z.std(ddof=0)),
        "condition_loo_abs_z_gt_2_pct": float((np.abs(z) > 2.0).mean() * 100.0),
        "condition_loo_abs_z_gt_3_pct": float((np.abs(z) > 3.0).mean() * 100.0),
        "condition_loo_mae": float(ae.mean()),
        "condition_loo_max_abs_error": float(ae.max()),
    }


def summarize_model(
    hrp: str,
    model_name: str,
    model: SingleTaskGP,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
    cond: pd.DataFrame,
    extra: dict[str, float],
) -> dict[str, Any]:
    ls_tmb, ls_h2o2 = lengthscales(model)
    y_sd = float(train_y.std(unbiased=True).item()) if train_y.shape[0] > 1 else 1.0
    row: dict[str, Any] = {
        "HRP": hrp,
        "model": model_name,
        "n_train": int(train_y.shape[0]),
        "n_conditions": int(cond.shape[0]),
        "train_y_mean": float(train_y.mean().item()),
        "train_y_sd": y_sd,
        "lengthscale_TMB_normalized": ls_tmb,
        "lengthscale_H2O2_normalized": ls_h2o2,
    }
    if train_yvar is None:
        noise_std = math.sqrt(float(model.likelihood.noise.item())) * y_sd
        row.update(
            {
                "noise_mode": "inferred_global",
                "inferred_noise_standardized": float(model.likelihood.noise.item()),
                "inferred_noise_std_orig_y": noise_std,
            }
        )
    else:
        yvar = train_yvar.detach().cpu().numpy().reshape(-1)
        row.update(
            {
                "noise_mode": "fixed_sem",
                "train_yvar_min": float(yvar.min()),
                "train_yvar_median": float(np.median(yvar)),
                "train_yvar_max": float(yvar.max()),
                "train_yvar_sd_median": float(np.sqrt(np.median(yvar))),
                "train_yvar_sd_max": float(np.sqrt(yvar.max())),
            }
        )
    row.update(extra)
    return row


def run_for_hrp(args: argparse.Namespace, hrp: str, out_dir: Path) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    bounds = make_bounds(device)
    data_path = Path(args.input_dir) / f"{args.input_prefix}_LHS_HRP_{hrp}_res.xlsx"
    df = pd.read_excel(data_path)
    cond = prepare_condition_table(df, args.target, args.shrink_alpha)
    cond_out = out_dir / f"fixed_noise_gp_HRP_{hrp}_condition_noise.csv"
    cond.to_csv(cond_out, index=False)

    grid_x = make_grid(bounds, args.grid_size, device)
    summary_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        print(f"\n=== HRP={hrp} model={model_name} ===", flush=True)
        train_x, train_y, train_yvar, baseline_df = model_train_data(
            df, cond, args.target, model_name, device
        )
        model = fit_model(train_x, train_y, bounds, train_yvar)

        loo_stats: dict[str, float] = {}
        if args.loo:
            print("  running condition-level LOO refits...", flush=True)
            loo_stats = condition_loo(df, cond, args.target, model_name, bounds, device)

        cand_x = generate_candidates(
            model=model,
            bounds=bounds,
            baseline_x=train_x,
            q=args.q,
            seed=args.seed,
            num_restarts=args.num_restarts,
            raw_samples=args.raw_samples,
        )
        cand = posterior_frame(model, cand_x)
        cand = cand.rename(
            columns={
                "latent_mean": "predicted_value",
                "latent_std": "uncertainty_std",
            }
        )
        cand_path = out_dir / f"fixed_noise_gp_HRP_{hrp}_{model_name}_candidates.csv"
        cand.to_csv(cand_path, index=False)

        cov_path = out_dir / f"fixed_noise_gp_HRP_{hrp}_{model_name}_candidate_latent_cov.csv"
        covariance_frame(model, cand_x).to_csv(cov_path, index=False)

        grid = posterior_frame(model, grid_x)
        grid_path = out_dir / f"fixed_noise_gp_HRP_{hrp}_{model_name}_posterior_grid.csv"
        grid.to_csv(grid_path, index=False)

        in_sample = posterior_frame(model, train_x)
        in_sample["training_target"] = train_y.detach().cpu().numpy().reshape(-1)
        in_sample_path = out_dir / f"fixed_noise_gp_HRP_{hrp}_{model_name}_posterior_train.csv"
        in_sample.to_csv(in_sample_path, index=False)

        extra = {
            "candidate_pred_max": float(cand["predicted_value"].max()),
            "candidate_pred_min": float(cand["predicted_value"].min()),
            "candidate_latent_std_mean": float(cand["uncertainty_std"].mean()),
            "candidate_latent_std_max": float(cand["uncertainty_std"].max()),
            "candidate_default_observation_std_mean": float(
                cand["default_observation_std"].mean()
            ),
            "candidate_file": str(cand_path),
            "posterior_grid_file": str(grid_path),
            "posterior_train_file": str(in_sample_path),
            "candidate_cov_file": str(cov_path),
            "baseline_rows": int(baseline_df.shape[0]),
        }
        extra.update(loo_stats)
        summary_rows.append(
            summarize_model(hrp, model_name, model, train_y, train_yvar, cond, extra)
        )

    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare replicate-level inferred-noise GP with condition-mean fixed-noise GP."
    )
    parser.add_argument("--input_dir", default="May_5_full_log")
    parser.add_argument("--input_prefix", default="05_05")
    parser.add_argument("--target", default="AUC")
    parser.add_argument("--hrp", nargs="+", default=["1", "0.01", "0.0001"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=["replicate_inferred", "fixed_sem_raw", "fixed_sem_shrunk"],
        choices=["replicate_inferred", "fixed_sem_raw", "fixed_sem_shrunk"],
    )
    parser.add_argument("--out_dir", default="May_5_full_log")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--q", type=int, default=32)
    parser.add_argument("--num_restarts", type=int, default=20)
    parser.add_argument("--raw_samples", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=41)
    parser.add_argument("--shrink_alpha", type=float, default=0.5)
    parser.add_argument("--loo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Fixed-noise probe config: "
        + json.dumps(vars(args), ensure_ascii=True, sort_keys=True),
        flush=True,
    )
    warnings.filterwarnings("default")

    rows: list[dict[str, Any]] = []
    for hrp in args.hrp:
        rows.extend(run_for_hrp(args, hrp, out_dir))

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "fixed_noise_gp_probe_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary -> {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
