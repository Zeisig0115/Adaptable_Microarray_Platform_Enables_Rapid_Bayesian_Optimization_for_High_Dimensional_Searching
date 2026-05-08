from __future__ import annotations

import argparse
import json
import math
import random
import time
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
RUNS = ["LHS", "BO1", "BO2"]
HRPS = ["1", "0.01", "0.0001"]
COMBOS = {
    "LHS_BO1": ["LHS", "BO1"],
    "LHS_BO1_BO2": ["LHS", "BO1", "BO2"],
}
MODELS = ["replicate_inferred", "fixed_sem_raw", "fixed_sem_shrunk"]


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_bounds(device: torch.device, lower: float) -> torch.Tensor:
    return torch.tensor(
        [[math.log10(lower), math.log10(lower)], [0.0, 0.0]],
        dtype=torch.double,
        device=device,
    )


def read_run(base: Path, run: str, hrp: str) -> pd.DataFrame:
    df = pd.read_excel(base / f"04_29_{run}_HRP_{hrp}_res.xlsx")
    df = df.copy()
    df["source_run"] = run
    return df


def condition_key(df: pd.DataFrame) -> pd.Series:
    return df["TMB"].round(12).astype(str) + "|" + df["H2O2"].round(12).astype(str)


def prepare_condition_table(df: pd.DataFrame, target: str, shrink_alpha: float) -> pd.DataFrame:
    grouped = (
        df.groupby(ESSENTIALS, as_index=False)
        .agg(
            mean=(target, "mean"),
            median=(target, "median"),
            sd=(target, lambda s: float(s.std(ddof=1))),
            min_y=(target, "min"),
            max_y=(target, "max"),
            count=(target, "count"),
            n_runs=("source_run", "nunique"),
            runs=("source_run", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .sort_values(ESSENTIALS)
        .reset_index(drop=True)
    )
    grouped["var"] = grouped["sd"].pow(2)
    grouped["sem_var_raw"] = grouped["var"] / grouped["count"]
    median_sem = float(np.nanmedian(grouped["sem_var_raw"]))
    floor = max(median_sem * 0.05, 1e-9)
    grouped["sem_var_raw"] = grouped["sem_var_raw"].clip(lower=floor)
    pooled_var = float(np.average(grouped["var"], weights=np.maximum(grouped["count"] - 1, 1)))
    pooled_sem = max(pooled_var / float(grouped["count"].median()), floor)
    grouped["sem_var_shrunk"] = (
        (1.0 - shrink_alpha) * grouped["sem_var_raw"] + shrink_alpha * pooled_sem
    )
    grouped["sem_sd_raw"] = np.sqrt(grouped["sem_var_raw"])
    grouped["sem_sd_shrunk"] = np.sqrt(grouped["sem_var_shrunk"])
    return grouped


def tensor_x(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    x = np.log10(df[ESSENTIALS].to_numpy(dtype=float))
    return torch.tensor(x, dtype=torch.double, device=device)


def tensor_y(values: pd.Series | np.ndarray, device: torch.device) -> torch.Tensor:
    y = np.asarray(values, dtype=float).reshape(-1, 1)
    return torch.tensor(y, dtype=torch.double, device=device)


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
        raise ValueError(model_name)
    return tensor_x(cond, device), tensor_y(cond["mean"], device), tensor_y(cond[yvar_col], device), cond.copy()


def fit_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
    bounds: torch.Tensor,
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


def get_noise_vector(model: SingleTaskGP, n: int, like: torch.Tensor) -> torch.Tensor:
    noise = model.likelihood.noise.detach().to(like).reshape(-1)
    if noise.numel() == 1:
        return noise.expand(n)
    if noise.numel() == n:
        return noise
    return noise[-n:]


def residual_and_matrix_stats(
    model: SingleTaskGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_yvar: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        x_model = model.input_transform(train_x) if hasattr(model, "input_transform") else train_x
        k = model.covar_module(x_model).evaluate()
        noise_vec = get_noise_vector(model, k.shape[0], k)
        eig = torch.linalg.eigvalsh(k + torch.diag(noise_vec))
        post_lat = model.posterior(train_x, observation_noise=False)
        post_obs = model.posterior(train_x, observation_noise=True)
        mean = post_lat.mean.squeeze(-1)
        residual = (train_y.squeeze(-1) - mean).abs()

    y_np = train_y.squeeze(-1).detach().cpu().numpy()
    res_np = residual.detach().cpu().numpy()
    order = np.argsort(y_np)
    half = max(len(order) // 2, 1)
    low = float(res_np[order[:half]].mean())
    high = float(res_np[order[half:]].mean())
    out = {
        "matrix_min_eig": float(eig.min().item()),
        "matrix_max_eig": float(eig.max().item()),
        "matrix_condition": float((eig.max() / eig.min().clamp_min(1e-18)).abs().item()),
        "noise_stdzd_min": float(noise_vec.min().item()),
        "noise_stdzd_median": float(noise_vec.median().item()),
        "noise_stdzd_max": float(noise_vec.max().item()),
        "train_avg_abs_residual": float(residual.mean().item()),
        "train_max_abs_residual": float(residual.max().item()),
        "train_low_y_residual_mean": low,
        "train_high_y_residual_mean": high,
        "train_high_low_residual_ratio": high / max(low, 1e-12),
        "train_latent_std_mean": float(post_lat.variance.sqrt().mean().item()),
        "train_obs_std_mean": float(post_obs.variance.sqrt().mean().item()),
    }
    if train_yvar is None:
        y_sd = float(train_y.std(unbiased=True).item()) if train_y.shape[0] > 1 else 1.0
        out["inferred_noise_std_orig_y"] = math.sqrt(float(model.likelihood.noise.item())) * y_sd
    else:
        yvar = train_yvar.detach().cpu().numpy().reshape(-1)
        out["train_yvar_sd_median"] = float(np.sqrt(np.median(yvar)))
        out["train_yvar_sd_max"] = float(np.sqrt(yvar.max()))
    return out


def lengthscales(model: SingleTaskGP) -> tuple[float | None, float | None]:
    ls = getattr(model.covar_module, "lengthscale", None)
    if ls is None:
        return None, None
    vals = ls.detach().cpu().reshape(-1).tolist()
    return (float(vals[0]), float(vals[1]) if len(vals) > 1 else None)


def posterior_at(model: SingleTaskGP, query_x: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        lat = model.posterior(query_x, observation_noise=False)
        obs = model.posterior(query_x, observation_noise=True)
    return (
        lat.mean.squeeze(-1).detach().cpu().numpy(),
        lat.variance.sqrt().squeeze(-1).detach().cpu().numpy(),
        obs.variance.sqrt().squeeze(-1).detach().cpu().numpy(),
    )


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
            train_cond = prepare_condition_table(train_df, target, 0.5)
        else:
            train_df = df.copy()
            train_cond = cond.drop(index=i).reset_index(drop=True)
        tx, ty, tyv, _ = model_train_data(train_df, train_cond, target, model_name, device)
        model = fit_model(tx, ty, tyv, bounds)
        qx = tensor_x(pd.DataFrame([row]), device)
        with torch.no_grad():
            post = model.posterior(qx, observation_noise=False)
        mu = float(post.mean.item())
        latent_var = float(post.variance.item())
        denom = math.sqrt(max(latent_var + float(row["sem_var_raw"]), 1e-12))
        z_scores.append((float(row["mean"]) - mu) / denom)
        abs_errors.append(abs(float(row["mean"]) - mu))
    z = np.asarray(z_scores)
    ae = np.asarray(abs_errors)
    return {
        "loo_z_mean": float(z.mean()),
        "loo_z_std": float(z.std(ddof=0)),
        "loo_abs_z_gt_2_pct": float((np.abs(z) > 2).mean() * 100.0),
        "loo_abs_z_gt_3_pct": float((np.abs(z) > 3).mean() * 100.0),
        "loo_mae": float(ae.mean()),
        "loo_max_abs_error": float(ae.max()),
    }


def holdout_bo2_validation(
    model: SingleTaskGP,
    bo2_cond: pd.DataFrame,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    qx = tensor_x(bo2_cond, device)
    mean, latent_std, obs_std = posterior_at(model, qx)
    denom = np.sqrt(np.maximum(latent_std**2 + bo2_cond["sem_var_raw"].to_numpy(float), 1e-12))
    err = bo2_cond["mean"].to_numpy(float) - mean
    z = err / denom
    pred = bo2_cond[ESSENTIALS + ["mean", "sd", "count"]].copy()
    pred["pred_mean"] = mean
    pred["latent_std"] = latent_std
    pred["default_obs_std"] = obs_std
    pred["error_actual_minus_pred"] = err
    pred["z"] = z
    metrics = {
        "bo2_holdout_bias_actual_minus_pred": float(err.mean()),
        "bo2_holdout_mae": float(np.abs(err).mean()),
        "bo2_holdout_rmse": float(np.sqrt(np.mean(err**2))),
        "bo2_holdout_z_mean": float(z.mean()),
        "bo2_holdout_z_std": float(z.std(ddof=0)),
        "bo2_holdout_abs_z_gt_2_pct": float((np.abs(z) > 2).mean() * 100.0),
        "bo2_holdout_abs_z_gt_3_pct": float((np.abs(z) > 3).mean() * 100.0),
    }
    return metrics, pred


def generate_candidates(
    model: SingleTaskGP,
    bounds: torch.Tensor,
    baseline_x: torch.Tensor,
    seed: int,
    q: int,
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
    cand, _ = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        sequential=False,
    )
    return cand.detach()


def candidate_stats_and_rows(
    model: SingleTaskGP,
    cand_x: torch.Tensor,
    train_x: torch.Tensor,
    hrp: str,
    combo: str,
    model_name: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    mean, latent_std, obs_std = posterior_at(model, cand_x)
    z = cand_x.detach().cpu().numpy()
    phys = 10.0**z
    train_np = train_x.detach().cpu().numpy()
    nearest = np.sqrt(((z[:, None, :] - train_np[None, :, :]) ** 2).sum(axis=-1)).min(axis=1)
    pair = []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            pair.append(float(np.sqrt(((z[i] - z[j]) ** 2).sum())))
    rows = []
    for i in range(len(z)):
        rows.append(
            {
                "HRP": hrp,
                "train_combo": combo,
                "model": model_name,
                "rank_input_order": i + 1,
                "TMB": float(phys[i, 0]),
                "H2O2": float(phys[i, 1]),
                "pred_mean": float(mean[i]),
                "latent_std": float(latent_std[i]),
                "default_obs_std": float(obs_std[i]),
                "nearest_train_logdist": float(nearest[i]),
            }
        )
    stats = {
        "candidate_pred_max": float(mean.max()),
        "candidate_pred_min": float(mean.min()),
        "candidate_latent_std_mean": float(latent_std.mean()),
        "candidate_latent_std_max": float(latent_std.max()),
        "candidate_default_obs_std_mean": float(obs_std.mean()),
        "candidate_TMB_min": float(phys[:, 0].min()),
        "candidate_TMB_max": float(phys[:, 0].max()),
        "candidate_H2O2_min": float(phys[:, 1].min()),
        "candidate_H2O2_max": float(phys[:, 1].max()),
        "candidate_nearest_train_logdist_median": float(np.median(nearest)),
        "candidate_pairwise_logdist_median": float(np.median(pair)) if pair else math.nan,
    }
    return stats, rows


def run_drift_summary(data_by_run: dict[str, pd.DataFrame], target: str) -> pd.DataFrame:
    pieces = []
    for run, df in data_by_run.items():
        cond = prepare_condition_table(df, target, 0.5)
        cond["source_run"] = run
        pieces.append(cond)
    all_cond = pd.concat(pieces, ignore_index=True)
    rows = []
    for hrp_run in RUNS:
        sub = all_cond[all_cond["source_run"] == hrp_run]
        rows.append(
            {
                "source_run": hrp_run,
                "n_conditions": int(sub.shape[0]),
                "condition_mean_AUC_mean": float(sub["mean"].mean()),
                "condition_mean_AUC_max": float(sub["mean"].max()),
                "condition_sd_median": float(sub["sd"].median()),
                "condition_sd_max": float(sub["sd"].max()),
            }
        )
    return pd.DataFrame(rows)


def format_table(df: pd.DataFrame, cols: list[str]) -> str:
    tmp = df.copy()
    for col in cols:
        if col not in tmp:
            tmp[col] = np.nan
    tmp = tmp[cols].copy()
    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3g}"
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [
        "| " + " | ".join(fmt(row[col]) for col in cols) + " |"
        for _, row in tmp.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def build_report(
    summary: pd.DataFrame,
    drift: pd.DataFrame,
    candidates: pd.DataFrame,
    holdout: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines = []
    lines.append("# Apr 29 LHS/BO1/BO2 GP Validation Report\n")
    lines.append("This report uses extracted replicate-level objective tables in `Apr_29_full_log`.\n")
    lines.append(
        "Important design fact: BO1 and BO2 contain the same 32 conditions for each HRP; "
        "BO2 is therefore a repeat / temporal validation of BO1 conditions, not a new spatial BO round.\n"
    )
    lines.append(f"Log-space physical bounds used here: `[log10({args.lower_bound}), 0]` for both TMB and H2O2.\n")
    lines.append("## Run-Level Data Summary\n")
    lines.append(format_table(drift, ["HRP", "source_run", "n_conditions", "condition_mean_AUC_mean", "condition_mean_AUC_max", "condition_sd_median", "condition_sd_max"]))
    lines.append("\n## Matrix and In-Sample Diagnostics\n")
    lines.append(format_table(summary, ["HRP", "train_combo", "model", "n_train", "n_conditions", "matrix_condition", "noise_stdzd_median", "noise_stdzd_max", "train_avg_abs_residual", "train_high_low_residual_ratio"]))
    lines.append("\n## Condition LOO Calibration\n")
    lines.append(format_table(summary, ["HRP", "train_combo", "model", "loo_z_std", "loo_abs_z_gt_2_pct", "loo_abs_z_gt_3_pct", "loo_mae", "loo_max_abs_error"]))
    lines.append("\n## BO2 Holdout Validation for LHS+BO1 Models\n")
    hcols = ["HRP", "model", "bo2_holdout_bias_actual_minus_pred", "bo2_holdout_mae", "bo2_holdout_z_std", "bo2_holdout_abs_z_gt_2_pct", "bo2_holdout_abs_z_gt_3_pct"]
    lines.append(format_table(summary[summary["train_combo"] == "LHS_BO1"], hcols))
    lines.append("\n## Acquisition Stability and Candidate Distribution\n")
    lines.append(format_table(summary, ["HRP", "train_combo", "model", "acq_success", "acq_elapsed_sec", "acq_n_warnings", "candidate_pred_max", "candidate_latent_std_mean", "candidate_nearest_train_logdist_median", "candidate_pairwise_logdist_median"]))
    lines.append("\n## Top Generated Candidates\n")
    top = candidates.sort_values(["HRP", "train_combo", "model", "pred_mean"], ascending=[True, True, True, False]).groupby(["HRP", "train_combo", "model"]).head(3)
    lines.append(format_table(top, ["HRP", "train_combo", "model", "TMB", "H2O2", "pred_mean", "latent_std", "nearest_train_logdist"]))
    lines.append("\n## Interpretation\n")
    lines.append(
        "- Fixed-noise GP improves matrix conditioning because repeated observations are collapsed to condition means and the likelihood uses condition-specific diagonal noise.\n"
        "- Raw SEM is unstable with only four replicates for LHS-only conditions; the shrunk SEM variant is the more defensible fixed-noise baseline.\n"
        "- BO2 holdout errors mainly measure run-to-run / temporal drift at the BO1-selected conditions. Fixed noise cannot fully correct this drift because it is not a batch-effect model.\n"
        "- If future BO will use candidates generated from Apr29-like data, use the shrunk fixed-noise GP or add a learned extra global noise term on top of SEM; do not rely on raw SEM alone.\n"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate current vs fixed-noise GP on Apr 29 LHS/BO1/BO2 data.")
    parser.add_argument("--data_dir", default="Apr_29_full_log")
    parser.add_argument("--target", default="AUC")
    parser.add_argument("--lower_bound", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--q", type=int, default=32)
    parser.add_argument("--num_restarts", type=int, default=20)
    parser.add_argument("--raw_samples", type=int, default=512)
    parser.add_argument("--shrink_alpha", type=float, default=0.5)
    parser.add_argument("--skip_loo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device("cpu")
    base = Path(args.data_dir)
    bounds = make_bounds(device, args.lower_bound)
    summary_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    holdout_rows: list[pd.DataFrame] = []
    drift_rows: list[pd.DataFrame] = []

    for hrp in HRPS:
        data_by_run = {run: read_run(base, run, hrp) for run in RUNS}
        drift = run_drift_summary(data_by_run, args.target)
        drift.insert(0, "HRP", hrp)
        drift_rows.append(drift)

        bo2_cond = prepare_condition_table(data_by_run["BO2"], args.target, args.shrink_alpha)

        for combo, runs in COMBOS.items():
            train_df = pd.concat([data_by_run[r] for r in runs], ignore_index=True)
            cond = prepare_condition_table(train_df, args.target, args.shrink_alpha)

            for model_name in MODELS:
                print(f"HRP={hrp} combo={combo} model={model_name}", flush=True)
                train_x, train_y, train_yvar, _ = model_train_data(
                    train_df, cond, args.target, model_name, device
                )
                with warnings.catch_warnings(record=True) as fit_warnings:
                    warnings.simplefilter("always")
                    model = fit_model(train_x, train_y, train_yvar, bounds)
                ls_tmb, ls_h2o2 = lengthscales(model)

                row: dict[str, Any] = {
                    "HRP": hrp,
                    "train_combo": combo,
                    "model": model_name,
                    "n_train": int(train_x.shape[0]),
                    "n_conditions": int(cond.shape[0]),
                    "train_y_mean": float(train_y.mean().item()),
                    "train_y_sd": float(train_y.std(unbiased=True).item()),
                    "lengthscale_TMB_normalized": ls_tmb,
                    "lengthscale_H2O2_normalized": ls_h2o2,
                    "fit_n_warnings": len(fit_warnings),
                }
                row.update(residual_and_matrix_stats(model, train_x, train_y, train_yvar))

                if not args.skip_loo:
                    row.update(condition_loo(train_df, cond, args.target, model_name, bounds, device))

                if combo == "LHS_BO1":
                    hold_metrics, pred = holdout_bo2_validation(model, bo2_cond, device)
                    row.update(hold_metrics)
                    pred.insert(0, "model", model_name)
                    pred.insert(0, "train_combo", combo)
                    pred.insert(0, "HRP", hrp)
                    holdout_rows.append(pred)

                t0 = time.time()
                acq_success = True
                acq_error = ""
                with warnings.catch_warnings(record=True) as acq_warnings:
                    warnings.simplefilter("always")
                    try:
                        cand_x = generate_candidates(
                            model,
                            bounds,
                            train_x,
                            seed=args.seed,
                            q=args.q,
                            num_restarts=args.num_restarts,
                            raw_samples=args.raw_samples,
                        )
                        cand_stats, cand_rows = candidate_stats_and_rows(
                            model, cand_x, train_x, hrp, combo, model_name
                        )
                        candidate_rows.extend(cand_rows)
                        row.update(cand_stats)
                    except Exception as exc:
                        acq_success = False
                        acq_error = f"{type(exc).__name__}: {exc}"
                row.update(
                    {
                        "acq_success": acq_success,
                        "acq_error": acq_error,
                        "acq_elapsed_sec": time.time() - t0,
                        "acq_n_warnings": len(acq_warnings),
                        "acq_n_optimization_warnings": sum(
                            "OptimizationWarning" in type(w.message).__name__
                            or "OptimizationWarning" in str(w.message)
                            for w in acq_warnings
                        ),
                        "acq_n_numerical_warnings": sum(
                            "NumericalWarning" in type(w.message).__name__
                            or "NumericalWarning" in str(w.message)
                            for w in acq_warnings
                        ),
                        "acq_warning_sample": " | ".join(str(w.message)[:220] for w in acq_warnings[:3]),
                    }
                )
                summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    candidates = pd.DataFrame(candidate_rows)
    holdout = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()
    drift = pd.concat(drift_rows, ignore_index=True)

    summary_path = base / "apr29_lhs_bo_validation_summary.csv"
    candidates_path = base / "apr29_lhs_bo_validation_candidates.csv"
    holdout_path = base / "apr29_lhs_bo_validation_bo2_holdout_predictions.csv"
    drift_path = base / "apr29_lhs_bo_validation_run_summary.csv"
    report_path = base / "apr29_lhs_bo_validation_report.md"

    summary.to_csv(summary_path, index=False)
    candidates.to_csv(candidates_path, index=False)
    holdout.to_csv(holdout_path, index=False)
    drift.to_csv(drift_path, index=False)
    report = build_report(summary, drift, candidates, holdout, args)
    report_path.write_text(report, encoding="utf-8")

    print(f"Saved {summary_path.resolve()}")
    print(f"Saved {candidates_path.resolve()}")
    print(f"Saved {holdout_path.resolve()}")
    print(f"Saved {drift_path.resolve()}")
    print(f"Saved {report_path.resolve()}")


if __name__ == "__main__":
    main()
