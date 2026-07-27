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

from data_loader_objectives import ADDITIVE_COLS, OBJECTIVE_COLS
from fit_model import matern_with_hvarfner_prior


torch.set_default_dtype(torch.double)

ESSENTIALS = ["TMB", "H2O2"]
PHYSICAL_BOUNDS = {"TMB": (0.005, 1.0), "H2O2": (0.005, 1.0)}
UNIFIED_OBJECTIVE_KEYS = {str(column).strip().lower() for column in OBJECTIVE_COLS}
LOGS_DIR = Path(__file__).with_name("logs")
DEFAULT_LOG_DIR = LOGS_DIR / "Jul_1_full_log"

# Experimental rounds in chronological order. BO is cumulative: a run is trained
# on every round up to and including run_type (pooled at the replicate level),
# and the candidate file is named by the round that FOLLOWS run_type. Diagnostics
# keep run_type because they describe the model fitted to the pooled data through
# that round. Example: run_type=BO1 pools {LHS, BO1} and targets candidates for BO2.
ROUND_SEQUENCE = ("LHS", "BO1", "BO2", "BO3", "BO4")


def cumulative_rounds(run_type: str) -> tuple[str, ...]:
    """Rounds whose replicate data is pooled to train the model for this run."""
    return ROUND_SEQUENCE[: ROUND_SEQUENCE.index(run_type) + 1]


def next_round(run_type: str) -> str:
    """Experimental round the generated candidates are meant to be run in."""
    return ROUND_SEQUENCE[ROUND_SEQUENCE.index(run_type) + 1]


def read_objective_table(path: Path, sheet_name: str = "Corrected Data") -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path.resolve()}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input suffix: {path.suffix}")


def resolve_target_column(df: pd.DataFrame, requested: str) -> str:
    requested_key = str(requested).strip().lower()
    if requested_key not in UNIFIED_OBJECTIVE_KEYS:
        raise ValueError(
            f"Target {requested!r} is not a unified objective. "
            f"Choose one of {OBJECTIVE_COLS}."
        )
    matches = [c for c in df.columns if str(c).strip().lower() == requested_key]
    if len(matches) == 1:
        return str(matches[0])
    available = [
        c for c in df.columns
        if str(c).strip().lower() in UNIFIED_OBJECTIVE_KEYS
    ]
    raise KeyError(
        f"Target column {requested!r} was not found uniquely. Available objectives: {available}"
    )


def infer_round_from_batch(batch: str) -> str | None:
    """Infer a canonical round from loader Batch labels such as BO2 or 7_1_BO2."""
    text = Path(str(batch).strip()).stem.upper()
    for round_name in ROUND_SEQUENCE:
        if text == round_name or text.endswith(f"_{round_name}"):
            return round_name
    return None


def validate_explicit_round_pool(
    df: pd.DataFrame,
    *,
    rounds_to_pool: tuple[str, ...],
    source: str,
) -> dict[str, str]:
    """Require an explicit unified input to cover exactly the cumulative rounds."""
    if "Batch" not in df.columns:
        raise KeyError(
            f"{source}: --input needs Batch labels so cumulative rounds can be audited. "
            "Use data_loader_objectives batches such as LHS/BO1/BO2/BO3, or pass "
            "--input_is_already_pooled as an explicit assertion."
        )
    labels = sorted(df["Batch"].astype(str).unique())
    inferred = {label: infer_round_from_batch(label) for label in labels}
    unknown = [label for label, round_name in inferred.items() if round_name is None]
    if unknown:
        raise ValueError(
            f"{source}: cannot infer experimental rounds from Batch labels {unknown}. "
            "Use labels ending in LHS/BO1/BO2/BO3, or pass "
            "--input_is_already_pooled as an explicit assertion."
        )
    observed = {round_name for round_name in inferred.values() if round_name is not None}
    expected = set(rounds_to_pool)
    if observed != expected:
        missing = sorted(expected - observed, key=ROUND_SEQUENCE.index)
        extra = sorted(observed - expected, key=ROUND_SEQUENCE.index)
        raise ValueError(
            f"{source}: --run_type requires cumulative rounds {list(rounds_to_pool)}, "
            f"but Batch implies observed rounds={sorted(observed, key=ROUND_SEQUENCE.index)}, "
            f"missing={missing}, extra={extra}. Do not label a single-round table as a "
            "cumulative BO run; pass --input_is_already_pooled only when the supplied "
            "table has already been pooled intentionally."
        )
    if "HRP" not in df.columns:
        raise KeyError(f"{source}: HRP is required to audit round coverage per model.")
    incomplete_hrp: list[dict[str, Any]] = []
    for hrp, group in df.groupby("HRP", sort=True):
        group_labels = set(group["Batch"].astype(str))
        group_rounds = {
            inferred[label] for label in group_labels if inferred[label] is not None
        }
        if group_rounds != expected:
            incomplete_hrp.append(
                {
                    "HRP": float(hrp),
                    "observed": sorted(group_rounds, key=ROUND_SEQUENCE.index),
                    "missing": sorted(expected - group_rounds, key=ROUND_SEQUENCE.index),
                    "extra": sorted(group_rounds - expected, key=ROUND_SEQUENCE.index),
                }
            )
    if incomplete_hrp:
        raise ValueError(
            f"{source}: cumulative round coverage must hold separately for every HRP "
            f"model; violations={incomplete_hrp}"
        )
    return {label: str(round_name) for label, round_name in inferred.items()}


def _finite_numeric(
    df: pd.DataFrame,
    columns: list[str],
    *,
    source: str,
) -> pd.DataFrame:
    values = df[columns].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(values.to_numpy(dtype=float))
    if invalid.any():
        bad_rows = np.flatnonzero(invalid.any(axis=1))[:5].tolist()
        raise ValueError(
            f"{source}: non-numeric or non-finite values in {columns}; "
            f"first zero-based row indices={bad_rows}"
        )
    return values.astype(float)


def _boolean_series(values: pd.Series, *, label: str, source: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        bad = values.loc[mapped.isna()].head(5).tolist()
        raise ValueError(f"{source}: invalid {label} values; first examples={bad}")
    return mapped.astype(bool)


def _bad_row_examples(df: pd.DataFrame, mask: pd.Series) -> list[dict[str, Any]]:
    trace = [
        c for c in ["Batch", "source_file", "source_sheet", "source_row"]
        if c in df.columns
    ]
    if not trace:
        return [{"row_index": int(i)} for i in df.index[mask][:5]]
    return df.loc[mask, trace].head(5).to_dict("records")


def validate_ess_training_table(
    df: pd.DataFrame,
    *,
    target: str,
    source: str,
) -> tuple[pd.DataFrame, str]:
    """Validate that every row is safe for the essentials-only fixed-noise model."""
    if df.empty:
        raise ValueError(f"{source}: input table has no rows.")
    normalized_columns = pd.Index([str(c).strip().lower() for c in df.columns])
    duplicate_columns = normalized_columns[normalized_columns.duplicated()].unique().tolist()
    if duplicate_columns:
        raise ValueError(
            f"{source}: duplicate columns after case normalization: {duplicate_columns}"
        )
    required = ["HRP", *ESSENTIALS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{source}: missing required columns {missing}")

    target_col = resolve_target_column(df, target)
    if target_col in required or target_col in ADDITIVE_COLS or target_col == "CTRL":
        raise ValueError(f"Target {target_col!r} is a recipe feature or metadata column.")
    out = df.copy()
    numeric = _finite_numeric(out, required + [target_col], source=source)
    out[required + [target_col]] = numeric
    if (out["HRP"] <= 0.0).any():
        raise ValueError(f"{source}: HRP must be strictly positive.")
    for essential in ESSENTIALS:
        lo, hi = PHYSICAL_BOUNDS[essential]
        outside = ~out[essential].between(lo, hi, inclusive="both")
        if outside.any():
            examples = out.loc[outside, essential].head(5).tolist()
            raise ValueError(
                f"{source}: {essential} must lie in [{lo}, {hi}]; "
                f"first out-of-bounds values={examples}"
            )

    present_additives = [c for c in ADDITIVE_COLS if c in out.columns]
    unified_markers = {"composition_type", "is_ess_only", "n_active", "source_layout"}
    present_markers = unified_markers.intersection(out.columns)
    if present_additives and len(present_additives) != len(ADDITIVE_COLS):
        missing_additives = [c for c in ADDITIVE_COLS if c not in out.columns]
        raise KeyError(f"{source}: incomplete additive schema; missing={missing_additives}")
    if present_markers and not present_additives:
        raise KeyError(f"{source}: unified trace columns are present but additive columns are missing.")
    if present_additives and present_markers != unified_markers:
        missing_markers = sorted(unified_markers - present_markers)
        raise KeyError(f"{source}: incomplete unified trace schema; missing={missing_markers}")
    if present_additives:
        additive_values = _finite_numeric(out, ADDITIVE_COLS, source=source)
        if (additive_values < 0.0).any().any():
            raise ValueError(f"{source}: additive concentrations must be non-negative.")
        out[ADDITIVE_COLS] = additive_values
        active_rows = (additive_values > 0.0).any(axis=1)
        if active_rows.any():
            raise ValueError(
                f"{source}: fixed_noise_ess_bo accepts only ESS rows, but found "
                f"{int(active_rows.sum())} rows with active additives; "
                f"examples={_bad_row_examples(out, active_rows)}"
            )

    if "n_active" in out.columns:
        n_active = pd.to_numeric(out["n_active"], errors="coerce")
        invalid = ~np.isfinite(n_active.to_numpy(dtype=float)) | (n_active != 0)
        if invalid.any():
            mask = pd.Series(invalid, index=out.index)
            raise ValueError(
                f"{source}: n_active must be zero for every row; "
                f"examples={_bad_row_examples(out, mask)}"
            )
        out["n_active"] = n_active.astype(int)
    if "is_ess_only" in out.columns:
        is_ess = _boolean_series(out["is_ess_only"], label="is_ess_only", source=source)
        if (~is_ess).any():
            raise ValueError(
                f"{source}: is_ess_only must be true for every row; "
                f"examples={_bad_row_examples(out, ~is_ess)}"
            )
        out["is_ess_only"] = is_ess
    if "composition_type" in out.columns:
        composition = out["composition_type"].astype(str).str.strip().str.upper()
        invalid = composition != "ESS"
        if invalid.any():
            raise ValueError(
                f"{source}: composition_type must be ESS for every row; "
                f"examples={_bad_row_examples(out, invalid)}"
            )
    if "Batch" in out.columns:
        batch_text = out["Batch"].astype("string").str.strip()
        if batch_text.isna().any() or batch_text.eq("").any():
            raise ValueError(f"{source}: Batch labels must be non-empty when present.")
        out["Batch"] = batch_text.astype(str)
    if "source_layout" in out.columns:
        layout = out["source_layout"].astype(str).str.strip().str.lower()
        unknown_layout = ~layout.isin({"essentials", "additive"})
        if unknown_layout.any():
            raise ValueError(
                f"{source}: invalid source_layout values; "
                f"examples={out.loc[unknown_layout, 'source_layout'].head(5).tolist()}"
            )
        out["source_layout"] = layout

    if "CTRL" in out.columns:
        ctrl = _finite_numeric(out, ["CTRL"], source=source)["CTRL"]
        if (ctrl < 0.0).any():
            raise ValueError(f"{source}: CTRL markers must be non-negative.")
        if (ctrl > 0.0).any():
            raise ValueError(f"{source}: nonzero CTRL rows are not valid ESS training rows.")
        out["CTRL"] = ctrl

    provenance = ["source_file", "source_sheet", "source_row"]
    present_provenance = [c for c in provenance if c in out.columns]
    if present_provenance and len(present_provenance) != len(provenance):
        raise KeyError(f"{source}: incomplete provenance columns {present_provenance}")
    if all(c in out.columns for c in provenance):
        if out[provenance].isna().any().any():
            raise ValueError(f"{source}: provenance columns must not contain missing values.")
        duplicated = out.duplicated(provenance, keep=False)
        if duplicated.any():
            raise ValueError(
                f"{source}: duplicate provenance rows detected; "
                f"examples={_bad_row_examples(out, duplicated)}"
            )
    if "n_frames" in out.columns:
        n_frames = pd.to_numeric(out["n_frames"], errors="coerce")
        if (
            (~np.isfinite(n_frames.to_numpy(dtype=float))).any()
            or (n_frames < 2).any()
            or (n_frames != np.floor(n_frames)).any()
            or n_frames.nunique() != 1
        ):
            raise ValueError(f"{source}: n_frames must be one finite, common value >= 2.")
        out["n_frames"] = n_frames.astype(int)
    return out, target_col


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
    insufficient = grouped["count"] < 2
    if insufficient.any():
        examples = grouped.loc[insufficient, ESSENTIALS + ["count"]].head(5).to_dict("records")
        raise ValueError(
            "Fixed-noise SEM requires at least two replicates per TMB/H2O2 "
            f"condition; examples={examples}"
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
        covar_module=matern_with_hvarfner_prior(d),
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
        sequential=True,
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


def run_for_hrp(
    args: argparse.Namespace,
    hrp: str,
    df: pd.DataFrame,
    out_dir: Path,
    diag_dir: Path,
    cand_path: Path,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    device = torch.device(args.device)
    bounds = make_bounds(device)
    cond = prepare_condition_table(df, args.target, args.shrink_alpha)
    if not args.candidates_only:
        cond_out = diag_dir / f"{args.run_type}_HRP_{hrp}.csv"
        cond.to_csv(cond_out, index=False)

    grid_x = None if args.candidates_only else make_grid(bounds, args.grid_size, device)
    summary_rows: list[dict[str, Any]] = []
    cand_frames: list[pd.DataFrame] = []

    for model_name in args.models:
        print(f"\n=== HRP={hrp} model={model_name} ===", flush=True)
        train_x, train_y, train_yvar, baseline_df = model_train_data(
            df, cond, args.target, model_name, device
        )
        model = fit_model(train_x, train_y, bounds, train_yvar)

        loo_stats: dict[str, float] = {}
        if args.loo and not args.candidates_only:
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
        cand.insert(0, "HRP", float(hrp))
        cand.insert(1, "model", model_name)
        cand_frames.append(cand)

        extra = {
            "candidate_pred_max": float(cand["predicted_value"].max()),
            "candidate_pred_min": float(cand["predicted_value"].min()),
            "candidate_latent_std_mean": float(cand["uncertainty_std"].mean()),
            "candidate_latent_std_max": float(cand["uncertainty_std"].max()),
            "candidate_default_observation_std_mean": float(
                cand["default_observation_std"].mean()
            ),
            "candidate_file": str(cand_path),
            "baseline_rows": int(baseline_df.shape[0]),
        }
        if not args.candidates_only:
            cov_path = diag_dir / f"{args.run_type}_HRP_{hrp}_{model_name}_candidate_latent_cov.csv"
            covariance_frame(model, cand_x).to_csv(cov_path, index=False)

            grid = posterior_frame(model, grid_x)
            grid_path = diag_dir / f"{args.run_type}_HRP_{hrp}_{model_name}_posterior_grid.csv"
            grid.to_csv(grid_path, index=False)

            in_sample = posterior_frame(model, train_x)
            in_sample["training_target"] = train_y.detach().cpu().numpy().reshape(-1)
            in_sample_path = diag_dir / f"{args.run_type}_HRP_{hrp}_{model_name}_posterior_train.csv"
            in_sample.to_csv(in_sample_path, index=False)

            extra.update(
                {
                    "posterior_grid_file": str(grid_path),
                    "posterior_train_file": str(in_sample_path),
                    "candidate_cov_file": str(cov_path),
                }
            )
        extra.update(loo_stats)
        summary_rows.append(
            summarize_model(hrp, model_name, model, train_y, train_yvar, cond, extra)
        )

    return summary_rows, cand_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare replicate-level inferred-noise GP with condition-mean fixed-noise GP."
    )
    parser.add_argument(
        "--input",
        default=None,
        help=(
            "Optional unified replicate-level .xlsx/.xls/.csv from "
            "data_loader_objectives. When supplied, this exact table replaces "
            "legacy cumulative round-file discovery."
        ),
    )
    parser.add_argument(
        "--input_sheet",
        default="Corrected Data",
        help="Worksheet used by --input for Excel files.",
    )
    parser.add_argument(
        "--input_is_already_pooled",
        action="store_true",
        help=(
            "Explicitly assert that --input is already pooled for --run_type when "
            "its Batch labels cannot prove cumulative-round coverage. This bypasses "
            "only the round-label check; active-additive rows are still rejected."
        ),
    )
    parser.add_argument("--input_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--input_prefix", default="7_1")
    parser.add_argument(
        "--run_type",
        choices=["LHS", "BO1", "BO2", "BO3"],
        default="BO3",
        help=(
            "Latest round of data on hand. The model is trained on every round up to "
            "and including this one (cumulative pooling at the replicate level); "
            "candidates are named for the next round."
        ),
    )
    parser.add_argument("--target", default="AUC")
    parser.add_argument(
        "--hrp",
        nargs="+",
        default=None,
        help="Optional HRP level(s); by default, run every HRP level present in the input.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["fixed_sem_shrunk"],
        choices=["fixed_sem_raw", "fixed_sem_shrunk"],
    )
    parser.add_argument("--out_dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--q", type=int, default=32)
    parser.add_argument("--num_restarts", type=int, default=20)
    parser.add_argument("--raw_samples", type=int, default=512)
    parser.add_argument("--grid_size", type=int, default=41)
    parser.add_argument("--shrink_alpha", type=float, default=0.5)
    parser.add_argument("--loo", action="store_true")
    parser.add_argument(
        "--candidates_only",
        default=False,
        action="store_true",
        help="Only save candidate CSV files; skip condition-noise, posterior grid, train posterior, covariance, summary, and LOO outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.shrink_alpha <= 1.0:
        raise ValueError("--shrink_alpha must lie in [0, 1].")
    if args.input_is_already_pooled and not args.input:
        raise ValueError("--input_is_already_pooled is valid only together with --input.")
    set_seeds(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diag"
    diag_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Fixed-noise probe config: "
        + json.dumps(vars(args), ensure_ascii=True, sort_keys=True),
        flush=True,
    )
    warnings.filterwarnings("default")

    rounds_to_pool = cumulative_rounds(args.run_type)
    if args.input:
        input_path = Path(args.input)
        df_all = read_objective_table(input_path, sheet_name=args.input_sheet)
        input_label = f"explicit unified table {input_path.resolve()}"
    else:
        input_dir = Path(args.input_dir)
        frames: list[pd.DataFrame] = []
        for r in rounds_to_pool:
            round_path = input_dir / f"{args.input_prefix}_{r}_HRP_all_res.xlsx"
            if not round_path.exists():
                raise FileNotFoundError(
                    f"Cumulative training needs round '{r}' but {round_path} is missing. "
                    f"Run: data_loader_objectives --run_type {r} --keep_all_hrp"
                )
            frames.append(pd.read_excel(round_path))
        df_all = pd.concat(frames, ignore_index=True)
        input_label = f"legacy cumulative rounds {list(rounds_to_pool)}"

    df_all, args.target = validate_ess_training_table(
        df_all,
        target=args.target,
        source=input_label,
    )

    available_hrp = np.sort(df_all["HRP"].unique())
    requested_hrp = available_hrp.tolist() if args.hrp is None else args.hrp
    if not requested_hrp:
        raise ValueError("No HRP levels were requested or found in the input.")
    hrp_jobs: list[tuple[float, pd.Series]] = []
    selected_hrp: list[float] = []
    for requested in requested_hrp:
        requested_value = float(requested)
        if not np.isfinite(requested_value) or requested_value <= 0.0:
            raise ValueError(f"Requested HRP must be finite and positive; got {requested!r}.")
        hrp_mask = np.isclose(
            df_all["HRP"].astype(float), requested_value, rtol=0.0, atol=1e-12
        )
        matched_levels = np.sort(df_all.loc[hrp_mask, "HRP"].unique())
        if len(matched_levels) != 1:
            raise ValueError(
                f"Requested HRP={requested!r} must match exactly one actual level; "
                f"matched={matched_levels.tolist()}, available={available_hrp.tolist()}"
            )
        actual_hrp = float(matched_levels[0])
        if actual_hrp in selected_hrp:
            raise ValueError(
                f"Requested HRP values contain duplicate matches to actual level {actual_hrp}."
            )
        selected_hrp.append(actual_hrp)
        hrp_jobs.append((actual_hrp, hrp_mask))
    selected_df = df_all.loc[df_all["HRP"].isin(selected_hrp)].copy()

    round_map: dict[str, str] = {}
    if args.input and not args.input_is_already_pooled:
        round_map = validate_explicit_round_pool(
            selected_df,
            rounds_to_pool=rounds_to_pool,
            source=input_label,
        )
    elif args.input:
        print(
            "[Data guard] --input_is_already_pooled asserted: Batch labels are not "
            "being used to verify cumulative-round coverage.",
            flush=True,
        )
    batch_counts = (
        selected_df["Batch"].astype(str).value_counts().sort_index().to_dict()
        if "Batch" in selected_df.columns else {}
    )
    print(
        f"Loaded {input_label} -> {len(df_all)} ESS replicate rows, "
        f"selected {len(selected_df)} rows and "
        f"{selected_df.groupby(ESSENTIALS + ['HRP']).ngroups} conditions for "
        f"actual HRP levels={selected_hrp}; "
        f"Batch counts={batch_counts}; inferred rounds={round_map}; "
        f"candidates target round {next_round(args.run_type)}.",
        flush=True,
    )
    cand_path = out_dir / f"{next_round(args.run_type)}_candidates_all.csv"

    rows: list[dict[str, Any]] = []
    cand_frames: list[pd.DataFrame] = []
    for actual_hrp, hrp_mask in hrp_jobs:
        df_hrp = df_all.loc[hrp_mask].copy()
        hrp_rows, hrp_cands = run_for_hrp(
            args, str(actual_hrp), df_hrp, out_dir, diag_dir, cand_path
        )
        rows.extend(hrp_rows)
        cand_frames.extend(hrp_cands)

    combined = pd.concat(cand_frames, ignore_index=True)
    combined.to_csv(cand_path, index=False)
    print(f"\nSaved combined candidates ({len(combined)} rows) -> {cand_path.resolve()}", flush=True)

    if not args.candidates_only:
        summary = pd.DataFrame(rows)
        summary_path = diag_dir / f"fixed_noise_gp_{args.run_type}_probe_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary -> {summary_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
