# -*- coding: utf-8 -*-
"""Run a per-additive Hamming-theta baseline experiment for add_bo.

This script intentionally leaves add_bo.py unchanged. It reuses the project data
loading, fitting, and diagnostics helpers, and adds one experimental baseline:

    k_cat(x, x') = exp(-sum_i theta_i * 1[b_i != b'_i])

where the original BaselineKernel uses one shared scalar theta.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import LogNormalPrior

import add_bo


class PerAdditiveThetaKernel(Kernel):
    r"""BaselineKernel variant with one Hamming decay per additive indicator."""

    has_lengthscale = False

    def __init__(
        self,
        ess_dims: list[int],
        cat_dims: list[int],
        conc_dims: list[int],
        additive_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        if len(cat_dims) != len(conc_dims):
            raise ValueError(
                f"cat_dims length {len(cat_dims)} does not match "
                f"conc_dims length {len(conc_dims)}."
            )
        if additive_names is not None and len(additive_names) != len(cat_dims):
            raise ValueError(
                f"additive_names length {len(additive_names)} does not match "
                f"cat_dims length {len(cat_dims)}."
            )
        self.additive_names = list(additive_names) if additive_names is not None else None
        self.register_buffer("ess_dims", torch.tensor(ess_dims, dtype=torch.long))
        self.register_buffer("cat_dims", torch.tensor(cat_dims, dtype=torch.long))
        self.register_buffer("conc_dims", torch.tensor(conc_dims, dtype=torch.long))

        n_ess = len(ess_dims)
        n_conc = len(conc_dims)
        n_cat = len(cat_dims)

        self.register_parameter(
            "raw_outputscale", torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        )
        self.register_parameter(
            "raw_ess_lengthscale",
            torch.nn.Parameter(torch.zeros(n_ess, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_conc_lengthscale",
            torch.nn.Parameter(torch.zeros(n_conc, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_theta", torch.nn.Parameter(torch.zeros(n_cat, dtype=torch.double))
        )
        for raw_name in (
            "raw_outputscale",
            "raw_ess_lengthscale",
            "raw_conc_lengthscale",
            "raw_theta",
        ):
            self.register_constraint(raw_name, Positive())

        init_outputscale = torch.tensor(1.0, dtype=torch.double)
        init_ess_lengthscale = torch.full((n_ess,), 0.35, dtype=torch.double)
        init_conc_lengthscale = torch.full((n_conc,), 0.3, dtype=torch.double)
        init_theta = torch.full((n_cat,), 0.5, dtype=torch.double)
        sigma = torch.tensor(0.75, dtype=torch.double)

        self.register_prior(
            "outputscale_prior",
            LogNormalPrior(loc=init_outputscale.log(), scale=sigma),
            lambda m: m.outputscale,
            lambda m, v: m._set_outputscale(v),
        )
        self.register_prior(
            "ess_lengthscale_prior",
            LogNormalPrior(
                loc=init_ess_lengthscale.log(),
                scale=torch.full_like(init_ess_lengthscale, 0.75),
            ),
            lambda m: m.ess_lengthscale,
            lambda m, v: m._set_ess_lengthscale(v),
        )
        self.register_prior(
            "conc_lengthscale_prior",
            LogNormalPrior(
                loc=init_conc_lengthscale.log(),
                scale=torch.full_like(init_conc_lengthscale, 0.75),
            ),
            lambda m: m.conc_lengthscale,
            lambda m, v: m._set_conc_lengthscale(v),
        )
        self.register_prior(
            "theta_prior",
            LogNormalPrior(
                loc=init_theta.log(),
                scale=torch.full_like(init_theta, 0.75),
            ),
            lambda m: m.theta,
            lambda m, v: m._set_theta(v),
        )

        self.initialize(
            raw_outputscale=self.raw_outputscale_constraint.inverse_transform(
                init_outputscale
            ),
            raw_ess_lengthscale=self.raw_ess_lengthscale_constraint.inverse_transform(
                init_ess_lengthscale
            ),
            raw_conc_lengthscale=self.raw_conc_lengthscale_constraint.inverse_transform(
                init_conc_lengthscale
            ),
            raw_theta=self.raw_theta_constraint.inverse_transform(init_theta),
        )

    @property
    def outputscale(self) -> torch.Tensor:
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    def _set_outputscale(self, value: torch.Tensor) -> None:
        value = value.to(self.raw_outputscale).clamp_min(1e-12)
        self.initialize(
            raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value)
        )

    @property
    def ess_lengthscale(self) -> torch.Tensor:
        return self.raw_ess_lengthscale_constraint.transform(self.raw_ess_lengthscale)

    def _set_ess_lengthscale(self, value: torch.Tensor) -> None:
        value = value.to(self.raw_ess_lengthscale).clamp_min(1e-12)
        self.initialize(
            raw_ess_lengthscale=self.raw_ess_lengthscale_constraint.inverse_transform(
                value
            )
        )

    @property
    def conc_lengthscale(self) -> torch.Tensor:
        return self.raw_conc_lengthscale_constraint.transform(self.raw_conc_lengthscale)

    def _set_conc_lengthscale(self, value: torch.Tensor) -> None:
        value = value.to(self.raw_conc_lengthscale).clamp_min(1e-12)
        self.initialize(
            raw_conc_lengthscale=self.raw_conc_lengthscale_constraint.inverse_transform(
                value
            )
        )

    @property
    def theta(self) -> torch.Tensor:
        return self.raw_theta_constraint.transform(self.raw_theta)

    def _set_theta(self, value: torch.Tensor) -> None:
        value = value.to(self.raw_theta).clamp_min(1e-12)
        self.initialize(raw_theta=self.raw_theta_constraint.inverse_transform(value))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "PerAdditiveThetaKernel does not support last_dim_is_batch."
            )
        e1 = x1[..., self.ess_dims]
        e2 = x2[..., self.ess_dims]
        c1 = x1[..., self.conc_dims]
        c2 = x2[..., self.conc_dims]
        b1 = (x1[..., self.cat_dims] > 0.5).to(dtype=x1.dtype)
        b2 = (x2[..., self.cat_dims] > 0.5).to(dtype=x2.dtype)

        ess_diff = (e1.unsqueeze(-2) - e2.unsqueeze(-3)) / self.ess_lengthscale
        ess_quad = ess_diff.pow(2).sum(dim=-1)
        conc_diff = (c1.unsqueeze(-2) - c2.unsqueeze(-3)) / self.conc_lengthscale
        conc_quad = conc_diff.pow(2).sum(dim=-1)
        k_cont = torch.exp(-0.5 * (ess_quad + conc_quad))

        mismatch = (b1.unsqueeze(-2) != b2.unsqueeze(-3)).to(dtype=x1.dtype)
        weighted_hamming = (mismatch * self.theta).sum(dim=-1)
        k_cat = torch.exp(-weighted_hamming)

        cov = self.outputscale * k_cont * k_cat
        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


def make_per_add_theta_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    codec: add_bo.FlatCodec,
    bounds: torch.Tensor,
) -> SingleTaskGP:
    covar_module = PerAdditiveThetaKernel(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
        additive_names=codec.adds,
    )
    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar_module,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def per_add_theta_parameter_summary(model: SingleTaskGP) -> dict[str, Any]:
    covar = model.covar_module
    if not isinstance(covar, PerAdditiveThetaKernel):
        raise TypeError(f"Expected PerAdditiveThetaKernel, got {type(covar).__name__}.")
    names = covar.additive_names or [f"add_{i}" for i in range(len(covar.theta))]
    ess_l = covar.ess_lengthscale.detach().cpu().numpy()
    conc_l = covar.conc_lengthscale.detach().cpu().numpy()
    theta = covar.theta.detach().cpu().numpy()
    return {
        "per_add_theta_outputscale": float(covar.outputscale.detach().cpu()),
        "per_add_theta_ess_lengthscale": [float(v) for v in ess_l.reshape(-1)],
        "per_add_theta_conc_lengthscale": {
            n: float(v) for n, v in zip(names, conc_l)
        },
        "per_add_theta_theta": {n: float(v) for n, v in zip(names, theta)},
        "per_add_theta_theta_min": float(np.min(theta)),
        "per_add_theta_theta_median": float(np.median(theta)),
        "per_add_theta_theta_max": float(np.max(theta)),
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                row[f"{key}__{sub_key}"] = sub_value
        else:
            row[key] = value
    return row


def summarize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    key_metrics = [
        "mll",
        "fit_seconds",
        "train_rmse",
        "train_mae",
        "train_r2",
        "train_z_std",
        "noise_standardized",
        "eig_min",
        "eig_max",
        "condition_number",
        "loo_rmse_model_space",
        "loo_mae_model_space",
        "loo_z_mean",
        "loo_z_std",
        "loo_frac_abs_z_gt_2",
        "loo_frac_abs_z_gt_3",
        "loo_cholesky_jitter",
        "kernel_diag_min",
        "kernel_diag_max",
    ]
    rows = []
    for model, group in df.groupby("model", sort=False):
        row = {"model": model, "n_seeds": int(group["seed"].nunique())}
        for metric in key_metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(
    output_dir: Path,
    data_summary: dict[str, Any],
    per_summary: pd.DataFrame,
    comparison_summary: pd.DataFrame | None,
    previous_rows_path: Path,
) -> Path:
    report_path = output_dir / "per_add_theta_report.md"

    def fmt(value: Any) -> str:
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return str(value)

    lines = [
        "# Per-additive Hamming theta experiment",
        "",
        "## Scope",
        "",
        "- New kernel: `PerAdditiveThetaKernel`, a `BaselineKernel` variant with `exp(-sum_i theta_i * mismatch_i)` instead of `exp(-theta * sum_i mismatch_i)`.",
        "- Priors and initial values otherwise match the scalar-theta baseline: outputscale 1.0, essential lengthscales 0.35, concentration lengthscales 0.3, theta/theta_i 0.5, LogNormal prior sigma 0.75.",
        "- Candidate generation and acquisition-stability diagnostics were skipped; this is a GP fit / fixed-hyperparameter LOO diagnostic experiment.",
        f"- Previous comparison source: `{previous_rows_path.as_posix()}`.",
        "",
        "## Data",
        "",
        f"- Filtered rows: {data_summary['rows_after_filters']}.",
        f"- Unique encoded rows: {data_summary['unique_encoded_rows']}.",
        f"- Duplicates merged: {data_summary['duplicates_merged']}.",
        f"- Encoded dimensions: {data_summary['encoded_dim']}; additives: {data_summary['n_additives']}.",
        f"- Active-count distribution: {data_summary['active_count_dist']}.",
        "",
        "## Per-additive theta result",
        "",
    ]
    per_row = per_summary.iloc[0]
    lines.extend(
        [
            "| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| {per_row['model']} | {fmt(per_row['mll_mean'])} | "
                f"{fmt(per_row['loo_rmse_model_space_mean'])} | "
                f"{fmt(per_row['loo_z_std_mean'])} | "
                f"{fmt(per_row['train_rmse_mean'])} | "
                f"{fmt(per_row['noise_standardized_mean'])} | "
                f"{fmt(per_row['condition_number_mean'])} |"
            ),
            "",
        ]
    )

    if comparison_summary is not None:
        lines.extend(
            [
                "## Comparison with previous kernels",
                "",
                "| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in comparison_summary.iterrows():
            lines.append(
                f"| {row['model']} | {fmt(row['mll_mean'])} | "
                f"{fmt(row['loo_rmse_model_space_mean'])} | "
                f"{fmt(row['loo_z_std_mean'])} | "
                f"{fmt(row['train_rmse_mean'])} | "
                f"{fmt(row['noise_standardized_mean'])} | "
                f"{fmt(row['condition_number_mean'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Cautions",
            "",
            "- The added theta_i parameters increase categorical flexibility from 1 to 22 parameters; this is weakly supported by a dataset containing only 0/1/2-active-additive observations.",
            "- LOO RMSE is in standardized model space, matching `add_bo.py`.",
            "- These diagnostics do not validate extrapolation to 3/4-additive recipes.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run per-additive Hamming theta GP/LOOCV diagnostics."
    )
    parser.add_argument(
        "--input",
        default="logs/May_09_full_log/data_corrected.xlsx",
    )
    parser.add_argument("--sheet", default="Corrected Data")
    parser.add_argument("--target", default="intensity")
    parser.add_argument("--hrp", type=float, default=0.0001)
    parser.add_argument("--hrp_atol", type=float, default=1e-12)
    parser.add_argument(
        "--filter_ctrl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--k_max", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--fit_maxiter", type=int, default=75)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument(
        "--output_dir",
        default="logs/May_09_full_log/add_bo_per_add_theta_gp_loocv",
    )
    parser.add_argument(
        "--previous_rows",
        default=(
            "logs/May_09_full_log/add_bo_multiseed_gp_loocv/"
            "all_seed_model_diagnostics.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    df, codec = add_bo.load_training_data(args)
    X, Y, encode_meta = add_bo.encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    data_summary = {**add_bo.dataset_summary(df, codec, args.target), **encode_meta}

    rows = []
    for seed in args.seeds:
        add_bo.set_seeds(seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        model = make_per_add_theta_model(X, Y, codec, bounds)
        fit_info = add_bo.fit_model(model, maxiter=args.fit_maxiter)
        diagnostics: dict[str, Any] = {"seed": seed, "model": "per_add_theta_kernel"}
        diagnostics.update(fit_info)
        diagnostics.update(add_bo.training_fit_diagnostics(model, X, Y))
        diagnostics.update(add_bo.matrix_diagnostics(model, X))
        diagnostics.update(add_bo.loo_diagnostics(model, X))
        diagnostics.update(per_add_theta_parameter_summary(model))

        row = flatten_metrics(diagnostics)
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            seed_dir / "add_bo_diagnostics.csv",
            index=False,
            encoding="utf-8",
        )
        with (seed_dir / "add_bo_diagnostics.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "data": data_summary,
                    "models": {"per_add_theta_kernel": diagnostics},
                },
                f,
                indent=2,
            )
        print(
            f"seed={seed} mll={fit_info['mll']:.6g} "
            f"fit_seconds={fit_info['fit_seconds']:.1f}"
        )

    per_df = pd.DataFrame(rows)
    per_rows_path = output_dir / "per_add_theta_all_seed_diagnostics.csv"
    per_df.to_csv(per_rows_path, index=False, encoding="utf-8")

    per_summary = summarize_metrics(per_df)
    per_summary_path = output_dir / "per_add_theta_metric_summary.csv"
    per_summary.to_csv(per_summary_path, index=False, encoding="utf-8")

    previous_rows_path = Path(args.previous_rows)
    comparison_summary = None
    if previous_rows_path.exists():
        previous_df = pd.read_csv(previous_rows_path)
        comparison_df = pd.concat([previous_df, per_df], ignore_index=True, sort=False)
        comparison_path = output_dir / "comparison_all_seed_diagnostics.csv"
        comparison_df.to_csv(comparison_path, index=False, encoding="utf-8")
        comparison_summary = summarize_metrics(comparison_df)
        comparison_summary_path = output_dir / "comparison_metric_summary.csv"
        comparison_summary.to_csv(
            comparison_summary_path,
            index=False,
            encoding="utf-8",
        )

    report_path = write_report(
        output_dir=output_dir,
        data_summary=data_summary,
        per_summary=per_summary,
        comparison_summary=comparison_summary,
        previous_rows_path=previous_rows_path,
    )

    print(f"[Output] per-seed rows -> {per_rows_path.resolve()}")
    print(f"[Output] per-additive theta summary -> {per_summary_path.resolve()}")
    if comparison_summary is not None:
        print(f"[Output] comparison summary -> {comparison_summary_path.resolve()}")
    print(f"[Output] report -> {report_path.resolve()}")


if __name__ == "__main__":
    main()
