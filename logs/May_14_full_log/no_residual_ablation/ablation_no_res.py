# -*- coding: utf-8 -*-
"""Ablation: AdditiveSetKernel with vs without the s_res * k_set * k_all_conc term.

The original kernel carries five scales (ess, set, main, pair, residual). The
residual fitted to ~0.02 in the May_13 baseline, suggesting it is dead weight.
This script fits the original five-scale kernel and a four-scale variant on
the same data with the same seed, and reports paired diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.priors import LogNormalPrior

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from add_bo import (  # noqa: E402
    AdditiveSetKernel,
    acq_stability_diagnostics,
    dataset_summary,
    encode_training_data,
    fit_model,
    load_training_data,
    loo_diagnostics,
    make_candidate_pool,
    matrix_diagnostics,
    set_seeds,
    training_fit_diagnostics,
)

torch.set_default_dtype(torch.double)


class AdditiveSetKernelNoRes(Kernel):
    """Four-scale variant: (ess, set, main, pair). Drops the residual term."""

    has_lengthscale = False

    def __init__(
        self,
        ess_dims: list[int],
        cat_dims: list[int],
        conc_dims: list[int],
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("ess_dims", torch.tensor(ess_dims, dtype=torch.long))
        self.register_buffer("cat_dims", torch.tensor(cat_dims, dtype=torch.long))
        self.register_buffer("conc_dims", torch.tensor(conc_dims, dtype=torch.long))
        self.register_parameter(
            "raw_scales",
            torch.nn.Parameter(torch.zeros(4, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_ess_lengthscale",
            torch.nn.Parameter(torch.zeros(len(ess_dims), dtype=torch.double)),
        )
        self.register_parameter(
            "raw_conc_lengthscale",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_scales", Positive())
        self.register_constraint("raw_ess_lengthscale", Positive())
        self.register_constraint("raw_conc_lengthscale", Positive())
        initial_scales = torch.tensor([0.15, 0.35, 0.75, 0.25], dtype=torch.double)
        self.register_prior(
            "scales_prior",
            LogNormalPrior(
                loc=initial_scales.log(),
                scale=torch.full_like(initial_scales, 0.75),
            ),
            lambda module: module.scales,
            lambda module, value: module._set_scales(value),
        )
        self.initialize(
            raw_scales=self.raw_scales_constraint.inverse_transform(initial_scales),
            raw_ess_lengthscale=self.raw_ess_lengthscale_constraint.inverse_transform(
                torch.full((len(ess_dims),), 0.35, dtype=torch.double)
            ),
            raw_conc_lengthscale=self.raw_conc_lengthscale_constraint.inverse_transform(
                torch.tensor([0.25], dtype=torch.double)
            ),
        )

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales_constraint.transform(self.raw_scales)

    def _set_scales(self, value: torch.Tensor) -> None:
        value = value.to(device=self.raw_scales.device, dtype=self.raw_scales.dtype)
        value = value.clamp_min(1e-12)
        self.initialize(raw_scales=self.raw_scales_constraint.inverse_transform(value))

    @property
    def ess_lengthscale(self) -> torch.Tensor:
        return self.raw_ess_lengthscale_constraint.transform(self.raw_ess_lengthscale)

    @property
    def conc_lengthscale(self) -> torch.Tensor:
        return self.raw_conc_lengthscale_constraint.transform(self.raw_conc_lengthscale)

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
                "AdditiveSetKernelNoRes does not support last_dim_is_batch."
            )
        e1 = x1[..., self.ess_dims]
        e2 = x2[..., self.ess_dims]
        b1 = (x1[..., self.cat_dims] > 0.5).to(dtype=x1.dtype)
        b2 = (x2[..., self.cat_dims] > 0.5).to(dtype=x2.dtype)
        c1 = x1[..., self.conc_dims]
        c2 = x2[..., self.conc_dims]

        ess_diff = (e1.unsqueeze(-2) - e2.unsqueeze(-3)) / self.ess_lengthscale
        k_ess = torch.exp(-0.5 * ess_diff.pow(2).sum(dim=-1))

        b1u = b1.unsqueeze(-2)
        b2u = b2.unsqueeze(-3)
        shared = b1u * b2u
        intersection = shared.sum(dim=-1)
        union = b1.sum(dim=-1).unsqueeze(-1) + b2.sum(dim=-1).unsqueeze(-2) - intersection
        k_set = torch.where(
            union > self.eps,
            intersection / union.clamp_min(self.eps),
            torch.ones_like(union),
        )

        conc_diff = (c1.unsqueeze(-2) - c2.unsqueeze(-3)) / self.conc_lengthscale
        k_conc_per_add = torch.exp(-0.5 * conc_diff.pow(2))
        main_components = shared * k_conc_per_add
        k_main = main_components.sum(dim=-1)
        k_pair = 0.5 * (k_main.pow(2) - main_components.pow(2).sum(dim=-1))

        s_ess, s_set, s_main, s_pair = self.scales
        cov = k_ess * (s_ess + s_set * k_set + s_main * k_main + s_pair * k_pair)
        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


def make_with_res_model(X, Y, codec, bounds):
    covar_module = AdditiveSetKernel(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
    )
    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar_module,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def make_no_res_model(X, Y, codec, bounds):
    covar_module = AdditiveSetKernelNoRes(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
    )
    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar_module,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def kernel_parameter_summary(model: SingleTaskGP) -> dict[str, Any]:
    covar = model.covar_module
    scales = covar.scales.detach().cpu().numpy().tolist()
    out: dict[str, Any] = {
        "scale_ess": float(scales[0]),
        "scale_set": float(scales[1]),
        "scale_main": float(scales[2]),
        "scale_pair": float(scales[3]),
        "ess_lengthscale": [
            float(v) for v in covar.ess_lengthscale.detach().cpu().reshape(-1)
        ],
        "conc_lengthscale": float(covar.conc_lengthscale.detach().cpu().item()),
    }
    if isinstance(covar, AdditiveSetKernel):
        out["scale_residual"] = float(scales[4])
    return out


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "logs" / "May_10_full_log" / "data_corrected.xlsx"),
    )
    p.add_argument("--sheet", type=str, default="Corrected Data")
    p.add_argument("--target", type=str, default="intensity")
    p.add_argument("--hrp", type=float, default=0.0001)
    p.add_argument("--hrp_atol", type=float, default=1e-12)
    p.add_argument("--filter_ctrl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
    )
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--fit_maxiter", type=int, default=75)
    p.add_argument("--acq_pool_size", type=int, default=512)
    p.add_argument("--stability_mc_samples", type=int, default=64)
    p.add_argument("--stability_repeats", type=int, default=5)
    p.add_argument("--eval_batch_size", type=int, default=256)
    return p.parse_args()


def run() -> None:
    args = build_args()
    set_seeds(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, codec = load_training_data(args)
    X, Y, encode_meta = encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    pool = make_candidate_pool(
        codec=codec,
        bounds=bounds,
        n=args.acq_pool_size,
        k_max=args.k_max,
        seed=args.seed + 123,
    )

    summary_data = {**dataset_summary(df, codec, args.target), **encode_meta}
    print("[Data]", json.dumps(summary_data, indent=2))

    builders = [
        ("with_residual", make_with_res_model),
        ("no_residual", make_no_res_model),
    ]
    results: dict[str, dict[str, Any]] = {}
    for label, builder in builders:
        set_seeds(args.seed)
        print(f"\n=== Fitting {label} ===")
        model = builder(X, Y, codec, bounds)
        t0 = time.time()
        fit_info = fit_model(model, maxiter=args.fit_maxiter)
        elapsed = time.time() - t0
        print(f"  fit seconds: {elapsed:.2f}")
        diag: dict[str, Any] = dict(fit_info)
        diag.update(training_fit_diagnostics(model, X, Y))
        diag.update(matrix_diagnostics(model, X))
        diag.update(loo_diagnostics(model, X))
        diag.update(kernel_parameter_summary(model))
        diag.update(
            acq_stability_diagnostics(
                model=model,
                X_baseline=X,
                pool=pool,
                mc_samples=args.stability_mc_samples,
                repeats=args.stability_repeats,
                seed=args.seed + 29,
                eval_batch_size=args.eval_batch_size,
            )
        )
        results[label] = diag

    summary_path = output_dir / "ablation_diagnostics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "data": summary_data,
                "models": results,
            },
            f,
            indent=2,
        )

    flat_rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update({k: v for k, v in metrics.items() if not isinstance(v, list)})
        flat_rows.append(row)
    csv_path = output_dir / "ablation_diagnostics.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8")

    print(f"\n[Output] {summary_path}")
    print(f"[Output] {csv_path}")

    metrics_to_compare = [
        "mll",
        "train_rmse",
        "train_r2",
        "loo_rmse_model_space",
        "loo_z_std",
        "loo_frac_abs_z_gt_2",
        "noise_standardized",
        "condition_number",
        "scale_ess",
        "scale_set",
        "scale_main",
        "scale_pair",
        "scale_residual",
        "conc_lengthscale",
        "acq_spearman_min",
        "acq_top25_jaccard_min",
    ]
    print("\n=== Paired comparison ===")
    header = f"{'metric':<30}{'with_residual':>18}{'no_residual':>18}{'delta':>15}"
    print(header)
    print("-" * len(header))
    a = results["with_residual"]
    b = results["no_residual"]
    for key in metrics_to_compare:
        va = a.get(key, float("nan"))
        vb = b.get(key, float("nan"))
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            delta = vb - va
            print(f"{key:<30}{va:>18.6g}{vb:>18.6g}{delta:>15.4g}")
        else:
            print(f"{key:<30}{str(va):>18}{str(vb):>18}{'-':>15}")


if __name__ == "__main__":
    run()
