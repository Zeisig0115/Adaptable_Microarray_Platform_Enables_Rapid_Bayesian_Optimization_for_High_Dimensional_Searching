# -*- coding: utf-8 -*-
"""One-shot kernel-variant comparison on the Jun_11 data_corrected dataset.

Does NOT modify add_bo.py: both experiment kernels subclass
``add_bo.BaselineKernel`` so they reuse its exact parameters / constraints /
priors; only one extra scalar parameter and the ``forward()`` composition
change. GP fit + fixed-hyperparameter LOO only (no candidates / acquisition),
matching the ``*_gp_loocv`` experiments.

Models compared
---------------
  1. baseline_kernel  : current BaselineKernel (multiplicative gate)
                        cov = of * k_ess * (c0 + s_main * k_main)
  2. per_add_alpha     : current production default (per-additive alpha + set)
                        cov = of * k_ess * (c0 + sum_j alpha_j m_j + s_set*Tani)
  3. add_composition   : EXPERIMENT 1 -- essentials/additive ADDITIVE split
                        cov = of * (k_ess + s_add * (c0 + s_main * k_main))
  4. baseline_set      : EXPERIMENT 2 -- BaselineKernel + ONE scalar Tanimoto
                        cov = of * k_ess * (c0 + s_main * k_main + s_set*Tani)

This log records EXPERIMENT 1 (additive composition); ``baseline_set`` is run
and tabulated for reference but its interpretation is handled in a separate
discussion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# --- Locate the repo root (where add_bo.py lives) so this provenance script
#     can be re-run from anywhere, including from inside the logs/ subfolder.
_HERE = Path(__file__).resolve()
_ROOT = None
for _p in [_HERE.parent, *_HERE.parents]:
    if (_p / "add_bo.py").exists():
        _ROOT = _p
        break
if _ROOT is None:
    raise RuntimeError("Could not locate add_bo.py above this script.")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import Positive
from gpytorch.priors import LogNormalPrior

import add_bo as ab

torch.set_default_dtype(torch.double)

_PRIOR_SCALE = torch.tensor(0.75, dtype=torch.double)
_DEFAULT_INPUT = _ROOT / "logs" / "Jun_11_full_log" / "data_corrected.xlsx"
_DEFAULT_OUTPUT = _ROOT / "logs" / "Jun_11_full_log" / "add_bo_additive_composition_gp_loocv"

METRIC_KEYS = [
    "n_params", "mll", "fit_seconds", "train_rmse", "train_mae", "train_r2",
    "train_z_std", "noise_standardized", "eig_min", "eig_max",
    "condition_number", "loo_rmse_model_space", "loo_mae_model_space",
    "loo_z_mean", "loo_z_std", "loo_frac_abs_z_gt_2", "loo_frac_abs_z_gt_3",
    "loo_cholesky_jitter", "kernel_diag_min", "kernel_diag_max",
]


def _k_ess_k_main(kernel: "ab.BaselineKernel", x1: torch.Tensor, x2: torch.Tensor):
    """Recompute k_ess and the shared/conc terms exactly as BaselineKernel does."""
    e1 = x1[..., kernel.ess_dims]
    e2 = x2[..., kernel.ess_dims]
    b1 = (x1[..., kernel.cat_dims] > 0.5).to(dtype=x1.dtype)
    b2 = (x2[..., kernel.cat_dims] > 0.5).to(dtype=x2.dtype)
    c1 = x1[..., kernel.conc_dims]
    c2 = x2[..., kernel.conc_dims]
    ess_diff = (e1.unsqueeze(-2) - e2.unsqueeze(-3)) / kernel.ess_lengthscale
    k_ess = torch.exp(-0.5 * ess_diff.pow(2).sum(dim=-1))
    shared = b1.unsqueeze(-2) * b2.unsqueeze(-3)
    conc_diff = (c1.unsqueeze(-2) - c2.unsqueeze(-3)) / kernel.conc_lengthscale
    k_conc = torch.exp(-0.5 * conc_diff.pow(2))
    k_main = (shared * k_conc).sum(dim=-1)
    return k_ess, k_main, b1, b2, shared


class AdditiveCompositionKernel(ab.BaselineKernel):
    r"""EXPERIMENT 1: cov = of * (k_ess + s_add * (c0 + s_main * k_main))."""

    def __init__(self, ess_dims, cat_dims, conc_dims, additive_names=None) -> None:
        super().__init__(ess_dims, cat_dims, conc_dims, additive_names)
        self.register_parameter(
            "raw_add_scale", torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        )
        self.register_constraint("raw_add_scale", Positive())
        init_add_scale = torch.tensor(1.0, dtype=torch.double)
        self.register_prior(
            "add_scale_prior",
            LogNormalPrior(loc=init_add_scale.log(), scale=_PRIOR_SCALE),
            lambda m: m.add_scale,
            lambda m, v: m._set_add_scale(v),
        )
        self.initialize(
            raw_add_scale=self.raw_add_scale_constraint.inverse_transform(init_add_scale)
        )

    @property
    def add_scale(self) -> torch.Tensor:
        return self.raw_add_scale_constraint.transform(self.raw_add_scale)

    def _set_add_scale(self, value: torch.Tensor) -> None:
        value = value.to(self.raw_add_scale).clamp_min(1e-12)
        self.initialize(
            raw_add_scale=self.raw_add_scale_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params: Any):
        if last_dim_is_batch:
            raise NotImplementedError
        k_ess, k_main, *_ = _k_ess_k_main(self, x1, x2)
        cov = self.outputscale * (
            k_ess + self.add_scale * (self.c0 + self.main_scale * k_main)
        )
        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


class BaselineSetKernel(ab.BaselineKernel):
    r"""EXPERIMENT 2: cov = of * k_ess * (c0 + s_main*k_main + s_set*Tanimoto)."""

    def __init__(self, ess_dims, cat_dims, conc_dims, additive_names=None, eps=1e-12) -> None:
        super().__init__(ess_dims, cat_dims, conc_dims, additive_names)
        self.eps = float(eps)
        self.register_parameter(
            "raw_set_scale", torch.nn.Parameter(torch.zeros((), dtype=torch.double))
        )
        self.register_constraint("raw_set_scale", Positive())
        init_set_scale = torch.tensor(0.35, dtype=torch.double)
        self.register_prior(
            "set_scale_prior",
            LogNormalPrior(loc=init_set_scale.log(), scale=_PRIOR_SCALE),
            lambda m: m.set_scale,
            lambda m, v: m._set_set_scale(v),
        )
        self.initialize(
            raw_set_scale=self.raw_set_scale_constraint.inverse_transform(init_set_scale)
        )

    @property
    def set_scale(self) -> torch.Tensor:
        return self.raw_set_scale_constraint.transform(self.raw_set_scale)

    def _set_set_scale(self, value: torch.Tensor) -> None:
        value = value.to(self.raw_set_scale).clamp_min(1e-12)
        self.initialize(
            raw_set_scale=self.raw_set_scale_constraint.inverse_transform(value)
        )

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params: Any):
        if last_dim_is_batch:
            raise NotImplementedError
        k_ess, k_main, b1, b2, shared = _k_ess_k_main(self, x1, x2)
        intersection = shared.sum(dim=-1)
        union = (
            b1.sum(dim=-1).unsqueeze(-1)
            + b2.sum(dim=-1).unsqueeze(-2)
            - intersection
        )
        k_set = torch.where(
            union > self.eps,
            intersection / union.clamp_min(self.eps),
            torch.ones_like(union),
        )
        cov = self.outputscale * k_ess * (
            self.c0 + self.main_scale * k_main + self.set_scale * k_set
        )
        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


def _make_variant(kernel_cls):
    def builder(X, Y, codec, bounds):
        covar = kernel_cls(
            ess_dims=list(range(codec.n_ess)),
            cat_dims=codec.cat_dims,
            conc_dims=codec.conc_dims,
            additive_names=codec.adds,
        )
        return SingleTaskGP(
            train_X=X,
            train_Y=Y,
            covar_module=covar,
            input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
            outcome_transform=Standardize(m=1),
        ).to(X)
    return builder


MODELS = [
    ("baseline_kernel", ab.make_baseline_model),
    ("per_add_alpha", ab.make_per_add_alpha_model),
    ("add_composition", _make_variant(AdditiveCompositionKernel)),
    ("baseline_set", _make_variant(BaselineSetKernel)),
]


def flatten_diag(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"model": label}
    for k, v in metrics.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                row[f"{k}__{sub_k}"] = sub_v
        elif isinstance(v, list):
            for i, sub_v in enumerate(v):
                row[f"{k}__{i}"] = sub_v
        else:
            row[k] = v
    return row


def run_seed(args, seed: int, seed_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ab.set_seeds(seed)
    device = torch.device(args.device)
    df, codec = ab.load_training_data(args)
    X, Y, encode_meta = ab.encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    data_block = {**ab.dataset_summary(df, codec, args.target), **encode_meta}

    results: dict[str, dict[str, Any]] = {}
    for label, builder in MODELS:
        ab.set_seeds(seed)
        t0 = time.time()
        model = builder(X, Y, codec, bounds)
        diag: dict[str, Any] = {"seed": seed, "model": label}
        diag["n_params"] = int(sum(p.numel() for p in model.covar_module.parameters()))
        diag.update(ab.fit_model(model, maxiter=args.fit_maxiter))
        diag.update(ab.training_fit_diagnostics(model, X, Y))
        diag.update(ab.matrix_diagnostics(model, X))
        diag.update(ab.loo_diagnostics(model, X))
        diag.update(ab.kernel_parameter_summary(model))
        cv = model.covar_module
        if hasattr(cv, "add_scale"):
            diag["variant_add_scale"] = float(cv.add_scale.detach().cpu())
        if hasattr(cv, "set_scale"):
            diag["variant_set_scale"] = float(cv.set_scale.detach().cpu())
        diag["wall_seconds_including_fit_call"] = float(time.time() - t0)
        results[label] = diag
        print(
            f"  [seed {seed}] {label:18s} npar={diag['n_params']:2d} "
            f"mll={diag['mll']:.4f} loo_rmse={diag['loo_rmse_model_space']:.4f} "
            f"noise={diag['noise_standardized']:.5f} cond={diag['condition_number']:.1f}"
        )

    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "add_bo_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {
                    "input": str(args.input), "sheet": args.sheet, "target": args.target,
                    "hrp": args.hrp, "hrp_atol": args.hrp_atol,
                    "filter_ctrl": args.filter_ctrl, "k_max": args.k_max,
                    "device": args.device, "seed": seed, "fit_maxiter": args.fit_maxiter,
                    "models": [label for label, _ in MODELS],
                },
                "data": data_block,
                "models": results,
            },
            f, indent=2,
        )
    flat = [flatten_diag(label, results[label]) for label, _ in MODELS]
    pd.DataFrame(flat).to_csv(seed_dir / "add_bo_diagnostics.csv", index=False, encoding="utf-8")
    return flat, data_block


def aggregate_metric_summary(all_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, _ in MODELS:
        sub = all_seed[all_seed["model"] == label]
        if sub.empty:
            continue
        row: dict[str, Any] = {"model": label, "n_seeds": int(sub["seed"].nunique())}
        for m in METRIC_KEYS:
            if m in sub.columns:
                vals = pd.to_numeric(sub[m], errors="coerce")
                row[f"{m}_mean"] = float(vals.mean())
                row[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{m}_min"] = float(vals.min())
                row[f"{m}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(path: Path, summary: pd.DataFrame, all_seed: pd.DataFrame,
                 data_block: dict[str, Any], n_seeds: int) -> None:
    def g(model: str, metric: str) -> float:
        row = summary[summary["model"] == model]
        return float(row[f"{metric}_mean"].iloc[0]) if not row.empty else float("nan")

    def npar(model: str) -> int:
        row = summary[summary["model"] == model]
        return int(row["n_params_mean"].iloc[0]) if not row.empty else -1

    def scalar(model: str, col: str) -> float:
        sub = all_seed[(all_seed["model"] == model) & (all_seed["seed"] == 0)]
        if sub.empty or col not in sub.columns:
            return float("nan")
        return float(pd.to_numeric(sub[col], errors="coerce").iloc[0])

    lines = [
        "# Essentials/additive additive-composition experiment (Exp1)",
        "",
        "## Scope",
        "",
        "- Records EXPERIMENT 1: replace the BaselineKernel multiplicative essentials",
        "  gate with an ADDITIVE essentials/additive split.",
        "- `baseline_kernel` (multiplicative gate): `of * k_ess * (c0 + s_main * k_main)`.",
        "- `add_composition` (Exp1): `of * (k_ess + s_add * (c0 + s_main * k_main))`.",
        "  Subclasses BaselineKernel; adds ONE scalar `s_add`; identical priors/lengthscales.",
        "- `per_add_alpha` (current production default) and `baseline_set` (Exp2) are run",
        "  and tabulated for reference only; Exp2 is handled in a separate discussion.",
        "- GP fit + fixed-hyperparameter LOO in standardized model space only; candidate",
        "  generation and acquisition stability are intentionally skipped.",
        "- Diagnostics are deterministic w.r.t. seed (MAP fit from fixed init); seeds",
        "  reproduce the format, not independent samples.",
        "- `k_main = sum_j b_j b'_j exp(-0.5((c_j-c'_j)/ell_conc_j)^2)`; `of`, `c0`,",
        "  `s_main`, `s_add` are partially scale-redundant (kept explicit).",
        "",
        "## Data",
        "",
        f"- Filtered rows: {data_block.get('rows_after_filters')}.",
        f"- Unique encoded rows: {data_block.get('unique_encoded_rows')}.",
        f"- Duplicates merged: {data_block.get('duplicates_merged')}.",
        f"- Encoded dimensions: {data_block.get('encoded_dim')}.",
        f"- Additives: {data_block.get('n_additives')}.",
        f"- Active-count distribution: {data_block.get('active_count_dist')}.",
        "",
        "## Result",
        "",
        "| model | n_params | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number | kernel diag max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _ in MODELS:
        lines.append(
            f"| {label} | {npar(label)} | {g(label,'mll'):.6g} | "
            f"{g(label,'loo_rmse_model_space'):.6g} | {g(label,'loo_z_std'):.6g} | "
            f"{g(label,'train_rmse'):.6g} | {g(label,'noise_standardized'):.6g} | "
            f"{g(label,'condition_number'):.6g} | {g(label,'kernel_diag_max'):.6g} |"
        )
    lines += [
        "",
        "## Fitted scalar hyperparameters (seed 0)",
        "",
        f"- baseline_kernel : of={scalar('baseline_kernel','baseline_outputscale'):.4g}, "
        f"c0={scalar('baseline_kernel','baseline_c0'):.4g}, "
        f"s_main={scalar('baseline_kernel','baseline_main_scale'):.4g}.",
        f"- add_composition : of={scalar('add_composition','baseline_outputscale'):.4g}, "
        f"c0={scalar('add_composition','baseline_c0'):.4g}, "
        f"s_main={scalar('add_composition','baseline_main_scale'):.4g}, "
        f"s_add={scalar('add_composition','variant_add_scale'):.4g}.",
        f"- baseline_set    : of={scalar('baseline_set','baseline_outputscale'):.4g}, "
        f"c0={scalar('baseline_set','baseline_c0'):.4g}, "
        f"s_main={scalar('baseline_set','baseline_main_scale'):.4g}, "
        f"s_set={scalar('baseline_set','variant_set_scale'):.4g}.",
        "",
        "## Interpretation (Exp1)",
        "",
        "- The additive split does NOT improve the measured fit: `add_composition` matches",
        "  `baseline_kernel` on MLL and is marginally worse on LOO RMSE, with essentially",
        "  unchanged noise and train RMSE. The learned `s_add` is O(1), i.e. the optimizer",
        "  finds no reason to prefer the additive form here.",
        "- First-principles reason: in this DOE the essentials are swept ONLY when all",
        "  additives are off (32 rows), and all additive chemistry was measured at a single",
        "  fixed essentials point (tmb=1.0, h2o2=0.01). The two data slices are disjoint, so",
        "  within each slice the multiplicative gate and the additive split are nearly",
        "  equivalent; they differ only in the UNOBSERVED joint region (additives present",
        "  AND essentials != (1.0, 0.01)), which LOO/MLL on this dataset cannot see.",
        "- Consequence: this dataset cannot adjudicate the essentials x additive coupling",
        "  form. The 'no measured gain' here does not by itself reject additive composition",
        "  for BO extrapolation honesty; it only shows the choice is unidentified from data.",
        "",
        "## Historical reference (prior Jun_11 logs, separate runs)",
        "",
        "| model | MLL | LOO RMSE | noise | condition number |",
        "|---|---:|---:|---:|---:|",
        "| old_mixed_hamming | -0.595856 | 0.331417 | 0.0246446 | 34615.4 |",
        "| additive_block_kernel (== current BaselineKernel) | -0.313154 | 0.289985 | 0.0725509 | 3938.07 |",
        "| per_add_alpha_kernel (+set, default) | -0.108224 | 0.270971 | 0.0127609 | 12282.1 |",
        "| new_additive_set | -0.164145 | 0.271278 | 0.0109057 | 15827.1 |",
        "",
        "## Cautions",
        "",
        "- This run checks GP fit and fixed-hyperparameter LOO only; it does not validate",
        "  acquisition behavior or 3/4-additive extrapolation.",
        "- `baseline_set` (Exp2) numbers are recorded here for reference; their",
        "  interpretation (why a single Tanimoto set term helps so much) is intentionally",
        "  deferred to a separate discussion.",
        f"- Seeds run: {n_seeds}.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    data_block: dict[str, Any] = {}
    for seed in range(args.n_seeds):
        print(f"=== seed {seed} ===")
        rows, data_block = run_seed(args, seed, out / f"seed_{seed}")
        all_rows.extend(rows)

    all_seed = pd.DataFrame(all_rows)
    summary = aggregate_metric_summary(all_seed)

    all_seed.to_csv(out / "kernel_variants_all_seed_diagnostics.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "kernel_variants_metric_summary.csv", index=False, encoding="utf-8")
    # Aliases for parity with the prior experiment folder layout.
    all_seed.to_csv(out / "comparison_all_seed_diagnostics.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "comparison_metric_summary.csv", index=False, encoding="utf-8")
    write_report(out / "additive_composition_report.md", summary, all_seed, data_block, args.n_seeds)
    print(f"[Output] -> {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Kernel-variant comparison (records Exp1).")
    p.add_argument("--input", type=str, default=str(_DEFAULT_INPUT))
    p.add_argument("--sheet", type=str, default="Corrected Data")
    p.add_argument("--target", type=str, default="intensity")
    p.add_argument("--hrp", type=float, default=0.0001)
    p.add_argument("--hrp_atol", type=float, default=1e-12)
    p.add_argument("--filter_ctrl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--fit_maxiter", type=int, default=75)
    p.add_argument("--n_seeds", type=int, default=3)
    p.add_argument("--output_dir", type=str, default=str(_DEFAULT_OUTPUT))
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
