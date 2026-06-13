# -*- coding: utf-8 -*-
"""Set/pair ablation over the per-additive-amplitude baseline.

Decomposes the AdditiveSetKernel advantage by toggling, one term at a time, the
Tanimoto active-set term (1a) and the closed-form pair term (1b) on top of the
``per_add_alpha`` kernel. Reuses add_bo.py's kernels and fixed-hyperparameter
LOO diagnostics, and writes per-seed plus aggregated artifacts in the same
layout as ``logs/Jun_11_full_log/add_bo_per_add_alpha_gp_loocv``.

GP fit + fixed-hyperparameter LOO only; candidate generation and acquisition
stability are intentionally skipped (matching the ``*_gp_loocv`` experiments).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize

import add_bo as ab

torch.set_default_dtype(torch.double)

# Aggregated scalar metrics (mean/std/min/max), matching the prior
# comparison_metric_summary.csv column family.
METRIC_KEYS = [
    "mll", "fit_seconds", "train_rmse", "train_mae", "train_r2", "train_z_std",
    "noise_standardized", "eig_min", "eig_max", "condition_number",
    "loo_rmse_model_space", "loo_mae_model_space", "loo_z_mean", "loo_z_std",
    "loo_frac_abs_z_gt_2", "loo_frac_abs_z_gt_3", "loo_cholesky_jitter",
    "kernel_diag_min", "kernel_diag_max",
]


def make_per_add_alpha_variant(use_set: bool, use_pair: bool) -> Callable:
    def builder(X, Y, codec, bounds):
        covar = ab.PerAddAlphaKernel(
            ess_dims=list(range(codec.n_ess)),
            cat_dims=codec.cat_dims,
            conc_dims=codec.conc_dims,
            additive_names=codec.adds,
            use_set=use_set,
            use_pair=use_pair,
        )
        return SingleTaskGP(
            train_X=X,
            train_Y=Y,
            covar_module=covar,
            input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
            outcome_transform=Standardize(m=1),
        ).to(X)
    return builder


# Decomposition ladder: current baseline -> per-additive amplitude ->
# +set (1a) / +pair (1b) / +both -> full AdditiveSetKernel reference.
MODELS: list[tuple[str, Callable]] = [
    ("baseline_kernel", ab.make_baseline_model),
    ("per_add_alpha", ab.make_per_add_alpha_model),
    ("per_add_alpha_set", make_per_add_alpha_variant(use_set=True, use_pair=False)),
    ("per_add_alpha_pair", make_per_add_alpha_variant(use_set=False, use_pair=True)),
    ("per_add_alpha_set_pair", make_per_add_alpha_variant(use_set=True, use_pair=True)),
    ("new_additive_set", ab.make_new_model),
]


def flatten_diag(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Explode nested per-additive dicts into ``<key>__<sub>`` columns."""
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


def run_seed(args, seed: int, seed_dir: Path) -> list[dict[str, Any]]:
    ab.set_seeds(seed)
    device = torch.device(args.device)
    df, codec = ab.load_training_data(args)
    X, Y, encode_meta = ab.encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    data_block = {**ab.dataset_summary(df, codec, args.target), **encode_meta}

    results: dict[str, dict[str, Any]] = {}
    for label, builder in MODELS:
        t0 = time.time()
        model = builder(X, Y, codec, bounds)
        diag: dict[str, Any] = {"seed": seed, "model": label}
        diag.update(ab.fit_model(model, maxiter=args.fit_maxiter))
        diag.update(ab.training_fit_diagnostics(model, X, Y))
        diag.update(ab.matrix_diagnostics(model, X))
        diag.update(ab.loo_diagnostics(model, X))
        diag.update(ab.kernel_parameter_summary(model))
        diag["wall_seconds_including_fit_call"] = float(time.time() - t0)
        results[label] = diag
        print(
            f"  [seed {seed}] {label:24s} mll={diag['mll']:.5f} "
            f"loo_rmse={diag['loo_rmse_model_space']:.5f} "
            f"noise={diag['noise_standardized']:.5f} cond={diag['condition_number']:.1f}"
        )

    seed_dir.mkdir(parents=True, exist_ok=True)
    with (seed_dir / "add_bo_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": {
                    "input": args.input, "sheet": args.sheet, "target": args.target,
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
    pd.DataFrame(flat).to_csv(
        seed_dir / "add_bo_diagnostics.csv", index=False, encoding="utf-8"
    )
    # Each flattened row already carries "seed" and "model" columns.
    return flat


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


def parameter_ranking(all_seed: pd.DataFrame, codec_adds: list[str], seed: int = 0) -> pd.DataFrame:
    sub = all_seed[(all_seed["model"] == "per_add_alpha") & (all_seed["seed"] == seed)]
    if sub.empty:
        return pd.DataFrame()
    r = sub.iloc[0]
    rows = []
    for a in codec_adds:
        rows.append({
            "additive": a,
            "alpha": float(r.get(f"per_add_alpha_alpha__{a}", np.nan)),
            "effective_alpha": float(r.get(f"per_add_alpha_effective_alpha__{a}", np.nan)),
            "conc_lengthscale": float(r.get(f"per_add_alpha_conc_lengthscale__{a}", np.nan)),
        })
    return pd.DataFrame(rows).sort_values("alpha", ascending=False).reset_index(drop=True)


def write_report(path: Path, summary: pd.DataFrame, data_block: dict[str, Any], n_seeds: int) -> None:
    def g(model: str, metric: str) -> float:
        row = summary[summary["model"] == model]
        return float(row[f"{metric}_mean"].iloc[0]) if not row.empty else float("nan")

    lines = [
        "# Set/pair ablation over the per-additive-amplitude baseline",
        "",
        "## Scope",
        "",
        "- Base kernel: `per_add_alpha` = `sigma_f^2 * k_ess * (c0 + sum_j alpha_j b_j b'_j k_conc_j)`.",
        "- 1a `per_add_alpha_set`: base + `s_set * Tanimoto(b,b')`.",
        "- 1b `per_add_alpha_pair`: base + `s_pair * k_pair`, `k_pair = sum_{i<j} m_i m_j` (closed form).",
        "- `per_add_alpha_set_pair`: base + both terms (folded-equivalent of AdditiveSetKernel).",
        "- Set/pair terms reuse AdditiveSetKernel's exact closed forms and scale priors; each ablation adds one global scale.",
        "- GP fit + fixed-hyperparameter LOO in standardized model space only; candidates / acquisition stability skipped.",
        "- Diagnostics are deterministic w.r.t. seed (MAP fit from fixed init); seeds reproduce the format, not independent samples.",
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
        "## Decomposition result",
        "",
        "| model | MLL | LOO RMSE | LOO z std | train RMSE | noise | condition number |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, _ in MODELS:
        lines.append(
            f"| {label} | {g(label,'mll'):.6g} | {g(label,'loo_rmse_model_space'):.6g} | "
            f"{g(label,'loo_z_std'):.6g} | {g(label,'train_rmse'):.6g} | "
            f"{g(label,'noise_standardized'):.6g} | {g(label,'condition_number'):.6g} |"
        )
    lines += [
        "",
        "## Historical reference (prior Jun_11 logs, separate runs)",
        "",
        "| model | MLL | LOO RMSE | noise | condition number |",
        "|---|---:|---:|---:|---:|",
        "| old_mixed_hamming | -0.595856 | 0.331417 | 0.0246446 | 34615.4 |",
        "| additive_block_kernel (== current BaselineKernel) | -0.313154 | 0.289985 | 0.0725509 | 3938.07 |",
        "| per_add_alpha_kernel | -0.261321 | 0.28758 | 0.0715065 | 3283.72 |",
        "| new_additive_set | -0.164145 | 0.271278 | 0.0109057 | 15827.1 |",
        "",
        "## Cautions",
        "",
        "- This run checks GP fit and fixed-hyperparameter LOO only; it does not validate acquisition behavior or 3/4-additive extrapolation.",
        "- The pair term is identically 0 unless two recipes share >=2 active additives; with only 0/1/2-active data its off-diagonal support is nearly empty.",
        "- The set/pair terms raise the covariance condition number relative to the per_add_alpha base.",
        f"- Seeds run: {n_seeds}.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, codec = ab.load_training_data(args)

    all_rows: list[dict[str, Any]] = []
    data_block: dict[str, Any] = {}
    for seed in range(args.n_seeds):
        print(f"=== seed {seed} ===")
        rows = run_seed(args, seed, out / f"seed_{seed}")
        all_rows.extend(rows)
        if seed == 0:
            with (out / "seed_0" / "add_bo_diagnostics.json").open(encoding="utf-8") as f:
                data_block = json.load(f)["data"]

    all_seed = pd.DataFrame(all_rows)
    summary = aggregate_metric_summary(all_seed)
    ranking = parameter_ranking(all_seed, codec.adds, seed=0)

    all_seed.to_csv(out / "set_pair_ablation_all_seed_diagnostics.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "set_pair_ablation_metric_summary.csv", index=False, encoding="utf-8")
    ranking.to_csv(out / "set_pair_ablation_parameter_ranking.csv", index=False, encoding="utf-8")
    # Aliases for parity with the prior experiment folder layout.
    all_seed.to_csv(out / "comparison_all_seed_diagnostics.csv", index=False, encoding="utf-8")
    summary.to_csv(out / "comparison_metric_summary.csv", index=False, encoding="utf-8")
    write_report(out / "set_pair_ablation_report.md", summary, data_block, args.n_seeds)
    print(f"[Output] -> {out.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Set/pair ablation over per_add_alpha.")
    p.add_argument("--input", type=str, default="logs/Jun_11_full_log/data_corrected.xlsx")
    p.add_argument("--sheet", type=str, default="Corrected Data")
    p.add_argument("--target", type=str, default="intensity")
    p.add_argument("--hrp", type=float, default=0.0001)
    p.add_argument("--hrp_atol", type=float, default=1e-12)
    p.add_argument("--filter_ctrl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--fit_maxiter", type=int, default=75)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument(
        "--output_dir", type=str,
        default="logs/Jun_11_full_log/add_bo_set_pair_ablation_gp_loocv",
    )
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
