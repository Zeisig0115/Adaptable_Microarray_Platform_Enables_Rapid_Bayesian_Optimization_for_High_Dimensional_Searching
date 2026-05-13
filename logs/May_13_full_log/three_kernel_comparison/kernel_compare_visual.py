"""Compute and visualise per-pair kernel similarity under all 3 kernels.

Re-fits the 3 models on logs/May_10_full_log/data_corrected.xlsx with seed 42,
then computes the normalized correlation kernel
    rho(x, y) = K(x, y) / sqrt(K(x, x) K(y, y))
for:

  Figure 1 : Heatmap of rho on all 446 training points, sorted by (active count,
             cat-bit pattern). One panel per kernel.
  Figure 2 : Concentration sweep of PEG20K (singular recipe, essentials fixed at
             training median). One line per kernel.
  Figure 3 : Cross-additive similarity at fixed conc. Two panels: reference =
             no-add, then reference = PEG20K singular.
  Figure 4 : Curated pair case studies (10 cases). Grouped bars per kernel.

Outputs are written next to this script. A CSV with numerical case-study values
is also saved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ_ROOT))

from add_bo import (
    CONC_DEFAULT,
    LOG_HI,
    LOG_LO,
    FlatCodec,
    encode_training_data,
    fit_model as fit_old_or_new,
    load_training_data,
    make_new_model,
    make_old_model,
    set_seeds,
)
from add_bo_mod import fit_model as fit_hier
from add_bo_mod import make_hierarchical_family_model

OUT_DIR = Path(__file__).parent
torch.set_default_dtype(torch.double)
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 150


def kernel_corr(model, X1: torch.Tensor, X2: torch.Tensor | None = None) -> np.ndarray:
    """Correlation kernel K(X1,X2) / sqrt(diag K11 * diag K22) using the trained
    covar_module. Applies the model's input_transform manually."""
    model.eval()
    if X2 is None:
        X2 = X1
    with torch.no_grad():
        it = getattr(model, "input_transform", None)
        X1_t = it(X1) if it is not None else X1
        X2_t = it(X2) if it is not None else X2
        K = model.covar_module(X1_t, X2_t).evaluate()
        d1 = model.covar_module(X1_t).evaluate().diagonal().clamp_min(1e-18).sqrt()
        d2 = model.covar_module(X2_t).evaluate().diagonal().clamp_min(1e-18).sqrt()
        corr = K / (d1.unsqueeze(-1) * d2.unsqueeze(-2))
    return corr.cpu().numpy()


def main() -> None:
    args = argparse.Namespace(
        input=str(PROJ_ROOT / "logs/May_10_full_log/data_corrected.xlsx"),
        sheet="Corrected Data",
        target="intensity",
        hrp=0.0001,
        hrp_atol=1e-12,
        filter_ctrl=True,
        k_max=4,
        fit_maxiter=75,
    )
    set_seeds(42)
    device = torch.device("cpu")
    df, codec = load_training_data(args)
    X, Y, meta = encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    print(f"data: {meta['unique_encoded_rows']} unique rows, {len(codec.adds)} additives")

    print("fitting old_mixed_hamming ...")
    set_seeds(42)
    old_model = make_old_model(X, Y, codec, bounds)
    fit_old_or_new(old_model, maxiter=75)

    print("fitting new_additive_set ...")
    set_seeds(42)
    add_model = make_new_model(X, Y, codec, bounds)
    fit_old_or_new(add_model, maxiter=75)

    print("fitting hierarchical_family_prior ...")
    set_seeds(42)
    hier_model = make_hierarchical_family_model(X, Y, codec, bounds)
    fit_hier(hier_model, maxiter=75)

    models = {
        "old_hamming": old_model,
        "additive_set": add_model,
        "hier_family": hier_model,
    }

    # ---------- helpers for synthetic recipes ----------
    ess_med = X[:, : codec.n_ess].median(dim=0).values

    def synth(active: dict[str, float] | None = None) -> torch.Tensor:
        x = torch.zeros(1, codec.d, dtype=torch.double)
        x[0, : codec.n_ess] = ess_med
        for j in range(codec.A):
            bc, cc = codec.cat_dims[j], codec.conc_dims[j]
            x[0, bc] = 0.0
            x[0, cc] = CONC_DEFAULT
        if active:
            for name, log_conc in active.items():
                bc, cc = codec.add_cols[name]
                x[0, bc] = 1.0
                x[0, cc] = float(log_conc)
        return x

    c = (LOG_LO + LOG_HI) / 2  # center log10 conc

    # =============== Figure 1: heatmaps ==================
    n = X.shape[0]
    b_cat = X[:, codec.cat_dims].cpu().numpy().astype(int)
    active_count = b_cat.sum(axis=1)
    # Pattern key: turn bit row into integer
    pat = np.zeros(n, dtype=np.int64)
    for j in range(codec.A):
        pat = pat * 2 + b_cat[:, j]
    keys = list(zip(active_count, pat))
    order = sorted(range(n), key=lambda i: keys[i])
    X_ord = X[order]

    # Find boundaries between active-count blocks for axis ticks
    ac_sorted = active_count[order]
    block_edges = [0]
    for i in range(1, n):
        if ac_sorted[i] != ac_sorted[i - 1]:
            block_edges.append(i)
    block_edges.append(n)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (name, model) in zip(axes, models.items()):
        corr = kernel_corr(model, X_ord)
        im = ax.imshow(corr, cmap="viridis", vmin=0, vmax=1, origin="upper", interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("training point (sorted by active count, then cat-bit pattern)")
        ax.set_ylabel("training point")
        for be in block_edges[1:-1]:
            ax.axhline(be - 0.5, color="white", linewidth=0.6)
            ax.axvline(be - 0.5, color="white", linewidth=0.6)
        # Block labels
        mids = [(block_edges[k] + block_edges[k + 1]) / 2 for k in range(len(block_edges) - 1)]
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(mids)
        ax2.set_xticklabels([f"k={int(ac_sorted[block_edges[k]])}\n(n={block_edges[k+1] - block_edges[k]})"
                             for k in range(len(block_edges) - 1)])
        ax2.tick_params(axis="x", labelsize=8, length=0)
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle("Normalized kernel rho(x_i, x_j) on the 446 training recipes (sorted by k = active count)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_heatmaps.png")
    plt.close(fig)
    print("saved fig1_heatmaps.png")

    # =============== Figure 2: PEG20K conc sweep ============
    n_sweep = 25
    log_concs = np.linspace(LOG_LO, LOG_HI, n_sweep)
    ref_conc = c
    x_ref = synth({"peg20k": ref_conc})
    x_sweep = torch.cat([synth({"peg20k": float(lc)}) for lc in log_concs], dim=0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name, model in models.items():
        corr = kernel_corr(model, x_ref, x_sweep).reshape(-1)
        ax.plot(log_concs, corr, marker="o", label=name)
    ax.axvline(ref_conc, color="gray", linestyle=":", alpha=0.6,
               label=f"reference at log10c={ref_conc:.2f}")
    ax.set_xlabel("log10([PEG20K])")
    ax.set_ylabel("normalized kernel similarity to reference")
    ax.set_title("Concentration sweep: singular PEG20K recipe, essentials at training median")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_peg20k_conc_sweep.png")
    plt.close(fig)
    print("saved fig2_peg20k_conc_sweep.png")

    # =============== Figure 3: cross-additive similarity ===========
    x_noadd = synth(None)
    x_peg20k = synth({"peg20k": c})

    additive_panel = [synth(None)] + [synth({a: c}) for a in codec.adds]
    x_panel = torch.cat(additive_panel, dim=0)
    panel_labels = ["(no add)"] + codec.adds

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    panel_refs = [("ref = no-add", x_noadd),
                  ("ref = PEG20K singular", x_peg20k)]
    bar_w = 0.27
    for ax, (title, ref) in zip(axes, panel_refs):
        xs = np.arange(len(panel_labels))
        for k_idx, (name, model) in enumerate(models.items()):
            corr = kernel_corr(model, ref, x_panel).reshape(-1)
            ax.bar(xs + (k_idx - 1) * bar_w, corr, width=bar_w, label=name)
        ax.set_xticks(xs)
        ax.set_xticklabels(panel_labels, rotation=60, ha="right", fontsize=9)
        ax.set_title(title)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("normalized kernel similarity to reference")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"Cross-additive similarity at fixed log10c={c:.2f} (singular recipes, essentials at training median)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_cross_additive.png")
    plt.close(fig)
    print("saved fig3_cross_additive.png")

    # =============== Figure 4: curated pair case studies ===============
    cases = [
        ("noadd vs noadd",         synth(None),                       synth(None)),
        ("noadd vs PEG20K@c",      synth(None),                       synth({"peg20k": c})),
        ("PEG20K low vs PEG20K high",
         synth({"peg20k": LOG_LO + 0.5}),
         synth({"peg20k": LOG_HI - 0.5})),
        ("PEG20K vs PEG6k @c",     synth({"peg20k": c}),              synth({"peg6k": c})),
        ("PEG20K vs PEG200k @c",   synth({"peg20k": c}),              synth({"peg200k": c})),
        ("PEG20K vs TW80 @c",      synth({"peg20k": c}),              synth({"tw80": c})),
        ("PEG20K vs BSA @c",       synth({"peg20k": c}),              synth({"bsa": c})),
        ("PEG20K vs MgSO4 @c",     synth({"peg20k": c}),              synth({"mgso4": c})),
        ("PEG20K+BSA vs PEG20K",   synth({"peg20k": c, "bsa": c}),    synth({"peg20k": c})),
        ("PEG20K+BSA vs PEG6k+BSA",
         synth({"peg20k": c, "bsa": c}),
         synth({"peg6k": c, "bsa": c})),
        ("PEG20K+BSA vs TW80+DMSO",
         synth({"peg20k": c, "bsa": c}),
         synth({"tw80": c, "dmso": c})),
    ]
    rows = []
    for case_name, xa, xb in cases:
        row = {"case": case_name}
        for name, model in models.items():
            row[name] = float(kernel_corr(model, xa, xb)[0, 0])
        rows.append(row)
    df_cases = pd.DataFrame(rows)
    df_cases.to_csv(OUT_DIR / "case_studies_kernel_similarity.csv", index=False, encoding="utf-8")
    print("saved case_studies_kernel_similarity.csv")
    print()
    print(df_cases.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(1, 1, figsize=(15, 6.5))
    xs = np.arange(len(cases))
    bar_w = 0.27
    for k_idx, name in enumerate(models.keys()):
        vals = [r[name] for r in rows]
        ax.bar(xs + (k_idx - 1) * bar_w, vals, width=bar_w, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels([r["case"] for r in rows], rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("normalized kernel similarity rho(x1, x2)")
    ax.set_title("Pair case studies (essentials at training median, log10c = -1.35 unless noted)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_pair_case_studies.png")
    plt.close(fig)
    print("saved fig4_pair_case_studies.png")


if __name__ == "__main__":
    main()
