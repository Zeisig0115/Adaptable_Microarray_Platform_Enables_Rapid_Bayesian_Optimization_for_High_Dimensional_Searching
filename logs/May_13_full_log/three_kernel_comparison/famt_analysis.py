"""4-way diagnostics + FAMT kernel structure visualisation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).parent
PROJ_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJ_ROOT))

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 150

# ---------- combine diagnostics across 4 kernels ----------
ab = json.loads((HERE / "add_bo_diag" / "add_bo_diagnostics.json").read_text())
abm = json.loads((HERE / "add_bo_mod_diag" / "add_bo_mod_diagnostics.json").read_text())
fa = json.loads((HERE / "famt_diag" / "add_bo_famt_diagnostics.json").read_text())

rows = [
    {"model": "old_mixed_hamming", **ab["models"]["old_mixed_hamming"]},
    {"model": "new_additive_set", **ab["models"]["new_additive_set"]},
    {"model": "hierarchical_family_prior", **abm["models"]["hierarchical_family_prior"]},
    {"model": "famt", **fa["models"]["famt"]},
]

scalar_keys = [
    "mll", "train_rmse", "train_r2", "train_z_std",
    "train_frac_abs_z_gt_2", "train_frac_abs_z_gt_3",
    "loo_rmse_model_space", "loo_z_std",
    "loo_frac_abs_z_gt_2", "loo_frac_abs_z_gt_3",
    "noise_standardized", "eig_max", "condition_number",
    "acq_value_mean", "acq_value_std",
    "acq_per_point_std_mean", "acq_spearman_mean",
    "acq_top25_jaccard_mean",
]
df_rows = []
for r in rows:
    row = {"model": r["model"]}
    for k in scalar_keys:
        row[k] = r.get(k, float("nan"))
    df_rows.append(row)
df_comp = pd.DataFrame(df_rows)
df_comp.to_csv(HERE / "four_way_diagnostics.csv", index=False, encoding="utf-8")
print("saved four_way_diagnostics.csv")
print()

# pretty print
print(f"{'metric':32s} " + " ".join(f"{m[:14]:>14s}" for m in df_comp['model']))
print("-" * (32 + 4 * 15))
for k in scalar_keys:
    vals = df_comp[k].tolist()
    print(f"{k:32s} " + " ".join(f"{v:>14.4f}" if isinstance(v, float) and not np.isnan(v) else f"{'NA':>14s}" for v in vals))


# ---------- FAMT-specific visualisations ----------
# Need to re-fit FAMT to get the actual G matrix (we only have G_diag in JSON)
import add_bo_famt
from add_bo import load_training_data, encode_training_data, set_seeds
import argparse

set_seeds(42)
args = argparse.Namespace(
    input=str(PROJ_ROOT / "logs/May_10_full_log/data_corrected.xlsx"),
    sheet="Corrected Data",
    target="intensity",
    hrp=0.0001,
    hrp_atol=1e-12,
    filter_ctrl=True,
    k_max=4,
)
device = torch.device("cpu")
df, codec = load_training_data(args)
X, Y, _ = encode_training_data(df, codec, args.target, device)
bounds = codec.get_bounds(device)
descriptors, _ = add_bo_famt.build_descriptor_matrix(codec.adds)
descriptors = descriptors.to(device=X.device, dtype=X.dtype)

print("\n=== Re-fitting FAMT to extract trained G ===")
set_seeds(42)
model = add_bo_famt.make_famt_model(X, Y, codec, bounds, descriptors, latent_rank=4)
add_bo_famt.fit_famt(model, maxiter=200)

covar: add_bo_famt.FAMTKernel = model.covar_module
with torch.no_grad():
    G = covar.G.detach().cpu().numpy()
    d = np.sqrt(np.clip(np.diag(G), 1e-12, None))
    G_corr = G / np.outer(d, d)
    # Decompose contributions
    sigma_j2 = covar.sigma_j.pow(2).detach().cpu().numpy()
    WF = (covar.W @ covar.F).detach().cpu().numpy()
    E = covar.E.detach().cpu().numpy()
    G_diag_part = np.diag(sigma_j2)
    G_chem_part = WF @ WF.T
    G_dev_part = E @ E.T
    G_cross_part = WF @ E.T + E @ WF.T

print("Decomposition of G (Frobenius norms):")
print(f"  ||diag(sigma_j^2)|| = {np.linalg.norm(G_diag_part):.4f}")
print(f"  ||WF (WF)^T||       = {np.linalg.norm(G_chem_part):.4f}")
print(f"  ||E E^T||           = {np.linalg.norm(G_dev_part):.4f}")
print(f"  ||cross terms||     = {np.linalg.norm(G_cross_part):.4f}")
print(f"  ||G||               = {np.linalg.norm(G):.4f}")
print(f"\nfractions of G's Frobenius norm:")
total = np.linalg.norm(G)
for label, part in [("diag(sigma_j^2)", G_diag_part), ("(WF)(WF)^T", G_chem_part),
                     ("E E^T", G_dev_part), ("cross", G_cross_part)]:
    print(f"  {label:20s}  {np.linalg.norm(part) / total:.3f}")

# ---------- Figure: G correlation heatmap + decomposition panels ----------
ordering = [
    "peg400", "peg6k", "peg20k", "peg200k", "pl127",
    "cmc", "pva", "paa",
    "tw20", "tw80", "tx100",
    "glycerol", "sucrose",
    "bsa",
    "dmso",
    "imidazole",
    "edta",
    "mgso4", "cacl2", "znso4",
    "mncl2", "feso4",
]
idx = [codec.adds.index(a) for a in ordering]
G_sorted = G[np.ix_(idx, idx)]
G_corr_sorted = G_corr[np.ix_(idx, idx)]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

im0 = axes[0].imshow(G_sorted, cmap="viridis", origin="upper")
axes[0].set_title("Learned G (raw, A x A coregionalisation matrix)")
axes[0].set_xticks(range(len(ordering))); axes[0].set_xticklabels(ordering, rotation=60, ha="right", fontsize=8)
axes[0].set_yticks(range(len(ordering))); axes[0].set_yticklabels(ordering, fontsize=8)
plt.colorbar(im0, ax=axes[0], fraction=0.045)

im1 = axes[1].imshow(G_corr_sorted, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
axes[1].set_title("Learned G correlation (G_jk / sqrt(G_jj G_kk))")
axes[1].set_xticks(range(len(ordering))); axes[1].set_xticklabels(ordering, rotation=60, ha="right", fontsize=8)
axes[1].set_yticks(range(len(ordering))); axes[1].set_yticklabels(ordering, fontsize=8)
plt.colorbar(im1, ax=axes[1], fraction=0.045)

fig.suptitle("FAMT learned G: raw and correlation")
fig.tight_layout()
fig.savefig(HERE / "fig6_famt_G.png")
plt.close(fig)
print("saved fig6_famt_G.png")

# ---------- Sigma_j bar plot ----------
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
sigma_j_dict = covar.sigma_j.detach().cpu().numpy()
sigma_j_sorted = [(codec.adds[i], float(sigma_j_dict[i])) for i in idx]
names = [s[0] for s in sigma_j_sorted]
vals = [s[1] for s in sigma_j_sorted]
ax.bar(range(len(names)), vals, color="steelblue")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=60, ha="right", fontsize=9)
ax.set_ylabel("sigma_j  (per-additive signal amplitude)")
ax.set_title(f"FAMT learned sigma_j (horseshoe).  tau = {covar.tau.item():.4f}")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig7_famt_sigma_j.png")
plt.close(fig)
print("saved fig7_famt_sigma_j.png")

# ---------- Pair similarity case studies (same 11 cases as fig4) ----------
from add_bo import EPS, LOG_LO, LOG_HI, CONC_DEFAULT
ess_med = X[:, : codec.n_ess].median(dim=0).values

def synth(active=None):
    x = torch.zeros(1, codec.d, dtype=torch.double)
    x[0, : codec.n_ess] = ess_med
    for j in range(codec.A):
        bc, cc = codec.cat_dims[j], codec.conc_dims[j]
        x[0, bc] = 0.0
        x[0, cc] = CONC_DEFAULT
    if active:
        for name, lc in active.items():
            bc, cc = codec.add_cols[name]
            x[0, bc] = 1.0
            x[0, cc] = float(lc)
    return x

c = (LOG_LO + LOG_HI) / 2

def kcorr_famt(x1, x2):
    model.eval()
    with torch.no_grad():
        it = model.input_transform
        X1t, X2t = it(x1), it(x2)
        K12 = model.covar_module(X1t, X2t).evaluate().item()
        K11 = model.covar_module(X1t).evaluate().diagonal().item()
        K22 = model.covar_module(X2t).evaluate().diagonal().item()
    return K12 / np.sqrt(max(K11 * K22, 1e-18))

cases = [
    ("noadd vs noadd",            synth(None),                        synth(None)),
    ("noadd vs PEG20K@c",         synth(None),                        synth({"peg20k": c})),
    ("PEG20K low vs PEG20K high", synth({"peg20k": LOG_LO + 0.5}),    synth({"peg20k": LOG_HI - 0.5})),
    ("PEG20K vs PEG6k @c",        synth({"peg20k": c}),               synth({"peg6k": c})),
    ("PEG20K vs PEG200k @c",      synth({"peg20k": c}),               synth({"peg200k": c})),
    ("PEG20K vs TW80 @c",         synth({"peg20k": c}),               synth({"tw80": c})),
    ("PEG20K vs BSA @c",          synth({"peg20k": c}),               synth({"bsa": c})),
    ("PEG20K vs MgSO4 @c",        synth({"peg20k": c}),               synth({"mgso4": c})),
    ("PEG20K+BSA vs PEG20K",      synth({"peg20k": c, "bsa": c}),     synth({"peg20k": c})),
    ("PEG20K+BSA vs PEG6k+BSA",   synth({"peg20k": c, "bsa": c}),     synth({"peg6k": c, "bsa": c})),
    ("PEG20K+BSA vs TW80+DMSO",   synth({"peg20k": c, "bsa": c}),     synth({"tw80": c, "dmso": c})),
    # also a few new ones involving high-sigma_j additives
    ("noadd vs CaCl2@c",          synth(None),                        synth({"cacl2": c})),
    ("noadd vs imidazole@c",      synth(None),                        synth({"imidazole": c})),
    ("CaCl2 vs MnCl2 @c",         synth({"cacl2": c}),                synth({"mncl2": c})),
    ("MnCl2 vs FeSO4 @c",         synth({"mncl2": c}),                synth({"feso4": c})),
]
results = []
for label, xa, xb in cases:
    results.append({"case": label, "famt": kcorr_famt(xa, xb)})
df_famt_cases = pd.DataFrame(results)
df_famt_cases.to_csv(HERE / "famt_case_studies.csv", index=False)
print()
print("FAMT pair-case similarities:")
print(df_famt_cases.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
