"""Reproduce fig4 case-study bar chart including FAMT alongside the original 3 kernels."""
from __future__ import annotations

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

import add_bo_famt
from add_bo import (
    CONC_DEFAULT, LOG_LO, LOG_HI,
    load_training_data, encode_training_data,
    make_old_model, make_new_model,
    fit_model as fit_old_or_new,
    set_seeds,
)
from add_bo_mod import (
    fit_model as fit_hier,
    make_hierarchical_family_model,
)
import argparse

args = argparse.Namespace(
    input=str(PROJ_ROOT / "logs/May_10_full_log/data_corrected.xlsx"),
    sheet="Corrected Data",
    target="intensity",
    hrp=0.0001,
    hrp_atol=1e-12,
    filter_ctrl=True,
    k_max=4,
)

set_seeds(42)
device = torch.device("cpu")
df, codec = load_training_data(args)
X, Y, _ = encode_training_data(df, codec, args.target, device)
bounds = codec.get_bounds(device)
descriptors, _ = add_bo_famt.build_descriptor_matrix(codec.adds)
descriptors = descriptors.to(device=X.device, dtype=X.dtype)

print("Fitting old_mixed_hamming ...")
set_seeds(42)
old_m = make_old_model(X, Y, codec, bounds); fit_old_or_new(old_m, maxiter=75)
print("Fitting new_additive_set ...")
set_seeds(42)
add_m = make_new_model(X, Y, codec, bounds); fit_old_or_new(add_m, maxiter=75)
print("Fitting hier_family ...")
set_seeds(42)
hier_m = make_hierarchical_family_model(X, Y, codec, bounds); fit_hier(hier_m, maxiter=75)
print("Fitting FAMT ...")
set_seeds(42)
famt_m = add_bo_famt.make_famt_model(X, Y, codec, bounds, descriptors, latent_rank=4)
add_bo_famt.fit_famt(famt_m, maxiter=200)

models = {
    "old_hamming": old_m,
    "additive_set": add_m,
    "hier_family": hier_m,
    "famt": famt_m,
}

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

def kcorr(model, x1, x2):
    model.eval()
    with torch.no_grad():
        it = getattr(model, "input_transform", None)
        X1t = it(x1) if it is not None else x1
        X2t = it(x2) if it is not None else x2
        K12 = model.covar_module(X1t, X2t).evaluate().item()
        K11 = model.covar_module(X1t).evaluate().diagonal().item()
        K22 = model.covar_module(X2t).evaluate().diagonal().item()
    return K12 / np.sqrt(max(K11 * K22, 1e-18))

c = (LOG_LO + LOG_HI) / 2

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
    ("noadd vs CaCl2@c",          synth(None),                        synth({"cacl2": c})),
    ("noadd vs imidazole@c",      synth(None),                        synth({"imidazole": c})),
    ("CaCl2 vs MnCl2 @c",         synth({"cacl2": c}),                synth({"mncl2": c})),
    ("MnCl2 vs FeSO4 @c",         synth({"mncl2": c}),                synth({"feso4": c})),
]
rows = []
for name, xa, xb in cases:
    row = {"case": name}
    for label, m in models.items():
        row[label] = float(kcorr(m, xa, xb))
    rows.append(row)
df_cases = pd.DataFrame(rows)
df_cases.to_csv(HERE / "case_studies_four_kernels.csv", index=False)
print()
print(df_cases.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

fig, ax = plt.subplots(1, 1, figsize=(17, 7))
xs = np.arange(len(cases))
bar_w = 0.20
colors = {"old_hamming": "C0", "additive_set": "C1", "hier_family": "C2", "famt": "C3"}
for k, name in enumerate(models.keys()):
    vals = [r[name] for r in rows]
    ax.bar(xs + (k - 1.5) * bar_w, vals, width=bar_w, label=name, color=colors[name])
ax.set_xticks(xs)
ax.set_xticklabels([r["case"] for r in rows], rotation=22, ha="right", fontsize=9)
ax.set_ylabel("normalized kernel similarity rho(x1, x2)")
ax.set_title("4-kernel pair case studies (essentials at training median, log10c = -1.35 unless noted)")
ax.axhline(0, color="black", linewidth=0.5)
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(HERE / "fig4_pair_case_studies_4kernels.png")
plt.close(fig)
print("saved fig4_pair_case_studies_4kernels.png")
