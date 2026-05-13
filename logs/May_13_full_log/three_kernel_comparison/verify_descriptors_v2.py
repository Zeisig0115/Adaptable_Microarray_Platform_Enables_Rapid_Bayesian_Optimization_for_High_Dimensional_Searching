"""Sanity-check the v2 descriptors by visualising the implied family cosine
similarity between additives and comparing it to hier_family's v1 cosine.

Outputs:
  fig5_descriptor_cosine.png  : 2-panel heatmap, v1 (hier_family) vs v2 (FAMT)
  descriptor_v2_table.csv     : full W matrix in CSV form
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
PROJ_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(HERE))

from add_bo_mod import ADDITIVE_METADATA as ADDITIVE_METADATA_V1
from additive_descriptors_v2 import (
    ADDITIVE_METADATA_V2,
    FEATURE_KEYS as KEYS_V2,
    build_descriptor_matrix,
)

mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 150

additives = list(ADDITIVE_METADATA_V2.keys())

# Build W_v1 (from hier_family's existing dict) using its keys.
KEYS_V1 = [
    "is_peg", "is_polymer", "is_surfactant", "is_polyol_sugar", "is_protein",
    "is_salt", "is_chloride", "is_sulfate", "is_chelator_or_buffer", "is_solvent",
    "log_mw",
]
W_v1 = np.array([[ADDITIVE_METADATA_V1[a][k] for k in KEYS_V1] for a in additives], dtype=float)
# z-score log_mw column (matches add_bo_mod's normalization)
log_idx_v1 = KEYS_V1.index("log_mw")
mu, sd = W_v1[:, log_idx_v1].mean(), W_v1[:, log_idx_v1].std()
W_v1[:, log_idx_v1] = (W_v1[:, log_idx_v1] - mu) / max(sd, 1e-8)

W_v2, _ = build_descriptor_matrix(additives)
log_idx_v2 = KEYS_V2.index("log_mw")
charge_idx_v2 = KEYS_V2.index("charge_at_pH7")
# z-score the two continuous columns
for col in (log_idx_v2, charge_idx_v2):
    mu, sd = W_v2[:, col].mean(), W_v2[:, col].std()
    W_v2[:, col] = (W_v2[:, col] - mu) / max(sd, 1e-8)


def cosine_gram(W):
    norm = np.linalg.norm(W, axis=1, keepdims=True)
    return (W / np.clip(norm, 1e-8, None)) @ (W / np.clip(norm, 1e-8, None)).T


C_v1 = cosine_gram(W_v1)
C_v2 = cosine_gram(W_v2)

# Save v2 W matrix as a CSV table
df_w = pd.DataFrame(W_v2, index=additives, columns=KEYS_V2)
df_w.to_csv(HERE / "descriptor_v2_table.csv")
print("saved descriptor_v2_table.csv")

# Sort additives by a sensible order for the heatmap
ordering = [
    "peg400", "peg6k", "peg20k", "peg200k", "pl127",   # PEG / PEG-block
    "cmc", "pva", "paa",                                # other polymers
    "tw20", "tw80", "tx100",                            # surfactants
    "glycerol", "sucrose",                              # polyols / sugars
    "bsa",                                              # protein
    "dmso",                                             # solvent
    "imidazole",                                        # buffer
    "edta",                                             # chelator
    "mgso4", "cacl2", "znso4",                          # non-redox salts
    "mncl2", "feso4",                                   # redox-active salts
]
assert set(ordering) == set(additives), set(additives) - set(ordering)
idx = [additives.index(a) for a in ordering]
C_v1 = C_v1[np.ix_(idx, idx)]
C_v2 = C_v2[np.ix_(idx, idx)]

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
for ax, C, title in zip(axes, [C_v1, C_v2],
                         ["v1 (hier_family, 11 features)",
                          "v2 (FAMT proposal, 13 features)"]):
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, origin="upper")
    ax.set_xticks(range(len(ordering)))
    ax.set_xticklabels(ordering, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(ordering)))
    ax.set_yticklabels(ordering, fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.045)
fig.suptitle("Additive descriptor cosine similarity (after z-scoring continuous columns)")
fig.tight_layout()
fig.savefig(HERE / "fig5_descriptor_cosine.png")
plt.close(fig)
print("saved fig5_descriptor_cosine.png")

# Print a few specific pair checks
def report(a, b, label):
    i, j = additives.index(a), additives.index(b)
    # Need to recompute on unsorted matrices
    Cv1 = cosine_gram(W_v1)
    Cv2 = cosine_gram(W_v2)
    print(f"  {label:35s}  v1 cos = {Cv1[i,j]:+.3f}   v2 cos = {Cv2[i,j]:+.3f}")

print()
print("Specific pair sanity checks (higher = more similar):")
report("peg20k", "peg6k",    "PEG20K vs PEG6k (same family)")
report("peg20k", "peg400",   "PEG20K vs PEG400 (same family, MW far)")
report("peg20k", "peg200k",  "PEG20K vs PEG200k (same family)")
report("peg20k", "pl127",    "PEG20K vs PL127 (PEG block copolymer)")
report("peg20k", "pva",      "PEG20K vs PVA (other polymer)")
report("peg20k", "bsa",      "PEG20K vs BSA (protein)")
report("peg20k", "mgso4",    "PEG20K vs MgSO4 (salt)")
report("mncl2",  "feso4",    "MnCl2 vs FeSO4 (both redox-active salt)")
report("mgso4",  "znso4",    "MgSO4 vs ZnSO4 (both kosmotrope salts)")
report("mncl2",  "cacl2",    "MnCl2 vs CaCl2 (chloride salts, redox differs)")
report("feso4",  "mgso4",    "FeSO4 vs MgSO4 (both sulfate, redox differs)")
report("edta",   "imidazole","EDTA vs imidazole (both chelator/buffer in v1)")
report("edta",   "cacl2",    "EDTA vs CaCl2")
report("glycerol","sucrose", "glycerol vs sucrose (polyol-sugar)")
report("bsa",    "pva",      "BSA vs PVA (both protein stabilizers)")
