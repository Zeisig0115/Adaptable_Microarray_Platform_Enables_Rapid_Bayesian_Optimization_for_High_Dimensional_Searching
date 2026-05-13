"""Combine diagnostics from add_bo.py and add_bo_mod.py into one comparison."""
import json
from pathlib import Path
import pandas as pd

root = Path(__file__).parent
add_bo = json.loads((root / "add_bo_diag" / "add_bo_diagnostics.json").read_text())
add_mod = json.loads((root / "add_bo_mod_diag" / "add_bo_mod_diagnostics.json").read_text())

rows = [
    {"model": "old_mixed_hamming", **add_bo["models"]["old_mixed_hamming"]},
    {"model": "new_additive_set", **add_bo["models"]["new_additive_set"]},
    {"model": "hierarchical_family_prior", **add_mod["models"]["hierarchical_family_prior"]},
]

# Sanity check baseline parity between the two scripts.
mod_old = add_mod["models"]["old_mixed_hamming"]
abm_old = add_bo["models"]["old_mixed_hamming"]
parity_keys = ("mll", "train_rmse", "loo_rmse_model_space", "acq_value_mean")
parity = all(
    abs(mod_old.get(k, 0) - abm_old.get(k, 0)) < 1e-9 for k in parity_keys
)
print("baseline parity between add_bo and add_bo_mod for old_mixed_hamming:", parity)

df = pd.DataFrame(rows)
out_csv = root / "combined_diagnostics.csv"
df.to_csv(out_csv, index=False, encoding="utf-8")
print("wrote", out_csv)

keys = [
    "mll",
    "train_rmse", "train_mae", "train_r2", "train_z_std", "train_frac_abs_z_gt_2",
    "loo_rmse_model_space", "loo_mae_model_space", "loo_z_std",
    "loo_frac_abs_z_gt_2", "loo_frac_abs_z_gt_3",
    "noise_standardized", "eig_max", "condition_number",
    "acq_value_mean", "acq_value_std", "acq_per_point_std_mean",
    "acq_spearman_mean", "acq_top25_jaccard_mean",
]

print()
print(f"{'metric':32s} {'old_hamming':>14s} {'new_addset':>14s} {'hier_family':>14s}")
print("-" * 80)
for k in keys:
    v = [r.get(k, float("nan")) for r in rows]
    print(f"{k:32s} {v[0]:>14.4f} {v[1]:>14.4f} {v[2]:>14.4f}")
