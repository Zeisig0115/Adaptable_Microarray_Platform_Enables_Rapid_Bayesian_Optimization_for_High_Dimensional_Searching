# -*- coding: utf-8 -*-
"""Determinism probe: compare seed_0 vs seed_1 add_bo diagnostics."""
import json
from pathlib import Path

keys = [
    "mll", "train_rmse", "train_r2",
    "loo_rmse_model_space", "loo_z_std", "loo_frac_abs_z_gt_2",
    "condition_number", "eig_min", "eig_max", "noise_standardized",
]
base = Path(__file__).parent
d0 = json.load(open(base / "seed_0" / "add_bo_diagnostics.json"))["models"]
d1 = json.load(open(base / "seed_1" / "add_bo_diagnostics.json"))["models"]

for m in d0:
    print("==== " + m + " ====")
    for k in keys:
        a = d0[m][k]
        b = d1[m][k]
        rel = abs(a - b) / (abs(a) + 1e-12)
        flag = "  <-- DIFF" if rel > 1e-9 else ""
        print("  %-28s seed0=%.10g  seed1=%.10g  reldiff=%.2e%s" % (k, a, b, rel, flag))
