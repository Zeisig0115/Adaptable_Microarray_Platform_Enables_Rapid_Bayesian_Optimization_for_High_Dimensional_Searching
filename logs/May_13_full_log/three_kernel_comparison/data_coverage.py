"""Verify the user's hypothesis that pair data is single-shot.

Check:
 - singleton coverage: how many concentration points per single additive?
 - pair coverage:      how many obs per (additive_X, additive_Y) pair, and
                       how many distinct (c_X, c_Y) per pair?
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJ_ROOT))

from add_bo import EPS, load_training_data

args = argparse.Namespace(
    input=str(PROJ_ROOT / "logs/May_10_full_log/data_corrected.xlsx"),
    sheet="Corrected Data",
    target="intensity",
    hrp=0.0001,
    hrp_atol=1e-12,
    filter_ctrl=True,
    k_max=4,
)
df, codec = load_training_data(args)
adds = codec.adds

bits = (df[adds].fillna(0).to_numpy() > EPS).astype(int)
active_counts = bits.sum(axis=1)
print("active count distribution:", dict(Counter(active_counts.tolist())))
print()

# Singleton coverage
print("=== singleton coverage ===")
sing_summary = []
for j, name in enumerate(adds):
    mask = (active_counts == 1) & (bits[:, j] == 1)
    n = int(mask.sum())
    if n > 0:
        concs = df.loc[mask, name].to_numpy()
        unique_concs = len(np.unique(np.round(np.log10(np.clip(concs, 1e-6, None)), 2)))
        sing_summary.append((name, n, unique_concs, float(concs.min()), float(concs.max())))
print(f"{'additive':12s} {'n_obs':>6s} {'unique_log10c (2dp)':>22s} {'min':>10s} {'max':>10s}")
for r in sorted(sing_summary, key=lambda r: -r[1]):
    print(f"{r[0]:12s} {r[1]:>6d} {r[2]:>22d} {r[3]:>10.4f} {r[4]:>10.4f}")
print(f"total singletons: {sum(r[1] for r in sing_summary)}")
print()

# Pair coverage
print("=== pair coverage ===")
pair_obs = Counter()
pair_unique_concs = {}
for i, row in enumerate(bits):
    if row.sum() == 2:
        idxs = tuple(sorted(np.flatnonzero(row).tolist()))
        a, b = adds[idxs[0]], adds[idxs[1]]
        key = f"{a}+{b}"
        pair_obs[key] += 1
        c_a = float(np.round(np.log10(max(df.iloc[i][a], 1e-6)), 2))
        c_b = float(np.round(np.log10(max(df.iloc[i][b], 1e-6)), 2))
        pair_unique_concs.setdefault(key, set()).add((c_a, c_b))

print(f"distinct pairs observed: {len(pair_obs)}")
print(f"total pair obs:          {sum(pair_obs.values())}")
dist = Counter(pair_obs.values())
print("how many pairs have how many obs:", dict(sorted(dist.items())))
multi_obs_pairs = {k: v for k, v in pair_obs.items() if v > 1}
print(f"pairs with > 1 obs:      {len(multi_obs_pairs)}")
multi_conc_pairs = {k: len(v) for k, v in pair_unique_concs.items() if len(v) > 1}
print(f"pairs with > 1 distinct (c_a, c_b): {len(multi_conc_pairs)}")
print()
print("top-10 most-observed pairs:")
for k, v in pair_obs.most_common(10):
    print(f"  {k:30s} n={v:3d}  unique (c_a,c_b)={len(pair_unique_concs[k])}")
