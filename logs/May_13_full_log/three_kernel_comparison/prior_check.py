"""Where do the fitted AdditiveSetKernel scales sit on the LogNormal prior?

Prior in add_bo.py:
    initial_scales = [0.15, 0.35, 0.75, 0.25, 0.05]  (ess, set, main, pair, residual)
    LogNormalPrior(loc=log(initial), scale=0.75)
"""
import math, json
from pathlib import Path
from scipy.stats import norm

root = Path(__file__).parent
fit = json.loads((root / "add_bo_diag" / "add_bo_diagnostics.json").read_text())["models"]["new_additive_set"]

components = [
    ("scale_ess",      0.15, fit["new_scale_ess"]),
    ("scale_set",      0.35, fit["new_scale_set"]),
    ("scale_main",     0.75, fit["new_scale_main"]),
    ("scale_pair",     0.25, fit["new_scale_pair"]),
    ("scale_residual", 0.05, fit["new_scale_residual"]),
]
sigma = 0.75
z90 = 1.6449

print(f"{'component':16s} {'init(median)':>14s} {'90% CI low':>12s} {'90% CI high':>12s} "
      f"{'MAP':>10s} {'z':>8s} {'percentile':>12s}")
for name, init, fitted in components:
    lo = init * math.exp(-z90 * sigma)
    hi = init * math.exp( z90 * sigma)
    z = math.log(fitted / init) / sigma
    pct = norm.cdf(z) * 100
    print(f"{name:16s} {init:>14.4f} {lo:>12.4f} {hi:>12.4f} "
          f"{fitted:>10.4f} {z:>8.3f} {pct:>11.3f}%")
