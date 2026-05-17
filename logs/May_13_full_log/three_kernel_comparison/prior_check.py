"""Where do the fitted AdditiveSetKernel scales sit on the LogNormal prior?

Supports two schemas:

* legacy 5-scale (ess, set, main, pair, residual) — the original kernel before
  the residual term was dropped;
* current 3-scale (ess, set, pair) — the per-additive upgrade also reports
  per-additive ``new_conc_lengthscale`` and ``new_main_amp`` dicts, summarised
  here as min / median / max against their respective LogNormal priors.

Run with no args to read the historical add_bo_diag JSON in this folder, or
pass an alternate path::

    python prior_check.py path/to/add_bo_diagnostics.json [model_name]
"""
import argparse
import json
import math
import statistics
from pathlib import Path
from scipy.stats import norm


LEGACY_PRIOR = {
    "scale_ess":      0.15,
    "scale_set":      0.35,
    "scale_main":     0.75,
    "scale_pair":     0.25,
    "scale_residual": 0.05,
}
CURRENT_SCALE_PRIOR = {
    "scale_ess":  0.15,
    "scale_set":  0.35,
    "scale_pair": 0.25,
}
# Per-additive priors registered in the current AdditiveSetKernel.
CURRENT_PER_ADDITIVE_PRIOR = {
    "conc_lengthscale": 0.25,
    "main_amp":         0.75,
}
SIGMA = 0.75
Z90 = 1.6449


def fmt_row(name: str, init: float, fitted: float) -> str:
    lo = init * math.exp(-Z90 * SIGMA)
    hi = init * math.exp( Z90 * SIGMA)
    z = math.log(fitted / init) / SIGMA
    pct = norm.cdf(z) * 100
    return (
        f"{name:32s} {init:>12.4f} {lo:>12.4f} {hi:>12.4f} "
        f"{fitted:>10.4f} {z:>8.3f} {pct:>11.3f}%"
    )


def print_table(rows: list[str]) -> None:
    header = (
        f"{'component':32s} {'init(median)':>12s} {'90% CI low':>12s} "
        f"{'90% CI high':>12s} {'fitted':>10s} {'z':>8s} {'percentile':>12s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(r)


def detect_schema(fit: dict) -> str:
    has_new_main = "new_scale_main" in fit
    has_new_residual = "new_scale_residual" in fit
    has_per_additive = isinstance(fit.get("new_conc_lengthscale"), dict)
    if has_per_additive:
        return "current"
    if has_new_main or has_new_residual:
        return "legacy"
    raise KeyError(
        "Could not detect AdditiveSetKernel schema from JSON; expected either "
        "'new_scale_main' / 'new_scale_residual' (legacy) or a dict-valued "
        "'new_conc_lengthscale' (current per-additive upgrade)."
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "json_path",
        nargs="?",
        default=str(Path(__file__).parent / "add_bo_diag" / "add_bo_diagnostics.json"),
        help="Path to an add_bo_diagnostics.json.",
    )
    p.add_argument(
        "model_name",
        nargs="?",
        default="new_additive_set",
        help="Top-level key under 'models' to inspect.",
    )
    args = p.parse_args()

    payload = json.loads(Path(args.json_path).read_text())
    if args.model_name not in payload["models"]:
        raise SystemExit(
            f"Model '{args.model_name}' not in {list(payload['models'])}."
        )
    fit = payload["models"][args.model_name]
    schema = detect_schema(fit)
    print(f"Detected schema: {schema}  (model={args.model_name})")
    print()

    if schema == "legacy":
        rows = [
            fmt_row(name, init, fit[f"new_{name}"])
            for name, init in LEGACY_PRIOR.items()
        ]
        print_table(rows)
        return

    # Current per-additive schema.
    rows = [
        fmt_row(name, init, fit[f"new_{name}"])
        for name, init in CURRENT_SCALE_PRIOR.items()
    ]
    print("--- Global scales ---")
    print_table(rows)
    print()
    print(
        "--- Per-additive parameters (summary across additives vs their "
        "shared LogNormal prior) ---"
    )
    summary_rows = []
    for short, init in CURRENT_PER_ADDITIVE_PRIOR.items():
        per_add = fit[f"new_{short}"]
        vals = list(per_add.values())
        for agg, fn in (("min", min), ("median", statistics.median), ("max", max)):
            summary_rows.append(fmt_row(f"{short} [{agg}]", init, fn(vals)))
    print_table(summary_rows)

    # Show the per-additive outliers (most extreme z on lengthscale).
    print()
    print("--- Top |z| per-additive concentration lengthscale ---")
    pairs = [
        (name, math.log(v / CURRENT_PER_ADDITIVE_PRIOR["conc_lengthscale"]) / SIGMA)
        for name, v in fit["new_conc_lengthscale"].items()
    ]
    pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    for name, z in pairs[:6]:
        print(f"  {name:16s} z = {z:+.3f}  (l = {fit['new_conc_lengthscale'][name]:.4f})")


if __name__ == "__main__":
    main()
