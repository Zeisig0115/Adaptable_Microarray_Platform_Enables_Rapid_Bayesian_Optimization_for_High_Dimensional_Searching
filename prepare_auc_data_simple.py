from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def find_frame_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).strip().lower().startswith("frame")]
    if not cols:
        raise ValueError("No frame columns found. Expected columns like 'frame 0', 'frame 1', ...")

    def frame_index(col) -> int:
        s = str(col).replace("_", " ").strip().split()
        return int(s[-1])

    return sorted(cols, key=frame_index)


def hampel_isolated_spikes(y: np.ndarray, window: int = 2, n_sigmas: float = 5.0, min_abs: float = 2.5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    out = y.copy()
    n = len(y)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbors = np.concatenate([y[lo:i], y[i + 1:hi]])
        if len(neighbors) < 3:
            continue
        med = np.median(neighbors)
        mad = np.median(np.abs(neighbors - med))
        sigma = 1.4826 * mad
        threshold = max(n_sigmas * sigma, min_abs)
        if abs(y[i] - med) > threshold:
            out[i] = med
    return out


def calc_auc(y: np.ndarray, method: str = "trapz") -> float:
    y = np.asarray(y, dtype=float)
    if method == "sum":
        return float(np.sum(y))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=1.0))
    return float(np.trapz(y, dx=1.0))


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def trim1_mean_by_mad(x: pd.Series) -> float:
    """
    Drop the single replicate farthest from the group median,
    then average the rest.

    For your design n=4, this keeps 3 replicates.
    """
    values = x.to_numpy(dtype=float)
    n = len(values)

    if n <= 2:
        return float(np.mean(values))

    med = np.median(values)
    deviation = np.abs(values - med)

    # Drop exactly one value: the replicate farthest from the median.
    drop_idx = int(np.argmax(deviation))
    kept = np.delete(values, drop_idx)

    return float(np.mean(kept))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert frame-wise blueness curves into BO data.xlsx with TMB, H2O2, AUC.")
    parser.add_argument("--input", default="./Apr_29_full_log/LHS_all_data.xlsx")
    parser.add_argument("--output", default="data.xlsx", help="Output Excel file. Default: data.xlsx")
    parser.add_argument("--hrp", type=float, default=0.0001, help="HRP value to keep. Required if the input has multiple HRP levels.")
    parser.add_argument(
        "--agg",
        choices=["median", "mean", "trim1_mean"],
        default="trim1_mean",
        help="How to aggregate replicate AUCs. Default: trim1_mean",
    )
    parser.add_argument("--auc_method", choices=["trapz", "sum"], default="trapz", help="AUC method. Default: trapz")
    parser.add_argument("--no_filter", action="store_true", help="Disable conservative isolated-spike repair")
    args = parser.parse_args()

    df = read_table(Path(args.input))
    for col in ["TMB", "H2O2"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if "HRP" in df.columns:
        hrps = sorted(df["HRP"].dropna().astype(float).unique())
        if args.hrp is None and len(hrps) > 1:
            raise ValueError(f"Input has multiple HRP values {hrps}. Please pass one, e.g. --hrp {hrps[0]}")
        if args.hrp is not None:
            df = df[np.isclose(df["HRP"].astype(float), args.hrp)].copy()
            if df.empty:
                raise ValueError(f"No rows matched --hrp {args.hrp}")

    frame_cols = find_frame_cols(df)
    rows = []
    for _, row in df.iterrows():
        y = row[frame_cols].to_numpy(dtype=float)
        if np.isnan(y).any():
            raise ValueError("NaN found in frame columns. Please inspect the raw file first.")
        if not args.no_filter:
            y = hampel_isolated_spikes(y)
        rows.append({
            "TMB": float(row["TMB"]),
            "H2O2": float(row["H2O2"]),
            "AUC_rep": calc_auc(y, method=args.auc_method),
        })

    rep = pd.DataFrame(rows)
    if args.agg == "median":
        out = rep.groupby(["TMB", "H2O2"], as_index=False)["AUC_rep"].median()

    elif args.agg == "mean":
        out = rep.groupby(["TMB", "H2O2"], as_index=False)["AUC_rep"].mean()

    elif args.agg == "trim1_mean":
        out = (
            rep.groupby(["TMB", "H2O2"])["AUC_rep"]
            .apply(trim1_mean_by_mad)
            .reset_index(name="AUC_rep")
        )

    else:
        raise ValueError(f"Unknown agg: {args.agg}")
    out = out.rename(columns={"AUC_rep": "AUC"})
    out = out[["TMB", "H2O2", "AUC"]].sort_values(["TMB", "H2O2"]).reset_index(drop=True)

    output = Path(args.output)
    out.to_excel(output, index=False)
    print(f"Saved {len(out)} rows to {output.resolve()}")
    print("Columns: TMB, H2O2, AUC")


if __name__ == "__main__":
    main()
