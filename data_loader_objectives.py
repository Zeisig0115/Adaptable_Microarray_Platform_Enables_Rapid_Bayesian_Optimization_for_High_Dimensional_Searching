from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def find_frame_cols(df: pd.DataFrame) -> list[str]:
    """Find and numerically sort columns named like 'frame 0', 'frame_1', ..."""
    cols = [c for c in df.columns if str(c).strip().lower().startswith("frame")]
    if not cols:
        raise ValueError("No frame columns found. Expected columns like 'frame 0', 'frame 1', ...")

    def frame_index(col) -> int:
        s = str(col).replace("_", " ").strip().split()
        return int(s[-1])

    return sorted(cols, key=frame_index)


def hampel_isolated_spikes(
        y: np.ndarray,
        window: int = 2,
        n_sigmas: float = 5.0,
        min_abs: float = 2.5,
) -> np.ndarray:
    """
    Conservative repair for isolated frame spikes.

    A point is replaced by the local-neighbor median only if it is both far in
    robust-MAD units and far in absolute magnitude. This is intentionally mild.
    """
    y = np.asarray(y, dtype=float)
    out = y.copy()
    n = len(y)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        neighbors = np.concatenate([y[lo:i], y[i + 1: hi]])
        if len(neighbors) < 3:
            continue
        med = np.median(neighbors)
        mad = np.median(np.abs(neighbors - med))
        sigma = 1.4826 * mad
        threshold = max(n_sigmas * sigma, min_abs)
        if abs(y[i] - med) > threshold:
            out[i] = med
    return out


def moving_average(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average with edge padding. Set window <= 1 to disable."""
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()
    if window % 2 == 0:
        raise ValueError("--smooth_window must be odd, e.g. 3, 5, 7")
    pad = window // 2
    padded = np.pad(y, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def calc_auc(y: np.ndarray, method: str = "trapz") -> float:
    y = np.asarray(y, dtype=float)
    if method == "sum":
        return float(np.sum(y))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, dx=1.0))
    return float(np.trapz(y, dx=1.0))


def extract_objectives(
        y_raw: np.ndarray,
        *,
        baseline: float,
        signal_threshold: float,
        retention_last_n: int,
        smooth_window: int,
        auc_method: str,
) -> dict[str, float]:
    y = moving_average(np.asarray(y_raw, dtype=float), window=smooth_window)
    y_corr = y - baseline

    max_intensity = float(np.max(y_corr))
    auc = calc_auc(y_corr, method=auc_method)

    if max_intensity < signal_threshold:
        return {
            "max_intensity": max_intensity,
            "reaction_speed": 0.0,
            "color_retention": 0.0,
            "AUC": auc,
        }

    threshold_70 = 0.7 * max_intensity
    t70_candidates = np.flatnonzero(y_corr >= threshold_70)
    t70 = int(t70_candidates[0])
    reaction_speed = 1.0 - t70 / (len(y_corr) - 1.0)

    last_n = min(retention_last_n, len(y_corr))
    final_mean = float(np.mean(y_corr[-last_n:]))
    color_retention = final_mean / max_intensity
    color_retention = float(np.clip(color_retention, 0.0, 1.0))

    return {
        "max_intensity": max_intensity,
        "reaction_speed": float(reaction_speed),
        "color_retention": color_retention,
        "AUC": auc,
    }


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        raise ValueError("Output must be .xlsx, .xls, or .csv")


def summarize_replicates(rep: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols = list(group_cols)
    objective_cols = ["max_intensity", "reaction_speed", "color_retention", "AUC"]

    pieces = []
    grouped = rep.groupby(group_cols, as_index=False)
    for obj in objective_cols:
        tmp = grouped[obj].agg(
            **{
                f"{obj}_mean": "mean",
                f"{obj}_median": "median",
                f"{obj}_sd": lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan,
                f"{obj}_sem": lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
                f"{obj}_min": "min",
                f"{obj}_max": "max",
            }
        )
        pieces.append(tmp)

    summary = pieces[0]
    for tmp in pieces[1:]:
        summary = summary.merge(tmp, on=group_cols, how="outer")
    summary["n"] = rep.groupby(group_cols).size().to_numpy()
    return summary.sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert frame-wise blueness curves into replicate-level BO data. "
            "Automatically loops through specified HRP values."
        )
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="LHS",
        choices=["LHS", "BO1", "BO2"],
    )
    parser.add_argument("--input_dir", default="./May_5_full_log", help="输入文件所在的文件夹")
    parser.add_argument("--input_date_prefix", default="05_05", help="输入文件的前缀日期，默认为 05_05")
    parser.add_argument("--output_dir", default="./May_5_full_log")
    parser.add_argument("--hrp", type=float, nargs="+", default=[0.0001, 0.01, 1],
                        help="List of HRP values to process.")
    parser.add_argument(
        "--generate_summary",
        action="store_true",
        help="Generate a summary output file with mean/sd/sem per condition for each HRP.",
    )
    parser.add_argument(
        "--keep_all_hrp",
        action="store_true",
        help="Do not filter HRP; keep all HRP levels in one file.",
    )
    parser.add_argument("--baseline", type=float, default=22.0,
                        help="Baseline intensity to subtract before feature extraction.")
    parser.add_argument("--signal_threshold", type=float, default=5.0)
    parser.add_argument("--retention_last_n", type=int, default=5, help="Number of final frames used for retention.")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Odd window size for centered moving-average smoothing.")
    parser.add_argument("--auc_method", choices=["trapz", "sum"], default="trapz")
    parser.add_argument("--no_filter", action="store_true", help="Disable conservative isolated-spike repair")
    parser.add_argument(
        "--add_trace_cols",
        action="store_true",
        help="Append replicate_id and source_row for traceability. Off by default.",
    )
    args = parser.parse_args()

    # 动态生成输入文件路径
    input_filename = f"{args.input_date_prefix}_{args.run_type}.xlsx"
    input_path = Path(args.input_dir) / input_filename

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    df_full = read_table(input_path)

    for col in ["HRP", "TMB", "H2O2"]:
        if col not in df_full.columns:
            raise ValueError(f"Missing required column: {col}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 如果启用了 keep_all_hrp，只需处理一次，且不传入特定的 HRP；否则按照提供的列表循环
    hrps_to_process = [None] if args.keep_all_hrp else args.hrp

    for current_hrp in hrps_to_process:
        if current_hrp is not None:
            df = df_full[np.isclose(df_full["HRP"].astype(float), current_hrp)].copy()
            if df.empty:
                print(f"Warning: No rows matched --hrp {current_hrp}. Skipping this value.")
                continue

            # 格式化数字：若为整数形式显示1，否则显示0.01等
            hrp_str = str(int(current_hrp)) if float(current_hrp).is_integer() else str(current_hrp)
            # 输出文件名：加上日期前缀
            output_name = f"{args.input_date_prefix}_{args.run_type}_HRP_{hrp_str}_res.xlsx"
        else:
            df = df_full.copy()
            # 全量输出文件名：加上日期前缀
            output_name = f"{args.input_date_prefix}_{args.run_type}_HRP_all_res.xlsx"
            hrp_str = "all"

        output_path = output_dir / output_name

        frame_cols = find_frame_cols(df)

        rows: list[dict[str, float | int]] = []
        for raw_idx, row in df.iterrows():
            y = row[frame_cols].to_numpy(dtype=float)
            if np.isnan(y).any():
                raise ValueError(
                    f"NaN found in frame columns at input row index {raw_idx}. Please inspect the raw file first.")
            if not args.no_filter:
                y = hampel_isolated_spikes(y)

            features = extract_objectives(
                y,
                baseline=args.baseline,
                signal_threshold=args.signal_threshold,
                retention_last_n=args.retention_last_n,
                smooth_window=args.smooth_window,
                auc_method=args.auc_method,
            )
            rows.append(
                {
                    "HRP": float(row["HRP"]),
                    "TMB": float(row["TMB"]),
                    "H2O2": float(row["H2O2"]),
                    **features,
                    "source_row": int(raw_idx),
                }
            )

        rep = pd.DataFrame(rows)
        group_cols = ["HRP", "TMB", "H2O2"] if (args.keep_all_hrp or len(np.unique(rep["HRP"])) > 1) else ["TMB",
                                                                                                           "H2O2"]

        # Assign replicate number within each physical condition.
        rep = rep.sort_values(group_cols + ["source_row"]).reset_index(drop=True)
        rep["replicate_id"] = rep.groupby(group_cols).cumcount() + 1

        col_order = ["HRP", "TMB", "H2O2", "max_intensity", "reaction_speed", "color_retention", "AUC"]
        if args.add_trace_cols:
            col_order += ["replicate_id", "source_row"]
        rep = rep[col_order]

        write_table(rep, output_path)

        counts = rep.groupby(["HRP", "TMB", "H2O2"]).size()
        print(f"\n--- Processing [{args.input_date_prefix}_{args.run_type}] HRP = {hrp_str} ---")
        print(f"Saved replicate-level objective data: {len(rep)} rows -> {output_path.resolve()}")
        print(f"Unique conditions: {counts.size}")
        print(f"Replicate count per condition: min={counts.min()}, median={counts.median():.0f}, max={counts.max()}")

        if args.generate_summary:
            summary = summarize_replicates(rep, group_cols)
            # Summary文件：加上日期前缀
            summary_name = f"{args.input_date_prefix}_{args.run_type}_summary_HRP_{hrp_str}.xlsx"
            summary_path = output_dir / summary_name
            write_table(summary, summary_path)
            print(f"Saved replicate summary: {len(summary)} rows -> {summary_path.resolve()}")


if __name__ == "__main__":
    main()