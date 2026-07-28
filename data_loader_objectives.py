from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


META_COLS = ["HRP", "TMB", "H2O2"]
ADDITIVE_COLS = [
    "CMC", "DMSO", "PEG20K", "PL127", "BSA", "PVA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCl2", "Sucrose", "CaCl2",
    "ZnSO4", "PAA", "MnSO4", "PEG200K", "FeSO4", "PEG6000", "PEG400",
]
CONTROL_COLS = ["NaN3", "NaN4"]
OBJECTIVE_COLS = ["max_intensity", "reaction_speed", "color_retention", "AUC"]
UNIFIED_TRACE_COLS = [
    "composition_type", "is_ess_only", "n_active", "n_frames", "source_layout",
    "source_file", "source_sheet", "source_row",
]

# Quick-run entry: edit only "files" (and optionally "output"), then run this file.
DATA_LOADER_CONFIG: dict[str, object] = {
    "files": [
        "logs/Jul_9_full_log/0.001_PA_2.xlsx",
        "logs/Jul_9_full_log/0.001_PA_1.xlsx",
        "logs/Jul_9_full_log/0.01_PA_2.xlsx",
        "logs/Jul_9_full_log/0.01_PA_1.xlsx",
        "logs/Jul_9_full_log/0.1_PA_2.xlsx",
        "logs/Jul_9_full_log/0.1_PA_1.xlsx",
        "logs/Jul_9_full_log/0.001_SA.xlsx",
        "logs/Jul_9_full_log/0.01_SA.xlsx",
        "logs/Jul_9_full_log/0.1_SA.xlsx",
        "logs/Jul_1_full_log/7_1_BO3.xlsx",
        "logs/Jul_1_full_log/7_1_BO2.xlsx",
        "logs/Jul_1_full_log/7_1_BO1.xlsx",
        "logs/Jul_1_full_log/7_1_LHS.xlsx",
    ],
    "output": "logs/Jul_9_full_log/ESS_SA_PA.xlsx",
    "overwrite": True,
}


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


def read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError("Input must be .xlsx, .xls, or .csv")


def write_table(df: pd.DataFrame, path: Path, sheet_name: str = "Sheet1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df.to_excel(path, index=False, sheet_name=sheet_name)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8")
    else:
        raise ValueError("Output must be .xlsx, .xls, or .csv")


def _unnamed_frame_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed:")]

    def unnamed_index(col) -> int:
        return int(str(col).split(":", maxsplit=1)[-1].strip())

    return sorted(cols, key=unnamed_index)


def _validate_frame_sequence(
        frame_cols: Sequence[str], *, path: Path, expected_start: int
) -> None:
    if str(frame_cols[0]).strip().lower().startswith("frame"):
        indices = [
            int(str(col).replace("_", " ").strip().split()[-1])
            for col in frame_cols
        ]
    else:
        indices = [int(str(col).split(":", maxsplit=1)[-1].strip()) for col in frame_cols]

    expected = list(range(expected_start, expected_start + len(indices)))
    if indices != expected:
        raise ValueError(
            f"{path.name}: frame columns must be unique and contiguous from "
            f"{expected_start}; observed indices begin {indices[:5]} and end {indices[-5:]}."
        )


def detect_input_layout(df: pd.DataFrame, path: Path) -> tuple[str, list[str]]:
    """Detect an essentials-only or additive-plate table from its columns.

    Essentials-only tables contain only HRP/TMB/H2O2 plus explicit ``frame N``
    columns. Additive plates contain the complete 22-additive metadata block,
    NaN3/NaN4 placeholders, and either explicit frame columns or pandas
    ``Unnamed: N`` columns created from the headerless trace block.
    """
    missing_meta = [c for c in META_COLS if c not in df.columns]
    if missing_meta:
        raise KeyError(f"{path.name}: missing required metadata columns {missing_meta}")

    present_add = [c for c in ADDITIVE_COLS if c in df.columns]
    present_ctrl = [c for c in CONTROL_COLS if c in df.columns]
    is_additive_layout = bool(present_add or present_ctrl)

    if is_additive_layout:
        missing_add = [c for c in ADDITIVE_COLS if c not in df.columns]
        missing_ctrl = [c for c in CONTROL_COLS if c not in df.columns]
        if missing_add or missing_ctrl:
            raise KeyError(
                f"{path.name}: incomplete additive metadata block; "
                f"missing additives={missing_add}, missing controls={missing_ctrl}"
            )

        named_frames = [
            c for c in df.columns
            if str(c).strip().lower().startswith("frame")
        ]
        unnamed_frames = _unnamed_frame_cols(df)
        if named_frames and unnamed_frames:
            raise ValueError(
                f"{path.name}: mixed named and unnamed frame columns are ambiguous."
            )
        frame_cols = find_frame_cols(df) if named_frames else unnamed_frames
        if not frame_cols:
            raise ValueError(f"{path.name}: no frame columns detected for additive layout.")
        expected_start = 0 if named_frames else len(META_COLS + ADDITIVE_COLS + CONTROL_COLS)
        _validate_frame_sequence(frame_cols, path=path, expected_start=expected_start)

        known = set(META_COLS + ADDITIVE_COLS + CONTROL_COLS + frame_cols)
        unexpected = [c for c in df.columns if c not in known]
        if unexpected:
            raise ValueError(
                f"{path.name}: unexpected columns would otherwise be mistaken for frames: {unexpected}"
            )
        return "additive", frame_cols

    frame_cols = find_frame_cols(df)
    _validate_frame_sequence(frame_cols, path=path, expected_start=0)
    known = set(META_COLS + frame_cols)
    unexpected = [c for c in df.columns if c not in known]
    if unexpected:
        raise ValueError(
            f"{path.name}: essentials-only layout has unexpected columns: {unexpected}"
        )
    return "essentials", frame_cols


def _finite_float(value, *, label: str, path: Path, source_row: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path.name}: non-numeric {label} at source row {source_row}: {value!r}"
        ) from exc
    if not np.isfinite(result):
        raise ValueError(
            f"{path.name}: non-finite {label} at source row {source_row}: {value!r}"
        )
    return result


def _composition_type(n_active: int) -> str:
    if n_active == 0:
        return "ESS"
    if n_active == 1:
        return "SA"
    if n_active == 2:
        return "PA"
    return "higher_order"


def _validate_processing_options(
        *,
        baseline: float,
        signal_threshold: float,
        retention_last_n: int,
        smooth_window: int,
        auc_method: str,
        max_active: int | None,
) -> None:
    if not np.isfinite(baseline):
        raise ValueError("baseline must be finite.")
    if not np.isfinite(signal_threshold) or signal_threshold < 0.0:
        raise ValueError("signal_threshold must be finite and non-negative.")
    if not isinstance(retention_last_n, (int, np.integer)) or retention_last_n < 1:
        raise ValueError("retention_last_n must be a positive integer.")
    if (
            not isinstance(smooth_window, (int, np.integer))
            or smooth_window < 1
            or smooth_window % 2 == 0
    ):
        raise ValueError("smooth_window must be a positive odd integer.")
    if auc_method not in {"trapz", "sum"}:
        raise ValueError("auc_method must be 'trapz' or 'sum'.")
    if max_active is not None and (
            not isinstance(max_active, (int, np.integer)) or max_active < 0
    ):
        raise ValueError("max_active must be a non-negative integer or None.")


def process_mixed_input(
        path: Path,
        *,
        batch: str,
        input_sheet: str | int,
        baseline: float,
        signal_threshold: float,
        retention_last_n: int,
        smooth_window: int,
        auc_method: str,
        use_filter: bool,
        max_active: int | None,
) -> pd.DataFrame:
    """Standardize one essentials/additive workbook into the unified schema."""
    _validate_processing_options(
        baseline=baseline,
        signal_threshold=signal_threshold,
        retention_last_n=retention_last_n,
        smooth_window=smooth_window,
        auc_method=auc_method,
        max_active=max_active,
    )
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    df = read_table(path, sheet_name=input_sheet)
    if df.empty:
        raise ValueError(f"{path.name}: input table contains no data rows.")
    layout, frame_cols = detect_input_layout(df, path)
    if len(frame_cols) < 2:
        raise ValueError(f"{path.name}: at least two frame columns are required.")

    out_rows: list[dict[str, float | int | bool | str]] = []
    for raw_idx, row in df.iterrows():
        source_row = int(raw_idx)
        y = pd.to_numeric(row[frame_cols], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(
                f"{path.name}: non-finite value in frame columns at source row {source_row}."
            )
        if use_filter:
            y = hampel_isolated_spikes(y)

        features = extract_objectives(
            y,
            baseline=baseline,
            signal_threshold=signal_threshold,
            retention_last_n=retention_last_n,
            smooth_window=smooth_window,
            auc_method=auc_method,
        )

        rec: dict[str, float | int | bool | str] = {"Batch": batch}
        for metadata in META_COLS:
            value = _finite_float(
                row[metadata], label=metadata, path=path, source_row=source_row
            )
            if value < 0.0:
                raise ValueError(
                    f"{path.name}: negative {metadata} concentration at source row "
                    f"{source_row}: {value}"
                )
            rec[metadata] = value

        active: dict[str, float] = {}
        for additive in ADDITIVE_COLS:
            value = 0.0
            if layout == "additive" and pd.notna(row[additive]):
                value = _finite_float(
                    row[additive], label=additive, path=path, source_row=source_row
                )
                if value < 0.0:
                    raise ValueError(
                        f"{path.name}: negative {additive} concentration at "
                        f"source row {source_row}: {value}"
                    )
            rec[additive] = value
            if value > 0.0:
                active[additive] = value

        n_active = len(active)
        if max_active is not None and n_active > max_active:
            raise ValueError(
                f"{path.name}: source row {source_row} has {n_active} active additives "
                f"(> --max_active={max_active}): {active}"
            )

        ctrl_value = 0.0
        if layout == "additive":
            for control in CONTROL_COLS:
                if pd.notna(row[control]):
                    control_value = _finite_float(
                        row[control], label=control, path=path, source_row=source_row
                    )
                    if control_value < 0.0:
                        raise ValueError(
                            f"{path.name}: negative {control} marker at source row "
                            f"{source_row}: {control_value}"
                        )
                    ctrl_value += control_value

        rec["CTRL"] = ctrl_value
        rec.update(features)
        rec.update(
            {
                "composition_type": _composition_type(n_active),
                "is_ess_only": n_active == 0,
                "n_active": n_active,
                "n_frames": len(frame_cols),
                "source_layout": layout,
                "source_file": str(path.resolve()),
                "source_sheet": str(input_sheet),
                "source_row": source_row,
            }
        )
        out_rows.append(rec)

    col_order = (
        ["Batch"] + META_COLS + ADDITIVE_COLS + ["CTRL"] + OBJECTIVE_COLS
        + UNIFIED_TRACE_COLS
    )
    return pd.DataFrame(out_rows)[col_order]


def load_mixed_inputs(
        inputs: Sequence[str | Path],
        *,
        batches: Sequence[str] | None = None,
        input_sheet: str | int = "Sheet1",
        baseline: float = 16.0,
        signal_threshold: float = 5.0,
        retention_last_n: int = 5,
        smooth_window: int = 5,
        auc_method: str = "trapz",
        use_filter: bool = True,
        max_active: int | None = None,
) -> pd.DataFrame:
    """Load mixed raw workbooks into one replicate-level table without aggregation.

    Layout is inferred from the column schema, not the filename. Essentials-only
    inputs receive explicit zeroes in all additive columns, and any row with zero
    active additives is marked ``is_ess_only=True``.
    """
    if isinstance(inputs, (str, Path)):
        raise TypeError("inputs must be a sequence of paths, not a single path value.")
    if not inputs:
        raise ValueError("At least one input path is required.")
    if isinstance(batches, str):
        raise TypeError("batches must be a sequence of labels, not a single string.")
    paths = [Path(path) for path in inputs]
    batch_labels = list(batches) if batches is not None else [path.stem for path in paths]
    if len(batch_labels) != len(paths):
        raise ValueError("--batches count must match --inputs count.")

    parts = [
        process_mixed_input(
            path,
            batch=batch,
            input_sheet=input_sheet,
            baseline=baseline,
            signal_threshold=signal_threshold,
            retention_last_n=retention_last_n,
            smooth_window=smooth_window,
            auc_method=auc_method,
            use_filter=use_filter,
            max_active=max_active,
        )
        for path, batch in zip(paths, batch_labels)
    ]
    frame_counts = {
        str(path.resolve()): int(part["n_frames"].iloc[0])
        for path, part in zip(paths, parts)
    }
    if len(set(frame_counts.values())) != 1:
        raise ValueError(
            "Cannot combine inputs with different frame counts because AUC and "
            f"reaction-speed scales would not be directly comparable: {frame_counts}"
        )
    return pd.concat(parts, ignore_index=True)


_CONFIG_LOAD_KEYS = {
    "batches",
    "input_sheet",
    "baseline",
    "signal_threshold",
    "retention_last_n",
    "smooth_window",
    "auc_method",
    "use_filter",
    "max_active",
}
_CONFIG_KEYS = {"files", "output", "output_sheet", "overwrite"} | _CONFIG_LOAD_KEYS


def run_configured_loader(
        config: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run the mixed loader from ``DATA_LOADER_CONFIG`` and return its DataFrame.

    ``files`` and ``output`` are required. All scientific processing options are
    optional and otherwise retain the validated defaults of ``load_mixed_inputs``.
    Existing output files are protected unless ``overwrite`` is explicitly true.
    """
    cfg = dict(DATA_LOADER_CONFIG if config is None else config)
    unknown = sorted(set(cfg) - _CONFIG_KEYS)
    if unknown:
        raise KeyError(f"Unknown DATA_LOADER_CONFIG key(s): {unknown}")
    if "files" not in cfg:
        raise KeyError("DATA_LOADER_CONFIG must define 'files'.")
    if "output" not in cfg:
        raise KeyError("DATA_LOADER_CONFIG must define 'output'.")

    output_value = cfg["output"]
    if not isinstance(output_value, (str, Path)):
        raise TypeError("DATA_LOADER_CONFIG['output'] must be a path string or Path.")
    overwrite = cfg.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise TypeError("DATA_LOADER_CONFIG['overwrite'] must be True or False.")
    output_sheet = cfg.get("output_sheet", "Corrected Data")
    if not isinstance(output_sheet, str) or not output_sheet.strip():
        raise TypeError("DATA_LOADER_CONFIG['output_sheet'] must be a non-empty string.")

    output_path = Path(output_value)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Change 'output' or set "
            "DATA_LOADER_CONFIG['overwrite'] = True."
        )

    load_options = {key: cfg[key] for key in _CONFIG_LOAD_KEYS if key in cfg}
    combined = load_mixed_inputs(cfg["files"], **load_options)
    write_table(combined, output_path, sheet_name=output_sheet)

    type_counts = combined["composition_type"].value_counts().sort_index()
    print(f"Saved unified replicate-level data: {len(combined)} rows -> {output_path.resolve()}")
    print(f"Composition-type counts: {type_counts.to_dict()}")
    print(f"Frame count per curve: {int(combined['n_frames'].iloc[0])}")
    return combined


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
            "Convert essentials-only and additive frame-wise blueness curves "
            "into one replicate-level objective table."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=None,
        help=(
            "Input .xlsx/.xls/.csv paths for unified mixed loading. If omitted, "
            "the legacy date-prefix/run-type workflow is used."
        ),
    )
    parser.add_argument(
        "--batches",
        nargs="+",
        default=None,
        help="Optional batch labels matching --inputs; defaults to each file stem.",
    )
    parser.add_argument(
        "--output",
        default="combined_objectives.xlsx",
        help="Unified output path used with --inputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow unified mode to replace an existing output file.",
    )
    parser.add_argument(
        "--input_sheet",
        default="Sheet1",
        help="Worksheet read from every Excel input in unified mode.",
    )
    parser.add_argument(
        "--output_sheet",
        default="Corrected Data",
        help="Worksheet name for unified Excel output.",
    )
    parser.add_argument(
        "--max_active",
        type=int,
        default=None,
        help=(
            "Optional maximum active-additive count per row. The default allows "
            "SA, PA, and future higher-order recipes."
        ),
    )
    parser.add_argument(
        "--run_type",
        type=str,
        default="BO3",
        choices=["LHS", "BO1", "BO2", "BO3"],
    )
    parser.add_argument(
        "--input_dir",
        default="./logs/Jul_1_full_log",
        help="Directory containing the legacy essentials-only input file.",
    )
    parser.add_argument(
        "--input_date_prefix",
        default="7_1",
        help="Date prefix used in the legacy input filename.",
    )
    parser.add_argument("--output_dir", default="./logs/Jul_1_full_log")
    parser.add_argument("--hrp", type=float, nargs="+", default=[0.001, 0.01, 0.1],
                        help="List of HRP values to process.")
    parser.add_argument(
        "--generate_summary",
        action="store_true",
        help="Generate a summary output file with mean/sd/sem per condition for each HRP.",
    )
    parser.add_argument(
        "--keep_all_hrp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not filter HRP; keep all HRP levels in one file.",
    )
    parser.add_argument("--baseline", type=float, default=16.0,
                        help="Baseline intensity to subtract before feature extraction.")
    parser.add_argument("--signal_threshold", type=float, default=5.0)
    parser.add_argument("--retention_last_n", type=int, default=5, help="Number of final frames used for retention.")
    parser.add_argument("--smooth_window", type=int, default=5,
                        help="Odd window size for centered moving-average smoothing.")
    parser.add_argument("--auc_method", choices=["trapz", "sum"], default="trapz")
    parser.add_argument(
        "--no_filter",
        action="store_true",
        help="Disable conservative isolated-spike repair.",
    )
    parser.add_argument(
        "--add_trace_cols",
        action="store_true",
        help="Append replicate_id and source_row for traceability. Off by default.",
    )
    args = parser.parse_args()

    if args.max_active is not None and args.max_active < 0:
        parser.error("--max_active must be non-negative.")

    if args.inputs:
        run_configured_loader(
            {
                "files": args.inputs,
                "batches": args.batches,
                "output": args.output,
                "output_sheet": args.output_sheet,
                "overwrite": args.overwrite,
                "input_sheet": args.input_sheet,
                "baseline": args.baseline,
                "signal_threshold": args.signal_threshold,
                "retention_last_n": args.retention_last_n,
                "smooth_window": args.smooth_window,
                "auc_method": args.auc_method,
                "use_filter": not args.no_filter,
                "max_active": args.max_active,
            }
        )
        return

    if args.batches is not None:
        parser.error("--batches requires --inputs.")

    # 动态生成输入文件路径
    input_filename = f"{args.input_date_prefix}_{args.run_type}.xlsx"
    input_path = Path(args.input_dir) / input_filename

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

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
    if len(sys.argv) == 1:
        run_configured_loader()
    else:
        main()
