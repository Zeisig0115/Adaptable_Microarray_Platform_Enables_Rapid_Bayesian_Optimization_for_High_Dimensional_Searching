#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_extractor_from_npy.py  (AUC_mean 版本)

功能：
    1) 从 ./data 下读取 "<sheet_name> b.npy" 和 "<sheet_name> t.npy"
    2) 从 Excel 读取对应配方信息（仅 3 essentials + 22 additives，全部为连续值）
    3) 计算单一指标：AUC_mean = ∫ y(t) dt / max(t)
    4) 去重（25 维浓度完全一致的配方只保留一次）
    5) 输出 data.csv （25D 特征 + 1D 目标）

说明：
    - 不记录 *_sw（二值开关）
    - 不做平滑/阈值/基线扣除；如需改为除以持续时长，请把 DIVIDE_BY_TMAX 改为 False
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd

# ========== 配置 ==========
BASE_DIR   = Path("./data")
EXCEL_NAME = "3 dimensional data concentrations.xlsx"
OUT_NAME   = "data.csv"

# 是否用 max(t) 做归一化（True），否则用 (t[-1]-t[0])
DIVIDE_BY_TMAX = True

# 3 + 22 组件
ESSENTIALS = {"TMB": "tmb", "HRP": "hrp", "H2O2": "h2o2"}
ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PVA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PAA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]
# ==========================

def sanitize(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label).lower().strip("_")

ADDITIVE_MAP = {lab.upper().replace(" ", ""): sanitize(lab) for lab in ADDITIVE_LABELS}
ALL_COMPONENTS = {**ESSENTIALS, **ADDITIVE_MAP}  # 值无用，仅用于识别标签

def collect_block(df: pd.DataFrame, label_row: int, n_rows: int = 16) -> np.ndarray:
    rows = []
    r = label_row
    while r < len(df) and len(rows) < n_rows:
        first_cell_norm = str(df.iat[r, 0]).upper().replace(" ", "")
        if r != label_row and first_cell_norm in ALL_COMPONENTS:
            break
        vals = pd.to_numeric(df.iloc[r, 1:25], errors="coerce")
        if vals.notna().any():
            rows.append(vals.to_numpy(dtype=float, copy=True))
        r += 1
    if not rows:
        rows = [np.zeros(24, dtype=float) for _ in range(n_rows)]
    while len(rows) < n_rows:
        rows.append(rows[-1].copy())
    return np.vstack(rows[:n_rows])

def parse_sheet(df: pd.DataFrame) -> dict[str, np.ndarray]:
    comps: dict[str, np.ndarray] = {}
    col0 = df.iloc[:, 0].astype(str).str.upper().str.replace(" ", "")
    for label in ALL_COMPONENTS:
        matches = df.index[col0 == label]
        if not matches.empty:
            comps[label] = collect_block(df, matches[0])
        else:
            comps[label] = np.zeros((16, 24), dtype=float)
    return comps

def auc_mean(t: np.ndarray, y: np.ndarray) -> float:
    """AUC_mean = ∫ y(t) dt / max(t)（或 / duration）"""
    if t is None or y is None or t.size != y.size or t.size < 2:
        return np.nan
    m = np.isfinite(t) & np.isfinite(y)
    if not np.any(m):
        return np.nan
    t = t[m].astype(float)
    y = y[m].astype(float)
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # 积分
    area = float(np.trapezoid(y, t))

    # 归一化
    if DIVIDE_BY_TMAX:
        denom = float(np.max(t))
    else:
        denom = float(t[-1] - t[0])
    if denom <= 0:
        return np.nan
    return area / denom

def main():
    excel_path = BASE_DIR / EXCEL_NAME
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    xls = pd.ExcelFile(excel_path)

    # 列名（25D）
    feature_cols = [ESSENTIALS[k] for k in ["TMB", "HRP", "H2O2"]]
    additive_keys_inorder = [lab.upper().replace(" ", "") for lab in ADDITIVE_LABELS]
    feature_cols += [ADDITIVE_MAP[k] for k in additive_keys_inorder]
    columns = feature_cols + ["auc_mean"]

    records = []
    seen = set()

    for sheet_name in xls.sheet_names:
        if sheet_name.strip().lower() == "time points":
            continue

        b_path = BASE_DIR / f"{sheet_name.strip()} b.npy"
        t_path = BASE_DIR / f"{sheet_name.strip()} t.npy"
        if not b_path.exists() or not t_path.exists():
            print(f"⚠ 缺少 {b_path.name} 或 {t_path.name}，跳过 {sheet_name}")
            continue

        blue_tensor = np.load(b_path)  # (T,16,24)
        t_seconds   = np.load(t_path)  # (T,)

        df_comp = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        comps = parse_sheet(df_comp)

        for r in range(16):
            for c in range(24):
                # 25D 连续向量
                vec = [
                    comps["TMB"][r, c], comps["HRP"][r, c], comps["H2O2"][r, c]
                ] + [comps[k][r, c] for k in additive_keys_inorder]

                key = tuple(np.round(vec, 10))  # 去重键
                if key in seen:
                    continue
                seen.add(key)

                y = blue_tensor[:, r, c]
                yval = auc_mean(t_seconds, y)

                row = vec + [yval]
                records.append(row)

    df_out = pd.DataFrame.from_records(records, columns=columns)
    df_out.to_csv(OUT_NAME, index=False)
    print(f"✅ 写入 {OUT_NAME}（列：{', '.join(columns)}），共 {len(df_out)} 行（唯一配方）")

if __name__ == "__main__":
    main()
