#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_extractor_from_npy.py

功能：
    1. 从 ./data 下读取 "<sheet_name> b.npy" 和 "<sheet_name> t.npy"
    2. 从 Excel 读取对应配方信息（3 essentials + additives）
    3. 对每孔曲线计算三项指标：
         - speed       = (max - min) / Δt，如果 max 在 min 之前则取负
         - max_val     = 最大蓝度（已做背景校正）
         - tail_ratio  = |max - 末尾10%均值| / max
    4. 去重（浓度完全一致的配方只保留一次）
    5. 输出 data.csv 供 BO 使用
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd

# ========== 配置 ==========
BASE_DIR   = Path("./data")
EXCEL_NAME = "3 dimensional data concentrations.xlsx"
OUT_NAME   = "data.csv"

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
ALL_COMPONENTS = {**ESSENTIALS, **ADDITIVE_MAP}

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
    comps = {}
    col0 = df.iloc[:, 0].astype(str).str.upper().str.replace(" ", "")
    for label in ALL_COMPONENTS:
        matches = df.index[col0 == label]
        if not matches.empty:
            comps[label] = collect_block(df, matches[0])
        else:
            comps[label] = np.zeros((16, 24), dtype=float)
    return comps

def extract_metrics(t: np.ndarray, y: np.ndarray):
    """
    返回: (t_peak, I_max, D_blue)
      - t_peak: 达到平滑后全局最大值的最早时间点（单位与 t 一致）
      - I_max: 3点滑动平均平滑后的最大强度
      - D_blue: 强度 >= 0.5*I_max 的持续时长（单位与 t 一致，阈值处用线性插值）
    说明:
      - 非等间隔 t OK；若 t 未排序，会按 t 排序后计算
      - 平滑: 3 点移动平均，边界复制
      - 多个最大值并列 -> 取“最早”那个作为 t_peak
      - 若全程 I_max==0，则 D_blue=0
    """
    # 基础校验
    if t is None or y is None or y.size < 3 or t.size != y.size:
        return np.nan, np.nan, np.nan

    # 去除 NaN/Inf；若存在坏点，按 t 排序前先一起筛
    mask = np.isfinite(t) & np.isfinite(y)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    t = t[mask]
    y = y[mask]

    # 按时间排序（以防乱序）
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # 3点平滑（边界复制）
    if y.size < 3:
        # 极端短序列，直接返回简单值
        i_pk = int(np.argmax(y))
        t_peak = float(t[i_pk])
        I_max = float(y[i_pk])
        return t_peak, I_max, 0.0

    y_pad = np.pad(y, (1, 1), mode="edge")
    y_s = (y_pad[:-2] + y_pad[1:-1] + y_pad[2:]) / 3.0

    # I_max & t_peak（最早达到最大值的时间）
    I_max = float(np.max(y_s))
    # argmax 的“最早”实现
    idx_max_all = np.where(y_s == I_max)[0]
    i_pk = int(idx_max_all[0])
    t_peak = float(t[i_pk])

    # D_blue: >= 0.5 * I_max 的时长（线性插值）
    if I_max <= 0:
        return t_peak, I_max, 0.0

    thr = 0.8 * I_max

    # 找 t_up（首次上穿或起点即在阈上）
    t_up = None
    if y_s[0] >= thr:
        t_up = float(t[0])
    else:
        for i in range(1, len(y_s)):
            if (y_s[i-1] < thr) and (y_s[i] >= thr):
                # 线性插值 crossing
                y0, y1 = y_s[i-1], y_s[i]
                t0, t1 = t[i-1], t[i]
                # 避免除零
                if y1 == y0:
                    t_up = float(t1)
                else:
                    t_up = float(t0 + (thr - y0) * (t1 - t0) / (y1 - y0))
                break

    if t_up is None:
        # 从未到达 50% 阈值
        return t_peak, I_max, 0.0

    # 找 t_down（首次下穿）
    t_down = None
    # 从进入阈值后开始找
    start_i = max(i_pk, 1)  # 也可以从 i=1 开始，这里从峰位后仍然兼容
    # 更稳妥：从找到 t_up 的区间继续往后找
    # 先确定 t_up 所在的 i 起点
    for i in range(1, len(y_s)):
        # 找到 t_up 所在的区间并从此往后
        if (t[i-1] <= t_up <= t[i]) or (t_up == t[0] and i == 1):
            start_i = i
            break

    above = True  # 进入阈上状态
    for j in range(start_i, len(y_s)):
        # 判定从 >=thr 到 <thr 的首次下穿
        if (y_s[j-1] >= thr) and (y_s[j] < thr):
            y0, y1 = y_s[j-1], y_s[j]
            t0, t1 = t[j-1], t[j]
            if y1 == y0:
                t_down = float(t1)
            else:
                t_down = float(t0 + (thr - y0) * (t1 - t0) / (y1 - y0))
            break

    if t_down is None:
        # 一直在阈上到最后
        t_down = float(t[-1])

    D_blue = float(t_down - t_up)
    if D_blue < 0:
        # 理论上不应出现，防御性处理
        D_blue = 0.0

    return t_peak, I_max, D_blue

def main():
    excel_path = BASE_DIR / EXCEL_NAME
    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    xls = pd.ExcelFile(excel_path)
    additive_keys_inorder = [lab.upper().replace(" ", "") for lab in ADDITIVE_LABELS]
    columns = ["sheet"] + list(ESSENTIALS.values())
    for lab in ADDITIVE_LABELS:
        sanitized = ADDITIVE_MAP[lab.upper().replace(" ", "")]
        columns.append(f"{sanitized}_sw")
        columns.append(f"{sanitized}_lvl")
    columns += ["t_peak", "I_max", "D_blue"]

    records = []
    seen_keys = set()

    for sheet_name in xls.sheet_names:
        if sheet_name.strip().lower() == "time points":
            continue
        exp_name = sheet_name.strip()
        b_path = BASE_DIR / f"{exp_name} b.npy"
        t_path = BASE_DIR / f"{exp_name} t.npy"
        if not b_path.exists() or not t_path.exists():
            print(f"⚠ 缺少 {b_path.name} 或 {t_path.name}，跳过 {sheet_name}")
            continue

        blue_tensor = np.load(b_path)    # (T,16,24)
        t_seconds   = np.load(t_path)    # (T,)

        df_comp = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        comps = parse_sheet(df_comp)

        for r in range(16):
            for c in range(24):
                y = blue_tensor[:, r, c]
                t_peak, I_max, D_blue = extract_metrics(t_seconds, y)

                essentials_triplet = (comps["TMB"][r, c], comps["HRP"][r, c], comps["H2O2"][r, c])
                key = tuple(np.round(essentials_triplet, 10)) + tuple(np.round([comps[lab][r, c] for lab in additive_keys_inorder], 10))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                row = [sheet_name]
                for lab in ESSENTIALS:
                    row.append(essentials_triplet[list(ESSENTIALS.keys()).index(lab)])
                for lab in additive_keys_inorder:
                    lvl = comps[lab][r, c]
                    row.append(int(lvl > 0))
                    row.append(lvl)
                row.extend([t_peak, I_max, D_blue])
                records.append(row)

    df_out = pd.DataFrame.from_records(records, columns=columns)
    df_out.to_csv(OUT_NAME, index=False)
    print(f"✅ 写入 {OUT_NAME}（列：t_peak, I_max, D_blue），共 {len(df_out)} 行（唯一配方）")

if __name__ == "__main__":
    main()
