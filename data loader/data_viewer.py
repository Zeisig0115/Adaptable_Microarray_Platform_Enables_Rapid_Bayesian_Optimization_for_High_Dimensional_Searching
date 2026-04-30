#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
collapse_replicates_by_std.py

规则：
  - 按 25D 配方分组，先在当前数据上重新计算组内 auc_mean_std（样本标准差，ddof=1）
  - 若 auc_mean_std > THRESHOLD：该组仅保留 auc_mean 最大的那一条
  - 若 auc_mean_std <= THRESHOLD：该组仅保留一条，auc_mean 取组内平均
  - 为保留的那一条填入该组的 n_repeats（组内条数）与 auc_mean_std
  - 删除 roi_sd_q25/roi_sd_q50/roi_sd_q75 列（如果存在）

输入：./data/data_filtered.csv （若不存在则使用 ./data/data.csv）
输出：./data/data_collapsed.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re
from collections import defaultdict

# ================= 配置 =================
DATA_DIR = Path("./data")
IN_PRIMARY  = DATA_DIR / "data_filtered.csv"
IN_FALLBACK = DATA_DIR / "data.csv"
OUT_PATH    = DATA_DIR / "data_collapsed.csv"

THRESHOLD   = 1.8         # auc_mean_std 的阈值
ROUND_DECIMALS = 12       # 分组键四舍五入，避免浮点毛刺
# ======================================

# 与上游一致的列名定义
def sanitize(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label).lower().strip("_")

ESSENTIALS = {"TMB": "tmb", "HRP": "hrp", "H2O2": "h2o2"}
ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PVA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PAA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]
KEY_COLS = [ESSENTIALS[k] for k in ["TMB", "HRP", "H2O2"]] + [sanitize(lab) for lab in ADDITIVE_LABELS]

def main():
    # 读入
    in_path = IN_PRIMARY if IN_PRIMARY.exists() else IN_FALLBACK
    if not in_path.exists():
        raise FileNotFoundError(f"找不到输入：{IN_PRIMARY} 或 {IN_FALLBACK}")
    df = pd.read_csv(in_path)

    # 校验
    need = set(KEY_COLS + ["auc_mean"])
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"输入缺少必要列：{sorted(missing)}")

    # 构造配方键（25D 向量四舍五入）
    key_arr = np.round(df[KEY_COLS].to_numpy(dtype=float), ROUND_DECIMALS)
    keys = [tuple(row.tolist()) for row in key_arr]
    groups = defaultdict(list)
    for idx, k in enumerate(keys):
        groups[k].append(idx)

    rows_out = []
    for k, idxs in groups.items():
        sub = df.iloc[idxs].copy()

        # 组内 auc_mean 的样本标准差（NaN 会被忽略）
        auc_vals = sub["auc_mean"].to_numpy(dtype=float)
        auc_vals_finite = auc_vals[np.isfinite(auc_vals)]
        n = int(len(sub))
        if auc_vals_finite.size >= 2:
            auc_std = float(np.std(auc_vals_finite, ddof=1))
        else:
            auc_std = float('nan')

        if n == 1:
            # 只有一个重复，直接保留
            row = sub.iloc[0].to_dict()
            row["n_repeats"] = 1
            row["auc_mean_std"] = np.nan
            rows_out.append(row)
            continue

        if np.isfinite(auc_std) and auc_std > THRESHOLD:
            # 高方差：仅保留 auc_mean 最大的一条
            # 若全 NaN，np.nanargmax 会报错 → 退化为第一条
            if np.isfinite(auc_vals).any():
                keep_idx_local = int(np.nanargmax(auc_vals))
            else:
                keep_idx_local = 0
            row = sub.iloc[keep_idx_local].to_dict()
            row["n_repeats"] = n
            row["auc_mean_std"] = auc_std
            rows_out.append(row)
        else:
            # 低方差（含 NaN 情况）：保留一条，auc_mean = 组内平均（忽略 NaN）
            mean_val = float(np.nanmean(auc_vals)) if auc_vals_finite.size > 0 else np.nan
            row = sub.iloc[0].to_dict()  # 以第一条为模板（保留其 sheet、well 等元数据）
            row["auc_mean"] = mean_val
            row["n_repeats"] = n
            row["auc_mean_std"] = auc_std
            rows_out.append(row)

    df_out = pd.DataFrame(rows_out, columns=df.columns.tolist())

    # 移除 roi_sd_* 列（若存在）
    drop_cols = [c for c in ["roi_sd_q25", "roi_sd_q50", "roi_sd_q75"] if c in df_out.columns]
    if drop_cols:
        df_out = df_out.drop(columns=drop_cols)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"完成整理：输入 {len(df)} 行 → 输出 {len(df_out)} 行，写入：{OUT_PATH}")
    print(f"阈值：auc_mean_std > {THRESHOLD} → 取组内最大；否则 → 组内平均（仅保留一条）")

    # 方便交互使用
    global data
    data = df_out

if __name__ == "__main__":
    main()
