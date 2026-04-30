#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
postprocess_filter_and_collapse.py

步骤：
1) 读取 ./data/data.csv
2) 过滤：roi_sd_mean < ROI_SD_mean_CUTOFF（默认 100；与你当前数据列一致）
3) 以 25D 浓度向量分组（tmb/hrp/h2o2 + 22 additives，列名已 sanitize 为小写），对键 round(., 12)
4) 组内收缩：
   - 若 auc_mean_std(组内样本标准差, ddof=1) > AUC_STD_CUTOFF 且 n>=2：仅保留 auc_mean 最大的那一行
   - 若 auc_mean_std <= AUC_STD_CUTOFF 且 n>=2：保留一行，并将其 auc_mean 替换为组平均
   - 若 n==1：原样保留，但更新 n_repeats=1, auc_mean_std=NaN
5) 丢弃 roi_sd_q25/roi_sd_q50/roi_sd_q75（若存在）及中间列 _key
6) 写出 ./data/data_filtered.csv

备注：
- 你的数据只有 roi_sd_mean（没有 roi_sd_q50/q25/q75），因此过滤项用 roi_sd_mean。
- 为便于自检，脚本会打印聚合后的类别计数：
  0 个添加剂非 0（HRP/TMB/H2O2 网格）、1 个添加剂非 0（single + 特殊 22）、2 个添加剂非 0（pairwise）。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import re

# ========= 配置 =========
DATA_DIR = Path("./data")
IN_NAME = "data.csv"
OUT_NAME = "data_filtered.csv"

ROI_SD_mean_CUTOFF = 100.0   # 你的数据列：roi_sd_mean；默认 100 基本等于不过滤
AUC_STD_CUTOFF = 1.8
GROUP_ROUND_DECIMALS = 12
# =======================

# 组件（与生成 data.csv 的脚本保持一致）
ESSENTIALS = {"TMB": "tmb", "HRP": "hrp", "H2O2": "h2o2"}
ADDITIVE_LABELS = [
    "EtOH", "PEG20K", "DMSO", "PL127", "BSA", "PVA", "TW80", "Glycerol",
    "TW20", "Imidazole", "TX100", "EDTA", "MgCL2", "Sucrose", "CaCl2",
    "Zn2", "PAA", "Mn2", "PEG200K", "Fe2", "PEG5K", "PEG400"
]

def sanitize(label: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", label).lower().strip("_")

FEATURE_COLS = [ESSENTIALS[k] for k in ["TMB", "HRP", "H2O2"]]
ADDITIVE_COLS = [sanitize(lab) for lab in ADDITIVE_LABELS]
KEY_COLS = FEATURE_COLS + ADDITIVE_COLS  # 25D

def main():
    in_path = DATA_DIR / IN_NAME
    out_path = DATA_DIR / OUT_NAME

    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)

    # 必要列检查（与你的数据一致：有 roi_sd_mean）
    needed = set(["sheet", "well", "auc_mean", "n_repeats", "auc_mean_std", "roi_sd_mean"] + KEY_COLS)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"缺少列：{missing}")

    # 1) 过滤 roi_sd_mean
    before = len(df)
    df = df[df["roi_sd_mean"] < ROI_SD_mean_CUTOFF].copy()
    after_filter = len(df)
    print(f"[INFO] 过滤 roi_sd_mean >= {ROI_SD_mean_CUTOFF}: {before} → {after_filter}")

    if len(df) == 0:
        # 输出空框架（无 roi 列）
        cols_out = ["sheet", "well"] + KEY_COLS + ["auc_mean", "n_repeats", "auc_mean_std"]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=cols_out).to_csv(out_path, index=False)
        print(f"[INFO] 写出空文件：{out_path}")
        return

    # 2) 重新分组（按 25D；对键做 rounding 以避免浮点毛刺）
    key_array = np.round(df[KEY_COLS].to_numpy(dtype=float), GROUP_ROUND_DECIMALS)
    df["_key"] = [tuple(row.tolist()) for row in key_array]

    # 3) 组内统计与收缩
    rows_out = []
    for key, sub in df.groupby("_key", sort=False):
        n = len(sub)

        aucs = sub["auc_mean"].to_numpy(dtype=float)
        finite = np.isfinite(aucs)
        if finite.sum() >= 2:
            std = float(np.std(aucs[finite], ddof=1))  # 样本标准差
        else:
            std = float("nan")

        if n >= 2 and (std > AUC_STD_CUTOFF if np.isfinite(std) else False):
            # 高方差：取最大 auc_mean 的那一行
            argmax = int(np.nanargmax(aucs))
            row = sub.iloc[argmax].copy()
            row["n_repeats"] = n
            row["auc_mean_std"] = std
            rows_out.append(row)
        elif n >= 2:
            # 低/中方差：取均值，保留一行（基于第一行模板）
            mean_auc = float(np.nanmean(aucs))
            row = sub.iloc[0].copy()
            row["auc_mean"] = mean_auc
            row["n_repeats"] = n
            row["auc_mean_std"] = std
            rows_out.append(row)
        else:
            # 单样本：原样保留，但更新 n_repeats 与 auc_mean_std（=NaN）
            row = sub.iloc[0].copy()
            row["n_repeats"] = 1
            row["auc_mean_std"] = float("nan")
            rows_out.append(row)

    df_out = pd.DataFrame(rows_out)

    # 4) 丢掉 ROI 分位列（若存在）与中间列
    drop_cols = [c for c in ["roi_sd_q25", "roi_sd_q50", "roi_sd_q75", "_key"] if c in df_out.columns]
    df_out = df_out.drop(columns=drop_cols, errors="ignore")

    # 5) 调整列顺序：sheet, well, 25D, auc_mean, n_repeats, auc_mean_std
    cols_final = ["sheet", "well"] + KEY_COLS + ["auc_mean", "n_repeats", "auc_mean_std"]
    cols_final = [c for c in cols_final if c in df_out.columns] + [c for c in df_out.columns if c not in cols_final]
    df_out = df_out[cols_final]

    # —— 可选：打印一次聚合后的类别计数，便于快速核对 —— #
    add_mat = np.abs(df_out[ADDITIVE_COLS].to_numpy(dtype=float))
    nz_add = (add_mat > 0).sum(axis=1)
    cnt0 = int((nz_add == 0).sum())
    cnt1 = int((nz_add == 1).sum())
    cnt2 = int((nz_add == 2).sum())
    print(f"[CHECK] 聚合后唯一配方计数：总 {len(df_out)} = 0-add {cnt0} + 1-add {cnt1} + 2-add {cnt2}")
    # -------------------------------------------------- #

    # 6) 写出
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"✅ 写出 {out_path}，共 {len(df_out)} 行；去除 {before - after_filter} 行（roi 过滤）后完成分组收缩。")

if __name__ == "__main__":
    main()
