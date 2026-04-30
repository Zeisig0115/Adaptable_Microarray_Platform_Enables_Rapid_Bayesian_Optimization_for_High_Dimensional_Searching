#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spot_blueonly_pipeline_to_npy_refactored.py

变更：
  - 拆分为两个主要函数：
      1) get_time_axis(...)     → 负责读取/生成时间轴（含 MASTER 复用和外推/截断）
      2) blueonly_from_frames(...) → 负责标定与 BlueOnly 提取（含可视化）
  - 其它逻辑与原脚本保持一致。
"""

import os, glob, math, re
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PIL import Image, ExifTags
from datetime import datetime, timezone, timedelta

# ---------- 参数 ----------
EXP_NAME     = 'H2O2'                     # 当前要处理的实验名（决定输出文件名）
IMG_DIR      = './img/H2O2'               # 当前实验的图片目录
FILE_PATTERN = os.path.join(IMG_DIR, 'IMG_*.JPG')
SPOT_RADIUS  = 30
NUM_REFS     = 3
NUM_BG       = 1
NUM_VIS      = 9

# 基准时间轴设置
MASTER_TIME_EXP   = 'H2O2'                 # 只在这个实验上从图像计算时间，其它实验复用它
DATA_DIR          = './data'
MASTER_TIME_NPY   = os.path.join(DATA_DIR, f'{MASTER_TIME_EXP} t.npy')
USE_MASTER_TIME   = False                   # True：除 MASTER_TIME_EXP 外都复用时间轴
TAIL_FRAC_FOR_EXT = 0.20                   # 外推时用最后20%时间点估计步长
# ---------------------------

def numeric_key_from_filename(path: str) -> int:
    name = os.path.basename(path)
    m = re.findall(r'(\d+)', name)
    return int(m[-1]) if m else 0

def safe_crop(img, cx, cy, r):
    h, w = img.shape[:2]
    x1 = max(int(cx) - r, 0)
    y1 = max(int(cy) - r, 0)
    x2 = min(int(cx) + r + 1, w)
    y2 = min(int(cy) + r + 1, h)
    patch = img[y1:y2, x1:x2]
    rh = y2 - y1
    rw = x2 - x1
    yy, xx = np.ogrid[:rh, :rw]
    cy0 = min(r, rh - 1)
    cx0 = min(r, rw - 1)
    inds = np.where((xx - cx0)*(xx - cx0) + (yy - cy0)*(yy - cy0) <= r*r)
    return patch, inds

# ---------- EXIF 时间戳 ----------
def _parse_tz_offset_str(s: str):
    try:
        s = str(s).strip()
        sign = 1 if s[0] == '+' else -1
        hh, mm = s[1:].split(':')
        return timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
    except Exception:
        return None

def read_image_timestamp(path: str):
    try:
        im = Image.open(path)
        exif = im.getexif()
        if exif:
            tag_map = {ExifTags.TAGS.get(k, k): k for k in exif.keys()}
            dto_raw = exif.get(tag_map.get('DateTimeOriginal'))
            subsec  = exif.get(tag_map.get('SubsecTimeOriginal'))
            tz_off  = exif.get(tag_map.get('OffsetTimeOriginal'))
            if dto_raw:
                base = datetime.strptime(dto_raw, '%Y:%m:%d %H:%M:%S')
                if subsec:
                    subsec = str(subsec).strip()
                    us = int((subsec + '000000')[:6])
                    base = base.replace(microsecond=us)
                if tz_off:
                    tzinfo = _parse_tz_offset_str(tz_off)
                    if tzinfo is not None:
                        base = base.replace(tzinfo=tzinfo)
                return base, 'exif'
    except Exception:
        pass
    try:
        ts = os.path.getmtime(path)
        return datetime.fromtimestamp(ts), 'mtime'
    except Exception:
        return datetime.now(), 'now'

def to_epoch_seconds(dt: datetime) -> float:
    if dt.tzinfo is None:
        return dt.timestamp()
    return dt.astimezone(timezone.utc).timestamp()

def build_time_from_images(frames):
    timestamps = []
    sources = []
    for f in frames:
        dt, src = read_image_timestamp(f)
        timestamps.append(dt); sources.append(src)
    epoch_secs = np.array([to_epoch_seconds(d) for d in timestamps], dtype=np.float64)
    t = epoch_secs - epoch_secs[0]
    # 若有非增间隔，退化为等步长
    if np.any(np.diff(t) <= 0):
        print("⚠ 时间戳重复或倒序，改用等步长时间轴")
        step = (t[-1] - t[0]) / max(1, len(t)-1)
        if step <= 0: step = 1.0
        t = np.arange(len(t), dtype=np.float64) * step
    print(f"Timestamp base (t0): {timestamps[0].isoformat()}  [source={sources[0]}]")
    return t

def adapt_time_to_frames(master_t: np.ndarray, T: int, tail_frac: float) -> np.ndarray:
    """将 master_t 适配到长度 T：不够则按最后 tail_frac 平均步长外推，过长则截断。"""
    M = len(master_t)
    if M == T:
        return master_t.copy()
    if M > T:
        return master_t[:T].copy()
    # M < T，需要外推
    n_tail = max(1, int(np.ceil(tail_frac * M)))
    tail = master_t[-(n_tail+1):]  # 至少包含两点
    diffs = np.diff(tail)
    # 只用正的时间步长；全非正则退回全局正步长；仍无则设为1秒
    pos = diffs[diffs > 0]
    if pos.size == 0:
        diffs_all = np.diff(master_t)
        pos = diffs_all[diffs_all > 0]
    step = np.mean(pos) if pos.size else 1.0
    out = np.empty(T, dtype=np.float64)
    out[:M] = master_t
    for k in range(M, T):
        out[k] = out[k-1] + step
    return out

# =========================
#  函数 1：时间轴
# =========================
def get_time_axis(
    EXP_NAME: str,
    frames: list[str],
    DATA_DIR: str,
    MASTER_TIME_EXP: str,
    MASTER_TIME_NPY: str,
    USE_MASTER_TIME: bool,
    TAIL_FRAC_FOR_EXT: float
) -> np.ndarray:
    """根据策略返回当前实验的时间轴 t_seconds。必要时保存/复用 MASTER 时间轴。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if (not USE_MASTER_TIME) or (EXP_NAME == MASTER_TIME_EXP) or (not os.path.exists(MASTER_TIME_NPY)):
        # 自己算（作为基准或兜底）
        t_seconds = build_time_from_images(frames)
        # 如果这是基准实验，保存下来供其它实验复用
        if EXP_NAME == MASTER_TIME_EXP:
            np.save(MASTER_TIME_NPY, t_seconds)
            print(f"Saved master time axis: {MASTER_TIME_NPY} (len={len(t_seconds)})")
    else:
        # 复用基准时间轴，必要时外推/截断
        master_t = np.load(MASTER_TIME_NPY).astype(np.float64)
        t_seconds = adapt_time_to_frames(master_t, len(frames), TAIL_FRAC_FOR_EXT)
        print(f"Using master time from {MASTER_TIME_NPY} -> adapted len={len(t_seconds)}")
    return t_seconds

# =========================
#  函数 2：图像处理（标定 + BlueOnly）
# =========================
def blueonly_from_frames(
    frames: list[str],
    SPOT_RADIUS: int = 15,
    NUM_REFS: int = 2,
    NUM_BG: int = 2,
    NUM_VIS: int = 4
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], list[list[tuple[float,float]]]]:
    """
    从帧序列中进行交互式标定与 BlueOnly 提取。
    返回：
      - BlueOnly: (T,16,24)
      - gridXY:   每个参考帧的网格坐标 [(Xk,Yk), ...]
      - bgXY_list:每个参考帧的背景采样点 [[(x,y),...], ...]
    """
    T = len(frames)
    # ---------- 参考帧标定 ----------
    ref_idx = np.linspace(0, T - 1, NUM_REFS, dtype=int).tolist()
    gridXY, bgXY_list = [], []

    for ridx in ref_idx:
        img = cv2.imread(frames[ridx])
        if img is None:
            raise RuntimeError(f'Failed to read image: {frames[ridx]}')
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f'Ref frame {ridx} / file: {os.path.basename(frames[ridx])}\n'
                  f'Click 4 corners (TL, TR, BR, BL) + {NUM_BG} bg pts')
        pts = plt.ginput(4 + NUM_BG, timeout=0)
        plt.close()
        if len(pts) != 4 + NUM_BG:
            raise RuntimeError('Click count mismatch. Need 4 corners + NUM_BG background points.')
        corners = np.array(pts[:4], np.float32)
        bgXY_list.append([tuple(p) for p in pts[4:]])
        dst = np.array([[0, 0], [23, 0], [23, 15], [0, 15]], np.float32)
        H   = cv2.getPerspectiveTransform(dst, corners)
        U, V = np.meshgrid(np.arange(24), np.arange(16))
        uv   = np.stack([U.ravel(), V.ravel()], -1).astype(np.float32)
        mp   = cv2.perspectiveTransform(uv.reshape(-1, 1, 2), H).reshape(-1, 2)
        Xk, Yk = mp[:, 0].reshape(16, 24), mp[:, 1].reshape(16, 24)
        gridXY.append((Xk, Yk))

    print('Calibration done.')

    # ---------- 可视化若干帧 ----------
    vis_idx = np.linspace(0, T - 1, min(NUM_VIS, T), dtype=int)
    cols = int(math.ceil(math.sqrt(len(vis_idx))))
    rows = int(math.ceil(len(vis_idx) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axs = np.atleast_1d(axs).flatten()
    for ax, tidx in zip(axs, vis_idx):
        img = cv2.imread(frames[tidx])
        if img is None:
            ax.set_title(f'Failed to read frame {tidx}'); ax.axis('off'); continue
        ref_id = int(np.argmin(np.abs(np.array(ref_idx) - tidx)))
        Xk, Yk = gridXY[ref_id]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f'Frame {tidx+1} → Ref #{ref_id}')
        ax.axis('off')
        for i in range(16):
            for j in range(24):
                circ = plt.Circle((Xk[i, j], Yk[i, j]), SPOT_RADIUS,
                                  edgecolor='r', facecolor='none', linewidth=0.7)
                ax.add_patch(circ)
    for ax in axs[len(vis_idx):]:
        ax.axis('off')
    plt.tight_layout(); plt.show()

    # ---------- 提取 BlueOnly ----------
    BlueOnly = np.zeros((T, 16, 24), np.float32)

    for t_idx, fname in enumerate(frames):
        img = cv2.imread(fname)
        if img is None:
            raise RuntimeError(f'Failed to read image: {fname}')
        ref_id = int(np.argmin(np.abs(np.array(ref_idx) - t_idx)))
        Xk, Yk = gridXY[ref_id]
        bg_pts = bgXY_list[ref_id]

        blue_base_vals = []
        for cx, cy in bg_pts:
            patch, inds = safe_crop(img, cx, cy, SPOT_RADIUS)
            lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
            b = np.mean(lab[..., 2][inds]) - 128.0
            blue_base_vals.append(-b)
        b_bg = float(np.mean(blue_base_vals)) if blue_base_vals else 0.0

        for i in range(16):
            for j in range(24):
                cx, cy = Xk[i, j], Yk[i, j]
                patch, inds = safe_crop(img, cx, cy, SPOT_RADIUS)
                lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
                b = np.mean(lab[..., 2][inds]) - 128.0
                blue = (-b) - b_bg
                BlueOnly[t_idx, i, j] = max(0.0, float(blue))

        if (t_idx + 1) % 10 == 0 or t_idx == T - 1:
            print(f'Processed {t_idx + 1}/{T} frames')

    print('BlueOnly extraction finished.')
    return BlueOnly, gridXY, bgXY_list

# ========== 主流程 ==========
if __name__ == '__main__':
    # ---------- 收集帧 ----------
    frames = sorted(glob.glob(FILE_PATTERN), key=numeric_key_from_filename)
    if not frames:
        raise RuntimeError(f'No images found by pattern: {FILE_PATTERN}')
    T = len(frames)
    print(f'Found {T} frames for {EXP_NAME}')

    # # ---------- 构建时间轴 ----------
    # os.makedirs(DATA_DIR, exist_ok=True)
    # t_seconds = get_time_axis(
    #     EXP_NAME=EXP_NAME,
    #     frames=frames,
    #     DATA_DIR=DATA_DIR,
    #     MASTER_TIME_EXP=MASTER_TIME_EXP,
    #     MASTER_TIME_NPY=MASTER_TIME_NPY,
    #     USE_MASTER_TIME=USE_MASTER_TIME,
    #     TAIL_FRAC_FOR_EXT=TAIL_FRAC_FOR_EXT
    # )
    # plt.plot(t_seconds)
    # plt.show()

    # ---------- 图像处理 ----------
    BlueOnly, gridXY, bgXY_list = blueonly_from_frames(
        frames=frames,
        SPOT_RADIUS=SPOT_RADIUS,
        NUM_REFS=NUM_REFS,
        NUM_BG=NUM_BG,
        NUM_VIS=NUM_VIS
    )

    # ---------- 保存 ----------
    b_path = os.path.join(DATA_DIR, f"{EXP_NAME} b.npy")
    t_path = os.path.join(DATA_DIR, f"{EXP_NAME} t.npy")
    np.save(b_path, BlueOnly)
    # np.save(t_path, t_seconds)
    print(f"Saved {b_path} (shape {BlueOnly.shape})")
    # print(f"Saved {t_path} (shape {t_seconds.shape})")
