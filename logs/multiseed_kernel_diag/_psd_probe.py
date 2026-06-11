# -*- coding: utf-8 -*-
"""PSD stress-test for a candidate 'small' BaselineKernel edit.

Current BaselineKernel (add_bo.py) concentration block sums an RBF distance over
ALL additive conc dims, including the one-active-one-inactive case (inactive conc
is pinned at CONC_DEFAULT = LOG_LO). The tempting 'small fix' is to gate the conc
distance by shared activity (b_i * b_j), the way AdditiveSetKernel does. This
probe checks whether that gated *product* form stays positive semi-definite.

It does NOT modify add_bo.py; it re-implements the two k_cont variants on the
real May_09 encoded design and on a 3-point toy, and reports min eigenvalues.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import add_bo


def build_blocks(X, codec, l_ess=0.35, l_conc=0.30, theta=0.5, outputscale=1.0):
    ess = X[:, :codec.n_ess]
    c = X[:, codec.conc_dims]
    b = (X[:, codec.cat_dims] > 0.5).astype(float)

    # essentials RBF exponent (shared by both variants)
    de = (ess[:, None, :] - ess[None, :, :]) / l_ess
    ess_quad = (de ** 2).sum(-1)

    # concentration RBF exponent, per-additive lengthscale (scalar here)
    dc = (c[:, None, :] - c[None, :, :]) / l_conc            # (n, n, A)
    conc_quad_ungated = (dc ** 2).sum(-1)                    # current kernel
    shared = b[:, None, :] * b[None, :, :]                   # (n, n, A)
    conc_quad_gated = (shared * dc ** 2).sum(-1)             # 'fix'

    hamming = (b[:, None, :] != b[None, :, :]).sum(-1).astype(float)
    k_cat = np.exp(-theta * hamming)

    k_cont_ungated = np.exp(-0.5 * (ess_quad + conc_quad_ungated))
    k_cont_gated = np.exp(-0.5 * (ess_quad + conc_quad_gated))

    K_ungated = outputscale * k_cont_ungated * k_cat
    K_gated = outputscale * k_cont_gated * k_cat
    # also the bare gated concentration factor alone
    k_conc_factor_gated = np.exp(-0.5 * conc_quad_gated)
    return K_ungated, K_gated, k_conc_factor_gated


def min_eig(M):
    return float(np.linalg.eigvalsh(0.5 * (M + M.T)).min())


def main():
    torch.set_default_dtype(torch.double)
    args = add_bo.parse_args([
        "--input", "logs/May_09_full_log/data_corrected.xlsx",
    ])
    df, codec = add_bo.load_training_data(args)
    X, Y, meta = add_bo.encode_training_data(df, codec, args.target, torch.device("cpu"))
    Xnp = X.detach().cpu().numpy()
    print("encoded design:", meta)

    rng = np.random.default_rng(0)
    for n in (40, 80, 150, Xnp.shape[0]):
        idx = rng.choice(Xnp.shape[0], size=min(n, Xnp.shape[0]), replace=False)
        Xs = Xnp[idx]
        Ku, Kg, _ = build_blocks(Xs, codec)
        print("n=%-4d  current(ungated) min_eig=%+.3e   gated 'fix' min_eig=%+.3e   %s"
              % (len(idx), min_eig(Ku), min_eig(Kg),
                 "<-- NON-PSD" if min_eig(Kg) < -1e-10 else ""))

    # 3-point analytic toy: two co-active points far apart in conc + one inactive
    print("\n3-point toy on the bare gated conc factor (single additive):")
    l = 0.30
    for dc in (0.2, 0.5, 1.0, 2.0):
        # A=(active, c=0), B=(active, c=dc), C=(inactive, c=0)
        b = np.array([1.0, 1.0, 0.0])
        c = np.array([0.0, dc, 0.0])
        shared = b[:, None] * b[None, :]
        d = (c[:, None] - c[None, :]) / l
        Kf = np.exp(-0.5 * shared * d ** 2)
        print("  conc gap dc=%.1f  Gram=%s  min_eig=%+.4f%s"
              % (dc, np.round(Kf, 3).tolist(), min_eig(Kf),
                 "  <-- NON-PSD" if min_eig(Kf) < -1e-12 else ""))


if __name__ == "__main__":
    main()
