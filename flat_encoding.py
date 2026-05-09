# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import math
from itertools import combinations
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch

from fit_model import fit_gp

from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf_mixed

torch.set_default_dtype(torch.double)

FEATURE_WHITELIST = [
    "tmb", "h2o2", "cmc", "peg20k", "dmso", "pl127", "bsa",
    "pva", "tw80", "glycerol", "tw20", "imidazole",
    "tx100", "edta", "mgso4", "sucrose", "cacl2",
    "znso4", "paa", "mncl2", "peg200k", "feso4",
    "peg6k", "peg400"
]

ESSENTIALS = ["tmb", "h2o2"]


DEFAULT_Q = 32
DEFAULT_OUT_CSV = "bo_candidates_flat_gp.csv"
EPS = 1e-7

CONC_LO = 0.005
CONC_HI = 1.0
LOG_LO = math.log10(CONC_LO)   # -3.0
LOG_HI = math.log10(CONC_HI)   #  0.0
CONC_DEFAULT = LOG_LO


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_2d_y(y: np.ndarray | List[float]) -> np.ndarray:
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


class FlatCodec:

    def __init__(
        self,
        essentials: List[str],
        additives: List[str],
        k_max: int,
    ):
        self.E = list(essentials)
        self.adds = list(additives)
        self.A = len(self.adds)
        self.k_max = k_max
        self.n_ess = len(self.E)
        self.d = self.n_ess + 2 * self.A

        # Convenience maps: additive name → (binary_col, conc_col)
        self.add_cols: Dict[str, Tuple[int, int]] = {}
        for i, name in enumerate(self.adds):
            bin_col = self.n_ess + 2 * i
            conc_col = self.n_ess + 2 * i + 1
            self.add_cols[name] = (bin_col, conc_col)

        # Categorical dimension indices (the binary indicator columns)
        self.cat_dims: List[int] = [
            self.n_ess + 2 * i for i in range(self.A)
        ]

    # ---- encode ----
    def encode_row(self, row: Dict[str, float]) -> np.ndarray:
        z = np.zeros(self.d, dtype=np.float64)
        # essentials: always present, log10 concentration
        for i, e in enumerate(self.E):
            v = row.get(e, 0.0)
            z[i] = np.log10(np.clip(v, CONC_LO, CONC_HI)) if v > EPS else LOG_LO
        # additives: binary indicator + log10 concentration
        for name in self.adds:
            bin_col, conc_col = self.add_cols[name]
            v = row.get(name, 0.0)
            if v is not None and v > EPS:
                z[bin_col] = 1.0
                z[conc_col] = np.log10(np.clip(v, CONC_LO, CONC_HI))
            else:
                z[bin_col] = 0.0
                z[conc_col] = CONC_DEFAULT
        return z

    def encode(self, rows: List[Dict[str, float]]) -> np.ndarray:
        return np.array([self.encode_row(r) for r in rows])

    # ---- decode: encoded vector → human-readable dict ----
    def decode(self, Z: np.ndarray) -> List[Dict[str, float]]:
        Z = np.atleast_2d(Z)
        decoded: List[Dict[str, float]] = []
        for z in Z:
            row: Dict[str, float] = {}
            for i, e in enumerate(self.E):
                row[e] = float(10.0 ** np.clip(z[i], LOG_LO, LOG_HI))
            for name in self.adds:
                bin_col, conc_col = self.add_cols[name]
                if z[bin_col] > 0.5:  # binary indicator == 1
                    row[name] = float(10.0 ** np.clip(z[conc_col], LOG_LO, LOG_HI))
            decoded.append(row)
        return decoded

    # ---- postprocess: clamp to bounds, snap binary dims ----
    def postprocess_batch(self, Zt: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        Z = Zt.detach().cpu().clone()
        Z = Z.clamp(min=bounds[0].cpu(), max=bounds[1].cpu())
        # Snap binary indicator dims to nearest integer
        for bc in self.cat_dims:
            Z[..., bc] = Z[..., bc].round()
        # Force concentration = CONC_DEFAULT when indicator == 0
        for name in self.adds:
            bin_col, conc_col = self.add_cols[name]
            inactive = Z[..., bin_col] < 0.5
            Z[..., conc_col][inactive] = CONC_DEFAULT
        return Z.to(Zt.device, dtype=Zt.dtype)

    # ---- physical bounds ----
    def get_bounds(self, device: torch.device) -> torch.Tensor:
        lb = np.zeros(self.d, dtype=np.float64)
        ub = np.zeros(self.d, dtype=np.float64)
        # essentials: continuous log10
        lb[:self.n_ess] = LOG_LO
        ub[:self.n_ess] = LOG_HI
        # additives: binary + continuous
        for i in range(self.A):
            bin_col = self.n_ess + 2 * i
            conc_col = self.n_ess + 2 * i + 1
            lb[bin_col] = 0.0;  ub[bin_col] = 1.0
            lb[conc_col] = LOG_LO; ub[conc_col] = LOG_HI
        return torch.tensor(np.stack([lb, ub]), device=device, dtype=torch.double)

    # ---- fixed_features_list for optimize_acqf_mixed ----
    def build_fixed_features_list(self, k_max: int | None = None) -> List[Dict[int, float]]:
        k = k_max or self.k_max
        all_add_indices = list(range(self.A))
        ff_list: List[Dict[int, float]] = []

        for n_active in range(0, k + 1):
        # for n_active in range(1 if not self.E else 0, k + 1):
            for combo in combinations(all_add_indices, n_active):
                active_set = set(combo)
                fixed: Dict[int, float] = {}
                for j in range(self.A):
                    bin_col = self.n_ess + 2 * j
                    conc_col = self.n_ess + 2 * j + 1
                    if j in active_set:
                        # Active: fix binary = 1, let optimizer handle concentration
                        fixed[bin_col] = 1.0
                    else:
                        # Inactive: fix binary = 0 AND concentration = default
                        fixed[bin_col] = 0.0
                        fixed[conc_col] = CONC_DEFAULT
                ff_list.append(fixed)
        return ff_list


# ---------------------------------------------------------------------------
# FlatBO
# ---------------------------------------------------------------------------
class FlatBO:
    def __init__(
        self,
        df: pd.DataFrame,
        essentials: List[str],
        additives: List[str],
        k_max: int,
        target_col: str,
        device: str = "auto",
        seed: int = 42,
    ):
        self.E = list(essentials)
        self.adds = list(additives)
        self.k_max = int(k_max)
        self.target_col = target_col
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        set_seeds(self.seed)

        rows = df.to_dict("records")
        self.codec = FlatCodec(self.E, self.adds, self.k_max)
        Z_np = self.codec.encode(rows)
        y_np = _ensure_2d_y(df[self.target_col].values)

        unique_Z, _, inverse = np.unique(
            Z_np, axis=0, return_index=True, return_inverse=True
        )
        if len(unique_Z) < len(Z_np):
            print(f"[Warning] {len(Z_np) - len(unique_Z)} duplicates merged (mean).")
            unique_y = np.array([
                np.mean(y_np[inverse == i]) for i in range(len(unique_Z))
            ])
            Z_np = unique_Z
            y_np = _ensure_2d_y(unique_y)

        self.Z = torch.tensor(Z_np, dtype=torch.double, device=self.device)
        self.y = torch.tensor(y_np, dtype=torch.double, device=self.device)

        self.bounds = self.codec.get_bounds(self.device)

        n_cont = self.codec.n_ess + self.codec.A  # essential concs + additive concs
        n_cat = self.codec.A                        # binary indicators
        print(f"[Info] Hybrid encoding: d={self.codec.d} "
              f"({n_cont} continuous + {n_cat} categorical)")
        print(f"[Info] cat_dims = {self.codec.cat_dims}")
        print(f"[Info] Training: {self.Z.shape[0]} points, k_max={self.k_max}")
        print("Fitting MixedSingleTaskGP (Hamming kernel for indicators)...")
        self.model = fit_gp(
            self.Z, self.y,
            seed=self.seed,
            cat_dims=self.codec.cat_dims,
            bounds=self.bounds,
        )
        print("GP fitted.")

        self._print_kernel_diagnostics()
        self._print_loo_diagnostics()

    def _kernel_active_dims_str(self, kernel: Any) -> str | None:
        active_dims = getattr(kernel, "active_dims", None)
        if active_dims is None:
            return None
        if torch.is_tensor(active_dims):
            active_dims = active_dims.detach().cpu().tolist()
        else:
            active_dims = list(active_dims)
        return str(active_dims)

    def _print_kernel_tree(self, kernel: Any, indent: str = "    ") -> None:
        name = type(kernel).__name__
        parts: List[str] = []

        active_dims = self._kernel_active_dims_str(kernel)
        if active_dims is not None:
            parts.append(f"active_dims={active_dims}")

        if hasattr(kernel, "lengthscale"):
            try:
                ls = kernel.lengthscale.detach().cpu().reshape(-1).tolist()
                parts.append(
                    "lengthscale=[" + ", ".join(f"{v:.3g}" for v in ls) + "]"
                )
            except Exception:
                pass

        if hasattr(kernel, "outputscale"):
            try:
                os = kernel.outputscale.detach().cpu().item()
                parts.append(f"outputscale={os:.3g}")
            except Exception:
                pass

        suffix = f" ({'; '.join(parts)})" if parts else ""
        print(f"{indent}- {name}{suffix}")

        if hasattr(kernel, "base_kernel"):
            self._print_kernel_tree(kernel.base_kernel, indent + "  ")
        if hasattr(kernel, "kernels"):
            for child in kernel.kernels:
                self._print_kernel_tree(child, indent + "  ")

    def _print_kernel_diagnostics(self) -> None:
        """Print conditioning, noise, residuals, and mixed-kernel hyperparameters."""
        self.model.eval()
        with torch.no_grad():
            if getattr(self.model, "input_transform", None) is not None:
                Z_model = self.model.input_transform(self.Z)
            else:
                Z_model = self.Z

            K = self.model.covar_module(Z_model).evaluate()
            noise = self.model.likelihood.noise.item()
            eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
            K_full = K + noise * eye
            eigvals = torch.linalg.eigvalsh(K_full)

            print("\n[Kernel diagnostics]")
            print(f"  Minimum eigenvalue: {eigvals.min().item():.2e}")
            print(f"  Maximum eigenvalue: {eigvals.max().item():.2e}")
            print(
                f"  Condition number:   "
                f"{(eigvals.max() / eigvals.min().clamp_min(1e-12)).abs().item():.2e}"
            )

            y_std = self.y.std(unbiased=True).item() if self.y.shape[0] > 1 else 1.0
            print(f"  noise (standardized):   {noise:.3e}")
            print(f"  noise std (orig y):     {np.sqrt(noise) * y_std:.3g}")

            posterior = self.model.posterior(self.Z, observation_noise=False)
            mean_y = posterior.mean.squeeze(-1)
            residual_y = (self.y.squeeze(-1) - mean_y).abs()
            print(f"  Avg |residual| (orig y): {residual_y.mean().item():.3g}")
            print(f"  Max |residual| (orig y): {residual_y.max().item():.3g}")

            y_np = self.y.squeeze(-1).detach().cpu().numpy()
            res_np = residual_y.detach().cpu().numpy()
            order = np.argsort(y_np)
            half = len(y_np) // 2
            if half > 0:
                low_mean = res_np[order[:half]].mean()
                high_mean = res_np[order[half:]].mean()
                print(f"  Low-y half (n={half}) residual mean:  {low_mean:.3g}")
                print(
                    f"  High-y half (n={len(y_np) - half}) residual mean: {high_mean:.3g}"
                )
                print(
                    f"  High/low residual ratio:              "
                    f"{high_mean / max(low_mean, 1e-9):.2f}"
                    "  (>2 may indicate heteroscedasticity)"
                )

            if self.codec.A > 0:
                active_counts = self.Z[:, self.codec.cat_dims].sum(dim=-1)
                print(
                    f"  Active additives per row: min={active_counts.min().item():.0f}, "
                    f"mean={active_counts.mean().item():.2f}, "
                    f"max={active_counts.max().item():.0f}"
                )

        print("  Kernel structure:")
        self._print_kernel_tree(self.model.covar_module)

    def _print_loo_diagnostics(self) -> None:
        """Fast LOO diagnostics with fixed hyperparameters (no N refits).

        For exact GPs with Gaussian likelihood, the leave-one-out predictive mean
        and variance for all training points can be computed from a single fit via
        the full noisy covariance inverse, rather than refitting N separate models.
        This is a *virtual / fixed-hyperparameter* LOO diagnostic: it checks the
        calibration of the currently fitted model, but is not identical to retraining
        hyperparameters after removing each point.
        """
        n, _ = self.Z.shape
        if n < 2:
            print("\n[LOO diagnostics] skipped: need at least 2 training points.")
            return

        self.model.eval()
        with torch.no_grad():
            if getattr(self.model, "input_transform", None) is not None:
                Z_model = self.model.input_transform(self.Z)
            else:
                Z_model = self.Z

            mean = self.model.mean_module(Z_model).reshape(-1)
            K = self.model.covar_module(Z_model).evaluate()
            noise = self.model.likelihood.noise.to(K).reshape(-1)[0]
            eye = torch.eye(n, device=K.device, dtype=K.dtype)
            C = K + noise * eye

            jitter = 1e-8
            try:
                L = torch.linalg.cholesky(C)
            except RuntimeError:
                L = torch.linalg.cholesky(C + jitter * eye)

            y_model = self.model.train_targets.reshape(-1).to(K)
            alpha = torch.cholesky_solve((y_model - mean).unsqueeze(-1), L).squeeze(-1)
            C_inv_diag = torch.cholesky_inverse(L).diagonal().clamp_min(1e-18)

            loo_mean_model = y_model - alpha / C_inv_diag
            loo_var_model = (1.0 / C_inv_diag).clamp_min(1e-18)
            z_scores = ((y_model - loo_mean_model) / loo_var_model.sqrt()).detach().cpu().numpy()

        print(
            f"\n[LOO diagnostics] n={n}, fast exact-GP LOO with fixed hyperparameters "
            f"(single fit, no refits)"
        )
        print(f"  LOO z-score mean:  {z_scores.mean():+.2f}  (target ≈ 0)")
        print(f"  LOO z-score std:   {z_scores.std():.2f}   (target ≈ 1; >1 = over-confident)")
        print(f"  Fraction |z| > 2:  {(np.abs(z_scores) > 2).mean():.1%}  (target ≈ 5%)")
        print(f"  Fraction |z| > 3:  {(np.abs(z_scores) > 3).mean():.1%}  (target ≈ 0.3%)")

        flagged = np.where(np.abs(z_scores) > 2)[0]
        if len(flagged) == 0:
            print("  No holdout points with |z| > 2.")
            return

        print("  Holdout points with |z| > 2:")
        for i in flagged:
            z = z_scores[i]
            decoded = self.codec.decode(self.Z[i].detach().cpu().numpy())[0]
            feat_str = ", ".join(
                f"{k}={v:.3g}" for k, v in sorted(decoded.items())
            )
            print(f"    idx={i:3d}  z={z:+.2f}  y={self.y[i].item():.3g}  {feat_str}")

    # ---- Acquisition Function ----
    def _make_acqf(self, acq_type: str, mc_samples: int = 256, **opts: Any):
        t = acq_type.lower()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        best_f = self.y.max().item()

        if t in ("qlognei", "lognei"):
            return qLogNoisyExpectedImprovement(
                self.model, X_baseline=self.Z, sampler=sampler, cache_root=False, prune_baseline=True
            )
        if t in ("qei", "ei"):
            return qLogExpectedImprovement(
                self.model, best_f=best_f, sampler=sampler,
            )
        if t in ("ucb", "qucb"):
            return qUpperConfidenceBound(
                self.model, beta=opts.get("beta", 0.2), sampler=sampler,
            )
        if t in ("kg", "qkg", "knowledge_gradient"):
            return qKnowledgeGradient(
                self.model, num_fantasies=opts.get("kg_num_fantasies", 64),
            )
        raise ValueError(f"Unknown acq fn: '{acq_type}'")

    def ask(
        self,
        q: int = 8,
        num_restarts: int = 10,
        raw_samples: int = 256,
        acq_types: List[str] | None = None,
        acq_options: Dict | None = None,
        num_top_subspaces: int = 40,
        sobol_max_samples: int = 512,
        eval_batch_size: int = 256,
    ) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:

        set_seeds(self.seed)
        acq_types = acq_types or ["qlognei"]
        acq_options = acq_options or {}
        q_per_acq = max(1, math.ceil(q / len(acq_types)))

        all_fixed_features_list = self.codec.build_fixed_features_list()
        print(f"\n[Info] Total discrete subspaces: {len(all_fixed_features_list)}")
        print(f"[Info] Target candidates per acq fn: {q_per_acq}")

        all_candidates_tensors: List[torch.Tensor] = []

        lb = self.bounds[0]
        ub = self.bounds[1]
        d = lb.numel()

        for acq_name in acq_types:
            acqf_screen = self._make_acqf(acq_name, mc_samples=32, **acq_options)
            acqf_refine = self._make_acqf(acq_name, mc_samples=64, **acq_options)

            print(f"\n=== Acq: {acq_name}, generating {q_per_acq} candidates ===")

            max_n_free = max(
                d - len(fixed) for fixed in all_fixed_features_list
            )
            sobol_draw_size = min(sobol_max_samples, 2 ** max_n_free)
            sobol_engine = torch.quasirandom.SobolEngine(
                dimension=d, scramble=True, seed=self.seed,
            )
            sobol_base = sobol_engine.draw(sobol_draw_size).to(
                device=self.device, dtype=torch.double,
            )  # (sobol_draw_size, d) in [0, 1)

            best_val_per_space = torch.empty(
                len(all_fixed_features_list),
                device=self.device, dtype=torch.double,
            )
            sample_stats: Dict[int, int] = defaultdict(int)

            for si, fixed in enumerate(all_fixed_features_list):
                n_free = d - len(fixed)
                num_samples = min(sobol_max_samples, 2 ** n_free)
                sample_stats[n_free] = (
                    sample_stats.get(n_free, 0) + num_samples
                )

                # Scale Sobol samples to bounds and fix categorical dims
                X = lb + (ub - lb) * sobol_base[:num_samples].clone()
                for col_idx, val in fixed.items():
                    X[:, col_idx] = val

                cur_best_val = -float("inf")
                cur_best_x = None
                with torch.no_grad():
                    for i in range(0, num_samples, eval_batch_size):
                        Xb = X[i: i + eval_batch_size]
                        vb = acqf_screen(Xb.unsqueeze(1)).view(-1)
                        mv, mi = torch.max(vb, dim=0)
                        if mv.item() > cur_best_val:
                            cur_best_val = mv.item()
                            cur_best_x = Xb[mi].clone()

                best_val_per_space[si] = cur_best_val

            total_samples = sum(sample_stats.values())
            print(
                f"  [Prescreen] Sobol | "
                f"Total samples: {total_samples} | "
                f"Breakdown (n_free: samples): "
                f"{dict(sorted(sample_stats.items()))}"
            )

            k_top = min(num_top_subspaces, len(all_fixed_features_list))
            top_vals, top_idx = torch.topk(best_val_per_space, k=k_top)
            top_fixed = [
                all_fixed_features_list[i] for i in top_idx.tolist()
            ]
            print(
                f"  [Prescreen] Top-{k_top} scores: "
                f"best={top_vals[0].item():.4f}, "
                f"worst={top_vals[-1].item():.4f}"
            )

            print(
                f"  [Optimize] Running optimize_acqf_mixed "
                f"(q={q_per_acq}, top-{k_top} subspaces, "
                f"{num_restarts} restarts)..."
            )
            t0 = time.time()
            candidates_raw, acq_value = optimize_acqf_mixed(
                acq_function=acqf_refine,
                bounds=self.bounds,
                q=q_per_acq,
                fixed_features_list=top_fixed,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                options={
                    "batch_limit": num_restarts,
                    "maxiter": 100,
                },
                retry_on_optimization_warning=True,
            )
            elapsed = time.time() - t0
            print(
                f"  [Optimize] Done in {elapsed:.1f}s, "
                f"joint acq value: {acq_value.item():.4f}"
            )

            # Postprocess: clamp bounds, snap binary dims
            candidates = self.codec.postprocess_batch(
                candidates_raw, self.bounds,
            )
            all_candidates_tensors.append(candidates)

        if not all_candidates_tensors:
            print("[Warning] No candidates generated.")
            return [], np.array([]), np.array([])

        print(f"\n[Info] Decoding and finalizing results...")
        all_candidates = torch.cat(all_candidates_tensors, dim=0)
        all_rows = self.codec.decode(all_candidates.detach().cpu().numpy())

        seen_keys, unique_rows, unique_tensors = set(), [], []
        for r, t in zip(all_rows, all_candidates):
            z = self.codec.encode_row(r)
            key = tuple(np.round(z, 6))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_rows.append(r)
                unique_tensors.append(t)

        if not unique_rows:
            return [], np.array([]), np.array([])

        unique_cand = torch.stack(unique_tensors)

        final_acqf = self._make_acqf(acq_types[-1], **acq_options)
        with torch.no_grad():
            acq_vals = final_acqf(unique_cand.unsqueeze(1)).view(-1)
        acq_vals = acq_vals.detach().cpu().numpy()

        self.model.eval()
        with torch.no_grad():
            post = self.model.posterior(unique_cand)
            pred_means = post.mean.squeeze(-1).detach().cpu().numpy()
            pred_vars = post.variance.squeeze(-1).detach().cpu().numpy()

        combined = sorted(
            [
                {
                    "row": unique_rows[i],
                    "acq": float(acq_vals[i]),
                    "mean": float(pred_means[i]),
                    "sigma": float(np.sqrt(pred_vars[i])),
                }
                for i in range(len(unique_rows))
            ],
            key=lambda x: x["acq"],
            reverse=True,
        )

        final_rows = [c["row"] for c in combined[:q]]
        final_means = np.array([c["mean"] for c in combined[:q]])
        final_sigmas = np.array([c["sigma"] for c in combined[:q]])

        print(f"[Info] Successfully generated {len(final_rows)} candidates.\n")
        return final_rows, final_means, final_sigmas


def _setup_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BO with hybrid binary+log10 encoding and MixedGP"
    )
    p.add_argument("--input", type=str, default="data_corrected.xlsx")
    p.add_argument("--target", type=str, default="intensity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--q", type=int, default=DEFAULT_Q)
    p.add_argument("--out", type=str, default=DEFAULT_OUT_CSV)
    p.add_argument("--acq", type=str, default="qlognei")
    p.add_argument("--acq_opts", type=str, default="{}")
    return p


def _prepare_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    path = Path(file_path)
    if not path.exists():
        for alt in [Path("./data.xlsx"), Path("./data.csv")]:
            if alt.exists():
                print(f"[info] {path} not found, using {alt}")
                path = alt
                break
        else:
            raise FileNotFoundError(f"Not found: {path.resolve()}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported: '{path.suffix}'")

    df.columns = df.columns.str.lower()

    if "ctrl" in df.columns:
        n_before = len(df)
        ctrl_num = pd.to_numeric(df["ctrl"], errors="coerce").fillna(0)
        df = df.loc[ctrl_num == 0].copy()
        df = df.drop(columns=["ctrl"])
        print(f"[Info] CTRL filtered: kept {len(df)}/{n_before} rows (ctrl == 0)")
    else:
        print("[Warning] 'ctrl' column not found; no CTRL filtering applied.")

    present = [c for c in FEATURE_WHITELIST if c in df.columns]
    assert all(e in present for e in ESSENTIALS)
    additives = [c for c in present if c not in ESSENTIALS]

    return df, ESSENTIALS, additives


def _save_results(
    rows: List[Dict[str, float]],
    means: np.ndarray,
    sigmas: np.ndarray,
    essentials: List[str],
    out_path: str,
):
    if not rows:
        print("[warning] no candidate generated")
        return
    out_df = pd.DataFrame(rows).fillna(0.0)
    out_df["pred_mean"] = means
    out_df["pred_sigma"] = sigmas
    add_counts = Counter(
        k for r in rows for k in r
        if k not in essentials and r.get(k, 0.0) > EPS
    )
    sorted_adds = [
        k for k, _ in sorted(add_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    all_cols = essentials + sorted_adds + ["pred_mean", "pred_sigma"]
    for col in all_cols:
        if col not in out_df:
            out_df[col] = 0.0
    out_df = out_df[all_cols]
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[success] {len(rows)} candidates -> {Path(out_path).resolve()}")
    if add_counts:
        print(
            "[info] additive usage: ",
            ", ".join(f"{k}:{v}" for k, v in add_counts.most_common()),
        )


def main():
    t0 = time.time()
    args = _setup_arg_parser().parse_args()
    set_seeds(args.seed)

    df, essentials, additives = _prepare_data(args.input)

    bo = FlatBO(
        df=df,
        essentials=essentials,
        additives=additives,
        k_max=args.k_max,
        target_col=args.target,
        device=args.device,
        seed=args.seed,
    )

    acq_types = [s.strip() for s in args.acq.split(",") if s.strip()]
    try:
        acq_opts = json.loads(args.acq_opts)
        assert isinstance(acq_opts, dict)
    except (json.JSONDecodeError, AssertionError):
        raise ValueError("invalid --acq_opts")

    print(
        f"\nUsing {acq_types} (opts: {acq_opts}), "
        f"generating {args.q} candidates..."
    )
    rows, means, sigmas = bo.ask(
        q=args.q, acq_types=acq_types, acq_options=acq_opts,
    )
    _save_results(rows, means, sigmas, essentials, args.out)

    print(f"\n[Settings] device={bo.device}, hybrid encoding, k_max={args.k_max}")
    if torch.cuda.is_available() and bo.device.type == "cuda":
        torch.cuda.synchronize()
    print(f"[Timer] Total: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
