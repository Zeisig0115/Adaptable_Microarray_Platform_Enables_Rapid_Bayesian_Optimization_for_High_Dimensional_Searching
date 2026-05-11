# -*- coding: utf-8 -*-
"""Compare the current mixed GP prior with an additive-set GP prior.

This script is intentionally independent from ``flat_encoding.py`` so that the
existing BO workflow can stay untouched while we test whether a prior that more
closely matches the additive-set belief improves diagnostics and suggestions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf_mixed
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.double)

FEATURE_WHITELIST = [
    "tmb", "h2o2", "cmc", "peg20k", "dmso", "pl127", "bsa",
    "pva", "tw80", "glycerol", "tw20", "imidazole",
    "tx100", "edta", "mgso4", "sucrose", "cacl2",
    "znso4", "paa", "mncl2", "peg200k", "feso4",
    "peg6k", "peg400",
]

ESSENTIALS = ["tmb", "h2o2"]
EPS = 1e-7
CONC_LO = 0.001
CONC_HI = 2.0
LOG_LO = math.log10(CONC_LO)
LOG_HI = math.log10(CONC_HI)
CONC_DEFAULT = LOG_LO
LOGS_DIR = Path(__file__).with_name("logs")
PROFILE_DIR = Path(__file__).with_name("bo_profiles")


def load_bo_profiles(profile_dir: Path = PROFILE_DIR) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    if profile_dir.exists():
        for path in sorted(profile_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError(f"Profile must be a JSON object: {path}")
            args = payload.get("args", {})
            if not isinstance(args, dict):
                raise ValueError(f"Profile 'args' must be a JSON object: {path}")
            profiles[path.stem] = {
                "description": str(payload.get("description", "")),
                "args": args,
            }
    profiles.setdefault(
        "default",
        {
            "description": "Use argparse defaults: full diagnostics and moderate candidates.",
            "args": {},
        },
    )
    return profiles


def format_profile_help(profiles: dict[str, dict[str, Any]]) -> str:
    lines = ["Available BO profiles:"]
    for name, profile in sorted(profiles.items()):
        description = profile.get("description") or "No description."
        lines.append(f"  {name}: {description}")
    return "\n".join(lines)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_2d_y(y: np.ndarray | list[float]) -> np.ndarray:
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


class FlatCodec:
    """Flat binary+log-concentration codec shared by both compared models."""

    def __init__(self, essentials: list[str], additives: list[str], k_max: int):
        self.E = list(essentials)
        self.adds = list(additives)
        self.A = len(self.adds)
        self.k_max = int(k_max)
        self.n_ess = len(self.E)
        self.d = self.n_ess + 2 * self.A
        self.add_cols: dict[str, tuple[int, int]] = {}
        for i, name in enumerate(self.adds):
            bin_col = self.n_ess + 2 * i
            conc_col = self.n_ess + 2 * i + 1
            self.add_cols[name] = (bin_col, conc_col)
        self.cat_dims = [self.n_ess + 2 * i for i in range(self.A)]
        self.conc_dims = [self.n_ess + 2 * i + 1 for i in range(self.A)]
        self.cont_dims = list(range(self.n_ess)) + self.conc_dims

    def encode_row(self, row: dict[str, float]) -> np.ndarray:
        z = np.zeros(self.d, dtype=np.float64)
        for i, e in enumerate(self.E):
            v = row.get(e, 0.0)
            z[i] = np.log10(np.clip(v, CONC_LO, CONC_HI)) if v > EPS else LOG_LO
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

    def encode(self, rows: list[dict[str, float]]) -> np.ndarray:
        return np.array([self.encode_row(r) for r in rows])

    def decode(self, Z: np.ndarray) -> list[dict[str, float]]:
        Z = np.atleast_2d(Z)
        decoded: list[dict[str, float]] = []
        for z in Z:
            row: dict[str, float] = {}
            for i, e in enumerate(self.E):
                row[e] = float(10.0 ** np.clip(z[i], LOG_LO, LOG_HI))
            for name in self.adds:
                bin_col, conc_col = self.add_cols[name]
                if z[bin_col] > 0.5:
                    row[name] = float(10.0 ** np.clip(z[conc_col], LOG_LO, LOG_HI))
            decoded.append(row)
        return decoded

    def postprocess_batch(self, Zt: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        Z = Zt.detach().cpu().clone()
        Z = Z.clamp(min=bounds[0].cpu(), max=bounds[1].cpu())
        for bc in self.cat_dims:
            Z[..., bc] = Z[..., bc].round()
        for name in self.adds:
            bin_col, conc_col = self.add_cols[name]
            inactive = Z[..., bin_col] < 0.5
            Z[..., conc_col][inactive] = CONC_DEFAULT
        return Z.to(device=Zt.device, dtype=Zt.dtype)

    def get_bounds(self, device: torch.device) -> torch.Tensor:
        lb = np.zeros(self.d, dtype=np.float64)
        ub = np.zeros(self.d, dtype=np.float64)
        lb[: self.n_ess] = LOG_LO
        ub[: self.n_ess] = LOG_HI
        for i in range(self.A):
            bin_col = self.n_ess + 2 * i
            conc_col = self.n_ess + 2 * i + 1
            lb[bin_col] = 0.0
            ub[bin_col] = 1.0
            lb[conc_col] = LOG_LO
            ub[conc_col] = LOG_HI
        return torch.tensor(np.stack([lb, ub]), device=device, dtype=torch.double)

    def build_fixed_features_list(self, k_max: int | None = None) -> list[dict[int, float]]:
        k = self.k_max if k_max is None else int(k_max)
        ff_list: list[dict[int, float]] = []
        for n_active in range(0, k + 1):
            for combo in combinations(range(self.A), n_active):
                active_set = set(combo)
                fixed: dict[int, float] = {}
                for j in range(self.A):
                    bin_col = self.n_ess + 2 * j
                    conc_col = self.n_ess + 2 * j + 1
                    if j in active_set:
                        fixed[bin_col] = 1.0
                    else:
                        fixed[bin_col] = 0.0
                        fixed[conc_col] = CONC_DEFAULT
                ff_list.append(fixed)
        return ff_list


class AdditiveSetKernel(Kernel):
    """A prior over recipes based on active additive sets and shared effects.

    The kernel keeps the flat representation but changes the similarity belief:

    * active-set similarity uses Tanimoto/Jaccard, so common inactive additives
      do not make two sparse recipes look artificially similar;
    * additive main effects are shared only when both recipes contain the same
      additive;
    * pairwise effects are shared only through shared additive pairs, with a
      smaller initial scale for higher-order generalization;
    * a small residual term can model same/subset active-set smoothness without
      dominating the main/pair hierarchy.
    """

    has_lengthscale = False

    def __init__(
        self,
        ess_dims: list[int],
        cat_dims: list[int],
        conc_dims: list[int],
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("ess_dims", torch.tensor(ess_dims, dtype=torch.long))
        self.register_buffer("cat_dims", torch.tensor(cat_dims, dtype=torch.long))
        self.register_buffer("conc_dims", torch.tensor(conc_dims, dtype=torch.long))
        self.register_parameter(
            "raw_scales",
            torch.nn.Parameter(torch.zeros(5, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_ess_lengthscale",
            torch.nn.Parameter(torch.zeros(len(ess_dims), dtype=torch.double)),
        )
        self.register_parameter(
            "raw_conc_lengthscale",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_scales", Positive())
        self.register_constraint("raw_ess_lengthscale", Positive())
        self.register_constraint("raw_conc_lengthscale", Positive())
        self.initialize(
            raw_scales=self.raw_scales_constraint.inverse_transform(
                torch.tensor([0.15, 0.35, 0.75, 0.25, 0.05], dtype=torch.double)
            ),
            raw_ess_lengthscale=self.raw_ess_lengthscale_constraint.inverse_transform(
                torch.full((len(ess_dims),), 0.35, dtype=torch.double)
            ),
            raw_conc_lengthscale=self.raw_conc_lengthscale_constraint.inverse_transform(
                torch.tensor([0.25], dtype=torch.double)
            ),
        )

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales_constraint.transform(self.raw_scales)

    @property
    def ess_lengthscale(self) -> torch.Tensor:
        return self.raw_ess_lengthscale_constraint.transform(self.raw_ess_lengthscale)

    @property
    def conc_lengthscale(self) -> torch.Tensor:
        return self.raw_conc_lengthscale_constraint.transform(self.raw_conc_lengthscale)

    def _parts(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = x1[..., self.ess_dims]
        e2 = x2[..., self.ess_dims]
        b1 = (x1[..., self.cat_dims] > 0.5).to(dtype=x1.dtype)
        b2 = (x2[..., self.cat_dims] > 0.5).to(dtype=x2.dtype)
        c1 = x1[..., self.conc_dims]
        c2 = x2[..., self.conc_dims]

        ess_diff = (e1.unsqueeze(-2) - e2.unsqueeze(-3)) / self.ess_lengthscale
        k_ess = torch.exp(-0.5 * ess_diff.pow(2).sum(dim=-1))

        b1u = b1.unsqueeze(-2)
        b2u = b2.unsqueeze(-3)
        shared = b1u * b2u
        intersection = shared.sum(dim=-1)
        union = b1.sum(dim=-1).unsqueeze(-1) + b2.sum(dim=-1).unsqueeze(-2) - intersection
        k_set = torch.where(
            union > self.eps,
            intersection / union.clamp_min(self.eps),
            torch.ones_like(union),
        )

        conc_diff = (c1.unsqueeze(-2) - c2.unsqueeze(-3)) / self.conc_lengthscale
        k_conc_per_add = torch.exp(-0.5 * conc_diff.pow(2))
        main_components = shared * k_conc_per_add
        k_main = main_components.sum(dim=-1)
        k_pair = 0.5 * (k_main.pow(2) - main_components.pow(2).sum(dim=-1))

        k_all_conc = torch.exp(-0.5 * conc_diff.pow(2).sum(dim=-1))
        k_residual = k_set * k_all_conc
        return {
            "ess": k_ess,
            "set": k_set,
            "main": k_main,
            "pair": k_pair,
            "residual": k_residual,
        }

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("AdditiveSetKernel does not support last_dim_is_batch.")
        p = self._parts(x1=x1, x2=x2)
        s_ess, s_set, s_main, s_pair, s_res = self.scales
        cov = p["ess"] * (
            s_ess
            + s_set * p["set"]
            + s_main * p["main"]
            + s_pair * p["pair"]
            + s_res * p["residual"]
        )
        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


def load_training_data(args: argparse.Namespace) -> tuple[pd.DataFrame, FlatCodec]:
    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path.resolve()}")
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=args.sheet)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input suffix: {path.suffix}")
    df.columns = df.columns.str.lower()

    if args.hrp is not None and "hrp" in df.columns:
        hrp = pd.to_numeric(df["hrp"], errors="coerce")
        df = df.loc[np.isclose(hrp, args.hrp, rtol=0, atol=args.hrp_atol)].copy()
    if args.filter_ctrl and "ctrl" in df.columns:
        ctrl = pd.to_numeric(df["ctrl"], errors="coerce").fillna(0)
        df = df.loc[ctrl == 0].copy()
    if args.target not in df.columns:
        raise KeyError(f"Target column '{args.target}' not found in {list(df.columns)}")

    present = [c for c in FEATURE_WHITELIST if c in df.columns]
    missing_ess = [e for e in ESSENTIALS if e not in present]
    if missing_ess:
        raise KeyError(f"Missing essential columns: {missing_ess}")
    additives = [c for c in present if c not in ESSENTIALS]
    codec = FlatCodec(essentials=ESSENTIALS, additives=additives, k_max=args.k_max)
    return df, codec


def encode_training_data(
    df: pd.DataFrame,
    codec: FlatCodec,
    target_col: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    Z_np = codec.encode(df.to_dict("records"))
    y_np = ensure_2d_y(df[target_col].values)
    unique_Z, _, inverse = np.unique(Z_np, axis=0, return_index=True, return_inverse=True)
    n_duplicates = len(Z_np) - len(unique_Z)
    if n_duplicates > 0:
        unique_y = np.array([np.mean(y_np[inverse == i]) for i in range(len(unique_Z))])
        Z_np = unique_Z
        y_np = ensure_2d_y(unique_y)
    X = torch.tensor(Z_np, dtype=torch.double, device=device)
    Y = torch.tensor(y_np, dtype=torch.double, device=device)
    meta = {
        "raw_rows": int(len(df)),
        "unique_encoded_rows": int(X.shape[0]),
        "duplicates_merged": int(n_duplicates),
        "encoded_dim": int(X.shape[1]),
        "n_additives": int(codec.A),
    }
    return X, Y, meta


def make_old_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    codec: FlatCodec,
    bounds: torch.Tensor,
) -> MixedSingleTaskGP:
    return MixedSingleTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=codec.cat_dims,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def make_new_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    codec: FlatCodec,
    bounds: torch.Tensor,
) -> SingleTaskGP:
    covar_module = AdditiveSetKernel(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
    )
    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar_module,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def fit_model(model: SingleTaskGP, maxiter: int) -> dict[str, Any]:
    t0 = time.time()
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(model.train_targets)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": maxiter, "disp": False}},
    )
    elapsed = time.time() - t0
    model.train()
    model.likelihood.train()
    with torch.no_grad():
        try:
            mll_value = float(mll(model(*model.train_inputs), model.train_targets).item())
        except Exception:
            mll_value = float("nan")
    model.eval()
    return {"fit_seconds": elapsed, "mll": mll_value}


def transformed_train_X(model: SingleTaskGP, X: torch.Tensor) -> torch.Tensor:
    if getattr(model, "input_transform", None) is None:
        return X
    return model.input_transform(X)


def training_fit_diagnostics(model: SingleTaskGP, X: torch.Tensor, Y: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(X, observation_noise=True)
        mean = posterior.mean.squeeze(-1)
        var = posterior.variance.squeeze(-1).clamp_min(1e-18)
        y = Y.squeeze(-1)
        residual = y - mean
        rmse = residual.pow(2).mean().sqrt().item()
        mae = residual.abs().mean().item()
        y_var = y.var(unbiased=True).clamp_min(1e-18)
        r2 = 1.0 - residual.pow(2).sum() / ((y - y.mean()).pow(2).sum().clamp_min(1e-18))
        z = residual / var.sqrt()
    return {
        "train_rmse": float(rmse),
        "train_mae": float(mae),
        "train_r2": float(r2.item()),
        "train_z_mean": float(z.mean().item()),
        "train_z_std": float(z.std(unbiased=True).item()),
        "train_frac_abs_z_gt_2": float((z.abs() > 2).double().mean().item()),
        "train_frac_abs_z_gt_3": float((z.abs() > 3).double().mean().item()),
        "target_std": float(y_var.sqrt().item()),
    }


def matrix_diagnostics(model: SingleTaskGP, X: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        X_model = transformed_train_X(model, X)
        K = model.covar_module(X_model).evaluate()
        noise = model.likelihood.noise.to(K).reshape(-1)[0]
        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        K_full = K + noise * eye
        eigvals = torch.linalg.eigvalsh(K_full)
        cond = eigvals.max() / eigvals.min().clamp_min(1e-18)
    return {
        "kernel_diag_min": float(K.diagonal().min().item()),
        "kernel_diag_max": float(K.diagonal().max().item()),
        "noise_standardized": float(noise.item()),
        "eig_min": float(eigvals.min().item()),
        "eig_max": float(eigvals.max().item()),
        "condition_number": float(cond.item()),
    }


def loo_diagnostics(model: SingleTaskGP, X: torch.Tensor) -> dict[str, Any]:
    """Fast fixed-hyperparameter LOO diagnostics in model-output space."""
    n = X.shape[0]
    model.eval()
    with torch.no_grad():
        X_model = transformed_train_X(model, X)
        mean = model.mean_module(X_model).reshape(-1)
        K = model.covar_module(X_model).evaluate()
        noise = model.likelihood.noise.to(K).reshape(-1)[0]
        eye = torch.eye(n, device=K.device, dtype=K.dtype)
        C = K + noise * eye
        jitter = 1e-8
        try:
            L = torch.linalg.cholesky(C)
            used_jitter = 0.0
        except RuntimeError:
            L = torch.linalg.cholesky(C + jitter * eye)
            used_jitter = jitter
        y_model = model.train_targets.reshape(-1).to(K)
        alpha = torch.cholesky_solve((y_model - mean).unsqueeze(-1), L).squeeze(-1)
        C_inv_diag = torch.cholesky_inverse(L).diagonal().clamp_min(1e-18)
        loo_mean = y_model - alpha / C_inv_diag
        loo_var = (1.0 / C_inv_diag).clamp_min(1e-18)
        loo_resid = y_model - loo_mean
        z = loo_resid / loo_var.sqrt()
    return {
        "loo_rmse_model_space": float(loo_resid.pow(2).mean().sqrt().item()),
        "loo_mae_model_space": float(loo_resid.abs().mean().item()),
        "loo_z_mean": float(z.mean().item()),
        "loo_z_std": float(z.std(unbiased=True).item()),
        "loo_frac_abs_z_gt_2": float((z.abs() > 2).double().mean().item()),
        "loo_frac_abs_z_gt_3": float((z.abs() > 3).double().mean().item()),
        "loo_cholesky_jitter": float(used_jitter),
    }


def kernel_parameter_summary(model: SingleTaskGP) -> dict[str, Any]:
    covar = model.covar_module
    out: dict[str, Any] = {}
    if isinstance(covar, AdditiveSetKernel):
        scales = covar.scales.detach().cpu().numpy()
        out.update(
            {
                "new_scale_ess": float(scales[0]),
                "new_scale_set": float(scales[1]),
                "new_scale_main": float(scales[2]),
                "new_scale_pair": float(scales[3]),
                "new_scale_residual": float(scales[4]),
                "new_ess_lengthscale": [
                    float(v) for v in covar.ess_lengthscale.detach().cpu().reshape(-1)
                ],
                "new_conc_lengthscale": float(covar.conc_lengthscale.detach().cpu().item()),
            }
        )
    else:
        lengthscales: list[float] = []
        outputscales: list[float] = []
        cat_lengthscales: list[float] = []

        def walk(k: Any) -> None:
            if type(k).__name__ == "CategoricalKernel" and hasattr(k, "lengthscale"):
                cat_lengthscales.extend(
                    float(v) for v in k.lengthscale.detach().cpu().reshape(-1)
                )
            elif hasattr(k, "lengthscale") and k.lengthscale is not None:
                lengthscales.extend(
                    float(v) for v in k.lengthscale.detach().cpu().reshape(-1)
                )
            if hasattr(k, "outputscale"):
                outputscales.extend(
                    float(v) for v in k.outputscale.detach().cpu().reshape(-1)
                )
            if hasattr(k, "base_kernel"):
                walk(k.base_kernel)
            if hasattr(k, "kernels"):
                for child in k.kernels:
                    walk(child)

        walk(covar)
        if lengthscales:
            out.update(
                {
                    "old_cont_ls_min": float(np.min(lengthscales)),
                    "old_cont_ls_median": float(np.median(lengthscales)),
                    "old_cont_ls_max": float(np.max(lengthscales)),
                }
            )
        if cat_lengthscales:
            out.update(
                {
                    "old_cat_ls_min": float(np.min(cat_lengthscales)),
                    "old_cat_ls_median": float(np.median(cat_lengthscales)),
                    "old_cat_ls_max": float(np.max(cat_lengthscales)),
                }
            )
        if outputscales:
            out.update(
                {
                    "old_outputscale_min": float(np.min(outputscales)),
                    "old_outputscale_median": float(np.median(outputscales)),
                    "old_outputscale_max": float(np.max(outputscales)),
                }
            )
    return out


def make_qlognei(model: SingleTaskGP, X_baseline: torch.Tensor, mc_samples: int, seed: int):
    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([mc_samples]),
        seed=seed,
    )
    return qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=X_baseline,
        sampler=sampler,
        prune_baseline=True,
        cache_root=False,
    )


def make_candidate_pool(
    codec: FlatCodec,
    bounds: torch.Tensor,
    n: int,
    k_max: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    sobol = torch.quasirandom.SobolEngine(dimension=codec.d, scramble=True, seed=seed)
    base = sobol.draw(n).to(device=bounds.device, dtype=bounds.dtype)
    X = bounds[0] + (bounds[1] - bounds[0]) * base
    for row in range(n):
        k = int(rng.integers(0, k_max + 1))
        active = set(rng.choice(codec.A, size=k, replace=False).tolist()) if k else set()
        for j in range(codec.A):
            bin_col = codec.n_ess + 2 * j
            conc_col = codec.n_ess + 2 * j + 1
            if j in active:
                X[row, bin_col] = 1.0
            else:
                X[row, bin_col] = 0.0
                X[row, conc_col] = CONC_DEFAULT
    return X


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ar) < 1e-12 or np.std(br) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def acq_stability_diagnostics(
    model: SingleTaskGP,
    X_baseline: torch.Tensor,
    pool: torch.Tensor,
    mc_samples: int,
    repeats: int,
    seed: int,
    eval_batch_size: int,
) -> dict[str, Any]:
    values = []
    for r in range(repeats):
        acqf = make_qlognei(model, X_baseline, mc_samples=mc_samples, seed=seed + 1009 * r)
        chunks = []
        with torch.no_grad():
            for i in range(0, pool.shape[0], eval_batch_size):
                chunks.append(acqf(pool[i : i + eval_batch_size].unsqueeze(1)).view(-1))
        values.append(torch.cat(chunks).detach().cpu().numpy())
    V = np.vstack(values)
    corrs = [
        spearman_corr(V[i], V[j])
        for i in range(repeats)
        for j in range(i + 1, repeats)
    ]
    top_sets = []
    for row in V:
        top_sets.append(set(np.argsort(row)[-min(25, len(row)) :].tolist()))
    overlaps = []
    for i in range(repeats):
        for j in range(i + 1, repeats):
            union = top_sets[i] | top_sets[j]
            overlaps.append(len(top_sets[i] & top_sets[j]) / max(len(union), 1))
    return {
        "acq_pool_size": int(pool.shape[0]),
        "acq_mc_samples": int(mc_samples),
        "acq_repeats": int(repeats),
        "acq_value_mean": float(np.mean(V)),
        "acq_value_std": float(np.std(V)),
        "acq_per_point_std_mean": float(np.mean(np.std(V, axis=0))),
        "acq_spearman_mean": float(np.nanmean(corrs)) if corrs else float("nan"),
        "acq_spearman_min": float(np.nanmin(corrs)) if corrs else float("nan"),
        "acq_top25_jaccard_mean": float(np.mean(overlaps)) if overlaps else float("nan"),
        "acq_top25_jaccard_min": float(np.min(overlaps)) if overlaps else float("nan"),
        "acq_finite_fraction": float(np.isfinite(V).mean()),
    }


def screen_top_subspaces(
    model: SingleTaskGP,
    X_baseline: torch.Tensor,
    codec: FlatCodec,
    bounds: torch.Tensor,
    fixed_features_list: list[dict[int, float]],
    num_top_subspaces: int,
    sobol_max_samples: int,
    mc_samples: int,
    seed: int,
    eval_batch_size: int,
) -> tuple[list[dict[int, float]], dict[str, Any]]:
    acqf = make_qlognei(model, X_baseline, mc_samples=mc_samples, seed=seed)
    d = bounds.shape[-1]
    max_n_free = max(d - len(fixed) for fixed in fixed_features_list)
    sobol_draw_size = min(sobol_max_samples, 2 ** max_n_free)
    sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    sobol_base = sobol.draw(sobol_draw_size).to(device=bounds.device, dtype=bounds.dtype)
    lb, ub = bounds
    best_val_per_space = torch.empty(
        len(fixed_features_list), device=bounds.device, dtype=bounds.dtype
    )
    active_counts = []
    sample_stats: dict[int, int] = defaultdict(int)

    for si, fixed in enumerate(fixed_features_list):
        n_free = d - len(fixed)
        num_samples = min(sobol_max_samples, 2 ** n_free)
        sample_stats[n_free] += num_samples
        X = lb + (ub - lb) * sobol_base[:num_samples].clone()
        for col, val in fixed.items():
            X[:, col] = val
        cur_best = -float("inf")
        with torch.no_grad():
            for i in range(0, num_samples, eval_batch_size):
                vals = acqf(X[i : i + eval_batch_size].unsqueeze(1)).view(-1)
                cur_best = max(cur_best, float(vals.max().item()))
        best_val_per_space[si] = cur_best
        active_counts.append(sum(1 for col in codec.cat_dims if fixed.get(col, 0.0) > 0.5))

    k_top = min(num_top_subspaces, len(fixed_features_list))
    top_vals, top_idx = torch.topk(best_val_per_space, k=k_top)
    top_fixed = [fixed_features_list[i] for i in top_idx.tolist()]
    top_active_counts = [active_counts[i] for i in top_idx.tolist()]
    info = {
        "n_subspaces_total": int(len(fixed_features_list)),
        "n_subspaces_top": int(k_top),
        "screen_best": float(top_vals[0].item()),
        "screen_worst_top": float(top_vals[-1].item()),
        "screen_active_count_dist": dict(Counter(top_active_counts)),
        "screen_sample_stats_by_n_free": dict(sorted(sample_stats.items())),
    }
    return top_fixed, info


def summarize_candidates(
    rows: list[dict[str, float]],
    codec: FlatCodec,
    acq_values: np.ndarray | None = None,
) -> dict[str, Any]:
    active_counts = [sum(1 for a in codec.adds if r.get(a, 0.0) > EPS) for r in rows]
    add_counts = Counter(
        a for r in rows for a in codec.adds if r.get(a, 0.0) > EPS
    )
    pair_counts: Counter[str] = Counter()
    for r in rows:
        active = [a for a in codec.adds if r.get(a, 0.0) > EPS]
        for a, b in combinations(active, 2):
            pair_counts[f"{a}+{b}"] += 1
    out: dict[str, Any] = {
        "candidate_n": int(len(rows)),
        "candidate_active_count_dist": dict(Counter(active_counts)),
        "candidate_additive_counts": dict(add_counts.most_common()),
        "candidate_pair_counts": dict(pair_counts.most_common(20)),
    }
    if acq_values is not None and len(acq_values):
        out.update(
            {
                "candidate_acq_mean": float(np.mean(acq_values)),
                "candidate_acq_min": float(np.min(acq_values)),
                "candidate_acq_max": float(np.max(acq_values)),
            }
        )
    return out


def generate_candidates(
    model: SingleTaskGP,
    X_baseline: torch.Tensor,
    codec: FlatCodec,
    bounds: torch.Tensor,
    args: argparse.Namespace,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fixed_features_list = codec.build_fixed_features_list(k_max=args.k_max)
    top_fixed, screen_info = screen_top_subspaces(
        model=model,
        X_baseline=X_baseline,
        codec=codec,
        bounds=bounds,
        fixed_features_list=fixed_features_list,
        num_top_subspaces=args.num_top_subspaces,
        sobol_max_samples=args.sobol_max_samples,
        mc_samples=args.screen_mc_samples,
        seed=args.seed,
        eval_batch_size=args.eval_batch_size,
    )
    acqf = make_qlognei(
        model=model,
        X_baseline=X_baseline,
        mc_samples=args.refine_mc_samples,
        seed=args.seed + 17,
    )
    t0 = time.time()
    candidates_raw, joint_acq = optimize_acqf_mixed(
        acq_function=acqf,
        bounds=bounds,
        q=args.q,
        fixed_features_list=top_fixed,
        num_restarts=args.num_restarts,
        raw_samples=args.raw_samples,
        options={"batch_limit": args.num_restarts, "maxiter": args.acq_maxiter},
        retry_on_optimization_warning=True,
    )
    elapsed = time.time() - t0
    candidates = codec.postprocess_batch(candidates_raw, bounds)
    candidates_cpu = candidates.detach().cpu()

    seen: set[bytes] = set()
    unique_tensors: list[torch.Tensor] = []
    for row_cpu, tensor in zip(candidates_cpu, candidates):
        # De-duplicate directly in the canonical postprocessed encoding space.
        key = row_cpu.contiguous().numpy().tobytes()
        if key not in seen:
            seen.add(key)
            unique_tensors.append(tensor)
    unique_cand = torch.stack(unique_tensors) if unique_tensors else candidates[:0]
    unique_rows = codec.decode(unique_cand.detach().cpu().numpy())
    with torch.no_grad():
        acq_vals = acqf(unique_cand.unsqueeze(1)).view(-1).detach().cpu().numpy()
        post = model.posterior(unique_cand)
        means = post.mean.squeeze(-1).detach().cpu().numpy()
        sigmas = post.variance.squeeze(-1).clamp_min(0).sqrt().detach().cpu().numpy()

    cand_df = pd.DataFrame(unique_rows).fillna(0.0)
    cand_df["model"] = label
    cand_df["acq"] = acq_vals
    cand_df["pred_mean"] = means
    cand_df["pred_sigma"] = sigmas
    cand_df["active_count"] = [
        sum(1 for a in codec.adds if row.get(a, 0.0) > EPS) for row in unique_rows
    ]
    cand_df = cand_df.sort_values("acq", ascending=False).head(args.q)
    info = {
        **screen_info,
        "candidate_opt_seconds": float(elapsed),
        "candidate_joint_acq": float(joint_acq.item()),
        "candidate_unique_before_topq": int(len(unique_rows)),
        **summarize_candidates(cand_df.to_dict("records"), codec, cand_df["acq"].to_numpy()),
    }
    return cand_df, info


def dataset_summary(df: pd.DataFrame, codec: FlatCodec, target: str) -> dict[str, Any]:
    active = (df[codec.adds].fillna(0).to_numpy() > EPS).sum(axis=1)
    over_hi = (df[codec.adds] > CONC_HI).sum()
    over_hi = over_hi[over_hi > 0].sort_values(ascending=False)
    return {
        "rows_after_filters": int(len(df)),
        "target_mean": float(df[target].mean()),
        "target_std": float(df[target].std(ddof=1)),
        "target_min": float(df[target].min()),
        "target_max": float(df[target].max()),
        "active_count_dist": dict(Counter(int(v) for v in active)),
        "rows_any_additive_gt_encoding_hi": int((df[codec.adds] > CONC_HI).any(axis=1).sum()),
        "additive_gt_encoding_hi_counts": {k: int(v) for k, v in over_hi.to_dict().items()},
    }


def run(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, codec = load_training_data(args)
    X, Y, encode_meta = encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    pool = None
    if args.acq_stability_diagnostics:
        pool = make_candidate_pool(
            codec=codec,
            bounds=bounds,
            n=args.acq_pool_size,
            k_max=args.k_max,
            seed=args.seed + 123,
        )

    print("[Data]", json.dumps({**dataset_summary(df, codec, args.target), **encode_meta}, indent=2))
    results: dict[str, dict[str, Any]] = {}
    candidate_frames: list[pd.DataFrame] = []

    for label, builder in [
        ("old_mixed_hamming", make_old_model),
        ("new_additive_set", make_new_model),
    ]:
        print(f"\n=== Fitting {label} ===")
        model = builder(X, Y, codec, bounds)
        fit_info = fit_model(model, maxiter=args.fit_maxiter)
        print(f"  fit seconds: {fit_info['fit_seconds']:.1f}")
        diagnostics: dict[str, Any] = {}
        diagnostics.update(fit_info)
        if args.training_fit_diagnostics:
            diagnostics.update(training_fit_diagnostics(model, X, Y))
        if args.matrix_diagnostics:
            diagnostics.update(matrix_diagnostics(model, X))
        if args.loo_diagnostics:
            diagnostics.update(loo_diagnostics(model, X))
        if args.kernel_parameter_summary:
            diagnostics.update(kernel_parameter_summary(model))
        if args.acq_stability_diagnostics:
            assert pool is not None
            diagnostics.update(
                acq_stability_diagnostics(
                    model=model,
                    X_baseline=X,
                    pool=pool,
                    mc_samples=args.stability_mc_samples,
                    repeats=args.stability_repeats,
                    seed=args.seed + 29,
                    eval_batch_size=args.eval_batch_size,
                )
            )
        if not args.skip_candidates:
            print(f"  generating candidates for {label} ...")
            cand_df, cand_info = generate_candidates(
                model=model,
                X_baseline=X,
                codec=codec,
                bounds=bounds,
                args=args,
                label=label,
            )
            diagnostics.update(cand_info)
            candidate_frames.append(cand_df)
            cand_path = output_dir / f"{label}_candidates.csv"
            cand_df.to_csv(cand_path, index=False, encoding="utf-8")
            print(f"  candidates -> {cand_path.resolve()}")
        results[label] = diagnostics

    summary_path = output_dir / "add_bo_diagnostics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "data": {**dataset_summary(df, codec, args.target), **encode_meta},
                "models": results,
            },
            f,
            indent=2,
        )
    flat_rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        flat_rows.append(row)
    metrics_path = output_dir / "add_bo_diagnostics.csv"
    pd.DataFrame(flat_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    if candidate_frames:
        all_candidates_path = output_dir / "add_bo_candidates_all.csv"
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            all_candidates_path, index=False, encoding="utf-8"
        )
        print(f"[Output] candidates comparison -> {all_candidates_path.resolve()}")
    print(f"[Output] diagnostics JSON -> {summary_path.resolve()}")
    print(f"[Output] diagnostics CSV  -> {metrics_path.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    profiles = load_bo_profiles()
    p = argparse.ArgumentParser(
        description="Compare old MixedSingleTaskGP with a custom additive-set GP prior.",
        epilog=format_profile_help(profiles),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.bo_profiles = profiles  # type: ignore[attr-defined]

    presets = p.add_argument_group("BO run presets")
    presets.add_argument(
        "--bo_profile",
        choices=sorted(profiles),
        default="default",
        help=(
            "Optional preset for candidate-focused runs. Explicit command-line "
            "arguments override preset values. Profiles are loaded from bo_profiles/*.json."
        ),
    )

    data = p.add_argument_group("Data input and filtering")
    data.add_argument("--input", type=str, default="data_corrected.xlsx")
    data.add_argument("--sheet", type=str, default="Corrected Data")
    data.add_argument("--target", type=str, default="intensity")
    data.add_argument("--hrp", type=float, default=0.0001)
    data.add_argument("--hrp_atol", type=float, default=1e-12)
    data.add_argument(
        "--filter_ctrl",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    run_control = p.add_argument_group("Run control and output")
    run_control.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run_control.add_argument("--seed", type=int, default=42)
    run_control.add_argument("--output_dir", type=str, default=str(LOGS_DIR / "add_bo_compare"))

    search_space = p.add_argument_group("Search-space constraints")
    search_space.add_argument("--k_max", type=int, default=4)

    gp_fit = p.add_argument_group("GP model fitting")
    gp_fit.add_argument("--fit_maxiter", type=int, default=75)

    model_diag = p.add_argument_group("Model diagnostics")
    model_diag.add_argument(
        "--training_fit_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model_diag.add_argument(
        "--matrix_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model_diag.add_argument(
        "--loo_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    model_diag.add_argument(
        "--kernel_parameter_summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    stability = p.add_argument_group("Acquisition stability diagnostics")
    stability.add_argument(
        "--acq_stability_diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    stability.add_argument("--stability_mc_samples", type=int, default=64)
    stability.add_argument("--stability_repeats", type=int, default=5)
    stability.add_argument("--acq_pool_size", type=int, default=512)

    candidate = p.add_argument_group("Candidate generation")
    candidate.add_argument("--skip_candidates", action="store_true")
    candidate.add_argument("--q", type=int, default=32)

    acq_screen = p.add_argument_group("ACQ subspace screening")
    acq_screen.add_argument("--num_top_subspaces", type=int, default=40)
    acq_screen.add_argument("--sobol_max_samples", type=int, default=512)
    acq_screen.add_argument("--screen_mc_samples", type=int, default=32)

    acq_opt = p.add_argument_group("ACQ mixed optimization")
    acq_opt.add_argument("--num_restarts", type=int, default=10)
    acq_opt.add_argument("--raw_samples", type=int, default=256)
    acq_opt.add_argument("--acq_maxiter", type=int, default=100)
    acq_opt.add_argument("--refine_mc_samples", type=int, default=64)

    acq_eval = p.add_argument_group("ACQ evaluation performance")
    acq_eval.add_argument("--eval_batch_size", type=int, default=256)
    return p


def explicit_arg_dests(
    parser: argparse.ArgumentParser,
    argv: list[str] | None = None,
) -> set[str]:
    argv = sys.argv[1:] if argv is None else argv
    option_to_dest = {
        opt: action.dest
        for action in parser._actions
        for opt in action.option_strings
    }
    explicit: set[str] = set()
    for token in argv:
        opt = token.split("=", 1)[0]
        if opt in option_to_dest:
            explicit.add(option_to_dest[opt])
    return explicit


def apply_bo_profile(
    args: argparse.Namespace,
    explicit_dests: set[str],
    profiles: dict[str, dict[str, Any]],
    valid_dests: set[str],
) -> argparse.Namespace:
    profile = profiles[args.bo_profile]["args"]
    unknown = sorted(set(profile) - valid_dests)
    if unknown:
        raise ValueError(
            f"Unknown argument(s) in BO profile '{args.bo_profile}': {unknown}"
        )
    for dest, value in profile.items():
        if dest not in explicit_dests:
            setattr(args, dest, value)
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    profiles = parser.bo_profiles  # type: ignore[attr-defined]
    valid_dests = {action.dest for action in parser._actions}
    return apply_bo_profile(args, explicit_arg_dests(parser, argv), profiles, valid_dests)


if __name__ == "__main__":
    run(parse_args())
