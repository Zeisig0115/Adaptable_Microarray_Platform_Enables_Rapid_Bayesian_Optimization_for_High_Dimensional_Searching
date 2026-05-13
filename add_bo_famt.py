# -*- coding: utf-8 -*-
"""FAMT (Family-Aware Multi-Task) GP kernel for additive recipe BO.

Generative model
----------------
    f(x) = m_const + h_ess(x_ess) + mu_0 + sum_{j active} alpha_j(c_j)

with:
    h_ess  ~ GP(0, sigma_ess^2 * RBF(x_ess; ell_ess))  -- essentials structure
    mu_0   ~ N(0, sigma_0^2)                            -- shared baseline
    alpha_j ~ GP(0, G_jj * RBF(c; ell_conc))            -- per-additive curve
    Cov(alpha_j(c), alpha_k(c')) = G_jk * RBF(c, c')    -- chemistry coupling

The induced covariance is

    k(x, x') = sigma_ess^2 * RBF(x_ess, x'_ess)
             + sigma_0^2
             + sum_{j,k} b_j(x) b_k(x') G_{jk} RBF(c_j, c_k';  ell_conc)

with
    G = diag(sigma_j^2) + (W F + E)(W F + E)^T

where W (A x p, fixed) is the chemistry descriptor matrix from
ADDITIVE_METADATA_V2 below, F (p x r, learned) projects descriptors to a
shared r-dim latent space, and E (A x r, learned, sparse via lambda_E) lets
each additive deviate from its chemistry-implied embedding.

Belief mapping
--------------
  B1 (within-additive concentration response): k_conc on the (j=k) diagonal of G
  B2 (no-add vs singular X depends on X)     : independent sigma_j^2 on diag(G)
  B3 (singular X vs singular Y by chemistry) : (W F)(W F)^T term in G
  Robustness to wrong chemistry              : E term + sparsity prior on lambda_E

Priors (all on the constrained / positive side of each parameter)
-----------------------------------------------------------------
  sigma_j         = tau * lambda_j with tau ~ HalfCauchy(0.3),
                                     lambda_j ~ HalfCauchy(1)         (horseshoe)
  E               = lambda_E * E_raw,  E_raw_{i,k} ~ N(0, 1),
                                       lambda_E ~ HalfCauchy(0.1)
  F_{j,k}         ~ N(0, 1)
  sigma_0         ~ HalfNormal(0.5)
  sigma_ess       ~ HalfNormal(0.5)
  ell_ess (ARD)   ~ LogNormal(log 0.4, 0.5)
  ell_conc        ~ LogNormal(log 0.25, 0.5)
  likelihood sigma_n^2 : default BoTorch GammaPrior
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import (
    HalfCauchyPrior,
    HalfNormalPrior,
    LogNormalPrior,
    NormalPrior,
)

from add_bo import (
    CONC_DEFAULT,
    EPS,
    LOG_HI,
    LOG_LO,
    FlatCodec,
    acq_stability_diagnostics,
    dataset_summary,
    encode_training_data,
    load_training_data,
    loo_diagnostics,
    make_candidate_pool,
    matrix_diagnostics,
    set_seeds,
    training_fit_diagnostics,
)

torch.set_default_dtype(torch.double)

LOGS_DIR = Path(__file__).with_name("logs")


# ===================================================================
# Chemistry descriptors v2 (13 features per additive)
# ===================================================================
FEATURE_KEYS: list[str] = [
    # 7 structural / class flags
    "is_peg", "is_polymer", "is_surfactant", "is_polyol_sugar",
    "is_protein", "is_solvent", "is_salt",
    # 4 mechanism flags relevant to HRP / TMB / H2O2
    "is_redox_active", "is_protein_stabilizer",
    "is_chelator_strong", "is_kosmotrope_anion",
    # 2 continuous
    "log_mw", "charge_at_pH7",
]


def _row(**kw):
    """Helper to enforce that every entry covers every FEATURE_KEYS column."""
    missing = [k for k in FEATURE_KEYS if k not in kw]
    if missing:
        raise KeyError(f"Missing keys in descriptor row: {missing}")
    extra = [k for k in kw if k not in FEATURE_KEYS]
    if extra:
        raise KeyError(f"Unknown keys in descriptor row: {extra}")
    return {k: float(kw[k]) for k in FEATURE_KEYS}


ADDITIVE_METADATA_V2: dict[str, dict[str, float]] = {
    "cmc": _row(
        is_peg=0, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(90000.0), charge_at_pH7=-3.0,
    ),
    "peg20k": _row(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(20000.0), charge_at_pH7=0.0,
    ),
    "dmso": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=1, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(78.13), charge_at_pH7=0.0,
    ),
    "pl127": _row(
        is_peg=1, is_polymer=1, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(12600.0), charge_at_pH7=0.0,
    ),
    "bsa": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=1, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(66430.0), charge_at_pH7=-3.0,
    ),
    "pva": _row(
        is_peg=0, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(31000.0), charge_at_pH7=0.0,
    ),
    "tw80": _row(
        is_peg=0, is_polymer=0, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(1310.0), charge_at_pH7=0.0,
    ),
    "glycerol": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=1,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(92.09), charge_at_pH7=0.0,
    ),
    "tw20": _row(
        is_peg=0, is_polymer=0, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(1228.0), charge_at_pH7=0.0,
    ),
    "imidazole": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(68.08), charge_at_pH7=0.5,
    ),
    "tx100": _row(
        is_peg=0, is_polymer=0, is_surfactant=1, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(647.0), charge_at_pH7=0.0,
    ),
    "edta": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=1, is_kosmotrope_anion=0,
        log_mw=math.log10(292.24), charge_at_pH7=-3.0,
    ),
    "mgso4": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=1,
        log_mw=math.log10(120.37), charge_at_pH7=2.0,
    ),
    "sucrose": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=1,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(342.30), charge_at_pH7=0.0,
    ),
    "cacl2": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(110.98), charge_at_pH7=2.0,
    ),
    "znso4": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=1,
        log_mw=math.log10(161.44), charge_at_pH7=2.0,
    ),
    "paa": _row(
        is_peg=0, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(100000.0), charge_at_pH7=-3.0,
    ),
    "mncl2": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=1, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(125.84), charge_at_pH7=2.0,
    ),
    "peg200k": _row(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(200000.0), charge_at_pH7=0.0,
    ),
    "feso4": _row(
        is_peg=0, is_polymer=0, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=1,
        is_redox_active=1, is_protein_stabilizer=0,
        is_chelator_strong=0, is_kosmotrope_anion=1,
        log_mw=math.log10(151.91), charge_at_pH7=2.0,
    ),
    "peg6k": _row(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(6000.0), charge_at_pH7=0.0,
    ),
    "peg400": _row(
        is_peg=1, is_polymer=1, is_surfactant=0, is_polyol_sugar=0,
        is_protein=0, is_solvent=0, is_salt=0,
        is_redox_active=0, is_protein_stabilizer=1,
        is_chelator_strong=0, is_kosmotrope_anion=0,
        log_mw=math.log10(400.0), charge_at_pH7=0.0,
    ),
}


def build_descriptor_matrix(additives: list[str]) -> tuple[torch.Tensor, list[str]]:
    """Return (W, FEATURE_KEYS). Continuous columns (log_mw, charge_at_pH7) are
    z-scored across additives so they live on the same scale as the binary flags."""
    rows = []
    for name in additives:
        if name not in ADDITIVE_METADATA_V2:
            raise KeyError(f"Missing v2 metadata for additive '{name}'.")
        rows.append([ADDITIVE_METADATA_V2[name][k] for k in FEATURE_KEYS])
    arr = np.asarray(rows, dtype=np.float64)
    for col in ("log_mw", "charge_at_pH7"):
        idx = FEATURE_KEYS.index(col)
        mu = arr[:, idx].mean()
        sd = arr[:, idx].std()
        arr[:, idx] = (arr[:, idx] - mu) / max(sd, 1e-8)
    return torch.tensor(arr, dtype=torch.double), list(FEATURE_KEYS)


# ===================================================================
# FAMT kernel
# ===================================================================
class FAMTKernel(Kernel):
    """Family-Aware Multi-Task kernel; see the module docstring for the math."""

    has_lengthscale = False

    def __init__(
        self,
        ess_dims: list[int],
        cat_dims: list[int],
        conc_dims: list[int],
        descriptors: torch.Tensor,   # A x p, fixed chemistry features
        latent_rank: int = 4,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("ess_dims", torch.tensor(ess_dims, dtype=torch.long))
        self.register_buffer("cat_dims", torch.tensor(cat_dims, dtype=torch.long))
        self.register_buffer("conc_dims", torch.tensor(conc_dims, dtype=torch.long))
        self.register_buffer("W", descriptors)

        A = int(len(cat_dims))
        p = int(descriptors.shape[-1])
        r = int(latent_rank)
        self.A = A
        self.p = p
        self.r = r

        # essentials RBF
        self.register_parameter(
            "raw_ess_lengthscale",
            torch.nn.Parameter(torch.zeros(len(ess_dims), dtype=torch.double)),
        )
        self.register_constraint("raw_ess_lengthscale", Positive())
        self.register_parameter(
            "raw_ess_outputscale",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_ess_outputscale", Positive())

        # baseline variance sigma_0^2
        self.register_parameter(
            "raw_baseline_var",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_baseline_var", Positive())

        # horseshoe on per-additive scale: sigma_j = tau * lambda_j
        self.register_parameter(
            "raw_tau",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_tau", Positive())
        self.register_parameter(
            "raw_lambdas",
            torch.nn.Parameter(torch.zeros(A, dtype=torch.double)),
        )
        self.register_constraint("raw_lambdas", Positive())

        # chemistry-projection F (p x r) -- N(0, 1) prior, no constraint
        self.register_parameter(
            "F",
            torch.nn.Parameter(torch.zeros(p, r, dtype=torch.double)),
        )
        # per-additive deviation in non-centered form: E = lambda_E * E_raw
        self.register_parameter(
            "E_raw",
            torch.nn.Parameter(torch.zeros(A, r, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_lambda_E",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_lambda_E", Positive())

        # concentration RBF
        self.register_parameter(
            "raw_conc_lengthscale",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_conc_lengthscale", Positive())

        # Initialisations (in the constrained space)
        self.initialize(
            raw_ess_lengthscale=self.raw_ess_lengthscale_constraint.inverse_transform(
                torch.full((len(ess_dims),), 0.4, dtype=torch.double)
            ),
            raw_ess_outputscale=self.raw_ess_outputscale_constraint.inverse_transform(
                torch.tensor([0.3], dtype=torch.double)
            ),
            raw_baseline_var=self.raw_baseline_var_constraint.inverse_transform(
                torch.tensor([0.3], dtype=torch.double)
            ),
            raw_tau=self.raw_tau_constraint.inverse_transform(
                torch.tensor([0.3], dtype=torch.double)
            ),
            raw_lambdas=self.raw_lambdas_constraint.inverse_transform(
                torch.full((A,), 1.0, dtype=torch.double)
            ),
            F=0.1 * torch.randn(p, r, dtype=torch.double),
            E_raw=0.01 * torch.randn(A, r, dtype=torch.double),
            raw_lambda_E=self.raw_lambda_E_constraint.inverse_transform(
                torch.tensor([0.1], dtype=torch.double)
            ),
            raw_conc_lengthscale=self.raw_conc_lengthscale_constraint.inverse_transform(
                torch.tensor([0.25], dtype=torch.double)
            ),
        )

        # Priors
        self.register_prior(
            "tau_prior", HalfCauchyPrior(scale=0.3),
            lambda m: m.tau,
        )
        self.register_prior(
            "lambdas_prior", HalfCauchyPrior(scale=1.0),
            lambda m: m.lambdas,
        )
        self.register_prior(
            "lambda_E_prior", HalfCauchyPrior(scale=0.1),
            lambda m: m.lambda_E,
        )
        self.register_prior(
            "baseline_var_prior", HalfNormalPrior(scale=0.5),
            lambda m: m.baseline_var,
        )
        self.register_prior(
            "ess_outputscale_prior", HalfNormalPrior(scale=0.5),
            lambda m: m.ess_outputscale,
        )
        self.register_prior(
            "F_prior", NormalPrior(loc=0.0, scale=1.0),
            lambda m: m.F,
        )
        self.register_prior(
            "E_raw_prior", NormalPrior(loc=0.0, scale=1.0),
            lambda m: m.E_raw,
        )
        self.register_prior(
            "ess_lengthscale_prior",
            LogNormalPrior(loc=math.log(0.4), scale=0.5),
            lambda m: m.ess_lengthscale,
        )
        self.register_prior(
            "conc_lengthscale_prior",
            LogNormalPrior(loc=math.log(0.25), scale=0.5),
            lambda m: m.conc_lengthscale,
        )

    # ---- properties for transformed parameters ----
    @property
    def ess_lengthscale(self) -> torch.Tensor:
        return self.raw_ess_lengthscale_constraint.transform(self.raw_ess_lengthscale)

    @property
    def ess_outputscale(self) -> torch.Tensor:
        return self.raw_ess_outputscale_constraint.transform(self.raw_ess_outputscale)

    @property
    def baseline_var(self) -> torch.Tensor:
        return self.raw_baseline_var_constraint.transform(self.raw_baseline_var)

    @property
    def tau(self) -> torch.Tensor:
        return self.raw_tau_constraint.transform(self.raw_tau)

    @property
    def lambdas(self) -> torch.Tensor:
        return self.raw_lambdas_constraint.transform(self.raw_lambdas)

    @property
    def sigma_j(self) -> torch.Tensor:
        """Per-additive signal amplitude: sigma_j = tau * lambda_j."""
        return self.tau * self.lambdas

    @property
    def lambda_E(self) -> torch.Tensor:
        return self.raw_lambda_E_constraint.transform(self.raw_lambda_E)

    @property
    def E(self) -> torch.Tensor:
        """Per-additive deviation, non-centered: E = lambda_E * E_raw."""
        return self.lambda_E * self.E_raw

    @property
    def conc_lengthscale(self) -> torch.Tensor:
        return self.raw_conc_lengthscale_constraint.transform(self.raw_conc_lengthscale)

    @property
    def G(self) -> torch.Tensor:
        """Full A x A coregionalisation matrix."""
        sigma2_diag = self.sigma_j.pow(2)
        U = self.W @ self.F + self.E   # A x r
        return torch.diag(sigma2_diag) + U @ U.transpose(-1, -2)

    # ---- forward ----
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError("FAMTKernel does not support last_dim_is_batch.")

        e1 = x1[..., self.ess_dims]
        e2 = x2[..., self.ess_dims]
        b1 = (x1[..., self.cat_dims] > 0.5).to(dtype=x1.dtype)
        b2 = (x2[..., self.cat_dims] > 0.5).to(dtype=x2.dtype)
        c1 = x1[..., self.conc_dims]
        c2 = x2[..., self.conc_dims]

        # essentials part
        ess_diff = (e1.unsqueeze(-2) - e2.unsqueeze(-3)) / self.ess_lengthscale
        k_ess = self.ess_outputscale * torch.exp(-0.5 * ess_diff.pow(2).sum(dim=-1))

        G = self.G  # A x A

        if diag:
            # Compute k(x_i, x_i) only -- assumes x1 == x2
            c1k = c1.unsqueeze(-1)              # (..., n, A, 1)
            c1l = c1.unsqueeze(-2)              # (..., n, 1, A)
            conc_diff_diag = c1k - c1l          # (..., n, A, A)
            k_conc_diag = torch.exp(
                -0.5 * (conc_diff_diag / self.conc_lengthscale).pow(2)
            )
            mask_diag = b1.unsqueeze(-1) * b1.unsqueeze(-2)  # (..., n, A, A)
            additive_diag = (mask_diag * k_conc_diag * G).sum(dim=(-2, -1))
            ess_diag = self.ess_outputscale.expand_as(additive_diag)
            return ess_diag + self.baseline_var + additive_diag

        # Full kernel: (..., n1, n2)
        c1_e = c1.unsqueeze(-2).unsqueeze(-1)   # (..., n1, 1, A, 1)
        c2_e = c2.unsqueeze(-3).unsqueeze(-2)   # (..., 1, n2, 1, A)
        conc_diff = c1_e - c2_e                  # (..., n1, n2, A, A)
        k_conc = torch.exp(-0.5 * (conc_diff / self.conc_lengthscale).pow(2))

        b1_e = b1.unsqueeze(-2).unsqueeze(-1)
        b2_e = b2.unsqueeze(-3).unsqueeze(-2)
        mask = b1_e * b2_e                       # (..., n1, n2, A, A)

        weighted = mask * k_conc * G             # G broadcasts as (1, 1, A, A)
        additive_term = weighted.sum(dim=(-2, -1))

        return k_ess + self.baseline_var + additive_term


# ===================================================================
# Model builder, fit, diagnostics
# ===================================================================
def make_famt_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    codec: FlatCodec,
    bounds: torch.Tensor,
    descriptors: torch.Tensor,
    latent_rank: int = 4,
) -> SingleTaskGP:
    covar = FAMTKernel(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
        descriptors=descriptors,
        latent_rank=latent_rank,
    )
    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        covar_module=covar,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def fit_famt(model: SingleTaskGP, maxiter: int) -> dict[str, Any]:
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


def famt_kernel_parameter_summary(
    model: SingleTaskGP, additive_names: list[str]
) -> dict[str, Any]:
    covar = model.covar_module
    if not isinstance(covar, FAMTKernel):
        return {}
    out: dict[str, Any] = {}
    with torch.no_grad():
        out["famt_ess_outputscale"] = float(covar.ess_outputscale.item())
        out["famt_ess_lengthscale"] = [
            float(v) for v in covar.ess_lengthscale.detach().reshape(-1)
        ]
        out["famt_baseline_var"] = float(covar.baseline_var.item())
        out["famt_tau"] = float(covar.tau.item())
        out["famt_lambdas"] = {
            name: float(v) for name, v in zip(additive_names, covar.lambdas.detach())
        }
        out["famt_sigma_j"] = {
            name: float(v) for name, v in zip(additive_names, covar.sigma_j.detach())
        }
        out["famt_lambda_E"] = float(covar.lambda_E.item())
        out["famt_F_norm"] = float(covar.F.detach().norm().item())
        out["famt_E_raw_norm"] = float(covar.E_raw.detach().norm().item())
        out["famt_conc_lengthscale"] = float(covar.conc_lengthscale.item())
        G = covar.G.detach()
        out["famt_G_diag"] = {
            name: float(v) for name, v in zip(additive_names, G.diagonal())
        }
        d = G.diagonal().clamp_min(1e-12).sqrt()
        G_corr = (G / (d.unsqueeze(-1) * d.unsqueeze(-2))).cpu().numpy()
        # off-diagonal stats
        n = G.shape[0]
        offdiag = G_corr - np.eye(n)
        out["famt_G_corr_offdiag_min"] = float(offdiag.min())
        out["famt_G_corr_offdiag_max"] = float(offdiag.max())
        out["famt_G_corr_offdiag_mean"] = float(
            offdiag.sum() / (n * (n - 1))
        )
    return out


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seeds(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, codec = load_training_data(args)
    X, Y, encode_meta = encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
    descriptors, descriptor_keys = build_descriptor_matrix(codec.adds)
    descriptors = descriptors.to(device=X.device, dtype=X.dtype)

    pool = None
    if args.acq_stability_diagnostics:
        pool = make_candidate_pool(
            codec=codec,
            bounds=bounds,
            n=args.acq_pool_size,
            k_max=args.k_max,
            seed=args.seed + 123,
        )

    print(
        "[Data]",
        json.dumps(
            {**dataset_summary(df, codec, args.target), **encode_meta},
            indent=2,
        ),
    )
    print(
        f"[Descriptors] A={len(codec.adds)} additives, "
        f"p={descriptors.shape[1]} features, r={args.latent_rank} latent rank"
    )

    print("\n=== Fitting FAMT ===")
    model = make_famt_model(
        X, Y, codec, bounds, descriptors, latent_rank=args.latent_rank
    )
    fit_info = fit_famt(model, maxiter=args.fit_maxiter)
    print(
        f"  fit seconds: {fit_info['fit_seconds']:.1f}, "
        f"MLL (per point, standardized): {fit_info['mll']:.4f}"
    )

    diagnostics: dict[str, Any] = {}
    diagnostics.update(fit_info)
    if args.training_fit_diagnostics:
        diagnostics.update(training_fit_diagnostics(model, X, Y))
    if args.matrix_diagnostics:
        diagnostics.update(matrix_diagnostics(model, X))
    if args.loo_diagnostics:
        diagnostics.update(loo_diagnostics(model, X))
    if args.kernel_parameter_summary:
        diagnostics.update(famt_kernel_parameter_summary(model, codec.adds))
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

    out_json = output_dir / "add_bo_famt_diagnostics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "data": {**dataset_summary(df, codec, args.target), **encode_meta},
                "descriptor_keys": descriptor_keys,
                "models": {"famt": diagnostics},
            },
            f,
            indent=2,
            default=str,
        )
    print(f"[Output] diagnostics JSON -> {out_json.resolve()}")

    # Compact terminal summary
    print("\n[FAMT summary]")
    if "famt_sigma_j" in diagnostics:
        sj = diagnostics["famt_sigma_j"]
        ordered = sorted(sj.items(), key=lambda kv: -kv[1])
        print("  top per-additive sigma_j:")
        for name, val in ordered[:8]:
            print(f"    {name:12s}  sigma_j = {val:.4f}")
        print("  bottom per-additive sigma_j:")
        for name, val in ordered[-5:]:
            print(f"    {name:12s}  sigma_j = {val:.4f}")
    keys = (
        "famt_baseline_var", "famt_tau", "famt_lambda_E",
        "famt_conc_lengthscale", "famt_ess_outputscale",
        "famt_F_norm", "famt_E_raw_norm",
        "famt_G_corr_offdiag_mean", "famt_G_corr_offdiag_max",
        "famt_G_corr_offdiag_min",
    )
    for k in keys:
        if k in diagnostics:
            v = diagnostics[k]
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    print(
        f"  MLL: {diagnostics['mll']:.4f}"
        f"   LOO RMSE: {diagnostics.get('loo_rmse_model_space', float('nan')):.4f}"
        f"   train R2: {diagnostics.get('train_r2', float('nan')):.4f}"
    )
    return diagnostics


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit the FAMT kernel and run diagnostics."
    )
    data = p.add_argument_group("Data input and filtering")
    data.add_argument("--input", type=str, default="data_corrected.xlsx")
    data.add_argument("--sheet", type=str, default="Corrected Data")
    data.add_argument("--target", type=str, default="intensity")
    data.add_argument("--hrp", type=float, default=0.0001)
    data.add_argument("--hrp_atol", type=float, default=1e-12)
    data.add_argument(
        "--filter_ctrl", action=argparse.BooleanOptionalAction, default=True
    )

    rc = p.add_argument_group("Run control")
    rc.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    rc.add_argument("--seed", type=int, default=42)
    rc.add_argument(
        "--output_dir", type=str,
        default=str(LOGS_DIR / "May_13_full_log" / "three_kernel_comparison" / "famt_diag"),
    )

    ss = p.add_argument_group("Search-space / encoding")
    ss.add_argument("--k_max", type=int, default=4)

    fit_g = p.add_argument_group("FAMT model fitting")
    fit_g.add_argument("--fit_maxiter", type=int, default=200)
    fit_g.add_argument("--latent_rank", type=int, default=4)

    diag_g = p.add_argument_group("Model diagnostics")
    diag_g.add_argument(
        "--training_fit_diagnostics",
        action=argparse.BooleanOptionalAction, default=True,
    )
    diag_g.add_argument(
        "--matrix_diagnostics",
        action=argparse.BooleanOptionalAction, default=True,
    )
    diag_g.add_argument(
        "--loo_diagnostics",
        action=argparse.BooleanOptionalAction, default=True,
    )
    diag_g.add_argument(
        "--kernel_parameter_summary",
        action=argparse.BooleanOptionalAction, default=True,
    )

    stab = p.add_argument_group("Acquisition stability diagnostics")
    stab.add_argument(
        "--acq_stability_diagnostics",
        action=argparse.BooleanOptionalAction, default=True,
    )
    stab.add_argument("--stability_mc_samples", type=int, default=64)
    stab.add_argument("--stability_repeats", type=int, default=5)
    stab.add_argument("--acq_pool_size", type=int, default=512)
    stab.add_argument("--eval_batch_size", type=int, default=256)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
