"""Compare the original mixed GP prior with a hierarchical family-aware prior.

This module reuses the data loading, diagnostics, and candidate-generation flow
from ``add_bo.py`` and swaps in a new model that encodes two beliefs more
explicitly:

1. Similar additives should share statistical strength through family-aware
   descriptors instead of only through exact additive identity.
2. Higher-cardinality recipes should primarily inherit signal through explicit
   singleton and pairwise contributions, with only a small residual GP left to
   interpolate what the mean structure misses.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from gpytorch.constraints import Positive
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from gpytorch.mlls import ExactMarginalLogLikelihood

from add_bo import (
    ESSENTIALS,
    FEATURE_WHITELIST,
    AdditiveSetKernel,
    FlatCodec,
    acq_stability_diagnostics,
    dataset_summary,
    encode_training_data,
    ensure_2d_y,
    generate_candidates,
    kernel_parameter_summary as legacy_kernel_parameter_summary,
    load_training_data,
    make_candidate_pool,
    matrix_diagnostics,
    set_seeds,
    training_fit_diagnostics,
    loo_diagnostics,
)

torch.set_default_dtype(torch.double)


ADDITIVE_METADATA: dict[str, dict[str, float]] = {
    "cmc": {
        "is_peg": 0.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(90000.0),
    },
    "peg20k": {
        "is_peg": 1.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(20000.0),
    },
    "dmso": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 1.0,
        "log_mw": math.log10(78.13),
    },
    "pl127": {
        "is_peg": 0.0,
        "is_polymer": 1.0,
        "is_surfactant": 1.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(12600.0),
    },
    "bsa": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 1.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(66430.0),
    },
    "pva": {
        "is_peg": 0.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(31000.0),
    },
    "tw80": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 1.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(1310.0),
    },
    "glycerol": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 1.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(92.09),
    },
    "tw20": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 1.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(1228.0),
    },
    "imidazole": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 1.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(68.08),
    },
    "tx100": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 1.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(647.0),
    },
    "edta": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 1.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(292.24),
    },
    "mgso4": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 1.0,
        "is_chloride": 0.0,
        "is_sulfate": 1.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(120.37),
    },
    "sucrose": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 1.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(342.30),
    },
    "cacl2": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 1.0,
        "is_chloride": 1.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(110.98),
    },
    "znso4": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 1.0,
        "is_chloride": 0.0,
        "is_sulfate": 1.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(161.44),
    },
    "paa": {
        "is_peg": 0.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(100000.0),
    },
    "mncl2": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 1.0,
        "is_chloride": 1.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(125.84),
    },
    "peg200k": {
        "is_peg": 1.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(200000.0),
    },
    "feso4": {
        "is_peg": 0.0,
        "is_polymer": 0.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 1.0,
        "is_chloride": 0.0,
        "is_sulfate": 1.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(151.91),
    },
    "peg6k": {
        "is_peg": 1.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(6000.0),
    },
    "peg400": {
        "is_peg": 1.0,
        "is_polymer": 1.0,
        "is_surfactant": 0.0,
        "is_polyol_sugar": 0.0,
        "is_protein": 0.0,
        "is_salt": 0.0,
        "is_chloride": 0.0,
        "is_sulfate": 0.0,
        "is_chelator_or_buffer": 0.0,
        "is_solvent": 0.0,
        "log_mw": math.log10(400.0),
    },
}


def build_additive_descriptor_matrix(additives: list[str]) -> tuple[torch.Tensor, list[str]]:
    keys = [
        "is_peg",
        "is_polymer",
        "is_surfactant",
        "is_polyol_sugar",
        "is_protein",
        "is_salt",
        "is_chloride",
        "is_sulfate",
        "is_chelator_or_buffer",
        "is_solvent",
        "log_mw",
    ]
    rows: list[list[float]] = []
    for name in additives:
        if name not in ADDITIVE_METADATA:
            raise KeyError(f"Missing additive metadata for '{name}'.")
        rows.append([float(ADDITIVE_METADATA[name][k]) for k in keys])
    arr = np.asarray(rows, dtype=np.float64)
    log_idx = keys.index("log_mw")
    log_vals = arr[:, log_idx]
    arr[:, log_idx] = (log_vals - log_vals.mean()) / max(log_vals.std(ddof=0), 1e-8)
    return torch.tensor(arr, dtype=torch.double), keys


class HierarchicalFamilyMean(Mean):
    """Explicit singleton + pairwise recipe mean with family-aware sharing."""

    def __init__(
        self,
        ess_dims: list[int],
        cat_dims: list[int],
        conc_dims: list[int],
        descriptors: torch.Tensor,
        pair_rank: int = 4,
    ) -> None:
        super().__init__()
        self.register_buffer("ess_dims", torch.tensor(ess_dims, dtype=torch.long))
        self.register_buffer("cat_dims", torch.tensor(cat_dims, dtype=torch.long))
        self.register_buffer("conc_dims", torch.tensor(conc_dims, dtype=torch.long))
        self.register_buffer("descriptors", descriptors)
        self.A = len(cat_dims)
        self.desc_dim = int(descriptors.shape[-1])
        self.single_basis_dim = 3
        self.pair_basis_dim = 3

        self.register_parameter(
            "ess_weights",
            torch.nn.Parameter(torch.zeros(6, dtype=torch.double)),
        )
        self.register_parameter(
            "family_single_weights",
            torch.nn.Parameter(
                torch.zeros(self.desc_dim, self.single_basis_dim, dtype=torch.double)
            ),
        )
        self.register_parameter(
            "id_single_weights",
            torch.nn.Parameter(
                torch.zeros(self.A, self.single_basis_dim, dtype=torch.double)
            ),
        )
        self.register_parameter(
            "family_pair_proj",
            torch.nn.Parameter(
                torch.zeros(self.desc_dim, pair_rank, dtype=torch.double)
            ),
        )
        self.register_parameter(
            "id_pair_proj",
            torch.nn.Parameter(torch.zeros(self.A, pair_rank, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_pair_env_weights",
            torch.nn.Parameter(torch.zeros(self.pair_basis_dim, dtype=torch.double)),
        )
        self.register_constraint("raw_pair_env_weights", Positive())
        self.initialize(
            family_single_weights=0.01 * torch.randn_like(self.family_single_weights),
            id_single_weights=0.01 * torch.randn_like(self.id_single_weights),
            family_pair_proj=0.01 * torch.randn_like(self.family_pair_proj),
            id_pair_proj=0.01 * torch.randn_like(self.id_pair_proj),
            raw_pair_env_weights=self.raw_pair_env_weights_constraint.inverse_transform(
                torch.tensor([0.10, 0.10, 0.05], dtype=torch.double)
            ),
        )

    @property
    def pair_env_weights(self) -> torch.Tensor:
        return self.raw_pair_env_weights_constraint.transform(self.raw_pair_env_weights)

    def _ess_features(self, e: torch.Tensor) -> torch.Tensor:
        e1 = e[..., 0]
        e2 = e[..., 1]
        return torch.stack(
            [
                torch.ones_like(e1),
                e1,
                e2,
                e1 * e2,
                e1.pow(2),
                e2.pow(2),
            ],
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = x[..., self.ess_dims]
        b = (x[..., self.cat_dims] > 0.5).to(dtype=x.dtype)
        c = x[..., self.conc_dims]

        ess_term = torch.einsum("...f,f->...", self._ess_features(e), self.ess_weights)

        single_basis = torch.stack(
            [
                torch.ones_like(c),
                c,
                c.pow(2),
            ],
            dim=-1,
        )
        single_coeff = self.descriptors @ self.family_single_weights + self.id_single_weights
        single_term = (
            b.unsqueeze(-1)
            * single_basis
            * single_coeff.view(*([1] * (x.ndim - 1)), self.A, self.single_basis_dim)
        ).sum(dim=(-2, -1))

        pair_emb = self.descriptors @ self.family_pair_proj + self.id_pair_proj
        pair_scores = pair_emb @ pair_emb.transpose(-1, -2)
        avg_c = 0.5 * (c.unsqueeze(-1) + c.unsqueeze(-2))
        pair_basis = torch.stack(
            [
                torch.ones_like(avg_c),
                avg_c,
                avg_c.pow(2),
            ],
            dim=-1,
        )
        pair_env = torch.einsum("...ijm,m->...ij", pair_basis, self.pair_env_weights)
        active_pair = b.unsqueeze(-1) * b.unsqueeze(-2)
        pair_term = torch.triu(active_pair * pair_env * pair_scores, diagonal=1).sum(
            dim=(-2, -1)
        )
        return ess_term + single_term + pair_term


class FamilyAwareResidualKernel(Kernel):
    """Small residual GP over essentials and family-aware active-set similarity."""

    has_lengthscale = False

    def __init__(
        self,
        ess_dims: list[int],
        cat_dims: list[int],
        conc_dims: list[int],
        descriptors: torch.Tensor,
        descriptor_keys: list[str],
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("ess_dims", torch.tensor(ess_dims, dtype=torch.long))
        self.register_buffer("cat_dims", torch.tensor(cat_dims, dtype=torch.long))
        self.register_buffer("conc_dims", torch.tensor(conc_dims, dtype=torch.long))
        self.register_buffer("descriptors", descriptors)
        self.register_buffer(
            "identity_gram", torch.eye(len(cat_dims), dtype=torch.double)
        )

        log_idx = descriptor_keys.index("log_mw")
        desc_no_mw = descriptors.clone()
        desc_no_mw[:, log_idx] = 0.0
        desc_norm = desc_no_mw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        family_cos = (desc_no_mw / desc_norm) @ (desc_no_mw / desc_norm).transpose(-1, -2)
        self.register_buffer("family_cos_gram", family_cos)

        log_mw = descriptors[:, log_idx : log_idx + 1]
        mw_diff = log_mw - log_mw.transpose(-1, -2)
        self.register_buffer("mw_sqdist", mw_diff.pow(2))

        self.register_parameter(
            "raw_scales",
            torch.nn.Parameter(torch.zeros(5, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_ess_lengthscale",
            torch.nn.Parameter(torch.zeros(len(ess_dims), dtype=torch.double)),
        )
        self.register_parameter(
            "raw_same_conc_lengthscale",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_parameter(
            "raw_mw_lengthscale",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.double)),
        )
        self.register_constraint("raw_scales", Positive())
        self.register_constraint("raw_ess_lengthscale", Positive())
        self.register_constraint("raw_same_conc_lengthscale", Positive())
        self.register_constraint("raw_mw_lengthscale", Positive())
        self.initialize(
            raw_scales=self.raw_scales_constraint.inverse_transform(
                torch.tensor([0.05, 0.25, 0.25, 0.15, 0.10], dtype=torch.double)
            ),
            raw_ess_lengthscale=self.raw_ess_lengthscale_constraint.inverse_transform(
                torch.full((len(ess_dims),), 0.35, dtype=torch.double)
            ),
            raw_same_conc_lengthscale=self.raw_same_conc_lengthscale_constraint.inverse_transform(
                torch.tensor([0.25], dtype=torch.double)
            ),
            raw_mw_lengthscale=self.raw_mw_lengthscale_constraint.inverse_transform(
                torch.tensor([0.75], dtype=torch.double)
            ),
        )

    @property
    def scales(self) -> torch.Tensor:
        return self.raw_scales_constraint.transform(self.raw_scales)

    @property
    def ess_lengthscale(self) -> torch.Tensor:
        return self.raw_ess_lengthscale_constraint.transform(self.raw_ess_lengthscale)

    @property
    def same_conc_lengthscale(self) -> torch.Tensor:
        return self.raw_same_conc_lengthscale_constraint.transform(
            self.raw_same_conc_lengthscale
        )

    @property
    def mw_lengthscale(self) -> torch.Tensor:
        return self.raw_mw_lengthscale_constraint.transform(self.raw_mw_lengthscale)

    def _ess_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        e1 = x1[..., self.ess_dims]
        e2 = x2[..., self.ess_dims]
        diff = (e1.unsqueeze(-2) - e2.unsqueeze(-3)) / self.ess_lengthscale
        return torch.exp(-0.5 * diff.pow(2).sum(dim=-1))

    def _additive_gram(self) -> torch.Tensor:
        s_const, s_id, s_family, s_mw, _ = self.scales
        mw_kernel = torch.exp(
            -0.5 * self.mw_sqdist / self.mw_lengthscale.pow(2).clamp_min(1e-12)
        )
        return (
            s_id * self.identity_gram
            + s_family * self.family_cos_gram
            + s_mw * mw_kernel
            + s_const * torch.ones_like(self.identity_gram)
        )

    def _family_activation_kernel(
        self, b1: torch.Tensor, b2: torch.Tensor, gram: torch.Tensor
    ) -> torch.Tensor:
        num = torch.einsum("...a,ab,...b->...", b1.unsqueeze(-2), gram, b2.unsqueeze(-3))
        self1 = torch.einsum("...a,ab,...b->...", b1, gram, b1).clamp_min(self.eps)
        self2 = torch.einsum("...a,ab,...b->...", b2, gram, b2).clamp_min(self.eps)
        denom = self1.unsqueeze(-1).sqrt() * self2.unsqueeze(-2).sqrt()
        k = num / denom.clamp_min(self.eps)
        empty1 = b1.sum(dim=-1) < 0.5
        empty2 = b2.sum(dim=-1) < 0.5
        both_empty = empty1.unsqueeze(-1) & empty2.unsqueeze(-2)
        k = torch.where(both_empty, torch.ones_like(k), k)
        return k

    def _same_additive_conc_kernel(self, b1: torch.Tensor, c1: torch.Tensor, b2: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        shared = b1.unsqueeze(-2) * b2.unsqueeze(-3)
        conc_diff = (c1.unsqueeze(-2) - c2.unsqueeze(-3)) / self.same_conc_lengthscale
        same = (shared * torch.exp(-0.5 * conc_diff.pow(2))).sum(dim=-1)
        norm1 = b1.sum(dim=-1).clamp_min(1.0)
        norm2 = b2.sum(dim=-1).clamp_min(1.0)
        return same / (norm1.unsqueeze(-1) * norm2.unsqueeze(-2)).sqrt()

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> torch.Tensor:
        if last_dim_is_batch:
            raise NotImplementedError(
                "FamilyAwareResidualKernel does not support last_dim_is_batch."
            )
        b1 = (x1[..., self.cat_dims] > 0.5).to(dtype=x1.dtype)
        b2 = (x2[..., self.cat_dims] > 0.5).to(dtype=x2.dtype)
        c1 = x1[..., self.conc_dims]
        c2 = x2[..., self.conc_dims]
        gram = self._additive_gram()
        k_ess = self._ess_kernel(x1=x1, x2=x2)
        k_act = self._family_activation_kernel(b1=b1, b2=b2, gram=gram)
        k_same = self._same_additive_conc_kernel(b1=b1, c1=c1, b2=b2, c2=c2)
        s_const, _, _, _, s_same = self.scales
        cov = k_ess * (s_const + k_act + s_same * k_same)
        if diag:
            return torch.diagonal(cov, dim1=-2, dim2=-1)
        return cov


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


def make_hierarchical_family_model(
    X: torch.Tensor,
    Y: torch.Tensor,
    codec: FlatCodec,
    bounds: torch.Tensor,
) -> SingleTaskGP:
    descriptors, descriptor_keys = build_additive_descriptor_matrix(codec.adds)
    descriptors = descriptors.to(device=X.device, dtype=X.dtype)
    mean_module = HierarchicalFamilyMean(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
        descriptors=descriptors,
        pair_rank=4,
    )
    covar_module = FamilyAwareResidualKernel(
        ess_dims=list(range(codec.n_ess)),
        cat_dims=codec.cat_dims,
        conc_dims=codec.conc_dims,
        descriptors=descriptors,
        descriptor_keys=descriptor_keys,
    )
    return SingleTaskGP(
        train_X=X,
        train_Y=Y,
        mean_module=mean_module,
        covar_module=covar_module,
        input_transform=Normalize(d=codec.d, bounds=bounds, indices=codec.cont_dims),
        outcome_transform=Standardize(m=1),
    ).to(X)


def fit_model(model: SingleTaskGP, maxiter: int) -> dict[str, Any]:
    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(model.train_targets)
    fit_gpytorch_mll(
        mll,
        optimizer_kwargs={"options": {"maxiter": maxiter, "disp": False}},
    )
    model.train()
    model.likelihood.train()
    with torch.no_grad():
        try:
            mll_value = float(mll(model(*model.train_inputs), model.train_targets).item())
        except Exception:
            mll_value = float("nan")
    model.eval()
    return {"mll": mll_value}


def kernel_parameter_summary(model: SingleTaskGP) -> dict[str, Any]:
    covar = model.covar_module
    out: dict[str, Any] = {}
    if isinstance(covar, FamilyAwareResidualKernel):
        scales = covar.scales.detach().cpu().tolist()
        out.update(
            {
                "mod_scale_const": float(scales[0]),
                "mod_scale_id": float(scales[1]),
                "mod_scale_family": float(scales[2]),
                "mod_scale_mw": float(scales[3]),
                "mod_scale_same_conc": float(scales[4]),
                "mod_ess_lengthscale": [
                    float(v) for v in covar.ess_lengthscale.detach().cpu().reshape(-1)
                ],
                "mod_same_conc_lengthscale": float(
                    covar.same_conc_lengthscale.detach().cpu().item()
                ),
                "mod_mw_lengthscale": float(covar.mw_lengthscale.detach().cpu().item()),
            }
        )
        mean_module = model.mean_module
        if isinstance(mean_module, HierarchicalFamilyMean):
            out.update(
                {
                    "mod_mean_ess_weight_norm": float(
                        mean_module.ess_weights.detach().norm().cpu().item()
                    ),
                    "mod_mean_single_family_norm": float(
                        mean_module.family_single_weights.detach().norm().cpu().item()
                    ),
                    "mod_mean_single_id_norm": float(
                        mean_module.id_single_weights.detach().norm().cpu().item()
                    ),
                    "mod_mean_pair_family_norm": float(
                        mean_module.family_pair_proj.detach().norm().cpu().item()
                    ),
                    "mod_mean_pair_id_norm": float(
                        mean_module.id_pair_proj.detach().norm().cpu().item()
                    ),
                    "mod_mean_pair_env": [
                        float(v) for v in mean_module.pair_env_weights.detach().cpu()
                    ],
                }
            )
        return out
    return legacy_kernel_parameter_summary(model)


def run(args: argparse.Namespace) -> None:
    set_seeds(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, codec = load_training_data(args)
    X, Y, encode_meta = encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)
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
        ("hierarchical_family_prior", make_hierarchical_family_model),
    ]:
        print(f"\n=== Fitting {label} ===")
        model = builder(X, Y, codec, bounds)
        fit_info = fit_model(model, maxiter=args.fit_maxiter)
        diagnostics: dict[str, Any] = {}
        diagnostics.update(fit_info)
        diagnostics.update(training_fit_diagnostics(model, X, Y))
        diagnostics.update(matrix_diagnostics(model, X))
        diagnostics.update(loo_diagnostics(model, X))
        diagnostics.update(kernel_parameter_summary(model))
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

    summary_path = output_dir / "add_bo_mod_diagnostics.json"
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
    metrics_path = output_dir / "add_bo_mod_diagnostics.csv"
    pd.DataFrame(flat_rows).to_csv(metrics_path, index=False, encoding="utf-8")
    if candidate_frames:
        all_candidates_path = output_dir / "add_bo_mod_candidates_all.csv"
        pd.concat(candidate_frames, ignore_index=True).to_csv(
            all_candidates_path, index=False, encoding="utf-8"
        )
        print(f"[Output] candidates comparison -> {all_candidates_path.resolve()}")
    print(f"[Output] diagnostics JSON -> {summary_path.resolve()}")
    print(f"[Output] diagnostics CSV  -> {metrics_path.resolve()}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare the old mixed GP with a hierarchical family-aware prior."
    )
    p.add_argument("--input", type=str, default="data_corrected.xlsx")
    p.add_argument("--sheet", type=str, default="Corrected Data")
    p.add_argument("--target", type=str, default="intensity")
    p.add_argument("--hrp", type=float, default=0.0001)
    p.add_argument("--hrp_atol", type=float, default=1e-12)
    p.add_argument("--filter_ctrl", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_max", type=int, default=4)
    p.add_argument("--fit_maxiter", type=int, default=75)
    p.add_argument("--output_dir", type=str, default="add_bo_mod_compare")

    p.add_argument("--stability_mc_samples", type=int, default=64)
    p.add_argument("--stability_repeats", type=int, default=5)
    p.add_argument("--acq_pool_size", type=int, default=512)
    p.add_argument("--eval_batch_size", type=int, default=256)

    p.add_argument("--skip_candidates", action="store_true")
    p.add_argument("--q", type=int, default=32)
    p.add_argument("--num_restarts", type=int, default=10)
    p.add_argument("--raw_samples", type=int, default=256)
    p.add_argument("--acq_maxiter", type=int, default=100)
    p.add_argument("--num_top_subspaces", type=int, default=40)
    p.add_argument("--sobol_max_samples", type=int, default=512)
    p.add_argument("--screen_mc_samples", type=int, default=32)
    p.add_argument("--refine_mc_samples", type=int, default=64)
    return p


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
