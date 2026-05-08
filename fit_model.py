# -*- coding: utf-8 -*-
import torch
import numpy as np

from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.fit import fit_fully_bayesian_model_nuts

from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from botorch.models.fully_bayesian import FullyBayesianSingleTaskGP

from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models.map_saas import add_saas_prior

from botorch.models import MixedSingleTaskGP

from math import log, sqrt
from gpytorch.kernels import MaternKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan

SQRT2 = sqrt(2)
SQRT3 = sqrt(3)

def matern_with_hvarfner_prior(ard_num_dims: int, nu: float = 1.5):
    ls_prior = LogNormalPrior(loc=SQRT2 + log(ard_num_dims) * 0.5, scale=SQRT3)
    base_kernel = MaternKernel(
        nu=nu,
        ard_num_dims=ard_num_dims,
        lengthscale_prior=ls_prior,
        lengthscale_constraint=GreaterThan(
            2.5e-2, transform=None, initial_value=ls_prior.mode
        ),
    )
    return base_kernel


def _set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    try:
        import pyro
        pyro.set_rng_seed(seed)
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_bounds_from_train(X_train: torch.Tensor) -> torch.Tensor:
    lb = X_train.min(dim=0).values.clone()
    ub = X_train.max(dim=0).values.clone()
    same = ub <= lb
    if same.any():
        print("SINGULARITY")
        ub[same] = lb[same] + 1e-6
    return torch.stack([lb, ub])


def fit_gp(
        X_tr: torch.Tensor,
        y_tr: torch.Tensor,
        seed: int,
        cat_dims: list[int] | None = None,
        bounds: torch.Tensor | None = None,
):
    _set_seeds(seed)
    dev = X_tr.device
    d = X_tr.shape[1]

    if bounds is None:
        bounds = _make_bounds_from_train(X_tr)

    if cat_dims and len(cat_dims) > 0:
        print(f"  -> Using MixedSingleTaskGP for categorical dimensions: {cat_dims}")

        cont_dims = [i for i in range(d) if i not in cat_dims]

        model = MixedSingleTaskGP(
            train_X=X_tr,
            train_Y=y_tr,
            cat_dims=cat_dims,
            input_transform=Normalize(d=d, bounds=bounds, indices=cont_dims),
            outcome_transform=Standardize(m=1),
        ).to(dev)
    else:
        print("  -> Using standard SingleTaskGP (all features are continuous).")
        model = SingleTaskGP(
            train_X=X_tr,
            train_Y=y_tr,
            covar_module=matern_with_hvarfner_prior(d, nu=0.5),
            input_transform=Normalize(d=d, bounds=bounds),
            outcome_transform=Standardize(m=1),
        ).to(dev)

    mll = ExactMarginalLogLikelihood(model.likelihood, model).to(dev)
    fit_gpytorch_mll(mll)
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_saasbo(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    warmup: int, num_samples: int, thinning: int, seed: int
):
    _set_seeds(seed)
    dev = X_tr.device
    bounds_raw = _make_bounds_from_train(X_tr)
    model = SaasFullyBayesianSingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        input_transform=Normalize(d=X_tr.shape[1], bounds=bounds_raw),
        outcome_transform=Standardize(m=1),
    ).to(dev)
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=warmup,
        num_samples=num_samples,
        thinning=thinning,
        disable_progbar=True,
    )
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_fullyb_gp(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    warmup: int, num_samples: int, thinning: int, seed: int
):
    _set_seeds(seed)
    dev = X_tr.device
    d = X_tr.shape[1]
    model = FullyBayesianSingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        input_transform=Normalize(d=d, bounds=_make_bounds_from_train(X_tr)),
        outcome_transform=Standardize(m=1),
    ).to(dev)
    fit_fully_bayesian_model_nuts(
        model,
        warmup_steps=warmup,
        num_samples=num_samples,
        thinning=thinning,
        disable_progbar=False,
    )
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model


def fit_saas_gp(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    seed: int,
    nu: float = 2.5,
):
    _set_seeds(seed)
    dev = X_tr.device
    d = X_tr.shape[1]
    input_tf  = Normalize(d=d, bounds=_make_bounds_from_train(X_tr))
    outcome_tf = Standardize(m=1)
    base = MaternKernel(nu=nu, ard_num_dims=d)
    add_saas_prior(base_kernel=base, tau=None, log_scale=True)
    covar = ScaleKernel(base)
    model = SingleTaskGP(
        train_X=X_tr,
        train_Y=y_tr,
        covar_module=covar,
        input_transform=input_tf,
        outcome_transform=outcome_tf,
    ).to(dev)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    _ = model.posterior(X_tr[: min(5, X_tr.shape[0])], observation_noise=True)
    return model

