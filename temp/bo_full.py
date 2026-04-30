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
from botorch.models import SingleTaskGP
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


CONC_LO = 0.001
CONC_HI = 1.0
LOG_LO = math.log10(CONC_LO)   # -3.0
LOG_HI = math.log10(CONC_HI)   #  0.0
# Default concentration for inactive additives (used as placeholder;
# the binary indicator handles presence/absence semantics).
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
        for n_active in range(k + 1):
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

    # ---- Acquisition Function ----
    def _make_acqf(self, acq_type: str, mc_samples: int = 256, **opts: Any):
        t = acq_type.lower()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        best_f = self.y.max().item()

        if t in ("qlognei", "lognei"):
            return qLogNoisyExpectedImprovement(
                self.model, X_baseline=self.Z, sampler=sampler,
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
        num_restarts: int = 20,
        raw_samples: int = 512,
        acq_types: List[str] | None = None,
        acq_options: Dict | None = None,
    ) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:

        set_seeds(self.seed)
        acq_types = acq_types or ["qlognei"]
        acq_options = acq_options or {}
        q_per_acq = max(1, math.ceil(q / len(acq_types)))

        all_fixed_features_list = self.codec.build_fixed_features_list()
        print(f"\n[Info] Total discrete subspaces: {len(all_fixed_features_list)}")
        print(f"[Info] Target candidates per acq fn: {q_per_acq}")

        all_candidates_tensors: List[torch.Tensor] = []
        train_Z_np = self.Z.detach().cpu().numpy()
        occupied_keys = set(map(tuple, np.round(train_Z_np, 6)))

        for acq_name in acq_types:
            acqf_screen = self._make_acqf(acq_name, mc_samples=64, **acq_options)
            acqf_refine = self._make_acqf(acq_name, mc_samples=256, **acq_options)

            num_top_subspaces = 20
            print(f"\n=== Acq: {acq_name}, generating {q_per_acq} candidates (greedy) ===")

            base_per_axis = 2
            eval_batch_size = 32
            lb = self.bounds[0]
            ub = self.bounds[1]
            d = lb.numel()

            def _prescreen(
                acqf, pending_step: int
            ) -> Tuple[List[Dict[int, float]], List[torch.Tensor], List[float]]:
                best_val_per_space = torch.empty(
                    len(all_fixed_features_list),
                    device=self.device, dtype=torch.double,
                )
                best_X_list: List[torch.Tensor] = []
                best_afv_list: List[float] = []
                sample_stats: Dict[int, int] = defaultdict(int)

                for si, fixed in enumerate(all_fixed_features_list):
                    # Count free (optimizable) dims to scale sample count
                    n_free = d - len(fixed)
                    min_samples = int(math.ceil(base_per_axis ** 2))
                    max_samples = int(math.ceil(base_per_axis ** 6))
                    num_samples = int(np.clip(
                        math.ceil(base_per_axis ** n_free),
                        min_samples, max_samples,
                    ))
                    sample_stats[n_free] += num_samples

                    X = lb + (ub - lb) * torch.rand(
                        num_samples, d,
                        device=self.device, dtype=torch.double,
                    )
                    for col_idx, val in fixed.items():
                        X[:, col_idx] = val

                    cur_best_val = -float("inf")
                    cur_best_x = None
                    with torch.no_grad():
                        for i in range(0, num_samples, eval_batch_size):
                            Xb = X[i: i + eval_batch_size]
                            vb = acqf(Xb.unsqueeze(1)).view(-1)
                            mv, mi = torch.max(vb, dim=0)
                            if mv.item() > cur_best_val:
                                cur_best_val = mv.item()
                                cur_best_x = Xb[mi].clone()

                    best_val_per_space[si] = cur_best_val
                    best_X_list.append(
                        cur_best_x if cur_best_x is not None else X[0]
                    )
                    best_afv_list.append(
                        float(cur_best_val) if cur_best_x is not None
                        else float("-inf")
                    )

                print(
                    f"  [Prescreen][step={pending_step}] "
                    f"Total samples: {sum(sample_stats.values())} | "
                    f"Breakdown (n_free: samples): "
                    f"{dict(sorted(sample_stats.items()))}"
                )

                k_top = min(num_top_subspaces, len(all_fixed_features_list))
                top_vals, top_idx = torch.topk(best_val_per_space, k=k_top)
                top_fixed = [
                    all_fixed_features_list[i] for i in top_idx.tolist()
                ]
                print(
                    f"  [Prescreen][step={pending_step}] "
                    f"Top score: {top_vals[0].item():.4f}"
                )
                return top_fixed, best_X_list, best_afv_list

            base_X_pending_screen = acqf_screen.X_pending
            base_X_pending_refine = acqf_refine.X_pending
            pending = base_X_pending_refine
            step_candidates: List[torch.Tensor] = []

            for step in range(q_per_acq):
                acqf_screen.set_X_pending(pending)
                acqf_refine.set_X_pending(pending)

                top_fixed, best_X_list, best_afv_list = _prescreen(
                    acqf_screen, pending_step=step,
                )

                if step == 0 and best_X_list:
                    self._analyze_additives(
                        best_X_list, best_afv_list, acq_name,
                    )

                max_attempts = 5
                accepted = False
                for attempt in range(max_attempts):
                    candidate_raw, _ = optimize_acqf_mixed(
                        acq_function=acqf_refine,
                        bounds=self.bounds,
                        q=1,
                        fixed_features_list=top_fixed,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options={
                            "batch_limit": num_restarts,
                            "maxiter": 200,
                        },
                        retry_on_optimization_warning=True,
                    )
                    candidate = self.codec.postprocess_batch(
                        candidate_raw, self.bounds,
                    )
                    cand_key = tuple(
                        np.round(candidate.detach().cpu().numpy(), 6).ravel()
                    )
                    if cand_key not in occupied_keys:
                        occupied_keys.add(cand_key)
                        step_candidates.append(candidate)
                        pending = (
                            candidate if pending is None
                            else torch.cat([pending, candidate], dim=-2)
                        )
                        accepted = True
                        break
                    else:
                        set_seeds(
                            self.seed + 1000 * (step + 1) + attempt + 1
                        )

                if not accepted:
                    print(
                        f"  [Warning] step={step}: "
                        f"no non-dup after {max_attempts} attempts."
                    )

            print(f"  Done. Generated {len(step_candidates)} candidates.")
            if step_candidates:
                all_candidates_tensors.append(
                    torch.cat(step_candidates, dim=0)
                )

            acqf_screen.set_X_pending(base_X_pending_screen)
            acqf_refine.set_X_pending(base_X_pending_refine)

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

    def _analyze_additives(
        self,
        best_X_list: List[torch.Tensor],
        best_afv_list: List[float],
        acq_name: str,
    ):
        print(
            f"  [Analysis] AFV distribution per additive "
            f"across {len(best_X_list)} subspaces..."
        )
        all_best_X = torch.stack(best_X_list)
        all_best_rows = self.codec.decode(all_best_X.cpu().numpy())

        add2afv: Dict[str, List[float]] = defaultdict(list)
        for row, afv in zip(all_best_rows, best_afv_list):
            for name in self.adds:
                if row.get(name, 0.0) > EPS:
                    add2afv[name].append(afv)

        if not add2afv:
            print("  [Analysis] No additives in best candidates.")
            return

        stats = []
        for name, vals in add2afv.items():
            a = np.asarray(vals, dtype=float)
            stats.append({
                "additive": name, "n": int(a.size),
                "mean": float(np.mean(a)), "std": float(np.std(a)),
            })
        stats_df = pd.DataFrame(stats).sort_values(
            by=["mean", "n"], ascending=[False, False],
        ).reset_index(drop=True)

        try:
            labels = stats_df["additive"].tolist()
            means_ = stats_df["mean"].to_numpy()
            stds_ = stats_df["std"].to_numpy()

            plt.figure(figsize=(12, 6))
            plt.bar(labels, means_, edgecolor="black", alpha=0.8)
            plt.errorbar(
                np.arange(len(labels)), means_, yerr=stds_,
                fmt="none", ecolor="black", capsize=3, linewidth=1,
            )
            plt.title(
                f"Additive AFV ({len(best_X_list)} subspaces) | Acq={acq_name}"
            )
            plt.xlabel("Additive")
            plt.ylabel("Acquisition Function Value")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig_path = f"prescreen_{acq_name}_additive_afv.png"
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"  [Analysis] Saved plot -> {fig_path}")

            csv_path = f"prescreen_{acq_name}_additive_afv_stats.csv"
            stats_df.to_csv(csv_path, index=False, encoding="utf-8")
            print(f"  [Analysis] Saved stats -> {csv_path}")
        except Exception as e:
            print(f"  [Warning] Plotting failed: {e}")


def _setup_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BO with hybrid binary+log10 encoding and MixedGP"
    )
    p.add_argument("--input", type=str, default="data.xlsx")
    p.add_argument("--target", type=str, default="intensity")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
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