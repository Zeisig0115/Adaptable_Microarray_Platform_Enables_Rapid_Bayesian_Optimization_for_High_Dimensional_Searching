from __future__ import annotations
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import math
from itertools import combinations, product
import time
import matplotlib.pyplot as plt
import inspect

import numpy as np
import pandas as pd
import torch

from fit_model import fit_gp

from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
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

ESSENTIALS = ["tmb", "hrp", "h2o2"]
DEFAULT_Q = 32
DEFAULT_OUT_CSV = "bo_candidates_slots_gp.csv"
EPS = 1e-7


def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_2d_y(y: np.ndarray | List[float]) -> np.ndarray:
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _canonicalize_pairs(pairs: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    valid_pairs = [(int(aid), float(np.clip(r, 0.0, 1.0))) for aid, r in pairs if int(aid) > 0]
    bucket: Dict[int, float] = {}
    for aid, r in valid_pairs:
        bucket[aid] = float(np.clip(bucket.get(aid, 0.0) + r, 0.0, 1.0))
    return sorted(bucket.items(), key=lambda x: x[0])


class SlotCodec:
    def __init__(self, essentials: List[str], additives: List[str], ranges: Dict[str, Tuple[float, float]],
                 k_slots: int):
        self.E = list(essentials)
        self.adds = list(additives)
        self.A = len(self.adds)
        self.k = int(k_slots)
        self.name2id = {name: i + 1 for i, name in enumerate(self.adds)}
        self.id2name = {i + 1: name for i, name in enumerate(self.adds)}
        self.ranges = dict(ranges)

    def encode_row(self, row: Dict[str, float]) -> np.ndarray:
        z = np.zeros(3 + 2 * self.k, dtype=np.float64)
        for i, e in enumerate(self.E):
            z[i] = row.get(e, 0.0)
        pairs = []
        for a in self.adds:
            v = row.get(a, 0.0)
            if v is not None and abs(v) > EPS:
                lo, hi = self.ranges.get(a, (0.0, 1.0))
                denom = (hi - lo) if hi > lo else 1e-6
                r = np.clip((v - lo) / denom, 0.0, 1.0)
                pairs.append((self.name2id[a], r))
        pairs = _canonicalize_pairs(pairs)[:self.k]
        pairs += [(0, 0.0)] * (self.k - len(pairs))
        for s, (aid, r) in enumerate(pairs):
            z[3 + 2 * s] = float(aid)
            z[3 + 2 * s + 1] = float(r)
        return z

    def encode(self, rows: List[Dict[str, float]]) -> np.ndarray:
        return np.array([self.encode_row(r) for r in rows])

    def decode(self, Z: np.ndarray) -> List[Dict[str, float]]:
        Z = np.atleast_2d(Z)
        decoded_rows: List[Dict[str, float]] = []
        for z in Z:
            row = {e: z[i] for i, e in enumerate(self.E)}
            pairs = []
            for s in range(self.k):
                aid = int(round(z[3 + 2 * s]))
                r = float(z[3 + 2 * s + 1])

                if aid > 0:
                    aid = int(np.clip(aid, 1, self.A))
                    pairs.append((aid, np.clip(r, 0.0, 1.0)))

            for aid, r in _canonicalize_pairs(pairs):
                name = self.id2name[aid]
                lo, hi = self.ranges[name]
                row[name] = lo + r * (hi - lo)
            decoded_rows.append(row)
        return decoded_rows


    def postprocess_batch(self, Zt: torch.Tensor) -> torch.Tensor:
        if Zt.dim() not in [2, 3]:
            raise ValueError(f"Input Tensor dims should be 2 or 3, but get{Zt.dim()}")
        Z = Zt.detach().cpu().clone()
        is_2d = Z.dim() == 2
        if is_2d: Z = Z.unsqueeze(0)
        B, q, d = Z.shape
        expected_d = 3 + 2 * self.k
        assert d == expected_d, f"Encoding dims mismatch: expected{expected_d}, got{d}"
        for b in range(B):
            for i in range(q):
                ids = Z[b, i, 3::2].round().clamp_(0, self.A)
                ratios = Z[b, i, 4::2].clamp_(0.0, 1.0)
                ratios.masked_fill_(ids == 0, 0.0)
                pairs = [(int(ids[s].item()), ratios[s].item()) for s in range(self.k)]
                pairs = _canonicalize_pairs(pairs)[:self.k]
                pairs += [(0, 0.0)] * (self.k - len(pairs))
                for s, (aid, r) in enumerate(pairs):
                    Z[b, i, 3 + 2 * s] = float(aid)
                    Z[b, i, 3 + 2 * s + 1] = float(r)
        Z = Z.squeeze(0) if is_2d else Z
        return Z.to(Zt.device, dtype=Zt.dtype)

    def get_bounds(self, Z_hist: np.ndarray, device: torch.device) -> torch.Tensor:
        Z = np.asarray(Z_hist)
        lb_e = np.min(Z[:, :3], axis=0)
        ub_e = np.max(Z[:, :3], axis=0)
        same_bounds = ub_e <= lb_e
        ub_e[same_bounds] = lb_e[same_bounds] + 1e-6
        lb_slots = np.ravel([(0.0, 0.0)] * self.k)
        ub_slots = np.ravel([(float(self.A), 1.0)] * self.k)
        lb = np.concatenate([lb_e, lb_slots])
        ub = np.concatenate([ub_e, ub_slots])
        bounds_np = np.stack([lb, ub])
        return torch.tensor(bounds_np, device=device, dtype=torch.double)


class SlotBO:
    def __init__(
            self,
            df: pd.DataFrame,
            essentials: List[str],
            additives: List[str],
            ranges: Dict[str, Tuple[float, float]],
            k_slots: int,
            target_col: str,
            device: str = "auto",
            seed: int = 42,
    ):
        self.E = list(essentials)
        self.adds = list(additives)
        self.k = int(k_slots)
        self.target_col = target_col
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        set_seeds(self.seed)

        rows = df.to_dict('records')
        self.codec = SlotCodec(self.E, self.adds, ranges, self.k)
        Z_np = self.codec.encode(rows)
        y_np = _ensure_2d_y(df[self.target_col].values)

        unique_Z, unique_indices, inverse_indices = np.unique(
            Z_np, axis=0, return_index=True, return_inverse=True
        )

        if len(unique_Z) < len(Z_np):
            print(f"[Warning] Found {len(Z_np) - len(unique_Z)} duplicate rows in encoded data. Merging them.")
            unique_y = np.array([
                np.mean(y_np[inverse_indices == i]) for i in range(len(unique_Z))
            ])
            Z_np = unique_Z
            y_np = _ensure_2d_y(unique_y)

        self.Z = torch.tensor(Z_np, dtype=torch.double, device=self.device)
        self.y = torch.tensor(y_np, dtype=torch.double, device=self.device)

        self.cat_dims = list(range(3, 3 + 2 * self.k, 2))
        fit_kwargs = {"seed": self.seed, "cat_dims": self.cat_dims}

        print("Fitting GP...")
        self.model = fit_gp(self.Z, self.y, **fit_kwargs)
        print("GP fitted")

        self.bounds = self.codec.get_bounds(Z_np, self.device)
        self._supports_gpish = isinstance(self.model, (SingleTaskGP, MixedSingleTaskGP))

    def _make_acqf(self, acq_type: str, mc_samples: int = 256, **opts: Any):
        t = acq_type.lower()

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        def _maybe_add_sampler(acq_cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
            if "sampler" in inspect.signature(acq_cls.__init__).parameters:
                kwargs["sampler"] = sampler
            return kwargs

        best_f = self.y.max().item()
        if t in ("qlognei", "lognei"):
            kwargs = _maybe_add_sampler(qLogNoisyExpectedImprovement, {})
            return qLogNoisyExpectedImprovement(self.model, X_baseline=self.Z, **kwargs)
        if t in ("qei", "ei"):
            kwargs = _maybe_add_sampler(qLogExpectedImprovement, {"best_f": best_f})
            return qLogExpectedImprovement(self.model, **kwargs)
        if t in ("ucb", "qucb"):
            kwargs = _maybe_add_sampler(qUpperConfidenceBound, {"beta": opts.get("beta", 0.2)})
            return qUpperConfidenceBound(self.model, **kwargs)
        if t in ("kg", "qkg", "knowledge_gradient"):
            return qKnowledgeGradient(self.model, num_fantasies=opts.get("kg_num_fantasies", 64))

        raise ValueError(f"Unknown Acquisition Function: '{acq_type}'")

    def ask(
            self,
            q: int = 8,
            num_restarts: int = 20,
            raw_samples: int = 512,
            acq_types: List[str] | None = None,
            acq_options: Dict | None = None
    ) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:

        set_seeds(self.seed)
        acq_types = acq_types or ["qlognei"]
        acq_options = acq_options or {}

        q_per_acq = max(1, math.ceil(q / len(acq_types)))

        canonical_combinations = []
        additive_ids = range(1, self.codec.A + 1)
        for num_additives in range(self.k + 1):
            for combo in combinations(additive_ids, num_additives):
                padded_combo = combo + (0,) * (self.k - num_additives)
                canonical_combinations.append(padded_combo)

        all_fixed_features_list = []
        for combo in canonical_combinations:
            fixed_features = {}
            for i, additive_id in enumerate(combo):
                id_dim_index = self.cat_dims[i]
                ratio_dim_index = id_dim_index + 1
                fixed_features[id_dim_index] = float(additive_id)
                if additive_id == 0:
                    fixed_features[ratio_dim_index] = 0.0
            all_fixed_features_list.append(fixed_features)

        print(f"\n[Info] Total discrete subspaces defined: {len(all_fixed_features_list)}")
        print(f"[Info] Target candidates per acquisition function: {q_per_acq}")

        all_candidates_tensors = []

        train_Z_np = self.Z.detach().cpu().numpy()
        occupied_keys = set(map(tuple, np.round(train_Z_np, 6)))

        for acq_name in acq_types:

            acqf_screen = self._make_acqf(acq_name, mc_samples=64, **acq_options)
            acqf_refine = self._make_acqf(acq_name, mc_samples=256, **acq_options)

            num_top_subspaces = 20
            print(f"--- Stage 1/2 (B1): Pending-aware Screening + Refining for {q_per_acq} steps ---")

            base_per_axis = 2.2
            min_samples = int(math.ceil(base_per_axis ** 3))
            max_samples = int(math.ceil(base_per_axis ** 6))
            eval_batch_size = 32

            lb = self.bounds[0]
            ub = self.bounds[1]
            d = lb.numel()

            def _prescreen(pending_step: int) -> tuple[list[dict[int, float]], list[torch.Tensor], list[float]]:
                best_val_per_space = torch.empty(
                    len(all_fixed_features_list), device=self.device, dtype=torch.double
                )
                sample_stats = {0: 0, 1: 0, 2: 0, 3: 0}

                best_X_per_space_list: list[torch.Tensor] = []
                best_afv_per_space_list: list[float] = []

                for si, fixed in enumerate(all_fixed_features_list):
                    n_nonzero = sum(1 for dim in self.cat_dims if fixed.get(dim, 0.0) > 0.0)
                    d_eff = 3 + n_nonzero
                    num_screen_samples = int(np.clip(math.ceil(base_per_axis ** d_eff), min_samples, max_samples))
                    sample_stats[n_nonzero] += num_screen_samples

                    X = lb + (ub - lb) * torch.rand(num_screen_samples, d, device=self.device, dtype=torch.double)
                    for col_idx, val in fixed.items():
                        X[:, col_idx] = val

                    current_best_val = -float("inf")
                    current_best_x = None

                    with torch.no_grad():
                        for i in range(0, num_screen_samples, eval_batch_size):
                            Xb = X[i: i + eval_batch_size]
                            vb = acqf_screen(Xb.unsqueeze(1)).view(-1)
                            batch_max_val, batch_max_idx = torch.max(vb, dim=0)
                            if batch_max_val.item() > current_best_val:
                                current_best_val = batch_max_val.item()
                                current_best_x = Xb[batch_max_idx].clone()

                    best_val_per_space[si] = current_best_val
                    if current_best_x is not None:
                        best_X_per_space_list.append(current_best_x)
                        best_afv_per_space_list.append(float(current_best_val))
                    else:
                        best_X_per_space_list.append(X[0])
                        best_afv_per_space_list.append(float("-inf"))

                print(
                    f"    [Info][step={pending_step}] Total screening samples: {sum(sample_stats.values())} "
                    f"(Breakdown: {sample_stats})"
                )

                k_top = min(num_top_subspaces, len(all_fixed_features_list))
                top_vals, top_space_indices = torch.topk(best_val_per_space, k=k_top)
                top_fixed_features = [all_fixed_features_list[i] for i in top_space_indices.tolist()]
                print(
                    f"    [Done][step={pending_step}] Selected top {k_top} subspaces. Best score: {top_vals[0].item():.4f}")

                return top_fixed_features, best_X_per_space_list, best_afv_per_space_list

            base_X_pending_screen = acqf_screen.X_pending
            base_X_pending_refine = acqf_refine.X_pending

            pending = base_X_pending_refine
            step_candidates = []

            for step in range(q_per_acq):
                acqf_screen.set_X_pending(pending)
                acqf_refine.set_X_pending(pending)

                top_fixed_features, best_X_per_space_list, best_afv_per_space_list = _prescreen(pending_step=step)

                if step == 0:
                    if len(best_X_per_space_list) > 0:
                        print(
                            f"    [Analysis] analyzing AFV distribution per additive across {len(best_X_per_space_list)} subspaces...")

                        all_best_X_tensor = torch.stack(best_X_per_space_list)  # (N_spaces, d)
                        all_best_rows = self.codec.decode(all_best_X_tensor.cpu().numpy())

                        additive_to_afvs = defaultdict(list)

                        for row, afv in zip(all_best_rows, best_afv_per_space_list):
                            active_additives = [k for k in row.keys() if k not in self.E]
                            for add in active_additives:
                                additive_to_afvs[add].append(afv)

                        if len(additive_to_afvs) == 0:
                            print(
                                "    [Analysis] No additives found in best candidates (all subspaces chose 0 additives).")
                        else:
                            stats = []
                            for add, vals in additive_to_afvs.items():
                                arr = np.asarray(vals, dtype=float)
                                mean = float(np.mean(arr))
                                var = float(np.var(arr, ddof=0))
                                std = float(np.sqrt(var))
                                stats.append(
                                    {"additive": add, "n": int(arr.size), "mean": mean, "var": var, "std": std})

                            stats_df = pd.DataFrame(stats).sort_values(
                                by=["mean", "n"], ascending=[False, False]
                            ).reset_index(drop=True)

                            try:
                                labels = stats_df["additive"].tolist()
                                means_ = stats_df["mean"].to_numpy()
                                stds_ = stats_df["std"].to_numpy()

                                plt.figure(figsize=(12, 6))
                                x = np.arange(len(labels))
                                plt.bar(labels, means_, edgecolor="black", alpha=0.8)
                                plt.errorbar(x, means_, yerr=stds_, fmt="none", ecolor="black", capsize=3, linewidth=1)

                                plt.title(
                                    f"Additive AFV Distribution across {len(all_best_rows)} Subspaces\n"
                                    f"(bar=mean, errorbar=std; Acquisition={acq_name})"
                                )
                                plt.xlabel("Additive Name")
                                plt.ylabel("Acquisition Function Value (AFV)")
                                plt.xticks(rotation=45, ha="right")
                                plt.tight_layout()

                                fig_path = f"prescreen_{acq_name}_additive_afv_mean_std.png"
                                plt.savefig(fig_path, dpi=200)
                                plt.show()
                                print(f"    [Analysis] Saved plot -> {fig_path}")

                                csv_path = f"prescreen_{acq_name}_additive_afv_stats.csv"
                                stats_df.to_csv(csv_path, index=False, encoding="utf-8")
                                print(f"    [Analysis] Saved stats -> {csv_path}")

                            except Exception as e:
                                print(f"    [Warning] Plotting failed: {e}")

                max_attempts = 5
                accepted = False

                for attempt in range(max_attempts):
                    candidate_raw, _ = optimize_acqf_mixed(
                        acq_function=acqf_refine,
                        bounds=self.bounds,
                        q=1,
                        fixed_features_list=top_fixed_features,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                        options={"batch_limit": num_restarts, "maxiter": 200},
                        retry_on_optimization_warning=True,
                    )

                    candidate = self.codec.postprocess_batch(candidate_raw)  # (1,d)

                    cand_key = tuple(np.round(candidate.detach().cpu().numpy(), 6).ravel())
                    if cand_key not in occupied_keys:
                        occupied_keys.add(cand_key)
                        step_candidates.append(candidate)
                        pending = candidate if pending is None else torch.cat([pending, candidate], dim=-2)
                        accepted = True
                        break
                    else:
                        set_seeds(self.seed + 1000 * (step + 1) + attempt + 1)

                if not accepted:
                    print(
                        f"[Warning] step={step} failed to find a non-duplicate candidate after {max_attempts} attempts.")

            print(f"Done. Generated {q_per_acq} candidates (B1 greedy).")

            current_batch_candidates = torch.cat(step_candidates, dim=0)
            all_candidates_tensors.append(current_batch_candidates)

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

        unique_candidates_tensor = torch.stack(unique_tensors)

        final_acqf = self._make_acqf(acq_types[-1], **acq_options)
        with torch.no_grad():
            individual_acq_vals = final_acqf(unique_candidates_tensor.unsqueeze(1)).view(-1)

        individual_acq_vals = individual_acq_vals.detach().cpu().numpy()

        self.model.eval()
        with torch.no_grad():
            posterior = self.model.posterior(unique_candidates_tensor)
            pred_means = posterior.mean.squeeze(-1).detach().cpu().numpy()
            pred_variances = posterior.variance.squeeze(-1).detach().cpu().numpy()

        combined = []
        for i in range(len(unique_rows)):
            combined.append({
                "row": unique_rows[i],
                "acq": float(individual_acq_vals[i]),
                "mean": float(pred_means[i]),
                "sigma": float(np.sqrt(pred_variances[i])),
            })

        combined.sort(key=lambda x: x["acq"], reverse=True)

        final_rows = [item["row"] for item in combined[:q]]
        final_means = np.array([item["mean"] for item in combined[:q]])
        final_sigmas = np.array([item["sigma"] for item in combined[:q]])

        print(f"[Info] Successfully generated {len(final_rows)} candidates.\n")
        return final_rows, final_means, final_sigmas


def _setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian optimization script based on slot encoding and GP")
    # --- base setting ---
    parser.add_argument("--input", type=str, default="data_full.xlsx",
                        help="Path to the historical data file (.csv or .xlsx)")
    parser.add_argument("--target", type=str, default="auc", help="Column name for the target variable")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda", help="device")
    # --- BO setting ---
    parser.add_argument("--slots", type=int, default=3, help="# of slots (k)")
    parser.add_argument("--q", type=int, default=DEFAULT_Q, help="# of candidates per BO round")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT_CSV)
    # --- AF setting ---
    parser.add_argument("--acq", type=str, default="qlognei", help="Acquisition Function: qlognei, qei, qucb, qkg")
    parser.add_argument("--acq_opts", type=str, default="{}")
    return parser


def _prepare_data(file_path: str) -> Tuple[pd.DataFrame, List[str], List[str], Dict[str, Tuple[float, float]]]:
    path = Path(file_path)
    if not path.exists():
        alt_path_xlsx = Path("./data.xlsx")
        alt_path_csv = Path("./data.csv")
        if alt_path_xlsx.exists():
            print(f"[info] {path} not found, using alternative path {alt_path_xlsx}")
            path = alt_path_xlsx
        elif alt_path_csv.exists():
            print(f"[info] {path} not found, using alternative path {alt_path_csv}")
            path = alt_path_csv
        else:
            raise FileNotFoundError(f"Data file does not exist: {path.resolve()}")

    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: '{path.suffix}'. Please use .csv or .xlsx files.")

    df.columns = df.columns.str.lower()

    present_cols = [c for c in FEATURE_WHITELIST if c in df.columns]
    assert all(e in present_cols for e in ESSENTIALS), f"essentials {ESSENTIALS} must be included in columns"
    additives = [c for c in present_cols if c not in ESSENTIALS]
    ranges: Dict[str, Tuple[float, float]] = {}
    for a in additives:
        s = df[a].astype(float).values
        s_positive = s[s > 1e-6]
        if len(s_positive) == 0:
            lo, hi = 0.0, 1.0
        else:
            lo = float(np.nanmin(s_positive))
            hi = float(np.nanmax(s))
        if hi <= lo:
            hi = lo + 1e-6
        ranges[a] = (lo, hi)
    return df, ESSENTIALS, additives, ranges


def _save_results(rows: List[Dict[str, float]], means: np.ndarray, sigmas: np.ndarray, essentials: List[str],
                  out_path: str):
    if not rows:
        print("[warning] no candidate is generated")
        return

    out_df = pd.DataFrame(rows).fillna(0.0)
    out_df['pred_mean'] = means
    out_df['pred_sigma'] = sigmas
    add_counts = Counter(k for r in rows for k in r if k not in essentials and r.get(k, 0.0) > EPS)
    sorted_adds = [k for k, _ in sorted(add_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    all_cols = essentials + sorted_adds + ['pred_mean', 'pred_sigma']
    for col in all_cols:
        if col not in out_df:
            out_df[col] = 0.0
    out_df = out_df[all_cols]
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[success] {len(rows)} candidates are written into -> {Path(out_path).resolve()}")
    print(f"[info] column rank: {all_cols}")
    if add_counts:
        print("[info] additives used counter:", ", ".join(f"{k}:{v}" for k, v in add_counts.most_common()))


def main():
    start_time = time.time()

    parser = _setup_arg_parser()
    args = parser.parse_args()
    set_seeds(args.seed)

    df, essentials, additives, ranges = _prepare_data(args.input)

    bo = SlotBO(
        df=df,
        essentials=essentials,
        additives=additives,
        ranges=ranges,
        k_slots=args.slots,
        target_col=args.target,
        device=args.device,
        seed=args.seed,
    )
    acq_types = [s.strip() for s in args.acq.split(",") if s.strip()]
    try:
        acq_opts = json.loads(args.acq_opts)
        assert isinstance(acq_opts, dict)
    except (json.JSONDecodeError, AssertionError):
        raise ValueError(f"invalid --acq_opts")

    print(f"\n Currently using {acq_types} acquisition function (option: {acq_opts}) generating {args.q} candidates...")
    rows, means, sigmas = bo.ask(q=args.q, acq_types=acq_types, acq_options=acq_opts)

    _save_results(rows, means, sigmas, essentials, args.out)

    print(f"\n[Settings]")
    print(f"  - device: {bo.device}, model: GP (Mixed)")
    print(f"  - # of slots: {args.slots}, random seed: {args.seed}")

    if torch.cuda.is_available() and bo.device.type == 'cuda':
        torch.cuda.synchronize()

    end_time = time.time()
    elapsed_seconds = end_time - start_time

    print(f"\n[Timer] Total Execution Time: {elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()