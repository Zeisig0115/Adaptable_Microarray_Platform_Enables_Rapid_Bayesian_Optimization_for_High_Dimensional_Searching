from __future__ import annotations
import os
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch

from fit_model import fit_gp

from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf

torch.set_default_dtype(torch.double)


ESSENTIALS = ["TMB", "H2O2"]
DEFAULT_Q = 32

def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_2d_y(y: np.ndarray | List[float]) -> np.ndarray:
    y = np.asarray(y)
    return y.reshape(-1, 1) if y.ndim == 1 else y


class EssentialsBO:
    MODEL_FITTERS = {
        "gp": fit_gp
    }

    def __init__(
            self,
            df: pd.DataFrame,
            essentials: List[str],
            target_col: str,
            surrogate: str = "gp",
            device: str = "auto",
            seed: int = 42,
            nuts_cfg: Dict[str, int] | None = None,
            physical_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.E = list(essentials)
        self.target_col = target_col
        self.surrogate = surrogate.lower()
        self.nuts_cfg = nuts_cfg or {}
        self.seed = seed

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        set_seeds(self.seed)

        Z_np = df[self.E].values
        y_np = _ensure_2d_y(df[self.target_col].values)

        self.Z = torch.tensor(Z_np, dtype=torch.double, device=self.device)
        self.y = torch.tensor(y_np, dtype=torch.double, device=self.device)

        self.Z = torch.log10(self.Z)

        if physical_bounds is not None:
            try:
                b_min = [np.log10(physical_bounds[e][0]) for e in self.E]
                b_max = [np.log10(physical_bounds[e][1]) for e in self.E]
                self.bounds = torch.tensor([b_min, b_max], dtype=torch.double, device=self.device)
                print(f"  -> 已启用全局物理边界并转换至 Log 空间: min={b_min}, max={b_max}")
            except KeyError as e:
                raise ValueError(f"提供的 physical_bounds 缺少特征 {e} 的边界定义！")
        else:
            print("  -> [警告] 未提供物理边界，正在使用历史数据极值。")
            bounds_min = self.Z.min(dim=0).values
            bounds_max = self.Z.max(dim=0).values
            self.bounds = torch.stack([bounds_min, bounds_max])

        fitter = self.MODEL_FITTERS.get(self.surrogate)
        fit_kwargs = {"seed": self.seed, "bounds": self.bounds,}
        print(f"正在拟合模型 '{self.surrogate}'...")
        self.model = fitter(self.Z, self.y, **fit_kwargs)
        print("模型拟合完成。")

        self._print_kernel_diagnostics()
        self._print_loo_diagnostics()


    def _print_kernel_diagnostics(self) -> None:
        """打印核矩阵条件数、noise、lengthscale、残差，以及残差的高/低 y 分段对比。"""
        with torch.no_grad():
            Z_normalized = self.model.input_transform(self.Z)
            K = self.model.covar_module(Z_normalized).evaluate()
            noise = self.model.likelihood.noise.item()
            K_full = K + noise * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
            eigvals = torch.linalg.eigvalsh(K_full)

            print(f"\n[核矩阵诊断]")
            print(f"  Minimum eigenvalue: {eigvals.min().item():.2e}")
            print(f"  Maximum eigenvalue: {eigvals.max().item():.2e}")
            print(f"  Condition number:   {(eigvals.max() / eigvals.min()).abs().item():.2e}")

            # noise 同时给出标准化尺度和原 y 尺度
            y_std = self.y.std(unbiased=True).item()
            print(f"  noise (standardized):   {noise:.3e}")
            print(f"  noise std (orig y):     {np.sqrt(noise) * y_std:.1f}")

            # 训练点上的后验均值残差
            posterior = self.model.posterior(self.Z, observation_noise=False)
            mean_y = posterior.mean.squeeze(-1)
            residual_y = (self.y.squeeze(-1) - mean_y).abs()
            print(f"  Avg |residual| (orig y): {residual_y.mean().item():.1f}")
            print(f"  Max |residual| (orig y): {residual_y.max().item():.1f}")

            # 异方差检查：按 y 排序后分高/低两段看残差
            y_np = self.y.squeeze(-1).cpu().numpy()
            res_np = residual_y.cpu().numpy()
            order = np.argsort(y_np)
            half = len(y_np) // 2
            low_mean = res_np[order[:half]].mean()
            high_mean = res_np[order[half:]].mean()
            print(f"  低 y 段 (n={half}) 残差 mean:         {low_mean:.0f}")
            print(f"  高 y 段 (n={len(y_np) - half}) 残差 mean: {high_mean:.0f}")
            print(f"  高/低段残差比:                        {high_mean / max(low_mean, 1e-9):.2f}"
                  "  (>2 即异方差信号明显)")

        covar = self.model.covar_module
        print(f"  lengthscale: {covar.lengthscale.detach().cpu()}")
        print(f"  outputscale: 1.0 (fixed, no ScaleKernel wrapper)")


    def _print_loo_diagnostics(self) -> None:
        """Leave-one-out 交叉验证，用 z-score 判断 posterior 是否 over-confident。"""
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Standardize, Normalize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from botorch.fit import fit_gpytorch_mll

        n, d = self.Z.shape
        z_scores = np.zeros(n)
        print(f"\n[LOO 交叉验证] n={n}，正在逐点重拟合...")

        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool, device=self.device)
            mask[i] = False
            X_i, y_i = self.Z[mask], self.y[mask]

            m_i = SingleTaskGP(
                train_X=X_i,
                train_Y=y_i,
                input_transform=Normalize(d=d, bounds=self.bounds),
                outcome_transform=Standardize(m=1),
            ).to(self.device)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(m_i.likelihood, m_i))

            with torch.no_grad():
                post = m_i.posterior(self.Z[i:i + 1], observation_noise=True)
                mu = post.mean.item()
                sd = post.variance.sqrt().item()
            z_scores[i] = (self.y[i].item() - mu) / max(sd, 1e-12)

        print(f"  LOO z-score mean:     {z_scores.mean():+.2f}  (期望 ≈ 0)")
        print(f"  LOO z-score std:      {z_scores.std():.2f}   (期望 ≈ 1，>1 即 over-confident)")
        print(f"  |z| > 2 的比例:        {(np.abs(z_scores) > 2).mean():.1%}  (期望 ≈ 5%)")
        print(f"  |z| > 3 的比例:        {(np.abs(z_scores) > 3).mean():.1%}  (期望 ≈ 0.3%)")

        print("|z| > 2 的点:")
        for i in np.where(np.abs(z_scores) > 2)[0]:
            z = z_scores[i]
            x_phys = 10 ** self.Z[i].cpu().numpy()
            y_val = self.y[i].item()
            print(f"  idx={i:2d}  z={z:+.2f}  TMB={x_phys[0]:.3f}  H2O2={x_phys[1]:.3f}  y={y_val:.0f}")


    def _make_acqf(self, acq_type: str, **opts: Any):
        t = acq_type.lower()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        best_f = self.y.max().item()

        if t in ("qlognei", "lognei"):
            return qLogNoisyExpectedImprovement(self.model, X_baseline=self.Z, sampler=sampler, cache_root=False, prune_baseline=True)
        if t in ("qei", "ei"):
            return qLogExpectedImprovement(self.model, best_f=best_f)
        if t in ("ucb", "qucb"):
            return qUpperConfidenceBound(self.model, beta=opts.get("beta", 0.2))
        if t in ("kg", "qkg"):
            return qKnowledgeGradient(self.model, num_fantasies=opts.get("kg_num_fantasies", 64))
        raise ValueError(f"Unknown Acquisition Function'{acq_type}'")

    def ask(self, q: int = 8, num_restarts: int = 20, raw_samples: int = 512,
            acq_types: List[str] | None = None, acq_options: Dict | None = None) -> Tuple[
        List[Dict[str, float]], np.ndarray, np.ndarray]:
        acq_types = acq_types or ["qlognei"]
        acq_options = acq_options or {}

        acqf = self._make_acqf(acq_types[0], **acq_options)

        candidates, _ = optimize_acqf(
            acq_function=acqf,
            bounds=self.bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=False,
        )

        with torch.no_grad():
            posterior = self.model.posterior(candidates)
            predicted_means = posterior.mean.squeeze(-1)
            predicted_variance = posterior.variance.squeeze(-1)
            predicted_stddev = torch.sqrt(predicted_variance)

        candidates_np = candidates.detach().cpu().numpy()
        predicted_means_np = predicted_means.detach().cpu().numpy()
        predicted_stddev_np = predicted_stddev.detach().cpu().numpy()

        candidates_physical = 10 ** candidates_np

        rows = [dict(zip(self.E, c)) for c in candidates_physical]

        return rows, predicted_means_np, predicted_stddev_np


def _setup_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="必需品浓度组合的贝叶斯优化脚本")
    parser.add_argument("--datafile", type=str, default="data.xlsx", help="包含历史数据的CSV或Excel文件路径")
    parser.add_argument("--target", type=str, default="AUC", help="文件中的目标列名")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda", help="计算设备")
    parser.add_argument(
        "--model", type=str, default="gp",
        choices=["gp", "saasbo", "fullyb_gp", "saas_gp"],
        help="选择代理模型"
    )
    parser.add_argument("--q", type=int, default=DEFAULT_Q, help="每次迭代推荐的候选点数量")
    parser.add_argument(
        "--acq", type=str, default="qlognei",
        help="采集函数, 多个用逗号分隔 (例如 'qei,ucb')。常用: qlognei, qei, qucb, qkg, qnei"
    )
    parser.add_argument(
        "--acq_opts", type=str, default="{}",
        help='采集函数选项的JSON字符串, 例如 \'{"beta":0.3}\''
    )
    parser.add_argument("--nuts_warmup", type=int, default=512, help="NUTS预热步数")
    parser.add_argument("--nuts_samples", type=int, default=256, help="NUTS采样数")
    parser.add_argument("--nuts_thinning", type=int, default=16, help="NUTS thinning参数")
    return parser


def _prepare_data(file_path: str) -> Tuple[pd.DataFrame, List[str]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在: {path.resolve()}")

    file_suffix = path.suffix.lower()
    if file_suffix == '.csv':
        df = pd.read_csv(path)
    elif file_suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件格式: '{file_suffix}'。请使用 .csv 或 .xlsx 文件。")

    assert all(e in df.columns for e in ESSENTIALS), \
        f"必需组分 {ESSENTIALS} 必须全部存在于数据文件列中。"

    return df, ESSENTIALS


def _save_results(
    rows: List[Dict[str, float]],
    predicted_values: np.ndarray,
    uncertainties: np.ndarray,
    essentials: List[str],
    out_path: str
):
    if not rows:
        print("[警告] 没有生成任何候选点，不创建输出文件。")
        return

    out_df = pd.DataFrame(rows)
    out_df['predicted_value'] = predicted_values
    out_df['uncertainty_std'] = uncertainties

    column_order = essentials + ['predicted_value', 'uncertainty_std']
    out_df = out_df[column_order]

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\n[成功] {len(rows)}个候选点及其预测值和不确定性已写入 -> {Path(out_path).resolve()}")


def main():
    parser = _setup_arg_parser()
    args = parser.parse_args()
    set_seeds(args.seed)

    df, essentials = _prepare_data(args.datafile)
    assert args.target in df.columns, f"目标列 '{args.target}' 在文件中不存在。"

    physical_bounds = {
        "TMB": (0.001, 1.0),
        "H2O2": (0.001, 1.0)
    }

    bo = EssentialsBO(
        df=df,
        essentials=essentials,
        target_col=args.target,
        surrogate=args.model,
        device=args.device,
        seed=args.seed,
        nuts_cfg={"warmup": args.nuts_warmup, "num_samples": args.nuts_samples, "thinning": args.nuts_thinning},
        physical_bounds=physical_bounds,
    )

    acq_types = [s.strip() for s in args.acq.split(",") if s.strip()]
    try:
        acq_opts = json.loads(args.acq_opts)
        assert isinstance(acq_opts, dict)
    except (json.JSONDecodeError, AssertionError):
        raise ValueError(f"无效的 --acq_opts。必须是JSON对象字符串。")

    print(f"\n正在使用采集函数 {acq_types} (选项: {acq_opts}) 生成 {args.q} 个候选点...")

    rows, predicted_values, uncertainties = bo.ask(
        q=args.q,
        acq_types=acq_types,
        acq_options=acq_opts,
        num_restarts=40,    # default 20
        raw_samples=1024,    # default 512
    )

    output_filename = f"./Apr_29_full_log/ESS_BO_1_HRP_0.0001.csv"
    final_output_path = output_filename

    _save_results(rows, predicted_values, uncertainties, essentials, final_output_path)

    print(f"\n[配置回顾]")
    print(f"  - 设备: {bo.device}, 模型: {args.model}")
    print(f"  - 随机种子: {args.seed}")


if __name__ == "__main__":
    main()