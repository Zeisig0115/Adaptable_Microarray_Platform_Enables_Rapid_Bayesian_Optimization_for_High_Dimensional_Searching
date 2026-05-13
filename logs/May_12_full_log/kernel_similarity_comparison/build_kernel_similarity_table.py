from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import add_bo
import add_bo_mod


DATA_PATH = ROOT / "logs" / "May_10_full_log" / "data_corrected.xlsx"
OUT_DIR = ROOT / "logs" / "kernel_similarity_comparison"


PAIRS = [
    (460, 461, "Blank vs blank; different TMB/H2O2"),
    (0, 277, "Same additive and dose; A/B replicate recipe"),
    (276, 280, "Same additive; different CMC dose"),
    (0, 105, "Same family polymer; CMC vs PVA"),
    (270, 273, "Same family PEG; PEG6k vs PEG400"),
    (231, 266, "Same family salt; CaCl2 vs FeSO4"),
    (0, 45, "Different family; polymer vs solvent"),
    (231, 45, "Different family; salt vs solvent"),
    (0, 273, "Disjoint additives; both polymers"),
    (460, 0, "Blank vs single additive"),
    (278, 21, "Subset relation; CMC vs CMC+PEG400"),
]


def active_recipe_label(row: pd.Series, codec: add_bo.FlatCodec) -> str:
    active = []
    for name in codec.adds:
        value = float(row.get(name, 0.0) or 0.0)
        if value > add_bo.EPS:
            active.append(f"{name}={value:g}")
    additives = "+".join(active) if active else "blank"
    return f"r{int(row.name):03d} {additives}"


def encode_original_row(df: pd.DataFrame, codec: add_bo.FlatCodec, idx: int) -> torch.Tensor:
    if idx not in df.index:
        raise KeyError(f"row {idx} is not present after data filtering")
    z = codec.encode_row(df.loc[idx].to_dict())
    return torch.tensor(z, dtype=torch.double).unsqueeze(0)


def normalized_kernel_similarity(model: torch.nn.Module, z1: torch.Tensor, z2: torch.Tensor) -> float:
    model.eval()
    z = torch.cat([z1, z2], dim=0).to(next(model.parameters()).device)
    with torch.no_grad():
        if getattr(model, "input_transform", None) is not None:
            z = model.input_transform(z)
        k = model.covar_module(z).evaluate()
        sim = k[0, 1] / (k[0, 0].clamp_min(1e-18).sqrt() * k[1, 1].clamp_min(1e-18).sqrt())
    return float(sim.detach().cpu().item())


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.double)
    add_bo.set_seeds(42)
    device = torch.device("cpu")
    args = SimpleNamespace(
        input=str(DATA_PATH),
        sheet="Corrected Data",
        target="intensity",
        hrp=0.0001,
        hrp_atol=1e-12,
        filter_ctrl=True,
        k_max=4,
    )
    df, codec = add_bo.load_training_data(args)
    x_train, y_train, encode_meta = add_bo.encode_training_data(df, codec, args.target, device)
    bounds = codec.get_bounds(device)

    model_specs = [
        ("OLD", "old_mixed_hamming", add_bo.make_old_model, add_bo.fit_model),
        ("NEW", "new_additive_set", add_bo.make_new_model, add_bo.fit_model),
        (
            "MOD",
            "hierarchical_family_prior",
            add_bo_mod.make_hierarchical_family_model,
            add_bo_mod.fit_model,
        ),
    ]
    models = {}
    fit_info = {}
    for short_name, label, builder, fitter in model_specs:
        model = builder(x_train, y_train, codec, bounds)
        fit_info[label] = fitter(model, maxiter=75)
        models[short_name] = model

    rows = []
    for left_idx, right_idx, meaning in PAIRS:
        left = df.loc[left_idx]
        right = df.loc[right_idx]
        z_left = encode_original_row(df, codec, left_idx)
        z_right = encode_original_row(df, codec, right_idx)
        out = {
            "Pair": f"{active_recipe_label(left, codec)} vs {active_recipe_label(right, codec)}",
            "Meaning": meaning,
            "OLD": normalized_kernel_similarity(models["OLD"], z_left, z_right),
            "NEW": normalized_kernel_similarity(models["NEW"], z_left, z_right),
            "MOD": normalized_kernel_similarity(models["MOD"], z_left, z_right),
            "left_tmb": float(left["tmb"]),
            "left_h2o2": float(left["h2o2"]),
            "right_tmb": float(right["tmb"]),
            "right_h2o2": float(right["h2o2"]),
        }
        rows.append(out)

    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "kernel_similarity_table.csv", index=False, encoding="utf-8")

    display = table.copy()
    for col in ["OLD", "NEW", "MOD"]:
        display[col] = display[col].map(lambda v: f"{v:.2f}")
    md = [
        "# Kernel Similarity Comparison",
        "",
        f"- Data: `{DATA_PATH}`",
        "- Sheet/filter: `Corrected Data`, `HRP=0.0001`, `CTRL=0`",
        "- Similarity: normalized covariance `K(x, y) / sqrt(K(x, x) K(y, y))` after each model's input transform",
        "- Fit: imported `add_bo` / `add_bo_mod` builders, `fit_maxiter=75`, no candidates",
        "",
        to_markdown_table(display[["Pair", "Meaning", "OLD", "NEW", "MOD"]]),
        "",
        "## Fit Metadata",
        "",
        "```json",
        json.dumps({"encode_meta": encode_meta, "fit_info": fit_info}, indent=2),
        "```",
        "",
    ]
    (OUT_DIR / "kernel_similarity_table.md").write_text("\n".join(md), encoding="utf-8")

    summary = {"data": str(DATA_PATH), "encode_meta": encode_meta, "fit_info": fit_info, "rows": rows}
    (OUT_DIR / "kernel_similarity_table.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
