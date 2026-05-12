from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "logs" / "kernel_similarity_comparison"
CSV_PATH = OUT_DIR / "kernel_similarity_table.csv"
PNG_PATH = OUT_DIR / "kernel_similarity_bars.png"
FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


COLORS = {
    "OLD": (110, 120, 130),
    "NEW": (53, 116, 181),
    "MOD": (25, 155, 125),
}


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    fallback = Path(r"C:\Windows\Fonts\calibri.ttf")
    return ImageFont.truetype(str(path if path.exists() else fallback), size=size)


def shorten(text: str, max_chars: int = 44) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "."


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    models = ["OLD", "NEW", "MOD"]

    left = 420
    right = 58
    top = 96
    bottom = 72
    group_h = 72
    bar_h = 14
    gap = 6
    axis_w = 720
    width = left + axis_w + right
    height = top + bottom + group_h * len(df)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = font(FONT_BOLD, 30)
    label_font = font(FONT_REGULAR, 18)
    small_font = font(FONT_REGULAR, 15)
    tick_font = font(FONT_REGULAR, 14)
    legend_font = font(FONT_BOLD, 16)

    draw.text((left, 28), "Kernel Similarity by Recipe Pair", fill=(18, 18, 18), font=title_font)
    draw.text((left, 64), "Normalized covariance similarity; higher means the kernel treats two recipes as more similar.", fill=(85, 85, 85), font=small_font)

    x0 = left
    x1 = left + axis_w
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = x0 + int(axis_w * tick)
        draw.line((x, top - 8, x, height - bottom + 8), fill=(232, 232, 232), width=1)
        label = f"{tick:.2f}".rstrip("0").rstrip(".")
        tw = draw.textbbox((0, 0), label, font=tick_font)[2]
        draw.text((x - tw / 2, height - bottom + 18), label, fill=(75, 75, 75), font=tick_font)
    draw.line((x0, height - bottom + 8, x1, height - bottom + 8), fill=(120, 120, 120), width=1)

    legend_x = width - 300
    for i, model in enumerate(models):
        lx = legend_x + i * 90
        draw.rectangle((lx, 36, lx + 20, 50), fill=COLORS[model])
        draw.text((lx + 28, 33), model, fill=(18, 18, 18), font=legend_font)

    for row_idx, row in df.iterrows():
        y = top + row_idx * group_h
        draw.text((28, y + 17), shorten(str(row["Meaning"])), fill=(20, 20, 20), font=label_font)
        for model_idx, model in enumerate(models):
            value = float(row[model])
            bar_y = y + model_idx * (bar_h + gap) + 6
            bar_x1 = x0 + int(axis_w * max(0.0, min(1.0, value)))
            draw.rounded_rectangle((x0, bar_y, bar_x1, bar_y + bar_h), radius=3, fill=COLORS[model])
            value_text = f"{value:.2f}"
            draw.text((bar_x1 + 7, bar_y - 2), value_text, fill=(45, 45, 45), font=small_font)

    img.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
