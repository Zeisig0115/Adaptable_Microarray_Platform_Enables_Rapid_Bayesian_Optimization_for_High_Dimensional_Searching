from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "logs" / "kernel_similarity_comparison"
CSV_PATH = OUT_DIR / "kernel_similarity_table.csv"
PNG_PATH = OUT_DIR / "kernel_similarity_table.png"
FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    fallback = Path(r"C:\Windows\Fonts\calibri.ttf")
    return ImageFont.truetype(str(path if path.exists() else fallback), size=size)


def fit_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_=+./-]+|\s+|.", text)
    lines: list[str] = []
    current = ""
    for word in words:
        trial = current + word
        width = draw.textbbox((0, 0), trial, font=fnt)[2]
        if width <= max_width or not current:
            current = trial
        else:
            lines.append(current.strip())
            current = word.lstrip()
    if current:
        lines.append(current.strip())
    return lines


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    for col in ["OLD", "NEW", "MOD"]:
        df[col] = df[col].map(lambda v: f"{float(v):.2f}")

    columns = ["Pair", "Meaning", "OLD", "NEW", "MOD"]
    widths = [470, 600, 115, 115, 115]
    margin_x = 28
    margin_y = 24
    header_h = 58
    row_h = 62
    width = margin_x * 2 + sum(widths)
    height = margin_y * 2 + header_h + row_h * len(df)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    regular = font(FONT_REGULAR, 22)
    small = font(FONT_REGULAR, 19)
    bold = font(FONT_BOLD, 22)
    header = font(FONT_BOLD, 24)

    xs = [margin_x]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)

    y = margin_y
    for col, x, w in zip(columns, xs, widths):
        align_right = col in {"OLD", "NEW", "MOD"}
        text_w = draw.textbbox((0, 0), col, font=header)[2]
        tx = x + w - 18 - text_w if align_right else x
        draw.text((tx, y + 12), col, fill=(18, 18, 18), font=header)
    draw.line((margin_x, y + header_h, width - margin_x, y + header_h), fill=(110, 110, 110), width=1)

    for i, row in df.iterrows():
        top = margin_y + header_h + i * row_h
        draw.line((margin_x, top + row_h, width - margin_x, top + row_h), fill=(195, 195, 195), width=1)
        pair_lines = fit_text(draw, str(row["Pair"]), small, widths[0] - 12)
        meaning_lines = fit_text(draw, str(row["Meaning"]), regular, widths[1] - 12)
        draw.text((xs[0], top + 13), "\n".join(pair_lines[:2]), fill=(20, 20, 20), font=small, spacing=2)
        draw.text((xs[1], top + 13), "\n".join(meaning_lines[:2]), fill=(20, 20, 20), font=regular, spacing=2)
        for col, x, w in zip(["OLD", "NEW", "MOD"], xs[2:], widths[2:]):
            text = str(row[col])
            fnt = bold if col == "MOD" else regular
            text_w = draw.textbbox((0, 0), text, font=fnt)[2]
            draw.text((x + w - 18 - text_w, top + 15), text, fill=(18, 18, 18), font=fnt)

    img.save(PNG_PATH)
    print(PNG_PATH)


if __name__ == "__main__":
    main()
