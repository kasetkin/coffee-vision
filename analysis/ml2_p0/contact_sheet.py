"""ML-2 P0: one 2x2 sheet per photo with every box-prompt output outlined, for the mask-selection look.

Reads the masks `python -m coffeecv.segment_beans --candidates` wrote. The sheets are overviews only;
full-resolution boundary tiles of the candidate that looks right come from --tiles.

    PYTHONPATH=. python analysis/ml2_p0/contact_sheet.py outputs/ml2_p0/candidates
    PYTHONPATH=. python analysis/ml2_p0/contact_sheet.py outputs/ml2_p0/candidates --tiles single
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from coffeecv.dataset import load_rgb_image

NAMES = ("single", "multi1", "multi2", "multi3")
MAGENTA = (255, 0, 255)


def outline(rgb: np.ndarray, mask: np.ndarray, px: int, tint: float = 0.0) -> np.ndarray:
    """Magenta outline; with `tint` > 0 the inside is also washed magenta, so which side of the
    outline is the mask can't be misread."""
    m = mask.astype(np.uint8)
    edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((2 * px + 1, 2 * px + 1), np.uint8)) > 0
    out = rgb.copy()
    if tint:
        out[mask] = (out[mask] * (1 - tint) + np.array(MAGENTA) * tint).astype(np.uint8)
    out[edge] = MAGENTA
    return out


def sheet(photo: Path, cand_dir: Path, meta: dict, panel: int = 900) -> Image.Image:
    rgb = load_rgb_image(photo)
    h, w = rgb.shape[:2]
    s = panel / max(h, w)
    small = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    tiles = []
    for name in NAMES:
        m = np.array(Image.open(cand_dir / f"{photo.stem}__{name}.png")) > 0
        ms = cv2.resize(m.astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
        im = Image.fromarray(outline(small, ms, 1, tint=0.35))
        d = ImageDraw.Draw(im)
        g = meta.get("prompt")                     # absent in candidates.json written before 2026-09-30 evening
        if g and g["box"] and g["box"] != [0, 0, w - 1, h - 1]:
            d.rectangle([v * s for v in g["box"]], outline=(0, 255, 0), width=2)
        for (x, y), lab in zip(g["points"] if g else [], g["labels"] if g else []):
            d.ellipse([x * s - 4, y * s - 4, x * s + 4, y * s + 4],
                      fill=(0, 255, 0) if lab else (255, 0, 0), outline="black")
        d.text((8, 8), f"{name} iou={meta[name]['pred_iou']:.3f} area={meta[name]['area_frac']:.3f}",
                                fill=MAGENTA)
        tiles.append(im)
    tw, th = tiles[0].size
    out = Image.new("RGB", (2 * tw, 2 * th), "white")
    for i, t in enumerate(tiles):
        out.paste(t, ((i % 2) * tw, (i // 2) * th))
    return out


def boundary_tiles(photo: Path, cand_dir: Path, name: str, n: int = 4, size: int = 768) -> list[Image.Image]:
    """`n` full-resolution crops centred on points evenly spaced along the mask's largest contour."""
    rgb = load_rgb_image(photo)
    m = (np.array(Image.open(cand_dir / f"{photo.stem}__{name}.png")) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []
    c = max(contours, key=len)[:, 0, :]
    over = outline(rgb, m > 0, 2)
    h, w = m.shape
    out = []
    for k in range(n):
        x, y = c[(k * len(c)) // n]
        x0 = int(np.clip(x - size // 2, 0, max(0, w - size)))
        y0 = int(np.clip(y - size // 2, 0, max(0, h - size)))
        out.append(Image.fromarray(over[y0:y0 + size, x0:x0 + size]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cand_dir", type=Path)
    ap.add_argument("--tiles", choices=NAMES, help="also write full-resolution boundary tiles of this output")
    args = ap.parse_args()
    index = json.loads((args.cand_dir / "candidates.json").read_text())
    out_dir = args.cand_dir / "sheets"
    out_dir.mkdir(exist_ok=True)
    for photo, meta in index.items():
        photo = Path(photo)
        sheet(photo, args.cand_dir, meta).save(out_dir / f"{photo.stem}.jpg", quality=90)
        if args.tiles:
            for k, t in enumerate(boundary_tiles(photo, args.cand_dir, args.tiles)):
                t.save(out_dir / f"{photo.stem}__{args.tiles}__T{k + 1}.jpg", quality=92)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
