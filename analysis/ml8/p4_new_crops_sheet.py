"""Ticket ML-8 P4 (D11): the crops and masks of the photos new to the re-crop, for the owner to accept or decline
before they train (as ML-3 P5 did for its new sessions; a declined crop goes to labels/ml3/pool_exclude.csv).

    python analysis/ml8/p4_new_crops_sheet.py OLD_ROOT [--out outputs/ml8_p4]

The photos are those in data/segcropped/<session> that OLD_ROOT (ML-6 P8's crops) does not have. Writes, under
--out (gitignored: the photos are DVC data, never git):
    sheet.jpg            every new photo: the photo with its crop box (yellow) and mask outline (magenta), and
                         the crop as training reads it with the mask outline and the 40 D17 patch boxes (cyan)
    full/<n>_<file>.jpg  the crop at full resolution with the mask outline and the patch boxes
    index.csv            n, session, class folder, file, fallback, mask_area_frac, bean_frac_in_crop
A D18 fallback has no mask: the whole photo trains, every patch box accepted.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.dataset import load_bean_mask, load_rgb_image  # noqa: E402
from coffeecv.seg_overlay import outline  # noqa: E402
from p4_recrop_compare import NEW_ROOT, boxes, reports  # noqa: E402

TILE = 520
CYAN, YELLOW = (0, 230, 255), (255, 220, 0)


def draw_boxes(img: Image.Image, bxs: list[tuple], scale: float, colour, width: int) -> None:
    d = ImageDraw.Draw(img)
    for y0, y1, x0, x1 in bxs:
        d.rectangle([x0 * scale, y0 * scale, x1 * scale, y1 * scale], outline=colour, width=width)


def thumb(rgb: np.ndarray, side: int) -> tuple[Image.Image, float]:
    s = side / max(rgb.shape[:2])
    return Image.fromarray(rgb).resize((max(1, round(rgb.shape[1] * s)), max(1, round(rgb.shape[0] * s)))), s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_root", type=Path)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "ml8_p4")
    args = ap.parse_args()
    cfg = RunConfig.from_params_yaml()
    (args.out / "full").mkdir(parents=True, exist_ok=True)
    new = []
    for old_dir in sorted(d for d in args.old_root.iterdir() if d.is_dir()):
        old, now = reports(old_dir), reports(NEW_ROOT / old_dir.name)
        new += [(old_dir.name, c, f, now[(c, f)]) for c, f in sorted(set(now) - set(old))]
    tiles, rows = [], []
    font = ImageFont.load_default(size=18)
    for n, (session, cls, file, rep) in enumerate(new, 1):
        crop_path = NEW_ROOT / session / cls / Path(rep["out_path"]).name
        crop = load_rgb_image(crop_path)
        mask = load_bean_mask(crop_path, crop.shape[:2])
        bxs = boxes(crop_path, cfg)
        px = max(2, round(max(crop.shape[:2]) / 400))
        shown = outline(crop, mask.astype(bool), px) if mask is not None else crop.copy()
        full = Image.fromarray(shown)
        draw_boxes(full, bxs, 1.0, CYAN, px)
        full.save(args.out / "full" / f"{n:02d}_{Path(file).stem}.jpg", quality=90)

        raw = load_rgb_image(REPO_ROOT / "dataset" / session / cls / file)
        raw_t, s = thumb(raw, TILE)
        if rep["box"] is not None and mask is not None:
            x0, y0, w, h = rep["box"]
            canvas = np.zeros(raw.shape[:2], bool)
            canvas[y0:y0 + h, x0:x0 + w] = mask.astype(bool)
            raw_t = Image.fromarray(outline(np.asarray(raw_t), np.asarray(
                Image.fromarray(canvas).resize(raw_t.size, Image.NEAREST)), 2))
            ImageDraw.Draw(raw_t).rectangle([x0 * s, y0 * s, (x0 + w) * s, (y0 + h) * s], outline=YELLOW, width=3)
        crop_t, cs = thumb(shown, TILE)
        draw_boxes(crop_t, bxs, cs, CYAN, 1)
        tile = Image.new("RGB", (2 * TILE + 10, TILE + 30), "white")
        tile.paste(raw_t, (0, 30))
        tile.paste(crop_t, (TILE + 10, 30))
        label = (f"{n:02d}  {cls.split('__')[0]} {file}  "
                 + ("D18 FALLBACK: whole photo trains" if rep["fallback"] else
                    f"mask {rep['mask_area_frac']:.2f} of photo, beans {rep['bean_frac_in_crop']:.2f} of crop"))
        ImageDraw.Draw(tile).text((6, 5), label, fill="black", font=font)
        tiles.append(tile)
        rows.append({"n": n, "session": session, "class_folder": cls, "file": file, "fallback": rep["fallback"],
                     "mask_area_frac": round(rep["mask_area_frac"], 4),
                     "bean_frac_in_crop": rep["bean_frac_in_crop"] and round(rep["bean_frac_in_crop"], 4)})
    cols = 2
    sheet = Image.new("RGB", (cols * (2 * TILE + 20), -(-len(tiles) // cols) * (TILE + 40)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (2 * TILE + 20), (i // cols) * (TILE + 40)))
    sheet.save(args.out / "sheet.jpg", quality=88)
    with open(args.out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} new photos -> {args.out}/sheet.jpg, {args.out}/full/, {args.out}/index.csv")
    print(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
