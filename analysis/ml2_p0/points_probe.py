"""ML-2 P0 follow-up: can the D6 correction loop's prompt (whole-image box + include/exclude points)
make pretrained L0 draw the bean pile on a tray photo, where every fixed prompt fails?

The points stand in for a judge's corrective points. They were placed by hand from the P0 contact
sheets, as fractions of width/height: include points on the pile, exclude points on the table/paper
and the tray walls. This is a feasibility probe for the fine-tuning route (P5), not a label.

    PYTHONPATH=. python analysis/ml2_p0/points_probe.py [--variant l1]   # writes outputs/ml2_p0/points_probe[_l1]/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from analysis.ml2_p0.contact_sheet import outline
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import load_rgb_image
from coffeecv.sam_loader import VARIANTS, weights_for
from coffeecv.segment_beans import BeanSegmenter, SegParams

OUT = REPO_ROOT / "outputs" / "ml2_p0" / "points_probe"
GRID = [(x, y) for y in (0.38, 0.5, 0.62) for x in (0.4, 0.5, 0.6)]
TRAY_BOX_PICTURES = {"include": GRID,
                     "exclude": [(0.15, 0.5), (0.85, 0.5), (0.5, 0.15), (0.5, 0.8)]}
TRAY_BOX_TABLE = {"include": [(x, y) for y in (0.35, 0.48, 0.6) for x in (0.3, 0.5, 0.7)],
                  "exclude": [(0.5, 0.07), (0.5, 0.93), (0.03, 0.5), (0.96, 0.5), (0.5, 0.79)]}
PHOTOS = {
    "dataset/2026-08-07__box_pictures_all_classes/class_001__Ethiopia_Sidamo/PXL_20260807_082212145.jpg": TRAY_BOX_PICTURES,
    "dataset/2026-08-07__box_pictures_all_classes/class_008__Ethiopia_Kochere/PXL_20260807_084902479.jpg": TRAY_BOX_PICTURES,
    "dataset/2026-08-25__iphone/class_006__Brazil_Cerrado/IMG_6325.HEIC": TRAY_BOX_TABLE,
    "dataset/2026-08-25__oneplus/class_006__Brazil_Cerrado/PXL_20260825_192721855.jpg": TRAY_BOX_TABLE,
}


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="l0")
    v = ap.parse_args().variant
    out = OUT if v == "l0" else OUT.with_name(f"{OUT.name}_{v}")
    out.mkdir(parents=True, exist_ok=True)
    seg = BeanSegmenter(SegParams(mask_select="single", prompt="box", variant=v, weights=weights_for(v)))
    pr = seg.predictor
    for rel, pts in PHOTOS.items():
        rgb = load_rgb_image(REPO_ROOT / rel)
        h, w = rgb.shape[:2]
        seg._encode(rgb)
        xy = np.array([(fx * (w - 1), fy * (h - 1)) for fx, fy in pts["include"] + pts["exclude"]])
        labels = np.array([1] * len(pts["include"]) + [0] * len(pts["exclude"]))
        box = np.array([0, 0, w - 1, h - 1], float)
        for name, kwargs in (("points", {}), ("box+points", {"box": box})):
            masks, iou, _ = pr.predict(point_coords=xy, point_labels=labels, multimask_output=False, **kwargs)
            m = masks[0]
            s = 900 / max(h, w)
            small = np.array(Image.fromarray(rgb).resize((round(w * s), round(h * s))))
            ms = np.array(Image.fromarray(m).resize(small.shape[1::-1], Image.NEAREST))
            im = Image.fromarray(outline(small, ms, 1, tint=0.35))
            d = ImageDraw.Draw(im)
            for (x, y), lab in zip(xy * s, labels):
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(0, 255, 0) if lab else (255, 0, 0), outline="black")
            d.text((8, 8), f"{name} iou={float(iou[0]):.3f} area={m.mean():.3f}", fill=(255, 0, 255))
            im.save(out / f"{Path(rel).stem}__{name}.jpg", quality=90)
            print(Path(rel).name, name, f"iou={float(iou[0]):.3f} area={m.mean():.3f}")


if __name__ == "__main__":
    main()
