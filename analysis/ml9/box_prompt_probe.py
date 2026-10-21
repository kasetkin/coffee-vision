"""Ticket ML-9 P1 (and ML-10): does a segmenter follow a box prompt smaller than the photo?

    python analysis/ml9/box_prompt_probe.py > analysis/ml9/box_prompt_probe.txt

On 10 pool photos the owner accepted in pass3 (seed 7), each model's multi3 mask from three box prompts, against
the owner's mask: the whole photo (serving's prompt), the accepted mask's bounding box ("tight"), and that box's
left half ("half"). For the half box: IoU with the left half of the owner's mask, and the share of the model's
mask outside the box. A model that follows the box keeps that share near 0; one that ignores it near 0.5.
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coffeecv.config import REPO_ROOT  # noqa: E402
from coffeecv.dataset import load_rgb_image  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, named_params  # noqa: E402

MODELS = ("xl0_v3", "pretrained_xl0")
N, SEED = 10, 7


def decode(seg: BeanSegmenter, box: list[int]) -> np.ndarray:
    low, _ = seg.engine.decode(*seg._prompt(box, [], []), True)
    return seg._upsample(low[2])                                      # multi3, as the review page's redraws


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def main() -> None:
    torch.set_num_threads(4)
    rows = [r for r in csv.DictReader(open(REPO_ROOT / "data/ml5_labels/pass3/labels.csv"))
            if r["status"] == "accepted" and r["source"] == "pool"]
    random.Random(SEED).shuffle(rows)
    segs = {m: BeanSegmenter(named_params(m, "multi3")) for m in MODELS}
    res = {m: {"whole": [], "tight": [], "half_iou": [], "half_out": []} for m in MODELS}
    print("photo  " + "  ".join(f"{m}: half_out" for m in MODELS))
    for r in rows[:N]:
        rgb = load_rgb_image(REPO_ROOT / r["path"])
        lab = np.array(Image.open(REPO_ROOT / r["mask"]).convert("1"))
        h, w = lab.shape
        ys, xs = np.nonzero(lab)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        xm = (x0 + x1) // 2
        half = np.zeros_like(lab)
        half[:, x0:xm + 1] = True
        for m, seg in segs.items():
            seg._encode(rgb)
            res[m]["whole"].append(iou(decode(seg, [0, 0, w - 1, h - 1]), lab))
            res[m]["tight"].append(iou(decode(seg, [x0, y0, x1, y1]), lab))
            mask = decode(seg, [x0, y0, xm, y1])
            res[m]["half_iou"].append(iou(mask, lab & half))
            res[m]["half_out"].append(float((mask & ~half).sum() / max(mask.sum(), 1)))
        print(Path(r["path"]).name + "  " + "  ".join(f"{res[m]['half_out'][-1]:.3f}" for m in MODELS), flush=True)
    print()
    for m, d in res.items():
        print(m)
        for k, v in d.items():
            print(f"  {k:<9} median {np.median(v):.3f}  min {np.min(v):.3f}  max {np.max(v):.3f}")


if __name__ == "__main__":
    main()
