"""ML-37 P2's check: the models' inputs and the segmenter's masks with the float resizes (D11 (b)) against today's.

Runs the plan's P2 check (docs/ml-0037_plan_linear_canonical_photo.html): on the 13 canonical photos of D11's
research (analysis/ml37/d11_resize_parity.py's draw, one per capture folder, seed 37011), each read back as the
models read it today (canonical.load_canonical, 8-bit sRGB; P2 leaves the read-back alone):

    segmenter input   model_io.sam_resize (float, as built) against today's 8-bit SamResize (PIL "RGB" bilinear),
                      in 8-bit levels
    DINOv3 input      12 square crops per photo, sides log-uniform 224-1100 px, stored at 448 by PIL's 8-bit
                      bilinear (the patch store, unchanged, D10), then model_io.eval_resize to 224 (float, as built)
                      against today's 8-bit resize to 224, in 8-bit levels
    masks             the live segmenter (xl0_v7r_s123, box prompt, multi3, torch at 2 threads) through
                      BeanSegmenter.predict_mask as built, against the same with today's 8-bit sam_input; IoU

D11's research bound is 2.02 levels over the classifier's two resizes with a float read-back; with P2's 8-bit
read-back, only one rounding differs per resize, so the bound expected here is 1.0. Writes
analysis/ml37/p2_inputs_and_masks.txt. Reads data/canonical (DVC) and models_pretrained (not in git); writes
nothing else. --data-root reads the store from another checkout of the repo (a worktree has no DVC data).

    python analysis/ml37/p2_inputs_and_masks.py [--data-root /path/to/checkout]
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d11_resize_parity as d11  # noqa: E402

from coffeecv import canonical, model_io  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, named_params  # noqa: E402

OUT = Path(__file__).with_suffix(".txt")
STORE, RESIZE, SIDE = 448, 224, 1024
PATCHES = 12
THREADS = 2


def today_sam_input(rgb: np.ndarray, side: int):
    """model_io.sam_input before ML-37 P2: PIL's 8-bit bilinear, rounded, then / 255 and normalised."""
    h, w = rgb.shape[:2]
    nh, nw = model_io.sam_preprocess_shape(h, w, side)
    small = rgb if (nh, nw) == (h, w) else np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    x = np.zeros((1, 3, side, side), np.float32)
    x[0, :, :nh, :nw] = model_io._to_chw_normalised(small.astype(np.float32) / np.float32(255),
                                                     model_io.SAM_MEAN, model_io.SAM_STD)
    return x, (nh, nw)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=d11.REPO)
    d11.REPO = ap.parse_args().data_root
    torch.set_num_threads(THREADS)
    seg = BeanSegmenter(named_params("xl0_v7r_s123"))
    rng = np.random.default_rng(37011)
    lines = ["ML-37 P2: the float resizes (D11 (b)) against today's 8-bit ones, on D11's 13 photos. "
             "analysis/ml37/p2_inputs_and_masks.py", "Levels: |float x 255 - today's uint8|. Masks: xl0_v7r_s123, "
             f"box prompt, multi3, torch at {THREADS} threads; IoU of the mask as built against today's.", "",
             f"{'folder':<38}{'size':>11}{'seg max':>9}{'seg mean':>9}{'dino max':>9}{'dino mean':>10}"
             f"{'area':>8}{'IoU':>9}"]
    seg_max = dino_max = 0.0
    ious, seg_means, dino_means = [], [], []
    photos = d11.sample_photos()
    assert len(photos) == d11.N_PHOTOS
    for path in photos:
        rgb = canonical.load_canonical(path.read_bytes())
        h, w = rgb.shape[:2]
        nh, nw = model_io.sam_preprocess_shape(h, w, SIDE)
        today = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)).astype(np.float32)
        d = np.abs(model_io.sam_resize(rgb, SIDE) * 255 - today)
        s_max, s_mean = float(d.max()), float(d.mean())

        dd = []
        for _ in range(PATCHES):
            side = int(round(np.exp(rng.uniform(np.log(224), np.log(min(1100, h, w))))))
            y, x = int(rng.integers(0, h - side + 1)), int(rng.integers(0, w - side + 1))
            patch = Image.fromarray(rgb[y:y + side, x:x + side]).resize((STORE, STORE), Image.BILINEAR)
            old = np.asarray(patch.resize((RESIZE, RESIZE), Image.BILINEAR)).astype(np.float32)
            dd.append(np.abs(model_io.eval_resize(patch, RESIZE) * 255 - old))
        dd = np.stack(dd)
        p_max, p_mean = float(dd.max()), float(dd.mean())

        new_mask = seg.predict_mask(rgb)
        with mock.patch.object(model_io, "sam_input", today_sam_input):
            old_mask = seg.predict_mask(rgb)
        j = iou(new_mask, old_mask)

        folder = path.parts[path.parts.index("canonical") + 1]
        lines.append(f"{folder:<38}{f'{w}x{h}':>11}{s_max:>9.3f}{s_mean:>9.4f}{p_max:>9.3f}{p_mean:>10.4f}"
                     f"{float(old_mask.mean()):>8.4f}{j:>9.5f}")
        print(lines[-1], file=sys.stderr, flush=True)
        seg_max, dino_max = max(seg_max, s_max), max(dino_max, p_max)
        ious.append(j)
        seg_means.append(s_mean)
        dino_means.append(p_mean)
    lines += ["", f"segmenter input: max {seg_max:.3f} levels, mean {np.mean(seg_means):.4f}",
              f"DINOv3 input:    max {dino_max:.3f} levels, mean {np.mean(dino_means):.4f}",
              f"within 2.02 levels (D11's research bound): {max(seg_max, dino_max) <= 2.02}",
              f"mask IoU against today's: median {np.median(ious):.5f}, min {min(ious):.5f}"]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-5:]))


if __name__ == "__main__":
    main()
