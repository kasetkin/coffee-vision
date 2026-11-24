"""ML-37 bits for the classifier, levels 2 and 3: DINOv3 features of exp290's test patches in every arm.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_cls_embed.py [--limit N] [--threads 8]

exp290 is the live head's run (models/dino3b16_segxl0v7r_country_aug8_ml29gm_s123.pt, seed 123; webapp/app.py's
default checkpoint). Its test split is 266 photos the head never trained on, 160 patches a class, 1,600 in all
(predictions_test.csv). Each photo's boxes are drawn as _extract_photo drew them for exp290 (seed 123, split test,
the photo's rank among its class's test photos; bits_cls_check_boxes.txt checks the draw), from the float crop of
bits_cls_common. Per box, the model input of every arm (bits_cls_common.ARMS) and three yardsticks of changes that
should not matter:
  shift1   the f32 arm on the box moved by one crop pixel right and down (left/up at the crop's edge)
  fp16in   the f32 arm's normalised input rounded to float16 (the cast an fp16 serving graph applies first, ADR 0019)
  (the re-encode yardstick is bits_cls_noise.py's)
Each is embedded by the frozen dinov3_vitb16 (cls_mean, the shipped readout) in float32 on CPU. Also per arm, the
RMS and largest difference of the model input from the f32 arm's, in 8-bit levels (normalised difference * std * 255).

Features go to tmp/ml37/bits_cls/embed/<rank>.npz, one per photo (a rerun skips the photos done); bits_cls_predict.py
reads them. Log: tmp/ml37/bits_cls/embed.log.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from analysis.ml37 import bits_cls_common as bc
from analysis.ml37.bits_cls_check_boxes import exp290_test
from coffeecv.backbones import build_backbone
from coffeecv.geometry import Region

OUT_DIR = bc.SCRATCH / "embed"
FORMS = (*bc.ARMS, "shift1", "fp16in")
BATCH = 32


def shifted(box: Region, h: int, w: int) -> Region:
    dx = 1 if box.x1 + 1 <= w else -1
    dy = 1 if box.y1 + 1 <= h else -1
    return Region(y0=box.y0 + dy, y1=box.y1 + dy, x0=box.x0 + dx, x1=box.x1 + dx)


def embed(bb, xs: list[np.ndarray]) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), BATCH):
            out.append(bb.features(torch.from_numpy(np.stack(xs[i:i + BATCH])))["cls_mean"].float().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos (a toy run)")
    ap.add_argument("--threads", type=int, default=bc.THREADS)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    geo = bc.Geometry.from_params()
    entries = bc.pool_entries()
    photos = exp290_test()[:a.limit]
    bb = build_backbone("dinov3_vitb16")
    t_start = time.time()
    scale = (bc.STD * 255).astype(np.float32)
    for rank, t in enumerate(photos):
        out = OUT_DIR / f"{rank:04d}.npz"
        if out.exists():
            continue
        e = entries[t["photo"]]
        x, mask = bc.float_crop(e, a.threads)
        u8 = np.rint(x * 255).astype(np.uint8)
        boxes, pitch = geo.boxes(u8, mask, 123, t["class_idx"], t["photo_idx"], "test", t["n"])
        u8 = None
        h, w = x.shape[:2]
        inputs = {k: [] for k in FORMS}
        for box, _, _ in boxes:
            px = bc.cut(x, box)
            for arm in bc.ARMS:
                inputs[arm].append(bc.model_input(px, arm))
            inputs["shift1"].append(bc.model_input(bc.cut(x, shifted(box, h, w)), "f32"))
            inputs["fp16in"].append(inputs["f32"][-1].astype(np.float16).astype(np.float32))
        x = None
        ref = np.stack(inputs["f32"])
        diff = {}
        for k in FORMS:
            d = (np.stack(inputs[k]) - ref) * scale
            diff[k] = (float(np.sqrt(np.mean(d ** 2))), float(np.abs(d).max()))
        order = [k for k in FORMS]
        flat = [v for k in order for v in inputs[k]]
        inputs = None
        f = embed(bb, flat).reshape(len(order), len(boxes), -1)
        np.savez(out, **{k: f[i] for i, k in enumerate(order)},
                 boxes=np.array([[b.y0, b.y1, b.x0, b.x1] for b, _, _ in boxes]),
                 input_rms=np.array([diff[k][0] for k in order]), input_max=np.array([diff[k][1] for k in order]),
                 forms=np.array(order), photo=t["photo"], cls=t["class"], session=e["session"], pitch=pitch)
        el = time.time() - t_start
        print(f"[{time.strftime('%H:%M:%S')}] {rank + 1}/{len(photos)} {t['photo'][:50]} {len(boxes)} boxes, "
              f"{el / 60:.1f} min, rss {bc.rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
