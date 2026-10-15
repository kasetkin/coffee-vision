"""ML-7 P8: the segmenter's ports against the torch code they replaced, on P7's 782 answer-check photos (negatives
included), on one machine, so CPU differences do not enter.

Per photo: the encoder's input from coffeecv.model_io.sam_input against EfficientViT-SAM's transform (bit for bit),
then the served decoder's selected low-resolution mask (multi3, the box prompt) upsampled twice: by SAM's
postprocess_masks (F.interpolate, the code before P8) and by model_io.upsample_mask (cv2, the port). Reports the
pixels that differ, the IoU, and whether the D18 fallback or the D4 crop box changes, which is what training's
segcrop stage stores. The plan's rule: if the port moves stored mask bits, the owner hears how many before any DVC
stage is rerun or committed.

    python analysis/ml7_gpu/p8_ports.py --out analysis/ml7_gpu/p8_ports.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from coffeecv import model_io  # noqa: E402
from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.photo_io import load_rgb_image  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, d4_box, seg_params  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", type=Path, default=REPO / "analysis/ml7_gpu/p7_answer_photos.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    rows = [ln.split() for ln in args.list.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    cfg = RunConfig.from_params_yaml()
    p = seg_params(cfg)
    seg = BeanSegmenter(p)
    sam = sys.modules["efficientvit.models.efficientvit.sam"]
    transform = seg.model.transform
    stand_in = types.SimpleNamespace(image_size=seg.model.image_size)
    k = int(p.mask_select[-1]) - 1
    out, t0 = [], time.time()
    for i, (role, sha, path) in enumerate(rows, 1):
        rgb = load_rgb_image(REPO / path)
        x, _ = model_io.sam_input(rgb, seg.engine.image_side)
        with torch.inference_mode():
            same_input = bool(np.array_equal(x, transform(rgb).unsqueeze(0).numpy()))
        seg._encode(rgb)
        low, _ = seg._decode(*rgb.shape[:2], multimask=True)
        new = seg._upsample(low[k])
        with torch.inference_mode():
            old = (sam.EfficientViTSam.postprocess_masks(stand_in, torch.from_numpy(low[k])[None, None],
                                                         seg.input_size, seg.original_size)[0, 0] > 0).numpy()
        n_old, n_new, union = int(old.sum()), int(new.sum()), int((old | new).sum())
        area = rgb.shape[0] * rgb.shape[1]
        fb_old, fb_new = (n_old == 0 or n_old / area < p.min_area_frac), (n_new == 0 or n_new / area < p.min_area_frac)
        r = {"role": role, "sha256": sha, "path": path, "same_input": same_input,
             "pixels_differ": int((old != new).sum()), "iou": (int((old & new).sum()) / union) if union else 1.0,
             "fallback_old": fb_old, "fallback_new": fb_new,
             "box_same": None if fb_old or fb_new else d4_box(old, p.keep_frac) == d4_box(new, p.keep_frac)}
        out.append(r)
        if i % 25 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} {time.time() - t0:.0f} s; differing photos so far "
                  f"{sum(q['pixels_differ'] > 0 for q in out)}", flush=True)
    summary = {"photos": len(out), "negatives": sum(r["role"] == "neg_ood" for r in out),
               "input_not_bit_identical": sum(not r["same_input"] for r in out),
               "photos_with_mask_pixels_moved": sum(r["pixels_differ"] > 0 for r in out),
               "max_pixels_moved": max(r["pixels_differ"] for r in out),
               "least_iou": min(r["iou"] for r in out),
               "fallback_changed": sum(r["fallback_old"] != r["fallback_new"] for r in out),
               "d4_box_changed": sum(r["box_same"] is False for r in out),
               "threads": args.threads, "mask_select": p.mask_select}
    args.out.write_text(json.dumps({"summary": summary, "photos": out}, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
