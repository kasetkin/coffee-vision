"""ML-37 D11: speed of each float resizer against today's 8-bit PIL bilinear, on 2 threads.

Research only (analysis/ml37/research_d11_float_resize_and_augment.md). Writes analysis/ml37/d11_resize_speed.txt.

A 15 MP photo (a canonical 5056x3792 photo's centre, 4480x3348) to the SAM side (1024 long), and a batch of 64
patches 448 -> 224 and 64 crops 800 -> 448 (the patch store), float32 [0, 1] read back as in d11_resize_parity.py.
Median of 7 runs after one warm-up. "pool2" runs PIL's three channels on a 2-thread pool (Pillow releases the GIL
while it resamples: ImagingSectionEnter in Resample.c).
"""
from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv import model_io

import importlib.util

torch.set_num_threads(2)
cv2.setNumThreads(2)
_spec = importlib.util.spec_from_file_location("parity", Path(__file__).with_name("d11_resize_parity.py"))
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_suffix(".txt")
PHOTO = REPO / "data/canonical/2026-08-30__sony/class_010__Indonesia_Java/PIC_20260830_200703.JPG.jxl"
POOL = ThreadPoolExecutor(2)


def pil_f_pool(a, size):
    def one(c):
        return np.asarray(Image.fromarray(np.ascontiguousarray(a[..., c])).resize(size, Image.BILINEAR))
    return np.stack(list(POOL.map(one, range(3))), axis=2)


def torch_aa_chw(t, size):
    return F.interpolate(t, size=(size[1], size[0]), mode="bilinear", align_corners=False, antialias=True)


def bench(fn, n=7) -> float:
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def main() -> None:
    u8, f32 = parity.read_back(PHOTO)
    h, w = u8.shape[:2]
    ch, cw = 3348, 4480
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    u8 = np.ascontiguousarray(u8[y0:y0 + ch, x0:x0 + cw])
    f32 = np.ascontiguousarray(f32[y0:y0 + ch, x0:x0 + cw])
    nh, nw = model_io.sam_preprocess_shape(ch, cw, 1024)
    size = (nw, nh)
    lines = [f"ML-37 D11 resize speed, 2 threads (torch {torch.get_num_threads()}, cv2 {cv2.getNumThreads()}); "
             f"median of 7 runs, ms",
             f"photo {cw}x{ch} ({cw * ch / 1e6:.1f} MP) -> {nw}x{nh}:"]
    chw = torch.from_numpy(np.ascontiguousarray(f32.transpose(2, 0, 1)))[None]
    rows = [
        ("pil_u8 (today, uint8 RGB)", lambda: parity.pil_u8(u8, size)),
        ("pil_f, 3 channels in turn", lambda: parity.pil_f(f32, size)),
        ("pil_f, 3 channels on a 2-thread pool", lambda: pil_f_pool(f32, size)),
        ("cv2 INTER_LINEAR float32", lambda: parity.cv_linear(f32, size)),
        ("cv2 INTER_AREA float32", lambda: parity.cv_area(f32, size)),
        ("torch antialias, from HWC (incl. transposes)", lambda: parity.torch_aa(f32, size)),
        ("torch antialias, from CHW", lambda: torch_aa_chw(chw, size)),
    ]
    for name, fn in rows:
        lines.append(f"  {name:<46} {bench(fn):9.1f}")
    chw = None

    rng = np.random.default_rng(0)
    for src, dst, label in ((448, 224, "eval_resize"), (800, 448, "patch store")):
        n = 64
        ys = rng.integers(0, ch - src, n)
        xs = rng.integers(0, cw - src, n)
        pu = [np.ascontiguousarray(u8[y:y + src, x:x + src]) for y, x in zip(ys, xs)]
        pf = [np.ascontiguousarray(f32[y:y + src, x:x + src]) for y, x in zip(ys, xs)]
        batch = torch.from_numpy(np.stack([p.transpose(2, 0, 1) for p in pf]))
        d = (dst, dst)
        lines.append(f"batch of {n} patches {src} -> {dst} ({label}):")
        rows = [
            ("pil_u8 (today)", lambda: [parity.pil_u8(p, d) for p in pu]),
            ("pil_f, channels in turn", lambda: [parity.pil_f(p, d) for p in pf]),
            ("pil_f, 2-thread pool over patches", lambda: list(POOL.map(lambda p: parity.pil_f(p, d), pf))),
            ("cv2 INTER_LINEAR float32", lambda: [parity.cv_linear(p, d) for p in pf]),
            ("cv2 INTER_AREA float32", lambda: [parity.cv_area(p, d) for p in pf]),
            ("torch antialias, per patch from HWC", lambda: [parity.torch_aa(p, d) for p in pf]),
            ("torch antialias, one N x 3 x H x W call", lambda: torch_aa_chw(batch, d)),
        ]
        for name, fn in rows:
            lines.append(f"  {name:<46} {bench(fn):9.1f}")
        pu = pf = batch = None
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
