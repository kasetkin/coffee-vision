"""ML-37 D11: PIL "F" and torch antialias from a planar (3 x H x W) float32 picture, as a read-back could write it.

Research only. Synthetic smooth-plus-noise float32 [0, 1] at 15 MP (timing does not depend on content), 2 threads,
median of 7 runs. Writes analysis/ml37/d11_resize_speed_planar.txt.
"""
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

torch.set_num_threads(2)
H, W, SIZE = 3348, 4480, (1024, 765)
POOL = ThreadPoolExecutor(2)


def bench(fn, n=7):
    fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * statistics.median(ts)


def one(plane):
    return np.asarray(Image.fromarray(plane).resize(SIZE, Image.BILINEAR))


def main():
    rng = np.random.default_rng(0)
    chw = np.empty((3, H, W), np.float32)
    for c in range(3):
        chw[c] = rng.random((H, W), dtype=np.float32)
    t = torch.from_numpy(chw)[None]
    lines = ["ML-37 D11: from a planar 3 x 3348 x 4480 float32 picture to 1024 x 765, 2 threads, ms (median of 7)",
             f"  pil_f, planes in turn                {bench(lambda: [one(chw[c]) for c in range(3)]):8.1f}",
             f"  pil_f, planes on a 2-thread pool     {bench(lambda: list(POOL.map(one, chw))):8.1f}",
             f"  pil_f, one plane                     {bench(lambda: one(chw[0])):8.1f}",
             f"  torch antialias bilinear             "
             f"{bench(lambda: F.interpolate(t, size=(765, 1024), mode='bilinear', align_corners=False, antialias=True)):8.1f}"]
    Path(__file__).with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


main()
