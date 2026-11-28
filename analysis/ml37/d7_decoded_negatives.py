"""ML-37 D7: does a canonical photo whose light was clipped at 0 before the encode (option a) decode without
negatives? Lossy XYB coding error can itself make samples below 0, which the read-back then has to clip anyway.

    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_decoded_negatives.py

Four dataset photos of the d7_dataset_gamut photos sample (one per colour source kind), decoded as the store does,
converted to linear Rec.2020 by LittleCMS float and clipped at 0 (values above 1.0 kept), the centre 1024x1024,
encoded as D1's canonical photo (float16, BT2100 primaries, linear, level 95, effort 7, two threads) and decoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coffeecv import canonical as cn  # noqa: E402
import d7_dataset_gamut as g  # noqa: E402

ENC = dict(level=95, effort=7, photometric="RGB", primaries="BT2100", numthreads=2)


def main():
    files, kinds = [], set()
    for f in g.pick():
        k = g.describe(f).split(",")[0]
        if k not in kinds:
            kinds.add(k)
            files.append(f)
    print("| photo | px < 0 before encode | decoded px < 0 | decoded px < -1e-3 | decoded lowest | read-back levels "
          "changed by clipping the decoded negatives (max) |")
    print("|---|---|---|---|---|---|")
    for f in files[:6]:
        src = cn.decode(f.read_bytes(), f.name)
        h, w = src.pixels.shape[:2]
        t = 1024
        y0, x0 = max(0, h // 2 - t // 2), max(0, w // 2 - t // 2)
        part = src.pixels[y0:y0 + t, x0:x0 + t]
        _, light = cn._light(part, src.full, src.linearize, 3)
        if src.profile is cn.LINEAR_REC2020 and src.pixels.dtype == np.float32:
            c = light
        else:
            c = ic.cms_transform(light, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                                 outdtype=np.float32)
        before = float((c < 0).any(-1).mean())
        c = np.maximum(c, 0)
        d = ic.jpegxl_decode(ic.jpegxl_encode(c.astype(np.float16), **ENC), numthreads=2).astype(np.float64)
        flat = d.reshape(-1, 3)
        diff = np.abs(g.readback(np.maximum(flat, 0)) - g.readback(flat)).max()
        print(f"| {g.describe(f)} | {before:.3%} | {float((flat < 0).any(-1).mean()):.3%} "
              f"| {float((flat < -1e-3).any(-1).mean()):.3%} | {float(flat.min()):+.4f} | {diff:.2f} |", flush=True)
        del src, part, light, c, d


if __name__ == "__main__":
    main()
