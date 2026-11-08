"""ML-19: the share of a photo's power above the Nyquist limit of its 50 MP downscale, in three 2048 px tiles."""
import glob
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 2**28
TILE = 2048
for f in sorted(glob.glob("/tmp/ml16/q13/orig/*")):
    g = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
    h, w = g.shape
    scale = (50e6 / (w * h)) ** 0.5
    cut = 0.5 * scale  # the downscale's Nyquist, in cycles per original pixel
    shares = []
    for cy, cx in [(h // 2, w // 2), (h // 3, w // 3), (2 * h // 3, 2 * w // 3)]:
        t = g[cy - TILE // 2:cy + TILE // 2, cx - TILE // 2:cx + TILE // 2]
        t = (t - t.mean()) * np.outer(np.hanning(TILE), np.hanning(TILE))
        power = np.abs(np.fft.fft2(t)) ** 2
        fy = np.fft.fftfreq(TILE)[:, None]
        fx = np.fft.fftfreq(TILE)[None, :]
        shares.append(power[np.maximum(np.abs(fy), np.abs(fx)) > cut].sum() / power.sum())
    print(f"{os.path.basename(f)[:40]:40s} scale {scale:.2f}  power above the new Nyquist: "
          + ", ".join(f"{x * 100:.3f}%" for x in shares))
