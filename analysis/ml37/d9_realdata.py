"""ML-37 D9: the read-back's maths on real canonical photos (data/canonical, read only), four ways.

Today's canonical photos are Rec.2020 with the BT.709 curve in uint16, not D1's linear half-float, so D1's input is
emulated: the decoded samples are taken to linear Rec.2020 by LittleCMS (committed REC2020 -> LINEAR_REC2020, float32,
which undoes the profile's own curve exactly) and rounded to float16, as D1 stores them. The lossy re-encode is not
emulated: it is the same for every option. Then, with the clip as the tone map (nothing in today's photos is above
white):

  A  today:           rec2020_to_srgb on the uint16 samples (LittleCMS NOOPTIMIZE -> uint8)
  B  float, today:    LittleCMS REC2020 -> SRGB on the uint16 samples, float32 out (today's file without the 8-bit step)
  C  option (b):      LittleCMS LINEAR_REC2020 -> SRGB on the float16 linear samples, float32 out, clipped
  D  option (a):      numpy float32 on the float16 linear samples: exact matrix, clip, IEC sRGB curve in closed form
  R  reference:       the same as D in float64

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d9_realdata.py > analysis/ml37/d9_realdata.txt
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn

STORE = Path("/workspace/data/canonical")
SAMPLE = 20
STRIP = 256
P709 = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
P2020 = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))


def npm(primaries, white=(0.3127, 0.3290)):
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = white
    return xyz * np.linalg.solve(xyz, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))


M64 = np.linalg.inv(npm(P709)) @ npm(P2020)
M32 = M64.astype(np.float32)


def encode(lin: np.ndarray) -> np.ndarray:
    """IEC 61966-2-1 in the array's own dtype, in place where it can: lin is already clipped to [0, 1]."""
    t = lin.dtype.type
    low = lin <= t(0.0031308)
    out = np.power(lin, t(1 / 2.4))
    out *= t(1.055)
    out -= t(0.055)
    out[low] = lin[low] * t(12.92)
    return out


def numpy_readback(lin: np.ndarray, m: np.ndarray, dtype) -> np.ndarray:
    """(h, w, 3) linear Rec.2020 (any float) -> sRGB in `dtype`: matrix, clip to [0, 1], curve."""
    x = lin.astype(dtype, copy=False)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    out = np.empty(x.shape, dtype)
    for i in range(3):   # nine multiply-adds: exact per pixel, no BLAS threads
        o = out[..., i]
        np.multiply(r, dtype(m[i, 0]), out=o)
        o += g * dtype(m[i, 1])
        o += b * dtype(m[i, 2])
    np.clip(out, 0, 1, out=out)
    return encode(out)


def sample() -> list[Path]:
    paths = sorted(STORE.rglob("*.jxl"))
    step = len(paths) / SAMPLE
    return [paths[int(i * step)] for i in range(SAMPLE)]


class Acc:
    def __init__(self):
        self.max = 0.0
        self.sum = 0.0
        self.n = 0
        self.flips = 0

    def add(self, d: np.ndarray, flips: int = 0):
        a = np.abs(d)
        self.max = max(self.max, float(a.max()))
        self.sum += float(a.sum(dtype=np.float64))
        self.n += a.size
        self.flips += flips

    def line(self, scale=255.0, unit="8-bit levels"):
        return (f"max {self.max * scale:.5f} {unit}, mean {self.sum / self.n * scale:.6f}, "
                f"rounding flips {self.flips} of {self.n} samples ({self.flips / self.n:.4%})")


def main() -> None:
    files = sample()
    keys = ["C-R", "D-R", "C-D", "B-D", "A-D", "A-B"]
    total = {k: Acc() for k in keys}
    print(f"{len(files)} photos, every {len(sorted(STORE.rglob('*.jxl'))) // SAMPLE}th of the store's sorted paths")
    for p in files:
        rec = ic.jpegxl_decode(p.read_bytes(), numthreads=1)
        h, w, _ = rec.shape
        per = {k: Acc() for k in keys}
        neg = 0
        for y in range(0, h, STRIP):
            u = rec[y:y + STRIP]
            a = cn.rec2020_to_srgb(u, numthreads=1).astype(np.float32) / 255
            b = ic.cms_transform(u, cn.REC2020, cn.SRGB, intent=cn.RELATIVE, outdtype=np.float32)
            np.clip(b, 0, 1, out=b)
            lin = ic.cms_transform(u, cn.REC2020, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)
            neg += int((lin < 0).sum())
            lin16 = lin.astype(np.float16)
            c = ic.cms_transform(lin16, cn.LINEAR_REC2020, cn.SRGB, intent=cn.RELATIVE, outdtype=np.float32)
            np.clip(c, 0, 1, out=c)
            d = numpy_readback(lin16, M32, np.float32)
            r = numpy_readback(lin16, M64, np.float64)
            r16 = np.rint(r * 65535)
            for k, x, yv, flips in (
                    ("C-R", c, r, int((np.rint(c * 65535) != r16).sum())),
                    ("D-R", d, r, int((np.rint(d * 65535) != r16).sum())),
                    ("C-D", c, d, int((np.rint(c * 65535) != np.rint(d * 65535)).sum())),
                    ("B-D", b, d, int((np.rint(b * 255) != np.rint(d * 255)).sum())),
                    ("A-D", a, d, int((np.rint(a * 255) != np.rint(d * 255)).sum())),
                    ("A-B", a, b, int((np.rint(a * 255) != np.rint(b * 255)).sum()))):
                dd = x.astype(np.float64) - yv
                per[k].add(dd, flips)
                total[k].add(dd, flips)
        del rec
        print(f"\n{p.relative_to(STORE)} {w}x{h}; linear values below 0: {neg}")
        for k in keys:
            unit = ("16-bit codes", 65535.0) if k in ("C-R", "D-R", "C-D") else ("8-bit levels", 255.0)
            print(f"  {k}: {per[k].line(unit[1], unit[0])}")
    print("\n# all photos (flips: C-R, D-R, C-D at 16 bits, D10's crop store; the rest at 8 bits)")
    for k in keys:
        unit = ("16-bit codes", 65535.0) if k in ("C-R", "D-R", "C-D") else ("8-bit levels", 255.0)
        print(f"{k}: {total[k].line(unit[1], unit[0])};  in 8-bit levels max {total[k].max * 255:.5f} mean "
              f"{total[k].sum / total[k].n * 255:.6f}")


if __name__ == "__main__":
    main()
