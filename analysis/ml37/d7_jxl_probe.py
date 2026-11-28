"""ML-37 D7: what libjxl (imagecodecs 2026.10.10, libjxl 0.12.0) does with negative linear Rec.2020 samples in a lossy
float16 encode, and what comes back on decode.

    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_jxl_probe.py

Encodes like D1's canonical photo would be (float16, photometric RGB, primaries BT2100, linear transfer, D65, level 95,
effort 7, two threads) and decodes with imagecodecs.jpegxl_decode. Three probes:
  1. flat 64x64 patches of chosen colours, some with a negative channel, read back as the mean of each patch's inner
     32x32; beside each, libjxl's own opsin "mixed" values (enc_xyb.cc, OpsinAbsorbance: linear sRGB times
     kOpsinAbsorbanceMatrix times intensity_target / 255, plus kOpsinAbsorbanceBias), which the encoder clamps at 0;
  2. a ramp of each ProPhoto primary and secondary, converted by LittleCMS (float) to linear Rec.2020 unclipped;
  3. a natural photo's light (a dataset photo, linear Rec.2020) with one out-of-gamut square laid in, encoded with the
     square's negatives kept and clipped at 0: how far the in-gamut pixels around it move between the two.
Writes nothing but its printout (analysis/ml37/d7_jxl_probe.txt when redirected).
"""
from __future__ import annotations

import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import canonical as cn  # noqa: E402

THREADS = 2
# linear Rec.2020 -> linear sRGB (BT.709), D65, from the primaries (the read-back's D2 matrix)


def rgb_to_xyz(prim, white):
    xr, yr, xg, yg, xb, yb = prim
    xw, yw = white
    m = np.array([[xr / yr, xg / yg, xb / yb], [1, 1, 1], [(1 - xr - yr) / yr, (1 - xg - yg) / yg, (1 - xb - yb) / yb]])
    w = np.array([xw / yw, 1, (1 - xw - yw) / yw])
    return m * np.linalg.solve(m, w)


D65 = (0.3127, 0.3290)
REC2020_PRIM = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046)
SRGB_PRIM = (0.640, 0.330, 0.300, 0.600, 0.150, 0.060)
SRGB_FROM_2020 = np.linalg.inv(rgb_to_xyz(SRGB_PRIM, D65)) @ rgb_to_xyz(REC2020_PRIM, D65)
OPSIN = np.array([[0.30, 1 - 0.078 - 0.30, 0.078], [0.23, 1 - 0.078 - 0.23, 0.078],
                  [0.24342268924547819, 0.20476744424496821, 1 - 0.24342268924547819 - 0.20476744424496821]])
BIAS = 0.0037930732552754493
INTENSITY_TARGET = 255.0           # libjxl's default for a non-PQ file (image_metadata.cc SetIntensityTarget)

PROPHOTO = ic.cms_profile("rgb", whitepoint=(0.3457, 0.3585),
                          primaries=(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001), gamma=1.8)


def mixed(rec):
    return (np.asarray(rec, np.float64) @ SRGB_FROM_2020.T) @ OPSIN.T * (INTENSITY_TARGET / 255) + BIAS


def encode(a: np.ndarray) -> bytes:
    return ic.jpegxl_encode(a.astype(np.float16), level=95, effort=7, photometric="RGB", primaries="BT2100",
                            numthreads=THREADS)


def decode(b: bytes) -> np.ndarray:
    return ic.jpegxl_decode(b, numthreads=THREADS)


def patches(colours: np.ndarray, background=(0.18, 0.18, 0.18)) -> np.ndarray:
    n = len(colours)
    a = np.empty((64, 64 * n, 3), np.float32)
    a[...] = background
    for i, c in enumerate(colours):
        a[:, 64 * i:64 * i + 64] = c
    return a


def inner_means(a: np.ndarray, n: int) -> np.ndarray:
    return np.array([a[16:48, 64 * i + 16:64 * i + 48].astype(np.float64).reshape(-1, 3).mean(0) for i in range(n)])


def fmt(v):
    return "(" + ", ".join(f"{x:+.4f}" for x in v) + ")"


def probe_flat():
    print("## 1. Flat patches (inner 32x32 mean), float16 lossy q95 e7")
    colours = np.array([
        [0.50, 0.40, 0.30],        # in gamut, control
        [0.50, 0.40, -0.002],      # tiny negative: LittleCMS rounding size
        [0.50, 0.40, -0.02],
        [0.50, 0.40, -0.10],
        [-0.02, 0.30, 0.30],
        [-0.10, 0.30, 0.30],
        [0.30, -0.02, 0.30],
        [0.30, -0.10, 0.30],
        [0.02, -0.05, 0.40],       # violet-blue, past the Rec.2020 blue-green edge
        [0.00, 0.00, -0.01],       # black just below 0
        [1.50, 1.20, -0.05],       # above white and out of gamut
        [-0.30, 0.80, -0.05],      # far out: green past Rec.2020's
    ], np.float32)
    a = patches(colours)
    data = encode(a)
    d = decode(data)
    print(f"encoded {len(data)} bytes; decoded dtype {d.dtype}, shape {d.shape}; whole-image min {float(d.min()):+.4f}")
    got = inner_means(d, len(colours))
    print("| in (linear Rec.2020) | opsin mixed in (clamped at 0 if negative) | out (mean) | out - in |")
    print("|---|---|---|---|")
    for c, g in zip(colours, got):
        m = mixed(c)
        flag = " CLAMPED" if (m < 0).any() else ""
        print(f"| {fmt(c)} | {fmt(m)}{flag} | {fmt(g)} | {fmt(g - c)} |")
    return colours, got


def prophoto_ramp():
    print("\n## 2. ProPhoto primaries and secondaries at full and half value, through LittleCMS to linear Rec.2020")
    src = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1],
                    [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.5, 0, 0.5]], np.float32)
    rec = ic.cms_transform(src[None], PROPHOTO, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)[0]
    d = decode(encode(patches(rec)))
    got = inner_means(d, len(rec))
    print("| ProPhoto (gamma 1.8) | linear Rec.2020 (LittleCMS float, unclipped) | opsin mixed | decoded | decoded - in |")
    print("|---|---|---|---|---|")
    for s, c, g in zip(src, rec, got):
        m = mixed(c)
        flag = " CLAMPED" if (m < 0).any() else ""
        print(f"| {fmt(s)} | {fmt(c)} | {fmt(m)}{flag} | {fmt(g)} | {fmt(g - c)} |")


def natural():
    print("\n## 3. A natural photo with an out-of-gamut square: kept vs clipped at 0")
    path = REPO / "dataset/2026-08-09__pixel_cam"
    photo = sorted(path.rglob("*.jpg"))[0]
    src = cn.decode(photo.read_bytes(), photo.name, settings=cn.Settings("probe", "float", 7, 95, 4_000_000, 1_000_000,
                                                                          gain_maps=False))
    h, w = src.pixels.shape[:2]
    h, w = min(h, 768), min(w, 768)
    f = src.pixels[:h, :w, :3].astype(np.float32) / src.full
    lin = ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)
    print(f"photo: one {photo.parent.name} JPEG, crop {w}x{h}, min {float(lin.min()):+.5f} before the square")
    kept = lin.copy()
    y0, x0, s = h // 2 - 64, w // 2 - 64, 128
    kept[y0:y0 + s, x0:x0 + s] = (-0.10, 0.30, 0.30)  # cyan past Rec.2020 (red negative), opsin mixed > 0
    clipped = np.clip(kept, 0, None)
    dk, dc = decode(encode(kept)).astype(np.float32), decode(encode(clipped)).astype(np.float32)
    ring = np.zeros((h, w), bool)
    ring[y0 - 16:y0 + s + 16, x0 - 16:x0 + s + 16] = True
    ring[y0:y0 + s, x0:x0 + s] = False
    far = np.ones((h, w), bool)
    far[y0 - 32:y0 + s + 32, x0 - 32:x0 + s + 32] = False
    for name, m in (("16-px ring around the square", ring), ("everything 32 px or more away", far)):
        diff = np.abs(dk[m] - dc[m])
        print(f"{name}: |kept - clipped| mean {float(diff.mean()):.6f}, max {float(diff.max()):.5f} "
              f"(linear, 1.0 = white)")
    for name, d, ref in (("kept", dk, kept), ("clipped", dc, clipped)):
        e = np.abs(d[ring] - ref[ring])
        print(f"{name}: coding error in the ring, mean {float(e.mean()):.6f}, max {float(e.max()):.5f}")
    inner = dk[y0 + 16:y0 + s - 16, x0 + 16:x0 + s - 16].reshape(-1, 3).mean(0)
    print(f"square's inner mean, kept: {fmt(inner)}; clipped file: "
          f"{fmt(dc[y0 + 16:y0 + s - 16, x0 + 16:x0 + s - 16].reshape(-1, 3).mean(0))}")


if __name__ == "__main__":
    if sys.argv[1:] == ["natural"]:
        natural()
        sys.exit()
    print(f"imagecodecs {ic.__version__}; {[v for v in ic.version().split(', ') if v.startswith(('libjxl', 'lcms2'))]}")
    print("linear Rec.2020 -> linear sRGB matrix:\n" + np.array2string(SRGB_FROM_2020, precision=5))
    probe_flat()
    prophoto_ramp()
    natural()
