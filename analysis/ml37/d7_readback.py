"""ML-37 D7: what negative linear Rec.2020 samples would change at D2's read-back, and where they come from.

    OMP_NUM_THREADS=2 PYTHONPATH=. uv run python analysis/ml37/d7_readback.py [float_jxl]

1. The matrices from the primaries (BT.2020, Display P3 / DCI-P3 D65, BT.709/sRGB, D65): which Display P3 colours lie
   outside Rec.2020 and by how much, and the values above 1.0 that converting in-gamut-looking colours makes.
2. The read-back (D2: matrix to linear sRGB -> clip [0, 1] -> sRGB curve) of chosen colours with their negatives
   (a) clipped at 0 before it and (b) kept, in 8-bit levels; and D33's EETF on max(R, G, B) before the matrix, as
   coffeecv.tone_map.eetf_in_place runs it, for both.
3. Where the tone map's peak would be measured: max(R, G, B) of the primaries and secondaries of the dataset's colour
   sources in linear Rec.2020 and after the matrix in linear sRGB.
4. (with an argument: a float JPEG XL path) a real float file: its share of samples below 0, through canonicalization's
   own source profile into linear Rec.2020 unclipped, and the read-back's difference between (a) and (b).
"""
from __future__ import annotations

import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from coffeecv import canonical as cn  # noqa: E402
from coffeecv import tone_map as tm  # noqa: E402

D65 = (0.3127, 0.3290)
PRIMARIES = {"Rec.2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046),
             "Display P3": (0.680, 0.320, 0.265, 0.690, 0.150, 0.060),
             "sRGB": (0.640, 0.330, 0.300, 0.600, 0.150, 0.060),
             "Adobe RGB": (0.640, 0.330, 0.210, 0.710, 0.150, 0.060)}


def rgb_to_xyz(prim, white=D65):
    xr, yr, xg, yg, xb, yb = prim
    xw, yw = white
    m = np.array([[xr / yr, xg / yg, xb / yb], [1, 1, 1], [(1 - xr - yr) / yr, (1 - xg - yg) / yg, (1 - xb - yb) / yb]])
    return m * np.linalg.solve(m, np.array([xw / yw, 1, (1 - xw - yw) / yw]))


def matrix(src, dst):
    return np.linalg.inv(rgb_to_xyz(PRIMARIES[dst])) @ rgb_to_xyz(PRIMARIES[src])


SRGB_FROM_2020 = matrix("Rec.2020", "sRGB")


def srgb_curve(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def readback(rec, peak=None):
    """D2 on linear Rec.2020 (n, 3): optional EETF (eetf_in_place, in Rec.2020), matrix, clip, sRGB curve, 8-bit levels."""
    c = np.array(rec, np.float32, copy=True)
    if peak is not None:
        tm.eetf_in_place(c, peak)
    return srgb_curve(np.clip(c.astype(np.float64) @ SRGB_FROM_2020.T, 0, 1)) * 255


def fmt(v, p=4):
    return "(" + ", ".join(f"{x:+.{p}f}" for x in v) + ")"


def part1():
    print("## 1. Matrices from the primaries (D65)")
    for src in ("Display P3", "sRGB", "Adobe RGB"):
        m = matrix(src, "Rec.2020")
        print(f"linear Rec.2020 from linear {src}:\n{np.array2string(m, precision=5)}")
        print(f"  most negative coefficient {m.min():+.5f}; largest row sum of positives "
              f"{np.clip(m, 0, None).sum(1).max():.5f} (the highest value an in-range {src} colour reaches)")
    print(f"linear sRGB from linear Rec.2020 (D2's matrix):\n{np.array2string(SRGB_FROM_2020, precision=5)}")
    p3 = matrix("Display P3", "Rec.2020")
    for name, v in (("P3 red", [1, 0, 0]), ("P3 cyan", [0, 1, 1]), ("P3 magenta", [1, 0, 1]),
                    ("P3 (1, 0.2, 0)", [1, 0.2, 0])):
        print(f"  {name} {fmt(v, 1)} -> Rec.2020 {fmt(p3 @ v, 5)}")


def part2():
    print("\n## 2. Read-back of colours with negatives: (a) clipped at 0 first vs (b) kept; 8-bit sRGB levels")
    prophoto = ic.cms_profile("rgb", whitepoint=(0.3457, 0.3585),
                              primaries=(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001), gamma=1.8)
    pp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0.5, 0], [0, 0.5, 0.5]], np.float32)
    rec_pp = ic.cms_transform(pp[None], prophoto, cn.LINEAR_REC2020, intent=cn.RELATIVE, outdtype=np.float32)[0]
    rows = [("P3 red", matrix("Display P3", "Rec.2020") @ [1, 0, 0]),
            ("P3 (0.2, 0.02, 0.01), a dark red", matrix("Display P3", "Rec.2020") @ [0.2, 0.02, 0.01]),
            ("in gamut (0.5, 0.4, 0.3)", [0.5, 0.4, 0.3]),
            ("tiny: (0.5, 0.4, -0.00004)", [0.5, 0.4, -0.00004]),
            ("(0.02, -0.05, 0.40)", [0.02, -0.05, 0.40]),
            ("(-0.10, 0.30, 0.30)", [-0.10, 0.30, 0.30])]
    rows += [(f"ProPhoto {fmt(s, 1)}", r) for s, r in zip(pp, rec_pp)]
    rows += [("above white and out: (1.5, 1.2, -0.05)", [1.5, 1.2, -0.05])]
    print("| colour | linear Rec.2020 | linear sRGB (matrix) | (a) clip at 0, read back | (b) kept, read back | "
          "(b) - (a), levels | EETF at peak 2: (a) | (b) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r in rows:
        r = np.asarray(r, np.float64)
        a = readback(np.maximum(r, 0)[None])[0]
        b = readback(r[None])[0]
        ea = readback(np.maximum(r, 0)[None], 2.0)[0]
        eb = readback(r[None], 2.0)[0]
        print(f"| {name} | {fmt(r)} | {fmt(SRGB_FROM_2020 @ r)} | {fmt(a, 1)} | {fmt(b, 1)} | {fmt(b - a, 2)} "
              f"| {fmt(ea, 1)} | {fmt(eb, 1)} |")
    # exhaustive in-gamut check: any Rec.2020 colour with every channel >= 0 reads back identically under (a) and (b)
    rng = np.random.default_rng(7)
    x = rng.uniform(0, 2, (1_000_000, 3)).astype(np.float32)
    same = np.array_equal(readback(np.maximum(x, 0)), readback(x))
    print(f"\n1,000,000 random non-negative Rec.2020 colours in [0, 2): (a) and (b) identical: {same}")
    # max(R, G, B) and negatives
    y = rng.uniform(-0.2, 1.5, (1_000_000, 3)).astype(np.float32)
    has_neg = (y < 0).any(-1)
    same_max = np.array_equal(np.maximum(y, 0).max(-1)[has_neg], np.maximum(y.max(-1), 0)[has_neg])
    print(f"max(R, G, B) of a colour with a negative channel, clipped first vs after: identical: {same_max}")


def part3():
    print("\n## 3. max(R, G, B) of each source's primaries and secondaries: in linear Rec.2020 and after D2's matrix")
    print("| source | colour | max in Rec.2020 | max in linear sRGB |")
    print("|---|---|---|---|")
    for src in ("Display P3", "Adobe RGB", "sRGB"):
        m = matrix(src, "Rec.2020")
        top2020, topsrgb = 0, 0
        for v in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]):
            r = m @ v
            s = SRGB_FROM_2020 @ r
            if s.max() > topsrgb:
                topsrgb, top2020, which = s.max(), r.max(), v
        print(f"| {src} | {which} | {top2020:.4f} | {topsrgb:.4f} |")


def part4(path: Path):
    print("\n## 4. A float JPEG XL (a darktable 32-bit float sRGB export; not a dataset photo)")
    data = path.read_bytes()
    src = cn.decode(data, path.name, settings=cn.Settings("probe", "float", 7, 95, cn.KEEP_MAX_PIXELS, cn.CAP_PIXELS,
                                                          gain_maps=False))
    px = src.pixels
    h, w = px.shape[:2]
    print(f"decoder {src.decoder}, {w}x{h}, samples {float(px.min()):+.4f} to {float(px.max()):+.4f}; "
          f"source samples < 0: {float((px < 0).any(-1).mean()):.4%} of pixels")
    n = neg = neg3 = differ = 0
    lo, worst = 0.0, 0.0
    for y in range(0, h, cn.STRIP_ROWS):
        _, f = cn._light(px[y:y + cn.STRIP_ROWS], src.full, src.linearize, 3)
        c = ic.cms_transform(f, src.profile or cn.SRGB, cn.LINEAR_REC2020, intent=cn.RELATIVE,
                             outdtype=np.float32).reshape(-1, 3)
        mn = c.min(-1)
        n += len(c)
        neg += int((mn < 0).sum())
        neg3 += int((mn < -1e-3).sum())
        lo = min(lo, float(mn.min()))
        bad = mn < 0
        if bad.any():
            d = np.abs(readback(np.maximum(c[bad], 0)) - readback(c[bad])).max(-1)
            differ += int((d >= 0.5).sum())
            worst = max(worst, float(d.max()))
    print(f"linear Rec.2020 (LittleCMS float, unclipped): pixels < 0 {neg / n:.4%}, < -1e-3 {neg3 / n:.4%}, "
          f"lowest {lo:+.5f}")
    print(f"read-back (a) vs (b): pixels differing by >= 0.5 level {differ / n:.4%}, largest {worst:.2f} levels")


if __name__ == "__main__":
    if sys.argv[1:]:
        part4(Path(sys.argv[1]))
    else:
        part1()
        part2()
        part3()
