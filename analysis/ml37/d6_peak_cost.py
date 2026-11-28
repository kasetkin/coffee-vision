"""ML-37 D6: what the per-picture peak costs at the read-back, and whether the lossy half-float coding moves it.

    cd /workspace && OMP_NUM_THREADS=2 PYTHONPATH=/workspace uv run python analysis/ml37/d6_peak_cost.py

Research only: reads dataset/ (never writes there); scratch in tmp/ml37/d6/. For a few gain-map photos of the sample
in d6_readback_options.py (one 12.5 MP Pixel 9 Pro, one 12 MP iPhone 15, one 24 MP iPhone 15 Pro Max):

1. The original decoded as the store decodes it (canonical.decode, gain map applied): linear Rec.2020 light, negative
   light clipped at 0 (D7 (a)). Written as D1's canonical photo would be: imagecodecs.jpegxl_encode on float16,
   primaries BT2100, lossy at distance 0.55 (the store's quality 95), effort 3 (the encode's time is not the question
   here), 2 threads. Decoded back on 2 threads, timed: the read-back's first step.
2. The peak pass on the decoded half-float picture, timed: D33's peak (99.99th percentile of max(R, G, B), at least
   1) in Rec.2020 (tone_map.picture_peak on one float32 per pixel), and in linear sRGB (the matrix first); a strided
   1-in-16 pixel subsample's peak; and a streaming histogram (strip by strip, 4096 log-spaced bins from 2^-12 to
   2^6) as a strip-wise alternative. Each against the exact peak on the original light (before coding).
3. The read-back itself on the decoded picture, timed, strip by strip in float32: (b) the matrix, clip and sRGB curve;
   (a) the EETF in Rec.2020 at the measured peak, then the same. The memory each holds is stated from the arrays.
4. Crops: the peak a crop would measure on its own (the bean mask's bounding box, as the segmenter crops) against the
   whole photo's, and the share of the box's pixels whose read-back moves by over half a level between the two.

Prints one block a photo; the text is kept in d6_peak_cost.txt.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn
from coffeecv import tone_map as tm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from d6_readback_options import M, srgb_curve  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tmp" / "ml37" / "d6"
STRIP = 256
MASKS = REPO / "data" / "seg_masks_ml19" / "xl0_v7r_s123"


def sample_photos() -> list[tuple[str, str]]:
    import json
    rows = [r for r in json.loads((OUT / "readback_rows.json").read_text()) if not r["skipped"] and r["kind"] == "gainmap"]
    px = sorted((r for r in rows if r["camera"] == "JPEG Google Pixel 9 Pro"), key=lambda r: r["peak2020"])
    pm = max((r for r in rows if r["camera"] == "HEIC Apple iPhone 15 Pro Max"), key=lambda r: r["region_above_white"])
    big = max(rows, key=lambda r: r["h"] * r["w"])
    return [(r["camera"], r["photo"]) for r in (px[len(px) // 2], pm, big)]


def bean_box(photo: Path, h: int, w: int):
    from PIL import Image
    mp = MASKS / ("__".join(Path(str(photo).split("dataset/")[1]).with_suffix("").parts) + ".png")
    if not mp.exists():
        return None
    with Image.open(mp) as im:
        m = np.asarray(im.convert("L").resize((w, h), Image.NEAREST)) > 127
    ys, xs = np.nonzero(m.any(1))[0], np.nonzero(m.any(0))[0]
    return (ys[0], ys[-1] + 1, xs[0], xs[-1] + 1) if ys.size else None


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def hist_peak(img: np.ndarray, srgb: bool = False) -> float:
    edges = np.concatenate([[0.0], np.logspace(-12, 6, 4096, base=2.0), [np.inf]])
    hist = np.zeros(len(edges) - 1, np.int64)
    for y in range(0, img.shape[0], STRIP):
        c = img[y:y + STRIP].astype(np.float32)
        if srgb:
            c = c @ M.T
        hist += np.histogram(np.maximum(c.max(-1), 0), edges)[0]
    cdf = np.cumsum(hist) / hist.sum()
    i = int(np.searchsorted(cdf, tm.PEAK_PERCENTILE / 100))
    return float(min(max(edges[i + 1], 1.0), tm.MAX_PEAK))


def exact_peak(img: np.ndarray, srgb: bool = False) -> float:
    h, w = img.shape[:2]
    maxima = np.empty(h * w, np.float32)
    for y in range(0, h, STRIP):
        c = img[y:y + STRIP].astype(np.float32)
        if srgb:
            c = c @ M.T
            np.maximum(c, 0, out=c)
        maxima[y * w:(y + c.shape[0]) * w] = c.max(-1).ravel()
    return tm.picture_peak(maxima)


def readback(img: np.ndarray, peak: float | None) -> np.ndarray:
    out = np.empty(img.shape, np.float32)
    for y in range(0, img.shape[0], STRIP):
        c = img[y:y + STRIP].astype(np.float32)
        if peak is not None:
            tm.eetf_in_place(c, peak)
        s = c @ M.T
        np.clip(s, 0, 1, out=s)
        out[y:y + STRIP] = srgb_curve(s)
    return out


def main() -> None:
    for cam, rel in sample_photos():
        photo = Path(rel) if rel.startswith("/") else REPO / rel
        src = cn.decode(photo.read_bytes(), photo.name)
        light = src.pixels
        del src
        np.maximum(light, 0, out=light)
        h, w = light.shape[:2]
        p_true = exact_peak(light)
        p_true_srgb = exact_peak(light, srgb=True)
        data, t_enc = timed(ic.jpegxl_encode, light.astype(np.float16), distance=0.55, effort=3, primaries="BT2100",
                            numthreads=2)
        del light
        dec, t_dec = timed(ic.jpegxl_decode, data, numthreads=2)
        mp = h * w / 1e6
        print(f"== {cam}  {w}x{h} ({mp:.1f} MP)  file {len(data) / 1e6:.1f} MB, decoded {dec.dtype} {dec.nbytes / 1e6:.0f} MB")
        print(f"   encode (effort 3, 2 threads) {t_enc:.1f} s; decode (2 threads) {t_dec:.2f} s")
        p_dec, t_p = timed(exact_peak, dec)
        p_dec_s, t_ps = timed(exact_peak, dec, True)
        p_sub, t_sub = timed(exact_peak, np.ascontiguousarray(dec[::4, ::4]))
        p_hist, t_hist = timed(hist_peak, dec)
        print(f"   peak on the original light: Rec.2020 {p_true:.4f}, sRGB {p_true_srgb:.4f}")
        print(f"   peak pass on the decoded photo (float32 maxima, {h * w * 4 / 1e6:.0f} MB): Rec.2020 {p_dec:.4f} "
              f"in {t_p:.2f} s; sRGB (matrix first) {p_dec_s:.4f} in {t_ps:.2f} s")
        print(f"   1-in-16 subsample: {p_sub:.4f} in {t_sub:.2f} s; streaming histogram: {p_hist:.4f} in {t_hist:.2f} s")
        print(f"   knee at the decoded peak {tm.eetf_knee(p_dec):.4f}, at the original's {tm.eetf_knee(p_true):.4f}")
        rb, t_b = timed(readback, dec, None)
        del rb
        rb, t_a = timed(readback, dec, p_dec)
        nbytes = rb.nbytes
        del rb
        moved = 0
        for y in range(0, h, STRIP):
            a = readback(dec[y:y + STRIP], p_dec)
            b = readback(dec[y:y + STRIP], None)
            moved += int(((np.abs(a - b) * 255).max(-1) > 0.5).sum())
        print(f"   read-back, float32 strips, 1 thread: (b) clip {t_b:.2f} s, (a) EETF {t_a:.2f} s; "
              f"output float32 {nbytes / 1e6:.0f} MB; (a) vs (b): {moved / (h * w):.3%} of pixels differ by over half "
              f"a level")
        box = bean_box(photo, h, w)
        if box is not None:
            y0, y1, x0, x1 = box
            crop = np.ascontiguousarray(dec[y0:y1, x0:x1])
            p_crop = exact_peak(crop)
            moved = 0
            for y in range(0, crop.shape[0], STRIP):
                a = readback(crop[y:y + STRIP], p_dec)
                b = readback(crop[y:y + STRIP], p_crop)
                moved += int(((np.abs(a - b) * 255).max(-1) > 0.5).sum())
            print(f"   crop (bean box {x1 - x0}x{y1 - y0}, {(x1 - x0) * (y1 - y0) / (h * w):.0%} of the photo): its own peak "
                  f"{p_crop:.4f} (knee {tm.eetf_knee(p_crop):.4f}) against the photo's {p_dec:.4f}; read-back at its own "
                  f"peak vs the photo's: {moved / crop.shape[0] / crop.shape[1]:.3%} of its pixels differ by over half "
                  f"a level")
            del crop
        del dec


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    main()
