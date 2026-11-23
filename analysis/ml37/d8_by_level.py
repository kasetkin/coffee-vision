"""ML-37 D8: where the intensity target moves the lossy half-float coding's error, by light level.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d8_by_level.py [--scale 2] [--k 6]

From d8_intensity_target.py's sample (same seed), the --k photos with the highest peak (the gain-map photos, light
above white kept) plus two without a map. Each is encoded as lossy half-float linear Rec.2020 at the store's (95/7)
and the front's (90/3) settings with intensity_target 203 and 255 (libjxl 0.12.0, d8_intensity_target.encode), decoded,
and compared with its source in linear light. Pixels are binned by the source's luminance (BT.2100 weights; 1.0 =
reference white) in octaves from 2^-10 to 2^3; per bin: the median relative error of the luminance on 4 x 4 block
means (the coding's low-frequency error; a pixel alone is dominated by noise the encoder drops by design), and the
pixel count. Writes nothing but stdout (kept as analysis/ml37/d8_by_level.txt).
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import tone_map as tm

spec = importlib.util.spec_from_file_location("d8", Path(__file__).with_name("d8_intensity_target.py"))
d8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d8)

Y2100 = np.array(tm.BT2100_LUMINANCE, np.float32)
EDGES = 2.0 ** np.arange(-10, 4)          # 1/1024 ... 8 x white


def blocks4(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // 4) * 4, (a.shape[1] // 4) * 4
    return a[:h, :w].reshape(h // 4, 4, w // 4, 4).mean(axis=(1, 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    photos = d8.sample(20)
    gm = [p for p in photos if p["gain_map"]]
    peaks = []
    for p in gm:                                   # the peak needs the light: decode once, keep only the number
        light, _ = d8.light_of(p["file"], 4)
        peaks.append((tm.picture_peak(light.max(axis=-1).ravel().copy()), p))
        del light
    chosen = [p for _, p in sorted(peaks, key=lambda t: -t[0])[:args.k]]
    chosen += [p for p in photos if not p["gain_map"]][:2]
    acc: dict = {}
    for p in chosen:
        light, mapped = d8.light_of(p["file"], args.scale)
        y_src = blocks4(light @ Y2100)
        bins = np.digitize(y_src, EDGES)
        f16 = light.astype(np.float16)
        print(f"{Path(p['file']).name} {p['_k'][0]} map={mapped} max Y {y_src.max():.2f}", flush=True)
        for name, q, e in (("store", 95, 7), ("front", 90, 3)):
            for it in (203.0, 255.0):
                data = d8.encode(f16, level=q, effort=e, intensity_target=it, half=True)
                got = ic.jpegxl_decode(data, numthreads=d8.THREADS).astype(np.float32)
                y_got = blocks4(got @ Y2100)
                rel = np.abs(y_got - y_src) / np.maximum(y_src, 1e-6)
                for b in range(1, len(EDGES)):
                    sel = bins == b
                    if sel.any():
                        acc.setdefault((name, it, b), []).append(rel[sel])
        del light, f16
    print("\nmedian relative error of block luminance (4x4 block means), by source luminance; 1.0 = reference white")
    print(f"{'Y from':>9s} {'to':>7s} {'blocks':>9s} | {'store 203':>9s} {'store 255':>9s} {'ratio':>6s} | "
          f"{'front 203':>9s} {'front 255':>9s} {'ratio':>6s}")
    for b in range(1, len(EDGES)):
        if ("store", 203.0, b) not in acc:
            continue
        n = sum(a.size for a in acc[("store", 203.0, b)])
        m = {k: float(np.median(np.concatenate(acc[(k[0], k[1], b)]))) for k in
             (("store", 203.0), ("store", 255.0), ("front", 203.0), ("front", 255.0))}
        print(f"{EDGES[b - 1]:9.4f} {EDGES[b]:7.3f} {n:9d} | {m[('store', 203.0)]:9.5f} {m[('store', 255.0)]:9.5f} "
              f"{m[('store', 203.0)] / m[('store', 255.0)]:6.3f} | {m[('front', 203.0)]:9.5f} "
              f"{m[('front', 255.0)]:9.5f} {m[('front', 203.0)] / m[('front', 255.0)]:6.3f}")


if __name__ == "__main__":
    main()
