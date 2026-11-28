"""ML-37 D15: what the front pays to make the preview's picture itself, from its own canonical photo (form A).

    cd /workspace && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/workspace \
        taskset -c 0,1 uv run python analysis/ml37/preview_front_measure.py [--dir tmp/ml37/front_forms/A]

Reads the 40 form-A files front_forms_measure.py wrote (the front's settings: q90, e3, 15 MP cap, linear uint16,
intensity target 203). Per photo, one run, 2 CPUs pinned, libjxl 2 threads:
  same    decode (imagecodecs) -> D2/D3's read-back at full size in strips of 256 rows (front_forms_measure.read_back8,
          numpy float32, D9 (a)) -> the preview as webapp/app.py's _preview_part makes it (PIL thumbnail to 1024 px,
          JPEG quality 85): the picture yappy would send.
  shrink  decode -> area-average the linear light to the thumbnail's size (cv2 INTER_AREA, float32) -> read-back ->
          JPEG q85: cheaper, not the models' picture. Compared with `same`'s thumbnail in 8-bit levels.
Memory: on the largest photo, tracemalloc's peak of the read-back done whole against in strips, and whether the two
give identical uint8.
"""
from __future__ import annotations

import argparse
import io
import statistics as st
import time
import tracemalloc
from pathlib import Path

import cv2
import imagecodecs as ic
import numpy as np
from PIL import Image

from analysis.ml37.front_forms_measure import read_back8

STRIP = 256


def read_back_strips(u16: np.ndarray) -> np.ndarray:
    out = np.empty(u16.shape, np.uint8)
    for y in range(0, u16.shape[0], STRIP):
        out[y:y + STRIP] = read_back8(u16[y:y + STRIP].astype(np.float32) / 65535)
    return out


def jpeg(u8: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(u8).save(b, format="JPEG", quality=85)
    return b.getvalue()


def thumbnail(u8: np.ndarray) -> np.ndarray:
    t = Image.fromarray(u8)
    t.thumbnail((1024, 1024))
    return np.asarray(t)


def ms(rows: list[dict], k: str) -> str:
    v = [r[k] for r in rows]
    return f"{st.mean(v):.2f} ± {st.stdev(v):.2f} (min {min(v):.2f}, max {max(v):.2f})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="tmp/ml37/front_forms/A")
    paths = sorted(Path(ap.parse_args().dir).glob("*.jxl"))
    rows = []
    for p in paths:
        data = p.read_bytes()
        t0 = time.perf_counter()
        u16 = ic.jpegxl_decode(data, numthreads=2)
        t1 = time.perf_counter()
        u8 = read_back_strips(u16)
        t2 = time.perf_counter()
        th = thumbnail(u8)
        jpg = jpeg(th)
        t3 = time.perf_counter()
        small = cv2.resize(u16.astype(np.float32) / 65535, (th.shape[1], th.shape[0]), interpolation=cv2.INTER_AREA)
        th2 = read_back8(small)
        jpeg(th2)
        t4 = time.perf_counter()
        d = np.abs(th2.astype(np.int16) - th.astype(np.int16))
        r = dict(name=p.name[:2], mp=u16.shape[0] * u16.shape[1] / 1e6, decode=t1 - t0, readback=t2 - t1,
                 thumb_jpeg=t3 - t2, same=t3 - t0, shrink=(t1 - t0) + (t4 - t3), kb=len(jpg) / 1024,
                 d_mean=float(d.mean()), d_p99=float(np.percentile(d, 99)), d_max=float(d.max()))
        rows.append(r)
        print(f"{r['name']} {r['mp']:5.1f} MP  decode {r['decode']:.2f}  read-back {r['readback']:.2f}  "
              f"thumb+jpeg {r['thumb_jpeg']:.2f}  same {r['same']:.2f}  shrink {r['shrink']:.2f}  "
              f"jpeg {r['kb']:.0f} KB  shrink-vs-same mean {r['d_mean']:.2f} p99 {r['d_p99']:.0f} "
              f"max {r['d_max']:.0f}", flush=True)
    print(f"\n{len(rows)} photos, mean ± std (min, max); seconds, KB, 8-bit levels")
    for k in ("mp", "decode", "readback", "thumb_jpeg", "same", "shrink", "kb", "d_mean", "d_p99", "d_max"):
        print(f"  {k:10s} {ms(rows, k)}")

    big = max(paths, key=lambda q: np.prod(ic.jpegxl_decode(q.read_bytes(), numthreads=2).shape[:2]))
    u16 = ic.jpegxl_decode(big.read_bytes(), numthreads=2)
    tracemalloc.start()
    whole = read_back8(u16.astype(np.float32) / 65535)
    peak_whole = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    tracemalloc.start()
    strips = read_back_strips(u16)
    peak_strips = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    print(f"\nlargest photo {big.name[:2]}, {u16.shape[0] * u16.shape[1] / 1e6:.1f} MP: read-back peak memory "
          f"whole {peak_whole:.0f} MB, strips of {STRIP} rows {peak_strips:.0f} MB (incl. the {strips.nbytes / 1e6:.0f} MB "
          f"uint8 output); identical: {np.array_equal(whole, strips)}")


if __name__ == "__main__":
    main()
