# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "imagecodecs==2026.10.10",
#     "numpy==2.5.3",
#     "opencv-python-headless==5.0.0.93",
#     "pillow==12.3.0",
#     "pillow-heif==1.8.0",
#     "rawpy==0.27.1",
# ]
# ///
"""ML-19 D52: P1's pipeline with the chunked encoder: the source decoded once, each rectangle libjxl asks for
converted to Rec.2020 when it asks, so no full-size 16-bit frame exists and libjxl copies nothing.

    uv run --script stream.py canon PHOTO... --out DIR [--threads N] [--strip ROWS]

Prints one JSON line per photo, as P1's pipeline.py does, with the same file name (DIR/<stem>.jxl), so the two can be
compared byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "p1"))
sys.path.insert(0, str(HERE))

from PIL import Image  # noqa: E402

import pipeline as p1  # noqa: E402 -- decode(), cap(), to_rec2020() as measured in P1
from jxl_chunked import encode_chunked  # noqa: E402


def peak_mb() -> int:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def canonicalize(path: Path, effort: int, threads: int, strip: int, buffering: int | None) -> tuple[bytes, dict]:
    t0 = time.perf_counter()
    rgb, full, icc, kind = p1.decode(path)
    rgb = p1.cap(rgb)
    t1 = time.perf_counter()
    decoded = peak_mb()
    h, w = rgb.shape[:2]
    colour_s = [0.0]

    def region(y: int, x: int, ys: int, xs: int):
        t = time.perf_counter()
        r = p1.to_rec2020(rgb[y:y + ys, x:x + xs], full, icc, strip)
        colour_s[0] += time.perf_counter() - t
        return r

    stats: dict = {}
    jxl = encode_chunked(h, w, region, level=95, effort=effort, numthreads=threads, buffering=buffering, stats=stats)
    t2 = time.perf_counter()
    return jxl, {"decoded_as": kind, "size": [w, h], "decode_s": round(t1 - t0, 2),
                 "colour_s": round(colour_s[0], 2), "encode_s": round(t2 - t1 - colour_s[0], 2),
                 "total_s": round(t2 - t0, 2), "peak_after_decode_mb": decoded, "rects": stats["rects"],
                 "pixels_converted": stats["pixels"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("canon")
    c.add_argument("photos", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--effort", type=int, default=7)
    c.add_argument("--threads", type=int, default=1)
    c.add_argument("--strip", type=int, default=256)
    c.add_argument("--buffering", type=int, default=None, help="libjxl's BUFFERING option (default: libjxl's)")
    args = ap.parse_args()
    Image.MAX_IMAGE_PIXELS = 2**28
    args.out.mkdir(parents=True, exist_ok=True)
    for photo in args.photos:
        jxl, record = canonicalize(photo, args.effort, args.threads, args.strip, args.buffering)
        (args.out / f"{photo.stem}.jxl").write_bytes(jxl)
        print(json.dumps({"photo": photo.name, "effort": args.effort, "threads": args.threads, "buffering": args.buffering, "bytes": len(jxl),
                          "sha256": hashlib.sha256(jxl).hexdigest(), **record, "peak_mb": peak_mb()}), flush=True)


if __name__ == "__main__":
    main()
