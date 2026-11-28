"""ML-37: why form A's encode takes longer than B's: the same pictures through other writers and inputs.

    cd /workspace && OMP_NUM_THREADS=2 PYTHONPATH=/workspace taskset -c 0,1 uv run python \
        analysis/ml37/front_forms_encoders.py [--photos 0,5,23,38]

For a few photos of front_forms_measure.py's sample, the uint16 linear picture (A's input) and the 8-bit sRGB picture
(B's input) are made as there, then encoded at quality 90, effort 3, 2 threads, median of 3 wall times:
  ctA203   the ctypes strip encoder, uint16 linear Rec.2100, intensity target 203 (form A as measured);
  ctA255   the same at intensity target 255;
  icA      imagecodecs.jpegxl_encode(uint16, primaries BT2100, transfer LINEAR): whole picture, target 255;
  icB      imagecodecs uint8 sRGB (form B as measured);
  icB16    imagecodecs, B's picture as uint16 (x 257), sRGB: the bit depth alone;
  icLin8   imagecodecs, linear sRGB in uint16: the transfer alone (sRGB primaries, linear).
Prints bytes and seconds, and for ctA203, ctA255 and icB the 8-bit error against B's uncompressed picture (A read
back and rounded once, as front_forms_measure.py): pixel RMS / D27 block mean / largest block, in 8-bit levels
(kept as analysis/ml37/front_forms_encoders.txt).
"""
from __future__ import annotations

import argparse
import statistics as st
import sys
import time

import imagecodecs as ic
import numpy as np

from analysis.ml37 import canon_int_measure as cm
from analysis.ml37 import d8_intensity_target as d8
from analysis.ml37 import front_forms_measure as ff


def timed(fn, runs: int = 3):
    ts, out = [], None
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t)
    return out, st.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--photos", default="0,5,23,38")
    args = ap.parse_args()
    photos = ff.sample()
    print(f"# quality 90, effort 3, 2 threads, median of 3; load {__import__('os').getloadavg()[0]:.2f}")
    for i in (int(x) for x in args.photos.split(",")):
        _, _, info, (u16, u8) = ff.make(photos[i]["file"], None, True)
        lin_srgb = np.empty(u16.shape, np.uint16)
        for y in range(0, u16.shape[0], 256):
            c = (u16[y:y + 256].astype(np.float32) / 65535) @ d8.REC2020_TO_SRGB.T
            np.clip(c, 0, 1, out=c)
            lin_srgb[y:y + 256] = np.rint(c * 65535)
        cases = {
            "ctA203": lambda: cm.encode_linear_u16(u16, level=90, effort=3, intensity_target=203.0),
            "ctA255": lambda: cm.encode_linear_u16(u16, level=90, effort=3, intensity_target=255.0),
            "icA": lambda: ic.jpegxl_encode(u16, level=90, effort=3, primaries="BT2100", transfer="LINEAR",
                                            numthreads=2),
            "icB": lambda: ic.jpegxl_encode(u8, level=90, effort=3, numthreads=2),
            "icB16": lambda: ic.jpegxl_encode(u8.astype(np.uint16) * 257, level=90, effort=3, primaries="SRGB",
                                              transfer="SRGB", numthreads=2),
            "icLin8": lambda: ic.jpegxl_encode(lin_srgb, level=90, effort=3, primaries="SRGB", transfer="LINEAR",
                                               numthreads=2),
        }
        line, errs = [], []
        ref_blocks = ff.blocks8(u8)
        for name, fn in cases.items():
            data, t = timed(fn)
            line.append(f"{name} {len(data):8d} B {t:5.2f} s")
            if name in ("ctA203", "ctA255", "icB"):
                dec = ic.jpegxl_decode(data, numthreads=2)
                if dec.dtype == np.uint16:
                    got = np.empty(u8.shape, np.uint8)
                    for y in range(0, dec.shape[0], 256):
                        got[y:y + 256] = ff.read_back8(dec[y:y + 256].astype(np.float32) / 65535)
                    dec = got
                e = ff.errors(dec, u8, ref_blocks)
                errs.append(f"{name} {e['rms8']:.3f} / {e['blk_mean']:.3f} / {e['blk_max']:.2f}")
        print(f"{i:2d} {photos[i]['stratum'][:28]:28s} {info['w'] * info['h'] / 1e6:5.1f} MP | " + " | ".join(line),
              flush=True)
        print("   error " + " | ".join(errs), flush=True)
        u16 = u8 = lin_srgb = None


if __name__ == "__main__":
    sys.exit(main())
