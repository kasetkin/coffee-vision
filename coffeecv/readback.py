"""The read-back (ticket ML-37 D2, D3, D9): a canonical photo's linear Rec.2020 samples to the 8-bit sRGB both models
read, and the preview is made from.

    to_srgb(rec)        -> np.ndarray   H x W x 3 uint16 linear Rec.2020 (D1) to H x W x 3 uint8 sRGB
    srgb_levels(rec)    -> np.ndarray   the same before rounding: float32 levels, 0-255

uint16 / 65535 -> the 3 x 3 matrix from Rec.2020 to sRGB's primaries (both D65) -> clipped to [0, 1] -> the sRGB curve
(IEC 61966-2-1, closed form) -> x 255, rounded to the nearest level once (D3). numpy in float32 (D9 (a)), not
LittleCMS: the matrix is derived here from the two sets of primaries and D65 (SMPTE RP 177), and the tests hold it to
ITU-R BT.2087's published matrix, to a float64 reference and to LittleCMS. Done in strips of rows on a thread pool,
so no whole-photo float copy exists (ML-37 R10: 101 MB against 1,074 MB at 17 MP); a strip's pixels do not depend on
the strip size or the thread count. numpy releases the GIL in its loops.

No torch, no imagecodecs: the front, the host, the store and training import it. numpy is pinned as imagecodecs is in
coffeecv.canonical (D9): another version may round a float32 pixel differently, which is a new model input.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# D9: the version the read-back was tested at; tests/test_readback.py holds it to uv.lock's, and coffeecv.lock_check
# holds the three locks to one version. A new numpy is a re-test of every model input, as a new imagecodecs is.
NUMPY = "2.5.3"
if np.__version__ != NUMPY:
    raise ImportError(f"coffeecv.readback is written for numpy {NUMPY}, found {np.__version__}")

STRIP_ROWS = 256                         # ML-37 R10: 101 MB at 17 MP on the front's two threads

D65 = (0.3127, 0.3290)
REC2020_PRIMARIES = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))   # ITU-R BT.2020: R, G, B x, y
SRGB_PRIMARIES = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))            # ITU-R BT.709 = IEC 61966-2-1


def rgb_to_xyz(primaries, white=D65) -> np.ndarray:
    """The 3 x 3 linear RGB to CIE XYZ matrix of a set of primaries and a white point, in float64 (SMPTE RP 177):
    each primary's XYZ at Y = 1, scaled so that RGB (1, 1, 1) is the white at Y = 1."""
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = white
    return xyz * np.linalg.solve(xyz, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))


# Linear Rec.2020 to linear sRGB: derived in float64, applied in float32.
REC2020_TO_SRGB = np.linalg.inv(rgb_to_xyz(SRGB_PRIMARIES)) @ rgb_to_xyz(REC2020_PRIMARIES)
_MATRIX_T = (REC2020_TO_SRGB.T / 65535).astype(np.float32)   # uint16 straight to linear sRGB, row vectors


def srgb_levels(rec: np.ndarray) -> np.ndarray:
    """Linear Rec.2020 uint16 samples (..., 3) as sRGB levels before rounding, float32 in [0, 255]."""
    lin = np.matmul(rec.astype(np.float32), _MATRIX_T)
    np.clip(lin, 0, 1, out=lin)
    low = lin <= np.float32(0.0031308)
    out = np.power(lin, np.float32(1 / 2.4))
    out *= np.float32(1.055)
    out -= np.float32(0.055)
    np.multiply(lin, np.float32(12.92), out=out, where=low)
    out *= np.float32(255)
    return out


def usable_cpus() -> int:
    """The CPUs this process may run on: its affinity, which taskset and systemd's CPUAffinity= narrow."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:   # not Linux
        return os.cpu_count() or 1


def to_srgb(rec: np.ndarray, strip: int = STRIP_ROWS, numthreads: int | None = None) -> np.ndarray:
    """A canonical photo's linear Rec.2020 samples, H x W x 3 uint16, as 8-bit sRGB, H x W x 3 uint8 (D2, D3): in
    strips of `strip` rows on `numthreads` threads (None: every CPU this process may use). Each strip is converted
    alone, so the pixels depend on neither."""
    threads = numthreads or usable_cpus()
    out = np.empty(rec.shape, np.uint8)

    def convert(y: int) -> None:
        np.rint(srgb_levels(rec[y:y + strip]), out=out[y:y + strip], casting="unsafe")

    rows = range(0, rec.shape[0], strip)
    if threads == 1:
        for y in rows:
            convert(y)
    else:
        with ThreadPoolExecutor(threads) as pool:   # per call: no threads left over across gunicorn's fork
            list(pool.map(convert, rows))
    return out


__all__ = ["NUMPY", "REC2020_TO_SRGB", "STRIP_ROWS", "rgb_to_xyz", "srgb_levels", "to_srgb", "usable_cpus"]
