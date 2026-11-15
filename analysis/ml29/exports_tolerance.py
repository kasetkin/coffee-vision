"""Each export's canonical photo against the 8-bit sRGB export's, as the models read them (ticket ML-29, D27).

    python analysis/ml29/exports_tolerance.py [--settings store|front] FILE ...
    python analysis/ml29/exports_tolerance.py tmp/darktable_exported/*/* tmp/ml29_made/*

Each file is canonicalized with the given settings (canonical.canonicalize_with) and read back as 8-bit sRGB
(load_canonical); so is the reference, the darktable 8-bit sRGB JPEG XL (--reference). Both are cut into 32 x 32
blocks; a block's value is its mean per channel. Reported per file: the mean absolute difference over every block
and channel ("mean"), and the largest difference in any block and channel ("worst"), in 8-bit levels; D27's bar for
standard-range declarations is mean <= 1.0 and worst <= 6. A file canonicalization refuses is reported with the
message a visitor would read. For a file of another size (the files made here, area-averaged to a quarter in linear
light) the reference is area-averaged to its size, in 8-bit sRGB.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402

REFERENCE = REPO / "tmp" / "darktable_exported" / "jxl" / "20261113_172701_Canon_EOS_90D_Brazil-q90-8bit-sRGB.jxl"
BLOCK = 32


def blocks(a: np.ndarray) -> np.ndarray:
    h, w = (a.shape[0] // BLOCK) * BLOCK, (a.shape[1] // BLOCK) * BLOCK
    return a[:h, :w].reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, 3).mean((1, 3))


def read(path: Path, settings: cn.Settings) -> np.ndarray:
    return cn.load_canonical(cn.canonicalize_with(path.read_bytes(), path.name, settings).canonical)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--settings", choices=("store", "front"), default="store")
    ap.add_argument("--reference", type=Path, default=REFERENCE)
    a = ap.parse_args()
    settings = cn.STORE_SETTINGS if a.settings == "store" else cn.FRONT_SETTINGS
    ref_full = read(a.reference, settings)
    print(f"reference {a.reference.name}, settings {settings.label}")
    for path in a.files:
        t = time.monotonic()
        try:
            got = read(path, settings)
        except cn.Unreadable as exc:
            print(f"{path.name:70s} refused: {exc}")
            continue
        ref = ref_full
        if got.shape != ref.shape:                       # the files made here are a quarter of the size
            import cv2
            ref = cv2.resize(ref_full, (got.shape[1], got.shape[0]), interpolation=cv2.INTER_AREA)
        ref = blocks(ref.astype(np.float64))
        d = np.abs(blocks(got.astype(np.float64)) - ref)
        ok = "within D27" if d.mean() <= 1.0 and d.max() <= 6 else "OUTSIDE D27"
        print(f"{path.name:70s} mean {d.mean():5.2f}  worst {d.max():5.1f}  {ok}  ({time.monotonic() - t:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
