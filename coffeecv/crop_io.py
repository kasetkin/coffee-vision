"""The segmenter's crops as stored (ticket ML-19 D10, D32): `<stem>__cropped.jxl`, the crop's 8-bit sRGB pixels as
lossless JPEG XL, beside `<stem>__beanmask.png`, its 1-bit mask. Lossless, so what training reads is exactly the array
serving computes from the same canonical photo (tests/test_seg_parity.py); until ML-19 the crops were JPEG q95.

`<stem>` is the original photo's file name without its extension (labels/ml3/pool_exclude.csv names crops by it).
Effort 5: any effort decodes to the same pixels; 5 makes a 11 MP crop 6.2 MB in 2.2 s (7: 5.4 MB, 4.0 s; 2026-11-03).
"""
from __future__ import annotations

import os
from pathlib import Path

import imagecodecs as ic
import numpy as np

CROP_SUFFIX = "__cropped.jxl"
MASK_SUFFIX = "__beanmask.png"
EFFORT = 5


def crop_name(stem: str) -> str:
    return f"{stem}{CROP_SUFFIX}"


def stem_of(crop: Path) -> str:
    """The original photo's stem a crop was made from."""
    if not crop.name.endswith(CROP_SUFFIX):
        raise ValueError(f"{crop.name} is not a crop ({CROP_SUFFIX})")
    return crop.name[:-len(CROP_SUFFIX)]


def mask_path(crop: Path) -> Path:
    """The bean-region mask beside a crop: same stem, `__beanmask.png`. Absent on a D18 fallback photo."""
    return crop.with_name(stem_of(crop) + MASK_SUFFIX)


def write_crop(rgb: np.ndarray, path: Path) -> None:
    """An 8-bit RGB crop as lossless JPEG XL (sRGB, as the canonical photo's read-back is), written whole or not."""
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"a crop is H x W x 3 uint8, not {rgb.shape} {rgb.dtype}")
    data = ic.jpegxl_encode(np.ascontiguousarray(rgb), lossless=True, effort=EFFORT, numthreads=4)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def load_crop(path: Path) -> np.ndarray:
    """A stored crop's pixels, H x W x 3 uint8, exactly as written."""
    rgb = ic.jpegxl_decode(Path(path).read_bytes())
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"{path}: not an 8-bit RGB crop ({rgb.shape} {rgb.dtype})")
    return rgb
