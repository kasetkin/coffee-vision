"""The page's three wide-gamut previews, the trial of ticket OPS-25 (D1): the canonical photo's 16-bit Rec.2020 samples,
shrunk to the preview's size as the sRGB preview is (`shrink`), then three files (`encode_all`):

    p3_jpeg   8-bit JPEG, Display P3, its ICC profile embedded, quality 85 (the sRGB preview's)
    avif10    10-bit AVIF, CICP 9/13/9: Rec.2020 primaries, the sRGB transfer, the BT.2020 matrix; quality 80
    jxl       16-bit lossy JPEG XL, Rec.2020 primaries, the sRGB transfer, distance 1.0

The AVIF and the JPEG XL carry the sRGB curve, not the canonical photo's BT.709 (ticket OPS-25 D8): Chromium shows
BT.709-tagged files with a display gamma, about 11 levels darker than the canonical photo reads, and the trial
compares gamut only, so all three files carry the curve of today's sRGB preview.

Display only, as the sRGB preview is (GLOSSARY.md, "Preview"): nothing here is what a model reads. Standard range only
(D3): the canonical photo holds no light above white.
"""
from __future__ import annotations

import io
import struct
from concurrent.futures import ThreadPoolExecutor

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv.canonical import NOOPTIMIZE, REC2020, RELATIVE

D65 = (0.3127, 0.3290)
# Display P3: DCI-P3's primaries, D65, the sRGB curve. LittleCMS builds it; the curve is a 256-point table, which
# reads 8-bit sRGB samples exactly as LittleCMS's own sRGB does (tests/test_preview_formats.py), in 2,124 bytes.
_x = np.linspace(0.0, 1.0, 256)
SRGB_CURVE = np.round(np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4) * 65535).astype(np.uint16)
DISPLAY_P3 = ic.cms_profile("rgb", whitepoint=D65, primaries=(0.680, 0.320, 0.265, 0.690, 0.150, 0.060),
                            transferfunction=SRGB_CURVE)
# D8: Rec.2020's primaries with the sRGB curve, what the AVIF and the JPEG XL are tagged as. Never embedded, so its
# curve is a finer table, for the 10- and 16-bit samples converted into it.
_x = np.linspace(0.0, 1.0, 4096)
REC2020_SRGB = ic.cms_profile("rgb", whitepoint=D65, primaries=(0.708, 0.292, 0.170, 0.797, 0.131, 0.046),
                              transferfunction=np.round(np.where(_x <= 0.04045, _x / 12.92,
                                                                 ((_x + 0.055) / 1.055) ** 2.4) * 65535
                                                        ).astype(np.uint16))

JPEG_QUALITY = 85     # the sRGB preview's
AVIF_QUALITY = 80
AVIF_SPEED = 8        # 0.25 s for 1024 x 768 against libavif's default 6's 0.56 s, for 0.5% more bytes (plan P1)
AVIF_CICP = (9, 13, 9)  # primaries BT.2020, transfer sRGB (D8), matrix BT.2020 non-constant luminance (H.273)
JXL_DISTANCE = 1.0
JXL_EFFORT = 7        # the store's (canonical.EFFORT)

# The page's order of the tabs after the sRGB preview, and each file's type for its data: URL.
MIME = {"p3_jpeg": "image/jpeg", "avif10": "image/avif", "jxl": "image/jxl"}


def shrink(rec: np.ndarray, max_dim: int) -> np.ndarray:
    """An H x W x 3 uint16 photo at the size Pillow's `thumbnail((max_dim, max_dim))` gives the sRGB preview, by the
    same resampling, one channel at a time on three threads: as 32-bit integers (mode I), since Pillow has no 16-bit
    RGB mode and cannot reduce its 16-bit grey (I;16), which thumbnail does first over twice the size."""
    def one(k: int) -> np.ndarray:
        im = Image.fromarray(rec[..., k].astype(np.int32))
        im.thumbnail((max_dim, max_dim))
        return np.clip(np.asarray(im), 0, 65535).astype(np.uint16)   # bicubic overshoots at edges

    if max(rec.shape[:2]) <= max_dim:
        return rec
    with ThreadPoolExecutor(3) as pool:
        return np.stack(list(pool.map(one, range(3))), axis=-1)


def p3_jpeg(small: np.ndarray) -> bytes:
    p3 = ic.cms_transform(np.ascontiguousarray(small), REC2020, DISPLAY_P3, intent=RELATIVE, flags=NOOPTIMIZE,
                          outdtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(p3).save(buf, format="JPEG", quality=JPEG_QUALITY, icc_profile=DISPLAY_P3)
    return buf.getvalue()


def _srgb_curve(small: np.ndarray) -> np.ndarray:
    """The canonical photo's Rec.2020 samples, BT.709 curve, as Rec.2020 with the sRGB curve (D8), 16-bit."""
    return ic.cms_transform(np.ascontiguousarray(small), REC2020, REC2020_SRGB, intent=RELATIVE, flags=NOOPTIMIZE,
                            outdtype=np.uint16)


def avif10(small: np.ndarray) -> bytes:
    ten = ((_srgb_curve(small).astype(np.uint32) * 1023 + 32767) // 65535).astype(np.uint16)
    primaries, transfer, matrix = AVIF_CICP
    return ic.avif_encode(ten, level=AVIF_QUALITY, speed=AVIF_SPEED, bitspersample=10, primaries=primaries,
                          transfer=transfer, matrix=matrix)


def jxl(small: np.ndarray) -> bytes:
    """imagecodecs' own encoder: the preview is small, so jxl_encode's streaming (and its fixed BT.709) is not needed."""
    return ic.jpegxl_encode(_srgb_curve(small), distance=JXL_DISTANCE, effort=JXL_EFFORT, primaries="BT2100",
                            transfer="SRGB", numthreads=2)


def encode_all(small: np.ndarray) -> dict[str, bytes]:
    """The three files of `small`, the preview-sized Rec.2020 samples, in MIME's order."""
    return {"p3_jpeg": p3_jpeg(small), "avif10": avif10(small), "jxl": jxl(small)}


def avif_cicp(data: bytes) -> tuple[int, int, int, bool] | None:
    """An AVIF's colour tags from its first `colr` box of type `nclx`: (primaries, transfer, matrix, full range), or
    None without one. For the checks: imagecodecs' decoder does not return them."""
    at = data.find(b"colrnclx")
    if at < 0 or len(data) < at + 15:
        return None
    primaries, transfer, matrix = struct.unpack(">HHH", data[at + 8:at + 14])
    return primaries, transfer, matrix, bool(data[at + 14] & 0x80)


__all__ = ["AVIF_CICP", "DISPLAY_P3", "MIME", "avif_cicp", "encode_all", "shrink"]
