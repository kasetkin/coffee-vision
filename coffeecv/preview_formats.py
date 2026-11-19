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

import hashlib
import io
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv.canonical import NOOPTIMIZE, REC2020, RELATIVE

D65 = (0.3127, 0.3290)
# Display P3: DCI-P3's primaries, D65, the sRGB curve, as a 256-point table, which reads 8-bit sRGB samples exactly as
# LittleCMS's own sRGB does (tests/test_preview_formats.py), in 2,124 bytes. Committed, not built at import: LittleCMS
# stamps the time into a profile's header, so each process would embed different bytes (the first OPS-25 deploy's
# smoke). The committed file is `p3_profile()`'s output of 2026-11-19, held to it by the tests.
_x = np.linspace(0.0, 1.0, 256)
SRGB_CURVE = np.round(np.where(_x <= 0.04045, _x / 12.92, ((_x + 0.055) / 1.055) ** 2.4) * 65535).astype(np.uint16)
P3_PROFILE_PATH = Path(__file__).resolve().parent / "data" / "display_p3.icc"
P3_PROFILE_SHA256 = "c08b2f7e1849bad388dd1ca0e55a9ab625f49cfb41d89d12d0e451ef799fb49f"


def p3_profile() -> bytes:
    """Display P3 as LittleCMS builds it now: the committed profile but for its header's creation time."""
    return ic.cms_profile("rgb", whitepoint=D65, primaries=(0.680, 0.320, 0.265, 0.690, 0.150, 0.060),
                          transferfunction=SRGB_CURVE)


DISPLAY_P3 = P3_PROFILE_PATH.read_bytes()
if hashlib.sha256(DISPLAY_P3).hexdigest() != P3_PROFILE_SHA256:
    raise ImportError(f"{P3_PROFILE_PATH} is not the committed Display P3 profile (OPS-25)")
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


# D9: each preview names its own format in its bottom-right corner, so a phone's screenshot says which file the browser
# showed; D10: today's sRGB preview too, during the trial. The tabs' own words.
LABELS = {"srgb_jpeg": "sRGB JPEG", "p3_jpeg": "P3 JPEG", "avif10": "AVIF 10-bit", "jxl": "JPEG XL"}
LABEL_SIZE = 0.04     # the text's height, of the long side: 41 px at 1024, about 13 CSS px in the page's 20rem box
LABEL_MARGIN = 0.02   # from the corner, of the long side
LABEL_MIN_PX = 8      # smaller than this, no label


def label(a: np.ndarray, name: str) -> np.ndarray:
    """A copy of `a` (H x W x 3, uint8 or uint16) with LABELS[name] in its bottom-right corner: white, outlined in
    black, both the dtype's extremes on every channel, so they are the same grey in any colour space and the label
    adds no colour difference between the tabs. One font (Pillow's own, at Pillow's pinned version) on both hosts.
    An image too small for a readable label is returned unlabelled."""
    from PIL import ImageDraw, ImageFont

    h, w = a.shape[:2]
    size = round(LABEL_SIZE * max(h, w))
    out = a.copy()
    if size < LABEL_MIN_PX:
        return out
    font = ImageFont.load_default(size=size)
    stroke = max(1, round(size / 10))
    x0, y0, x1, y1 = font.getbbox(LABELS[name], stroke_width=stroke)
    margin = round(LABEL_MARGIN * max(h, w))
    if x1 - x0 + 2 * margin > w or y1 - y0 + 2 * margin > h:
        return out
    at = (w - margin - x1, h - margin - y1)
    shape, fill = Image.new("L", (w, h)), Image.new("L", (w, h))
    ImageDraw.Draw(shape).text(at, LABELS[name], font=font, fill=255, stroke_width=stroke, stroke_fill=255)
    ImageDraw.Draw(fill).text(at, LABELS[name], font=font, fill=255)
    shape, fill = np.asarray(shape) >= 128, np.asarray(fill) >= 128   # no antialiased levels: pure black, pure white
    out[shape] = 0
    out[fill] = np.iinfo(a.dtype).max
    return out


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
    """The three files of `small`, the preview-sized Rec.2020 samples, in MIME's order, each with its label (D9)."""
    return {"p3_jpeg": p3_jpeg(label(small, "p3_jpeg")), "avif10": avif10(label(small, "avif10")),
            "jxl": jxl(label(small, "jxl"))}


def avif_cicp(data: bytes) -> tuple[int, int, int, bool] | None:
    """An AVIF's colour tags from its first `colr` box of type `nclx`: (primaries, transfer, matrix, full range), or
    None without one. For the checks: imagecodecs' decoder does not return them."""
    at = data.find(b"colrnclx")
    if at < 0 or len(data) < at + 15:
        return None
    primaries, transfer, matrix = struct.unpack(">HHH", data[at + 8:at + 14])
    return primaries, transfer, matrix, bool(data[at + 14] & 0x80)


__all__ = ["AVIF_CICP", "DISPLAY_P3", "MIME", "avif_cicp", "encode_all", "shrink"]
