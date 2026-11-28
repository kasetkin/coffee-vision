"""The preview (GLOSSARY.md "Preview"; ticket ML-37 D4, D15; ADR 0043): the model picture of a canonical photo, its long
side at most MAX_DIM px, as an untagged sRGB JPEG at quality QUALITY. The front makes it from the canonical photo it has
just made and sends it as its answer's first line, while the photo crosses to the inference host; the host sends only
the mask and the box, which the page draws over it.

It is the picture the models read (OPS-7 D8's promise, ML-37 D4): `canonical.load_canonical`'s two steps, the decode
and `readback.to_srgb` in strips (ML-37 R10: about 101 MB more at 17 MP, not the 1,074 MB of a whole-photo float copy),
then Pillow's bicubic resize to `size`. Ships to the front (webapp/deploy/release_manifest.py, FRONT_CODE) and to
the inference host (LIBRARY), which scales the photo's mask to the same `size` (webapp/app.py), so the two agree to
the pixel.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from coffeecv import canonical

MAX_DIM = 1024   # px, the long side
QUALITY = 85


def size(width: int, height: int) -> tuple[int, int]:
    """The preview's (width, height) for a photo of `width` x `height`: its long side at most MAX_DIM, never enlarged,
    each side rounded to the nearest pixel, at least 1. The one rule for the preview and the host's mask: Pillow's
    thumbnail rounds its own way, a pixel off on some photos (4910 x 3776: 788 rows, not 787)."""
    scale = min(1.0, MAX_DIM / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def jpeg(picture: np.ndarray) -> bytes:
    """A model picture (H x W x 3 uint8 sRGB) as the preview's JPEG at `size`, by Pillow's bicubic resize with its
    reducing gap (thumbnail's, without its rounding); no ICC profile, so a browser reads it as sRGB."""
    thumb = Image.fromarray(picture)
    to = size(*thumb.size)
    if to != thumb.size:
        thumb = thumb.resize(to, Image.Resampling.BICUBIC, reducing_gap=2.0)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=QUALITY)
    return buf.getvalue()


def from_canonical(data: bytes, numthreads: int | None = None) -> bytes:
    """A canonical photo's preview: its model picture, read as the host reads it (`canonical.load_canonical`), as
    `jpeg` makes it. Raises as load_canonical does (NotCanonical, Unreadable)."""
    return jpeg(canonical.load_canonical(data, numthreads))
