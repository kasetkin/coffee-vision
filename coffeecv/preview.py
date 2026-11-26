"""The preview (GLOSSARY.md "Preview"; ticket ML-37 D4, D15; ADR 0043): the model picture of a canonical photo, its long
side at most MAX_DIM px, as an untagged sRGB JPEG at quality QUALITY. The front makes it from the canonical photo it has
just made and sends it as its answer's first line, while the photo crosses to the inference host; the host sends only
the mask and the box, which the page draws over it.

It is the picture the models read (OPS-7 D8's promise, ML-37 D4): `canonical.load_canonical`'s two steps, the decode
and `readback.to_srgb` in strips (ML-37 R10: about 101 MB more at 17 MP, not the 1,074 MB of a whole-photo float copy),
then Pillow's thumbnail. Ships to the front (webapp/deploy/release_manifest.py, FRONT_CODE).
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from coffeecv import canonical

MAX_DIM = 1024   # px, the long side; the host scales the mask to the same (webapp/app.py, PREVIEW_MAX_DIM)
QUALITY = 85


def jpeg(picture: np.ndarray) -> bytes:
    """A model picture (H x W x 3 uint8 sRGB) as the preview's JPEG: shrunk by Pillow's thumbnail to MAX_DIM on its
    long side, never enlarged, aspect kept; no ICC profile, so a browser reads it as sRGB."""
    thumb = Image.fromarray(picture)
    thumb.thumbnail((MAX_DIM, MAX_DIM))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=QUALITY)
    return buf.getvalue()


def from_canonical(data: bytes, numthreads: int | None = None) -> bytes:
    """A canonical photo's preview: its model picture, read as the host reads it (`canonical.load_canonical`), as
    `jpeg` makes it. Raises as load_canonical does (NotCanonical, Unreadable)."""
    return jpeg(canonical.load_canonical(data, numthreads))
