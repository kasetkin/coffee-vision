"""Photo decoding: every format the service and training read, to a uint8 RGB array. No torch, so the web service
imports it alone (ticket ML-7 P8); coffeecv.dataset re-exports it for training's callers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pillow_avif  # noqa: F401 -- import alone registers AVIF with Pillow
import pillow_heif
import pillow_jxl  # noqa: F401 -- import alone registers JPEG XL with Pillow
import rawpy
from PIL import Image

pillow_heif.register_heif_opener()
# AVIF and JPEG XL (pillow_avif/pillow_jxl above) need no explicit register_opener()
# call -- each self-registers with Pillow's `Image.ID` on import, same effect as this
# line has for HEIF, just triggered differently by each library's own __init__.


# Major-brand camera RAW extensions LibRaw (via rawpy) decodes. Unlike every
# other format this module registers, Pillow can't open these at all -- there's no opener to
# register, so load_rgb_image branches explicitly instead.
RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def load_rgb_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in RAW_EXTENSIONS:
        try:
            with rawpy.imread(str(path)) as raw:
                # output_bps=8: the rest of the pipeline (PIL.Image.fromarray in
                # patches_for_photo, bean_scale's luma weights) expects uint8,
                # not rawpy's 16-bit default. use_camera_wb: the color the camera
                # actually recorded, not a gray-world guess.
                return raw.postprocess(output_bps=8, use_camera_wb=True)
        except rawpy.LibRawError as exc:
            # Normalized to OSError so this joins the same "can't read this
            # photo" handling every other decode failure already gets
            # (classify_one's try/except, webapp's /analyze) -- rawpy's own
            # exceptions aren't OSError/ValueError.
            # Message is fixed rather than str(exc): LibRaw's own messages
            # come through as a bytes-repr ("b'Input/output error'"), which
            # would look broken surfaced straight to a user.
            raise OSError(f"unreadable RAW file ({type(exc).__name__})") from exc
    return np.array(Image.open(path).convert("RGB"))
