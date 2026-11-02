"""The front's check of an upload's format (ticket OPS-15 P1): is this file one the inference host can decode?

    refusal("IMG_0001.HEIC", first_bytes)   # None, or a reason a visitor can read

Mirrors how coffeecv.photo_io.load_rgb_image dispatches: a name with a RAW extension goes to rawpy, so its first
bytes must be that RAW family's signature; any other name goes to Pillow, which opens by content, so the content
alone decides. Only the formats the public contract names (webapp/static/llms.txt) pass: not GIF, BMP or a plain
TIFF, which Pillow would open too. The front has the standard
library only (D7) and cannot import coffeecv, so tests/test_front_upload_check.py holds this to photo_io: the same
RAW extensions, and every Pillow format accepted here decoded there. A file that passes may still fail to decode;
the inference host answers that one with its own 400.
"""
from __future__ import annotations

from pathlib import PurePath

# coffeecv.photo_io.RAW_EXTENSIONS, held equal by the test.
RAW_EXTENSIONS = frozenset({".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"})
# Enough for every signature below, an ISO-BMFF ftyp box's compatible brands included (a phone's HEIC: 44 bytes).
HEAD_BYTES = 64

_TIFF = (b"II*\x00", b"MM\x00*")
# Each RAW extension's signatures, as LibRaw identifies the family: TIFF-based for most, and the three that are not.
_RAW_SIGNATURES = {
    ".dng": _TIFF, ".cr2": _TIFF, ".nef": _TIFF, ".arw": _TIFF, ".pef": _TIFF, ".srw": _TIFF,
    ".orf": (b"IIRO", b"IIRS", b"MMOR"),
    ".rw2": (b"IIU\x00",),
    ".raf": (b"FUJIFILMCCD-RAW",),
}
_CR3_BRANDS = {b"crx "}
# HEIF (HEIC included) and AVIF image brands, ISO/IEC 23008-12 and the AVIF spec; mif1/msf1 are HEIF's generic ones.
_HEIF_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1",
                b"avif", b"avis"}
_JXL = (b"\xff\x0a", b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a")   # bare codestream, ISO BMFF container

_NOT_A_PHOTO = "not a JPEG, PNG, WebP, HEIC/HEIF, AVIF, JPEG XL or camera RAW file"


def _ftyp_brands(head: bytes) -> set[bytes]:
    """The major and compatible brands of an ISO-BMFF file's ftyp box, as far as `head` holds them."""
    if len(head) < 12 or head[4:8] != b"ftyp":
        return set()
    end = min(int.from_bytes(head[:4], "big"), len(head))
    return {head[8:12]} | {head[i:i + 4] for i in range(16, end - 3, 4)}


def _pillow_format(head: bytes) -> bool:
    return (head.startswith(b"\xff\xd8\xff")
            or head.startswith(b"\x89PNG\r\n\x1a\n")
            or (head[:4] == b"RIFF" and head[8:12] == b"WEBP")
            or head.startswith(_JXL)
            or bool(_ftyp_brands(head) & _HEIF_BRANDS))


def refusal(filename: str, head: bytes) -> str | None:
    """None if `head`, the file's first HEAD_BYTES bytes, may be a photo load_rgb_image reads; else why not."""
    ext = PurePath(filename).suffix.lower()
    if ext in RAW_EXTENSIONS:
        ok = bool(_ftyp_brands(head) & _CR3_BRANDS) if ext == ".cr3" else head.startswith(_RAW_SIGNATURES[ext])
        return None if ok else f"the file is named {ext} but is not a {ext[1:].upper()} camera RAW file"
    return None if _pillow_format(head) else _NOT_A_PHOTO
