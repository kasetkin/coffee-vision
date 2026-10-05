"""Ticket ML-3 P3 (owner, 2026-10-05): strip ground_coffee_016.jpg, which coffeecv.strip_metadata cannot.

    python analysis/ml3/fix_ground_coffee_016.py

An internet-proxy negative (a 2011 Photoshop export from Wikimedia). Its XMP sits in an APP1 segment whose
header is "XMP\\0://ns.adobe.com/xap/1.0/\\0" instead of "http://ns.adobe.com/xap/1.0/\\0", so exiftool
reads it but will not edit or delete it -- a photographer credit, rights and a camera serial among it.
The owner allowed any fix, re-encoding included. This one is lossless: the whole segment is dropped (the
entropy-coded image data is not touched), then the normal exiftool pass removes the IPTC owner fields.

The checks are coffeecv.strip_metadata's: decoded pixels equal, no deny-listed tag left, every other tag
byte for byte -- except the XMP packet, which goes whole and is listed in the row's `removed`. The row
goes into labels/ml3/strip_manifest.csv keyed by the original hash, as the bulk run's rows are, so P3's
re-key finds it; the bulk run then sees the file as already done.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from coffeecv import strip_metadata as sm  # noqa: E402

PHOTO = REPO_ROOT / "dataset/ood_negatives/2026-09__internet_proxy/ground_coffee/ground_coffee_016.jpg"
MANIFEST = REPO_ROOT / "labels/ml3/strip_manifest.csv"
BAD_XMP_HEADER = b"XMP\x00://ns.adobe.com/xap/1.0/\x00"


def drop_segments(data: bytes, drop) -> tuple[bytes, int]:
    """`data` without the APPn segments `drop(marker, payload)` selects; everything from SOS on is copied
    as is. Returns (new bytes, number dropped)."""
    assert data[:2] == b"\xff\xd8", "not a JPEG"
    out, i, n = bytearray(data[:2]), 2, 0
    while True:
        if data[i] != 0xFF:
            raise ValueError(f"no marker at {i}")
        marker = data[i + 1]
        if marker == 0xDA:                                   # SOS: the image data and the rest, untouched
            out += data[i:]
            return bytes(out), n
        length = int.from_bytes(data[i + 2:i + 4], "big")
        seg = data[i:i + 2 + length]
        if 0xE0 <= marker <= 0xEF and drop(marker, seg[4:]):
            n += 1
        else:
            out += seg
        i += 2 + length


def main() -> int:
    if str(PHOTO.relative_to(REPO_ROOT)) in sm.read_manifest(MANIFEST):
        print(f"{PHOTO.name} is already in the manifest")
        return 0
    before = sm.read_tags(PHOTO)
    sha_before = sm.sha256_file(PHOTO)
    pixels = sm.pixel_digest(PHOTO)
    data, n = drop_segments(PHOTO.read_bytes(), lambda m, p: m == 0xE1 and p.startswith(BAD_XMP_HEADER))
    if n != 1:
        raise SystemExit(f"expected one APP1 segment with the non-standard XMP header, found {n}")
    work = Path(tempfile.mkdtemp(dir=PHOTO.parent, prefix=".fix_"))
    try:
        dropped, out = work / "dropped.jpg", work / PHOTO.name
        dropped.write_bytes(data)
        subprocess.run(["exiftool", "-q", "-q", *sm.DELETE_ARGS, "-o", str(out), str(dropped)], check=True)
        after = sm.read_tags(out)
        removed, problems = sm.compare(before, after, PHOTO.suffix)
        xmp_gone = sorted(k for k in before if k.startswith("XMP:") and k not in after)
        problems = [p for p in problems if not (p.startswith("tag lost: XMP:") or p.startswith("kept field lost: XMP:"))]
        if sm.pixel_digest(out) != pixels:
            problems.append("decoded pixels differ")
        if any(k.startswith("XMP:") for k in after):
            problems.append("XMP left after the segment was dropped")
        if problems:
            print("FAIL\n  " + "\n  ".join(problems))
            return 1
        shutil.copymode(PHOTO, out)
        os.replace(out, PHOTO)
    finally:
        shutil.rmtree(work)
    gone = sorted(set(removed) | set(xmp_gone))
    sm.append_manifest(MANIFEST, [{"path": str(PHOTO.relative_to(REPO_ROOT)), "sha256_before": sha_before,
                                   "sha256_after": sm.sha256_file(PHOTO), "pixel_sha256": pixels,
                                   "removed": " ".join(gone)}], header=[])
    print(f"{PHOTO.name}: dropped the non-standard XMP segment ({len(xmp_gone)} XMP tags) and "
          f"{len(set(removed) - set(xmp_gone))} other private tags; pixels unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
