"""Does every dataset photo keep the colour conversion it has today under ADR 0040's rule? (ticket ML-29 P1, P2)

    python analysis/ml29/dataset_same_profile.py [--root dataset] [--jobs 8]

For each photo in an accepted format under --root: the source profile canonicalization used before ML-29 (a usable
ICC profile as its decoder finds it, else LittleCMS's sRGB; a JPEG XL's from its two enumerations; a RAW's linear
Rec.2020) against colour_declaration.conversion(declare(...)), the rule P2 wires in. Prints the count that differ
and each one's path; nothing is decoded but metadata. Exit 1 if any differs.
"""
from __future__ import annotations

import argparse
import collections
import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from PIL import Image  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import colour_declaration as cd  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def today(data: bytes, fmt: cn.AcceptedFormat) -> tuple[bytes | None, str]:
    """The profile decode() gave this photo before ML-29 (None: sRGB), and the ICC its decoder found."""
    if fmt.decoder == "pillow_heif":
        import pillow_heif
        icc = pillow_heif.open_heif(io.BytesIO(data)).info.get("icc_profile") or None
        return cd.usable_profile(icc), icc
    if fmt.decoder == "imagecodecs.jpegxl":
        c = cn.read_header(data).colour
        return {("sRGB", "sRGB"): None, ("BT2100", "BT709"): cn.REC2020}[(c.primaries, c.transfer)], None
    with Image.open(io.BytesIO(data), formats=(fmt.pillow_plugin,)) as im:
        icc = im.info.get("icc_profile") or None
    return cd.usable_profile(icc), icc


def check(path: Path) -> tuple[str, str, str]:
    data = path.read_bytes()
    fmt = cn.accepted_format(data, path.name)
    if fmt.is_raw:
        return str(path), fmt.id, "same (RAW: developed)"
    old, icc = today(data, fmt)
    try:
        d = cd.declare(data, fmt.id, icc)
        new = cd.conversion(d)
    except cd.Refused as exc:
        return str(path), fmt.id, f"DIFFERS: refused ({exc})"
    if new == cd.SourceColour(old):
        return str(path), fmt.id, f"same ({d.kind})"
    return str(path), fmt.id, f"DIFFERS: {d.kind} {d.source}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=REPO / "dataset")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    paths = sorted(p for p in a.root.rglob("*") if p.is_file() and cn.is_photo(p))
    with ProcessPoolExecutor(a.jobs) as pool:
        rows = list(pool.map(check, paths, chunksize=8))
    print(f"{len(rows)} photos under {a.root}")
    for (fid, verdict), n in sorted(collections.Counter((f, v.split(":")[0]) for _, f, v in rows).items()):
        print(f"{n:6d}  {fid:5s} {verdict}")
    bad = [r for r in rows if r[2].startswith("DIFFERS")]
    for path, fid, verdict in bad:
        print(f"{fid} {verdict}  {path}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
