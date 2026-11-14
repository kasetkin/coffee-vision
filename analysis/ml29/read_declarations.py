"""Each file's colour declaration as coffeecv.colour_declaration reads it (ticket ML-29 P1's check).

    python analysis/ml29/read_declarations.py tmp/darktable_exported/*/* tmp/ml29_made/*

One line per file: its accepted format, the declaration (kind, where it was read, primaries, white point, curve), and
whether conversion() would take it; or the refusal. The ICC profile is the one the format's decoder finds, as
coffeecv.canonical.decode reads it. Nothing is decoded.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from PIL import Image  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import colour_declaration as cd  # noqa: E402


def decoder_icc(data: bytes, fmt: cn.AcceptedFormat) -> bytes | None:
    if fmt.decoder == "pillow_heif":
        import pillow_heif
        return pillow_heif.open_heif(io.BytesIO(data)).info.get("icc_profile") or None
    if fmt.is_raw or fmt.decoder == "imagecodecs.jpegxl":
        return None
    with Image.open(io.BytesIO(data), formats=(fmt.pillow_plugin,)) as im:
        return im.info.get("icc_profile") or None


def describe(path: Path) -> str:
    data = path.read_bytes()
    fmt = cn.accepted_format(data, path.name)
    try:
        d = cd.declare(data, fmt.id, decoder_icc(data, fmt))
    except cd.Refused as exc:
        return f"{fmt.id:5s} REFUSED: {exc}"
    if d.kind == "icc":
        what = f"icc ({len(d.icc)} bytes)"
    elif d.kind == "none":
        what = "none (sRGB)"
    else:
        prim = ",".join(f"{v:g}" for v in d.primaries)
        what = f"{d.kind:11s} {d.source}: primaries {prim} white {d.white[0]:g},{d.white[1]:g} curve {d.curve}" + \
            (f" {d.gamma:.4f}" if d.gamma else "")
    try:
        cd.conversion(d)
        verdict = "converted"
    except cd.Refused:
        verdict = "HDR: refused until P5"
    return f"{fmt.id:5s} {what}  [{verdict}]"


def main() -> None:
    for p in sys.argv[1:]:
        print(f"{Path(p).name}: {describe(Path(p))}")


if __name__ == "__main__":
    main()
