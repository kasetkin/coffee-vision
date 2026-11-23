"""ML-37 D10: is imagecodecs' lossless JPEG XL bit-exact for uint16 and float16, over every code value?

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d10_exact.py

Every uint16 value 0..65535 and every float16 bit pattern (finite, then the non-finite ones apart) is laid out as an
H x W x 3 image (the three channels rotated, so each channel sees every value), encoded with jpegxl_encode(lossless=
True) at efforts 1, 5 and 7, decoded and compared bit for bit (float16 through .view(uint16)). The header each file
gets is read with coffeecv.jxl_header (bits, float, XYB, transfer): the tag the crop would carry. Writes
analysis/ml37/d10_exact.txt. Nothing under data/ is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import jxl_header

OUT = Path(__file__).with_suffix(".txt")
SRGB_TAG = dict(primaries=ic.JPEGXL.PRIMARIES.SRGB, transfer=ic.JPEGXL.TRANSFER_FUNCTION.SRGB)


def image_of(codes: np.ndarray, width: int = 256) -> np.ndarray:
    """1-D codes -> H x width x 3, channel c holding the codes rolled by c/3 of their length."""
    n = -(-codes.size // width) * width
    flat = np.resize(codes, n)
    chans = [np.roll(flat, c * n // 3) for c in range(3)]
    return np.stack(chans, axis=-1).reshape(-1, width, 3)


def check(name: str, img: np.ndarray, lines: list[str], **tag) -> None:
    for effort in (1, 5, 7):
        try:
            data = ic.jpegxl_encode(img, lossless=True, effort=effort, numthreads=2, **tag)
        except Exception as exc:     # report, do not stop: the non-finite floats may be refused
            lines.append(f"{name:34s} effort {effort}: encode refused: {type(exc).__name__}: {exc}")
            continue
        back = ic.jpegxl_decode(data, numthreads=2)
        same_dtype = back.dtype == img.dtype
        if img.dtype == np.float16 and same_dtype:
            exact = np.array_equal(back.view(np.uint16), img.view(np.uint16))
            bad = int(np.count_nonzero(back.view(np.uint16) != img.view(np.uint16)))
        else:
            exact = same_dtype and np.array_equal(back, img)
            bad = int(np.count_nonzero(back != img)) if back.shape == img.shape else -1
        h = jxl_header.read_header(data)
        lines.append(f"{name:34s} effort {effort}: {len(data):8d} B, decoded {back.dtype}, bit-exact {exact} "
                     f"({bad} samples differ); header {h.bits_per_sample}-bit float={h.float_sample} "
                     f"xyb={h.xyb_encoded} primaries={h.colour.primaries} transfer={h.colour.transfer}")


def main() -> None:
    lines = [f"imagecodecs {ic.__version__}, {ic.jpegxl_version()}, numpy {np.__version__}", ""]
    u16 = image_of(np.arange(65536, dtype=np.uint16))
    check("uint16 all 65536 values, sRGB tag", u16, lines, **SRGB_TAG)
    check("uint16 all 65536 values, no tag", u16, lines)

    bits = np.arange(65536, dtype=np.uint32).astype(np.uint16)
    f = bits.view(np.float16)
    finite = f[np.isfinite(f)]
    unit = finite[(finite >= 0) & (finite <= 1) & ~np.signbit(finite)]
    lines.append("")
    lines.append(f"float16: {finite.size} finite bit patterns, {unit.size} of them in [+0, 1] "
                 f"(subnormals included: {int(np.count_nonzero((unit > 0) & (unit < np.float16(2 ** -14))))})")
    check("float16 [+0, 1], sRGB tag", image_of(unit), lines, **SRGB_TAG)
    check("float16 every finite pattern, sRGB tag", image_of(finite), lines, **SRGB_TAG)
    special = np.array([np.inf, -np.inf, np.nan, -0.0], np.float16)
    check("float16 inf, -inf, nan, -0", image_of(np.resize(special, 256 * 3)), lines, **SRGB_TAG)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
