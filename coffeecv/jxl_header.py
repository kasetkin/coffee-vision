"""Read a JPEG XL file's image header: size, orientation, bit depth, extra channels and colour encoding.

    python -m coffeecv.jxl_header FILE.jxl ...

coffeecv.canonical's is_canonical (ML-19 D35) reads a file's colour encoding here. imagecodecs' jpegxl_decode does
not report it (ML-19 plan, finding 5), so this reads it from
the codestream: the SizeHeader and ImageMetadata bundles of ISO/IEC 18181-1 (section A.6, A.8, A.4 and E.2), up to
and including the colour encoding. Standard library only. Container files (the ISOBMFF box format) are unwrapped
first; the box types found are reported, since EXIF, XMP and JUMBF boxes are metadata. Checked against libjxl's own
reading on 31 files (analysis/ml19/p1/README.md) and by tests/test_jxl_header.py.
"""
from __future__ import annotations

import json
import struct
import sys
from dataclasses import asdict, dataclass, field

CODESTREAM_SIGNATURE = b"\xff\x0a"
CONTAINER_SIGNATURE = b"\x00\x00\x00\x0cJXL \r\n\x87\n"

COLOUR_SPACES = {0: "RGB", 1: "Grey", 2: "XYB", 3: "Unknown"}
WHITE_POINTS = {1: "D65", 2: "Custom", 10: "E", 11: "DCI"}
PRIMARIES = {1: "sRGB", 2: "Custom", 9: "BT2100", 11: "P3"}
TRANSFERS = {1: "BT709", 2: "Unknown", 8: "Linear", 13: "sRGB", 16: "PQ", 17: "DCI", 18: "HLG"}
INTENTS = {0: "Perceptual", 1: "Relative", 2: "Saturation", 3: "Absolute"}


class HeaderError(ValueError):
    pass


class Bits:
    """The codestream's bit reader: least significant bit of each byte first."""

    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def u(self, n: int) -> int:
        value = 0
        for i in range(n):
            byte = self.pos >> 3
            if byte >= len(self.data):
                raise HeaderError("the header runs past the end of the codestream")
            value |= ((self.data[byte] >> (self.pos & 7)) & 1) << i
            self.pos += 1
        return value

    def bool(self) -> bool:
        return bool(self.u(1))

    def u32(self, *dists: tuple[int, int]) -> int:
        """U32(d0, d1, d2, d3): each distribution is (offset, bits); Val(v) is (v, 0)."""
        offset, n = dists[self.u(2)]
        return offset + self.u(n)

    def enum(self) -> int:
        return self.u32((0, 0), (1, 0), (2, 4), (18, 6))

    def f16(self) -> None:
        self.u(16)


@dataclass
class ColourEncoding:
    want_icc: bool
    colour_space: str
    white_point: str | None = None
    primaries: str | None = None
    transfer: str | None = None
    gamma: float | None = None
    rendering_intent: str | None = None


@dataclass
class Header:
    width: int
    height: int
    orientation: int                 # 1 = as stored; 2-8 as EXIF's
    bits_per_sample: int
    float_sample: bool
    num_extra_channels: int
    extra_channel_types: list[str]
    have_preview: bool
    have_animation: bool
    xyb_encoded: bool
    colour: ColourEncoding
    boxes: list[str] = field(default_factory=list)   # container box types, empty for a bare codestream


def _size(b: Bits) -> tuple[int, int]:
    small = b.bool()
    if small:
        height = (b.u(5) + 1) * 8
    else:
        height = b.u32((1, 9), (1, 13), (1, 18), (1, 30))
    ratio = b.u(3)
    if ratio == 0:
        width = (b.u(5) + 1) * 8 if small else b.u32((1, 9), (1, 13), (1, 18), (1, 30))
    else:
        num, den = {1: (1, 1), 2: (12, 10), 3: (4, 3), 4: (3, 2), 5: (16, 9), 6: (5, 4), 7: (2, 1)}[ratio]
        width = height * num // den
    return width, height


def _preview(b: Bits) -> None:
    div8 = b.bool()
    dists = ((16, 0), (32, 0), (1, 5), (33, 9)) if div8 else ((1, 6), (65, 8), (321, 10), (1345, 12))
    b.u32(*dists)
    if b.u(3) == 0:
        b.u32(*dists)


def _bit_depth(b: Bits) -> tuple[int, bool]:
    if b.bool():   # float_sample
        bits = b.u32((32, 0), (16, 0), (24, 0), (1, 6))
        b.u(4)     # exp_bits_minus_one
        return bits, True
    return b.u32((8, 0), (10, 0), (12, 0), (1, 6)), False


def _extra_channel(b: Bits) -> str:
    if b.bool():   # d_alpha: an alpha channel with every default
        return "Alpha"
    kind = b.enum()
    _bit_depth(b)
    b.u32((0, 0), (3, 0), (4, 0), (1, 3))                              # dim_shift
    name_len = b.u32((0, 0), (0, 4), (16, 5), (48, 10))
    b.u(8 * name_len)
    if kind == 0:
        b.bool()                                                       # alpha_associated
    elif kind == 2:
        for _ in range(4):
            b.f16()                                                    # spot colour
    elif kind == 5:
        b.u32((1, 0), (0, 2), (3, 4), (19, 8))                         # cfa_channel
    return {0: "Alpha", 1: "Depth", 2: "SpotColour", 3: "SelectionMask", 4: "Black", 5: "CFA", 6: "Thermal",
            15: "NonOptional", 16: "Optional"}.get(kind, f"Reserved{kind}")


def _customxy(b: Bits) -> None:
    for _ in range(2):
        b.u32((0, 19), (524288, 19), (1048576, 20), (2097152, 21))


def _colour_encoding(b: Bits) -> ColourEncoding:
    if b.bool():   # all_default: sRGB, relative
        return ColourEncoding(False, "RGB", "D65", "sRGB", "sRGB", None, "Relative")
    want_icc = b.bool()
    space = COLOUR_SPACES.get(b.enum(), "Reserved")
    enc = ColourEncoding(want_icc, space)
    if want_icc:
        return enc
    if space != "XYB":
        wp = b.enum()
        enc.white_point = WHITE_POINTS.get(wp, f"Reserved{wp}")
        if wp == 2:
            _customxy(b)
        if space != "Grey":
            pr = b.enum()
            enc.primaries = PRIMARIES.get(pr, f"Reserved{pr}")
            if pr == 2:
                for _ in range(3):
                    _customxy(b)
    if b.bool():   # have_gamma
        enc.gamma = b.u(24) / 1e7
    else:
        tf = b.enum()
        enc.transfer = TRANSFERS.get(tf, f"Reserved{tf}")
    intent = b.enum()
    enc.rendering_intent = INTENTS.get(intent, f"Reserved{intent}")
    return enc


def codestream(data: bytes) -> tuple[bytes, list[str]]:
    """The codestream inside a file, and the container's box types (empty for a bare codestream)."""
    if data.startswith(CODESTREAM_SIGNATURE):
        return data, []
    if not data.startswith(CONTAINER_SIGNATURE):
        raise HeaderError("not a JPEG XL file")
    pos, boxes, parts = 0, [], []
    while pos < len(data):
        size, kind = struct.unpack(">I4s", data[pos:pos + 8])
        head = 8
        if size == 1:
            size, head = struct.unpack(">Q", data[pos + 8:pos + 16])[0], 16
        elif size == 0:
            size = len(data) - pos
        name = kind.decode("latin-1")
        boxes.append(name)
        body = data[pos + head:pos + size]
        if name == "jxlc":
            parts.append(body)
        elif name == "jxlp":
            parts.append(body[4:])
        pos += size
    if not parts:
        raise HeaderError("a JPEG XL container without a codestream box")
    return b"".join(parts), boxes


def read_header(data: bytes) -> Header:
    stream, boxes = codestream(data)
    if not stream.startswith(CODESTREAM_SIGNATURE):
        raise HeaderError("the codestream has no signature")
    b = Bits(stream[2:])
    width, height = _size(b)
    if b.bool():   # all_default: 8-bit sRGB, no extra channels, XYB
        return Header(width, height, 1, 8, False, 0, [], False, False, True,
                      ColourEncoding(False, "RGB", "D65", "sRGB", "sRGB", None, "Relative"), boxes)
    orientation, have_preview, have_animation = 1, False, False
    if b.bool():   # extra_fields
        orientation = b.u(3) + 1
        if b.bool():
            _size(b)                                                   # intrinsic size
        have_preview = b.bool()
        if have_preview:
            _preview(b)
        have_animation = b.bool()
        if have_animation:
            b.u32((100, 0), (1000, 0), (1, 10), (1, 30))
            b.u32((1, 0), (1001, 0), (1, 8), (1, 10))
            b.u32((0, 0), (0, 3), (0, 16), (0, 32))
            b.bool()
    bits, is_float = _bit_depth(b)
    b.bool()                                                           # modular_16bit_buffers
    num_extra = b.u32((0, 0), (1, 0), (2, 4), (1, 12))
    extra = [_extra_channel(b) for _ in range(num_extra)]
    xyb = b.bool()
    return Header(width, height, orientation, bits, is_float, num_extra, extra, have_preview, have_animation, xyb,
                  _colour_encoding(b), boxes)


def main() -> None:
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            print(json.dumps({"file": path, **asdict(read_header(f.read()))}))


if __name__ == "__main__":
    main()
