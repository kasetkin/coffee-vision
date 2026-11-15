"""What a photo file says its colour numbers mean, and what converting from it takes (ticket ML-29; ADR 0040).

    declare(data, format_id, icc)  -> Declaration   the file's colour declaration, by ADR 0040's precedence
    conversion(declaration)        -> SourceColour  the source profile, and the curve undone in our own code first

A colour declaration (GLOSSARY.md) is an ICC profile, a coded tag (HEIF's and AVIF's `nclx`, PNG's `cICP`, and PNG's
`sRGB` chunk), a JPEG XL header enumeration, or the EXIF interoperability tag R03 (Adobe RGB) of a JPEG. They are
honoured in that order (D4): a usable ICC profile first, then the coded tag or enumeration, then R03, then nothing,
which is read as sRGB (ML-19 D16); but a PNG's cICP comes before its ICC profile, as the PNG 3rd edition orders
(D30). "Unspecified" declares nothing; an unspecified primaries or curve, beside a specified other half, is sRGB's.
One that cannot be converted raises Refused, naming it. Not read (D7): PNG's gAMA and cHRM chunks, a BMP header's
calibrated endpoints. A BMP's embedded ICC profile is read here, since Pillow does not.

`icc` is the profile the format's decoder already found (Pillow's or pillow-heif's): this module reads only what
those leave out. A file with a profile LittleCMS cannot use as RGB is read as sRGB, whatever else it declares (ADR
0040's consequence of ML-19 D16), unless it is a PNG with a cICP chunk.

The builder (D20): a declaration that is not an ICC profile becomes a linear RGB profile with its primaries and white
point (LittleCMS), with its curve undone in numpy first, so one builder serves every curve. Two declarations keep the
exact profile canonicalization used for them before this module, so those photos keep their bytes: sRGB (LittleCMS's
own sRGB) and Rec.2020 with the BT.709 curve (the canonical photo's committed profile, ML-19 D46). PQ and HLG are
recognised and marked HDR (Declaration.is_hdr); conversion() refuses them until a tone-mapping operator is chosen
(the plan's P4, P5).
"""
from __future__ import annotations

import io
import struct
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv.jxl_header import ColourEncoding

Primaries = tuple[float, float, float, float, float, float]   # red x, y; green x, y; blue x, y
XY = tuple[float, float]

D65: XY = (0.3127, 0.3290)
D50: XY = (0.3457, 0.3585)
SRGB_PRIMARIES: Primaries = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06)
REC2020_PRIMARIES: Primaries = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046)
P3_PRIMARIES: Primaries = (0.680, 0.320, 0.265, 0.690, 0.150, 0.060)
ADOBE_RGB_PRIMARIES: Primaries = (0.64, 0.33, 0.21, 0.71, 0.15, 0.06)
ADOBE_RGB_GAMMA = 563 / 256                                     # Adobe RGB (1998), the gamma of its section 4.3

REC2020_PROFILE = (Path(__file__).resolve().parent / "data" / "rec2020_bt709.icc").read_bytes()   # ML-19 D46

# ITU-T H.273 (07/2024), Table 2 (ColourPrimaries) and Table 3 (TransferCharacteristics): the values honoured, as
# (primaries, white point) and as this module's curve names. 2 is "unspecified"; every other value is refused.
H273_PRIMARIES: dict[int, tuple[str, Primaries, XY]] = {
    1: ("BT.709", SRGB_PRIMARIES, D65),
    4: ("BT.470 System M", (0.67, 0.33, 0.21, 0.71, 0.14, 0.08), (0.310, 0.316)),
    5: ("BT.470 System B, G", (0.64, 0.33, 0.29, 0.60, 0.15, 0.06), D65),
    6: ("BT.601 525", (0.630, 0.340, 0.310, 0.595, 0.155, 0.070), D65),
    7: ("SMPTE 240M", (0.630, 0.340, 0.310, 0.595, 0.155, 0.070), D65),
    8: ("generic film", (0.681, 0.319, 0.243, 0.692, 0.145, 0.049), (0.310, 0.316)),
    9: ("BT.2020", REC2020_PRIMARIES, D65),
    11: ("DCI-P3", P3_PRIMARIES, (0.314, 0.351)),
    12: ("Display P3", P3_PRIMARIES, D65),
    22: ("EBU 3213-E", (0.630, 0.340, 0.295, 0.605, 0.155, 0.077), D65),
}
H273_TRANSFERS: dict[int, str] = {1: "BT709", 4: "gamma2.2", 5: "gamma2.8", 6: "BT709", 7: "SMPTE240M", 8: "linear",
                                  13: "sRGB", 14: "BT709", 15: "BT709", 16: "PQ", 17: "DCI", 18: "HLG"}
UNSPECIFIED = 2

# JPEG XL's header enumerations (ISO/IEC 18181-1, E.2), as coffeecv.jxl_header names them
JXL_PRIMARIES: dict[str, Primaries] = {"sRGB": SRGB_PRIMARIES, "BT2100": REC2020_PRIMARIES, "P3": P3_PRIMARIES}
JXL_WHITE_POINTS: dict[str, XY] = {"D65": D65, "DCI": (0.314, 0.351), "E": (1 / 3, 1 / 3)}
# JPEG XL's DCI curve is a pure 2.6 gamma (libjxl's jxl_cms), not H.273's SMPTE ST 428-1 form under the same number
JXL_TRANSFERS = {"BT709": "BT709", "Linear": "linear", "sRGB": "sRGB", "PQ": "PQ", "HLG": "HLG", "DCI": "gamma"}

CURVES = ("sRGB", "BT709", "linear", "gamma", "SMPTE240M", "DCI", "PQ", "HLG")


class Refused(ValueError):
    """A colour declaration canonicalization cannot convert from (ADR 0040). The message names it."""


@dataclass(frozen=True)
class Declaration:
    """A file's colour declaration. `source` says where it was read and what it said ("nclx 9/8", "ICC profile",
    "JPEG XL header", "R03", "none"). For an ICC profile, `icc`; otherwise primaries, white point and curve (`gamma` is
    the decoding exponent of curve "gamma": linear = encoded ** gamma). Kind "none" is read as sRGB."""
    kind: str                           # "icc", "coded", "enumeration", "r03", "none"
    source: str
    icc: bytes | None = None
    primaries: Primaries | None = None
    white: XY | None = None
    curve: str | None = None
    gamma: float | None = None

    @property
    def is_hdr(self) -> bool:
        """PQ or HLG: a high-dynamic-range curve (GLOSSARY.md, "HDR", the curve)."""
        return self.curve in ("PQ", "HLG")


NONE = Declaration("none", "none")


# ----- reading -----

def usable_profile(profile: bytes | None) -> bytes | None:
    """The photo's ICC profile if LittleCMS can convert from it as RGB; else None, so the photo is read as sRGB, as an
    untagged one is (ML-19 D16): a grey or broken profile, which a phone does not write, refuses no photo."""
    if not profile or profile[16:20] != b"RGB ":
        return None
    try:
        ic.cms_profile_validate(profile)
    except Exception:   # noqa: BLE001 -- imagecodecs raises its own CmsError, or ValueError on garbage
        return None
    return profile


def declare(data: bytes, format_id: str, icc: bytes | None = None) -> Declaration:
    """The colour declaration of a file in accepted format `format_id` (coffeecv/data/accepted_formats.csv's id), by
    precedence: `icc` (the profile its decoder found), then what this module reads from the bytes. A profile LittleCMS
    cannot use as RGB is sRGB (ML-19 D16), not a way to the next declaration. Raises Refused for a declaration that
    cannot be converted, or colour boxes that cannot be parsed. A RAW format declares nothing that applies: it is
    developed."""
    if format_id == "bmp" and not icc:
        icc = bmp_profile(data)
    if format_id == "png":                         # PNG 3rd edition, section 4.3: cICP before iCCP
        cicp = _parsed(format_id, _png_cicp, data)
        if cicp is not NONE:
            return cicp
    if icc:
        profile = usable_profile(icc)
        return Declaration("icc", "ICC profile", icc=profile) if profile else NONE
    reader = {"heif": _bmff, "avif": _bmff, "png": _png_srgb, "jxl": _jxl, "jpeg": _jpeg}.get(format_id)
    return NONE if reader is None else _parsed(format_id, reader, data)


def _parsed(format_id: str, reader: Callable[[bytes], Declaration], data: bytes) -> Declaration:
    try:
        return reader(data)
    except (struct.error, IndexError) as exc:      # an upload's boxes or chunks running past their own ends
        raise Refused(f"a {format_id.upper()} file whose colour declaration cannot be parsed") from exc


def _cannot(what: str) -> Refused:
    return Refused(f"a colour declaration canonicalization does not read: {what}")


def _coded(source: str, primaries: int, transfer: int) -> Declaration:
    """An H.273 pair (an nclx box, a cICP chunk) as a declaration: "unspecified" in both is none, in one is sRGB's."""
    what = f"{source} {primaries}/{transfer}"
    if primaries == UNSPECIFIED and transfer == UNSPECIFIED:
        return NONE
    if primaries != UNSPECIFIED and primaries not in H273_PRIMARIES:
        raise _cannot(f"{what}, primaries {primaries}")
    if transfer != UNSPECIFIED and transfer not in H273_TRANSFERS:
        raise _cannot(f"{what}, transfer characteristics {transfer}")
    name, xy, white = H273_PRIMARIES.get(primaries, H273_PRIMARIES[1])
    curve = H273_TRANSFERS.get(transfer, "sRGB")
    gamma = {"gamma2.2": 2.2, "gamma2.8": 2.8}.get(curve)
    return Declaration("coded", f"{what} ({name}, {curve})", primaries=xy, white=white,
                       curve="gamma" if gamma else curve, gamma=gamma)


# ISO BMFF (HEIF, AVIF): the primary item's colr properties (ISO/IEC 23008-12, 6.5.5; 14496-12, 12.1.5)

def _boxes(d: bytes, s: int, e: int):
    while s + 8 <= e:
        n, t = struct.unpack(">I4s", d[s:s + 8])
        h = 8
        if n == 1:
            n, h = struct.unpack(">Q", d[s + 8:s + 16])[0], 16
        if n == 0:
            n = e - s
        if n < h or s + n > e:
            return
        yield t, s + h, s + n
        s += n


def primary_properties(d: bytes) -> list[tuple[bytes, bytes]]:
    """The properties (type, body) associated with a HEIF's or AVIF's primary item, in association order."""
    meta = next(((a, b) for t, a, b in _boxes(d, 0, len(d)) if t == b"meta"), None)
    if not meta:
        return []
    primary, props, assoc = None, [], {}
    for t, a, b in _boxes(d, meta[0] + 4, meta[1]):            # meta is a full box: version and flags first
        if t == b"pitm":
            primary = struct.unpack(">H" if d[a] == 0 else ">I", d[a + 4:a + (6 if d[a] == 0 else 8)])[0]
        elif t == b"iprp":
            for t2, a2, b2 in _boxes(d, a, b):
                if t2 == b"ipco":
                    props = [(t3, d[a3:b3]) for t3, a3, b3 in _boxes(d, a2, b2)]
                elif t2 == b"ipma":
                    version, flags = d[a2], d[a2 + 3]
                    p = a2 + 8
                    for _ in range(struct.unpack(">I", d[a2 + 4:a2 + 8])[0]):
                        wide = 4 if version >= 1 else 2
                        item = int.from_bytes(d[p:p + wide], "big")
                        p += wide
                        count = d[p]
                        p += 1
                        found = []
                        for _ in range(count):
                            if flags & 1:
                                found.append(struct.unpack(">H", d[p:p + 2])[0] & 0x7FFF)
                                p += 2
                            else:
                                found.append(d[p] & 0x7F)
                                p += 1
                        assoc[item] = assoc.get(item, []) + found
    return [props[i - 1] for i in assoc.get(primary, []) if 1 <= i <= len(props)]


def primary_colr(d: bytes) -> list[bytes]:
    """The bodies of the primary item's colr properties."""
    return [body for kind, body in primary_properties(d) if kind == b"colr"]


def primary_bit_depth(d: bytes) -> int | None:
    """The primary item's bits per channel, from its pixi property (ISO/IEC 23008-12, 6.5.6), the first channel's;
    None without one."""
    body = next((b for kind, b in primary_properties(d) if kind == b"pixi"), b"")
    return body[5] if len(body) >= 6 and body[4] >= 1 else None


def _bmff(data: bytes) -> Declaration:
    """A HEIF's or AVIF's primary item: its nclx colr box. Its ICC colr box is the decoder's to read (`icc`)."""
    for body in primary_colr(data):
        if body[:4] == b"nclx" and len(body) >= 10:
            primaries, transfer = struct.unpack(">HH", body[4:8])
            return _coded("nclx", primaries, transfer)
    return NONE


def _png_chunks(data: bytes):
    """A PNG's chunks before its image data, as (type, body)."""
    pos = 8
    while pos + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind in (b"IDAT", b"IEND"):
            return
        yield kind, data[pos + 8:pos + 8 + n]
        pos += 12 + n


def _png_cicp(data: bytes) -> Declaration:
    """A PNG's cICP chunk, read before its ICC profile as the PNG 3rd edition's section 4.3 orders."""
    for kind, body in _png_chunks(data):
        if kind == b"cICP" and len(body) == 4:
            if body[2] != 0:
                raise _cannot(f"cICP {body[0]}/{body[1]} with matrix coefficients {body[2]}")
            if body[3] != 1:                     # narrow range: PNG decoders do not scale it, so it is not read
                raise _cannot(f"cICP {body[0]}/{body[1]} in narrow range")
            return _coded("cICP", body[0], body[1])
    return NONE


def _png_srgb(data: bytes) -> Declaration:
    """A PNG's sRGB chunk, read after its ICC profile; gAMA and cHRM are not read (D7)."""
    if any(kind == b"sRGB" for kind, _ in _png_chunks(data)):
        return Declaration("coded", "PNG sRGB chunk", primaries=SRGB_PRIMARIES, white=D65, curve="sRGB")
    return NONE


def _jxl(data: bytes) -> Declaration:
    from coffeecv.jxl_header import read_header
    return jxl_declaration(read_header(data).colour)


def jxl_declaration(c: ColourEncoding) -> Declaration:
    """A JPEG XL header's colour encoding as a declaration (D5): any primaries, white point and curve; an embedded ICC
    profile is refused (D22)."""
    if c.want_icc:
        raise Refused("a JPEG XL with an embedded ICC profile, which canonicalization cannot read yet: export it with "
                      "the colour space in the header (an enumeration)")
    what = f"JPEG XL header {c.colour_space} {c.white_point} {c.primaries} {c.transfer or f'gamma {c.gamma}'}"
    if c.colour_space not in ("RGB", "Grey"):
        raise _cannot(what)
    white = c.white_xy if c.white_point == "Custom" else JXL_WHITE_POINTS.get(c.white_point or "")
    if white is None:
        raise _cannot(what)
    if c.colour_space == "Grey":
        primaries = SRGB_PRIMARIES          # grey has none: any primaries with its white point give the same grey
    else:
        primaries = c.primaries_xy if c.primaries == "Custom" else JXL_PRIMARIES.get(c.primaries or "")
        if primaries is None:
            raise _cannot(what)
    if c.gamma is not None:
        if c.gamma <= 0:
            raise _cannot(what)
        return Declaration("enumeration", what, primaries=primaries, white=white, curve="gamma", gamma=1 / c.gamma)
    curve = JXL_TRANSFERS.get(c.transfer or "")
    if curve is None:
        raise _cannot(what)
    return Declaration("enumeration", what, primaries=primaries, white=white, curve=curve,
                       gamma=2.6 if curve == "gamma" else None)


def _jpeg(data: bytes) -> Declaration:
    """A JPEG with no ICC profile: Adobe RGB if its EXIF interoperability index is R03 (DCF 2.0, D8)."""
    from PIL import Image
    try:
        with Image.open(io.BytesIO(data), formats=("JPEG",)) as im:
            index = im.getexif().get_ifd(0xA005).get(0x0001)
    except Exception:   # noqa: BLE001 -- broken EXIF declares nothing; decode itself reports a broken file
        return NONE
    if isinstance(index, bytes):
        index = index.decode("latin-1")
    if isinstance(index, str) and index.strip("\0 ") == "R03":
        return Declaration("r03", "EXIF interoperability index R03 (Adobe RGB)", primaries=ADOBE_RGB_PRIMARIES,
                           white=D65, curve="gamma", gamma=ADOBE_RGB_GAMMA)
    return NONE


def bmp_profile(data: bytes) -> bytes | None:
    """A BMP's embedded ICC profile: a V5 info header with bV5CSType PROFILE_EMBEDDED ('MBED'). Its calibrated
    endpoints and gamma (LCS_CALIBRATED_RGB) are not read (D7)."""
    if len(data) < 14 + 124 or int.from_bytes(data[14:18], "little") != 124:
        return None
    info = 14
    if data[info + 56:info + 60] != b"DEBM":        # 'MBED' as a little-endian DWORD
        return None
    offset, size = struct.unpack("<II", data[info + 112:info + 120])
    start = info + offset
    return data[start:start + size] if size and start + size <= len(data) else None


# ----- the builder -----

@dataclass(frozen=True)
class SourceColour:
    """What converting a source takes: `linearize`, if not None, undoes the curve in place on float32 samples scaled so
    1.0 is white; then `profile` (None: LittleCMS's own sRGB) is the source's ICC profile. Every profile the builder
    makes comes with a `linearize` (a no-op for a linear curve), so `linearize is None` means one of the profiles
    canonicalization used before ML-29: the front's integer path is for those alone (D26)."""
    profile: bytes | None
    linearize: Callable[[np.ndarray], None] | None = None


def conversion(d: Declaration) -> SourceColour:
    """D20: a declaration's source profile, and the curve undone first. Refused for PQ and HLG, whose rendering into
    the standard range waits for the tone-mapping operator (P4, P5)."""
    if d.kind == "none":
        return SourceColour(None)
    if d.kind == "icc":
        return SourceColour(d.icc)
    if d.is_hdr:
        raise Refused(f"an HDR photo ({d.curve}, {d.source}), which canonicalization does not read yet: export it "
                      "in a standard-range colour space such as sRGB or Display P3")
    if d.curve not in CURVES:
        raise ValueError(f"no curve {d.curve!r}")
    if d.white == D65 and d.primaries == SRGB_PRIMARIES and d.curve == "sRGB":
        return SourceColour(None)                       # exactly what an untagged photo is read as
    if d.white == D65 and d.primaries == REC2020_PRIMARIES and d.curve == "BT709":
        return SourceColour(REC2020_PROFILE)            # the canonical photo's own: a canonical JPEG XL's
    undo = _power(d.gamma) if d.curve == "gamma" else CURVE_DECODERS[d.curve]
    return SourceColour(linear_profile(d.primaries, d.white), undo)


@lru_cache(maxsize=32)
def linear_profile(primaries: Primaries, white: XY) -> bytes:
    """A linear-light RGB ICC profile with these primaries and white point (LittleCMS adapts it to D50, Bradford)."""
    return ic.cms_profile("rgb", whitepoint=white, primaries=primaries, gamma=1.0)


# Curves undone (encoded -> linear, 1.0 white), in place on float32. Values below 0 or above 1 keep their sign and
# their curve's extension; to_rec2020 clips after the conversion.

def _srgb(a: np.ndarray) -> None:            # IEC 61966-2-1, 4.3
    s = np.sign(a)
    np.abs(a, out=a)
    low = a <= 0.04045
    lin = np.power((a + 0.055) / 1.055, 2.4, dtype=np.float32)
    np.divide(a, 12.92, out=a)
    np.copyto(a, lin, where=~low)
    a *= s


def _bt709(a: np.ndarray) -> None:           # ITU-R BT.709-6, 1.2, inverted
    s = np.sign(a)
    np.abs(a, out=a)
    low = a < 0.081
    lin = np.power((a + 0.099) / 1.099, 1 / 0.45, dtype=np.float32)
    np.divide(a, 4.5, out=a)
    np.copyto(a, lin, where=~low)
    a *= s


def _smpte240m(a: np.ndarray) -> None:       # SMPTE ST 240, inverted (H.273 transfer 7)
    s = np.sign(a)
    np.abs(a, out=a)
    low = a < 4 * 0.0228
    lin = np.power((a + 0.1115) / 1.1115, 1 / 0.45, dtype=np.float32)
    np.divide(a, 4, out=a)
    np.copyto(a, lin, where=~low)
    a *= s


def _dci(a: np.ndarray) -> None:             # SMPTE ST 428-1: E' = (48 L / 52.37) ** (1 / 2.6); 48 cd/m2 is white
    np.maximum(a, 0, out=a)                  # the one curve that does not keep a sign: XYZ-coded values are not below 0
    np.power(a, 2.6, out=a)
    a *= 52.37 / 48


def _linear(a: np.ndarray) -> None:
    """Nothing to undo; still a curve, so the source takes the float path (D26)."""


def _power(gamma: float) -> Callable[[np.ndarray], None]:
    def undo(a: np.ndarray) -> None:
        s = np.sign(a)
        np.abs(a, out=a)
        np.power(a, gamma, out=a)
        a *= s
    return undo


CURVE_DECODERS: dict[str, Callable[[np.ndarray], None]] = {"sRGB": _srgb, "BT709": _bt709, "linear": _linear,
                                                           "SMPTE240M": _smpte240m, "DCI": _dci}

__all__ = ["Declaration", "NONE", "Refused", "SourceColour", "bmp_profile", "conversion", "declare",
           "jxl_declaration", "linear_profile", "primary_bit_depth", "primary_colr",
           "primary_properties", "usable_profile"]
