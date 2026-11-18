"""A photo's gain map, read from its bytes, and applied to its main image's linear light (ML-29 P7; D10, D14).

WHERE TO LOOK FIRST (ML-29 D37, the owner, 2026-11-16): every gain map is read and applied by this module's own
code, not by a library. libultrahdr (in imagecodecs) is only the check on it. So a gain-map photo that comes out
wrong (too bright or too dark, a colour cast, banding, the map turned or stretched wrong) is most likely this file's:
  1. compare with libultrahdr on the same JPEG: `python analysis/ml29/gain_map_check.py --sections 2` (it agreed
     within 0.006-0.014 stops on 60 dataset JPEGs, 2026-11-16) and tests.test_gain_map.Jpeg;
  2. then read, in this order: parse_iso / parse_hdrgm (the numbers), apply (the formula, its weight and offsets),
     _stretched (the map's upsampling), _map_item / _heic_aux / _apple (which image is the map), and canonical's
     _with_gain_map (the map turned as the main image was, the light in linear Rec.2020).
A canonical photo made with a map names it in its Source.decoder ("jpeg+Ultra HDR", "heif8+ISO 21496-1").
The colour space (ML-29 D38): canonical applies every map to linear Rec.2020 light. That is exact for a one-channel
map with one offset on every channel (every dataset photo's), and not for a three-channel map, which ISO 21496-1
applies in the base image's or the alternate's primaries: one of those coming out wrong is D38's case.

Behind canonical.Settings.gain_maps, on in the store's and the front's settings since the owner adopted it (ML-29 P11,
2026-11-18); a map that does not scale the channels alike is left unapplied (D38).

The readers, in D14's order, ISO 21496-1 first:
- ISO 21496-1: the metadata in a HEIF's or AVIF's `tmap` item (whose `dimg` references are the main image, then the
  map), or in a JPEG gain map image's APP2 segment "urn:iso:std:iso:ts:21496:-1" (the main image's own APP2 holds only
  the version);
- Ultra HDR: the gain map JPEG's `hdrgm` XMP (Google's format, version 1.0), attribute or per-channel `rdf:Seq` form;
- Apple: a HEIC's auxiliary image `urn:com:apple:photo:2020:aux:hdrgainmap` with its XMP `HDRGainMapHeadroom`.
A JPEG's gain map is the second image its MPF index lists. An AVIF's map item is decoded by pointing a copy's `pitm`
at it; a HEIC's through pillow-heif's auxiliary images, which libheif turns as it turns the main image.

Applying (ISO 21496-1, as libavif 1.4's avifRGBImageApplyGainMap does it), per channel, at full weight:
    log2 gain = lerp(gain_min, gain_max, map ** (1 / gamma))
    alternate = (base + base_offset) * 2 ** (log2 gain * weight) - alternate_offset
weight 1 for a map that leads up from a standard-range main image, -1 for one that leads down from an HDR main image
(its alternate headroom below its base's). Apple's: alternate = base * (1 + (headroom - 1) * sRGB-decoded map). Light
is relative to standard-range white, 1.0 at 203 cd/m2 (BT.2408), as the tone map (D33) reads it.

libultrahdr (in imagecodecs) applies Ultra HDR and ISO maps to a JPEG itself; it is the check on this arithmetic
(analysis/ml29/gain_map_check.py), not the reader (D37): it refuses the per-channel hdrgm form, reads no HEIF or AVIF,
stops at 8192 x 8192, and imagecodecs exposes no libavif gain-map call.
"""
from __future__ import annotations

import io
import re
import struct
from dataclasses import dataclass

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv.colour_declaration import CURVE_DECODERS, _boxes, primary_bit_depth

ISO_URN = b"urn:iso:std:iso:ts:21496:-1\0"
XMP_URN = b"http://ns.adobe.com/xap/1.0/\0"
APPLE_AUX = "urn:com:apple:photo:2020:aux:hdrgainmap"


@dataclass(frozen=True)
class Metadata:
    """What a gain map means. Per map channel (1 or 3): gain_min and gain_max in log2, gamma, the two offsets in
    linear light. The headrooms are log2: how far above white each rendition reaches. Apple's formula has only its
    linear headroom (apple_headroom), and the other fields at their identity."""
    gain_min: tuple[float, ...]
    gain_max: tuple[float, ...]
    gamma: tuple[float, ...]
    base_offset: tuple[float, ...]
    alternate_offset: tuple[float, ...]
    base_headroom: float
    alternate_headroom: float
    use_base_colour_space: bool = True
    apple_headroom: float | None = None

    @property
    def leads_down(self) -> bool:
        """The main image is the HDR rendition and the map leads down to standard range."""
        return self.alternate_headroom < self.base_headroom


@dataclass(frozen=True)
class GainMap:
    """A file's gain map: which reader found it, its metadata, and its pixels, float32 (h, w, 1 or 3) in 0-1, in the
    orientation the main image's decoder gives (a JPEG's and an AVIF's as stored, a HEIC's turned by libheif)."""
    form: str
    metadata: Metadata
    pixels: np.ndarray

    @property
    def alike(self) -> bool:
        """The map scales red, green and blue alike: one channel, and one value in each per-channel field. Only such a
        map's result is the same in every set of primaries with D65's white, so only it may be applied in linear
        Rec.2020 (ML-29 D38)."""
        m = self.metadata
        return self.pixels.shape[2] == 1 and all(
            len(set(v)) == 1 for v in (m.gain_min, m.gain_max, m.gamma, m.base_offset, m.alternate_offset))


def read(data: bytes) -> GainMap | None:
    """The file's gain map by D14's order, or None: no map, or one no reader here understands."""
    return next(iter(read_all(data)), None)


def read_all(data: bytes) -> list[GainMap]:
    """Every form of gain map the file carries that a reader here understands, in D14's order (ISO 21496-1 first);
    a file with two is how the ISO reader is checked against the vendor's (R5)."""
    if data[:3] == b"\xff\xd8\xff":
        return _jpeg(data)
    if data[4:8] == b"ftyp":
        return _heif(data)
    return []


def apply(base: np.ndarray, gain_map: GainMap) -> np.ndarray:
    """The other rendition, float32 (h, w, 3), from `base`: the main image's linear light (h, w, 3), relative to
    standard-range white, in the colour space the map's maths uses (the main image's, when
    metadata.use_base_colour_space) and in the map's orientation. The map is stretched to the main image bilinearly."""
    m = gain_map.metadata
    h, w = base.shape[:2]
    stretched = _stretched(gain_map.pixels, h, w)
    if m.apple_headroom is not None:
        CURVE_DECODERS["sRGB"](stretched)
        return (base * (1 + (m.apple_headroom - 1) * stretched)).astype(np.float32)
    weight = -1.0 if m.leads_down else 1.0
    if m.base_headroom == m.alternate_headroom:    # libavif's rule: the spec leaves it undefined, the map is not used
        weight = 0.0
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        k = c if len(m.gain_min) == 3 else 0
        g = stretched[..., k if stretched.shape[2] == 3 else 0]
        log2_gain = m.gain_min[k] + (m.gain_max[k] - m.gain_min[k]) * g ** (1 / m.gamma[k])
        out[..., c] = (base[..., c] + m.base_offset[k]) * np.exp2(log2_gain * weight) - m.alternate_offset[k]
    return out


# ----- ISO 21496-1 metadata -----

def parse_iso(payload: bytes) -> Metadata | None:
    """ISO 21496-1's GainMapMetadata from its minimum_version field on, or None: a minimum_version other than 0 (a
    reader must then ignore the map), or the version alone (a JPEG main image's APP2)."""
    if len(payload) < 5 or struct.unpack(">H", payload[:2])[0] != 0:
        return None
    flags, p = payload[4], 5
    channels = 3 if flags & 0x80 else 1
    if flags & 0x08:                              # use_common_denominator
        den, base_n, alt_n = struct.unpack(">III", payload[p:p + 12])
        p += 12
        rows = [struct.unpack(">iiIii", payload[p + 20 * c:p + 20 * c + 20]) for c in range(channels)]
        if den == 0:
            return None
        base, alt = base_n / den, alt_n / den
        values = [[v / den for v in row] for row in rows]
    else:
        base_n, base_d, alt_n, alt_d = struct.unpack(">IIII", payload[p:p + 16])
        p += 16
        rows = [struct.unpack(">iIiIIIiIiI", payload[p + 40 * c:p + 40 * c + 40]) for c in range(channels)]
        if 0 in (base_d, alt_d) or any(row[i] == 0 for row in rows for i in (1, 3, 5, 7, 9)):
            return None
        base, alt = base_n / base_d, alt_n / alt_d
        values = [[row[i] / row[i + 1] for i in range(0, 10, 2)] for row in rows]
    if any(v[2] <= 0 for v in values):            # a gamma of 0 is invalid (libavif's avifGainMapValidateMetadata)
        return None
    gain_min, gain_max, gamma, base_offset, alt_offset = (tuple(v[i] for v in values) for i in range(5))
    return Metadata(gain_min, gain_max, gamma, base_offset, alt_offset, base, alt, bool(flags & 0x40))


# ----- Ultra HDR's hdrgm XMP -----

def parse_hdrgm(xmp: bytes) -> Metadata | None:
    """Ultra HDR's gain map metadata (hdrgm, version 1.0) from a gain map image's XMP, or None without GainMapMax and
    HDRCapacityMax. Defaults are the format's: GainMapMin 0, Gamma 1, the offsets 1/64, HDRCapacityMin 0."""
    found: dict[str, list[bytes]] = {}
    for name, value in re.findall(rb'hdrgm:(\w+)="([^"]*)"', xmp):
        found[name.decode()] = [value]
    for name, body in re.findall(rb"<hdrgm:(\w+)>(.*?)</hdrgm:\1>", xmp, re.S):
        found[name.decode()] = re.findall(rb"<rdf:li>\s*([^<\s]+)\s*</rdf:li>", body) or [body.strip()]
    if found.get("Version", [b"1.0"])[0] != b"1.0" or "GainMapMax" not in found or "HDRCapacityMax" not in found:
        return None
    try:
        number = {k: tuple(float(x) for x in v) for k, v in found.items() if k not in ("Version", "BaseRenditionIsHDR")}
    except ValueError:
        return None
    channels = max(len(v) for v in number.values())
    if channels not in (1, 3) or any(len(v) not in (1, channels) for v in number.values()):
        return None

    def each(name: str, default: float) -> tuple[float, ...]:
        v = number.get(name, (default,))
        return v * channels if len(v) == 1 else v

    hdr_base = found.get("BaseRenditionIsHDR", [b"False"])[0].lower() == b"true"
    sdr_offset, hdr_offset = each("OffsetSDR", 1 / 64), each("OffsetHDR", 1 / 64)
    low, high = number.get("HDRCapacityMin", (0.0,))[0], number["HDRCapacityMax"][0]
    if any(g <= 0 for g in each("Gamma", 1.0)):
        return None
    return Metadata(each("GainMapMin", 0.0), each("GainMapMax", 0.0), each("Gamma", 1.0),
                    hdr_offset if hdr_base else sdr_offset, sdr_offset if hdr_base else hdr_offset,
                    high if hdr_base else low, low if hdr_base else high)


# ----- JPEG: the MPF's second image -----

def _jpeg_segments(d: bytes, start: int = 0):
    """A JPEG's marker segments before its scan, as (marker, body, where the body starts)."""
    p = start + 2
    while p + 4 <= len(d) and d[p] == 0xFF:
        marker = d[p + 1]
        if marker == 0xDA:
            return
        n = struct.unpack(">H", d[p + 2:p + 4])[0]
        yield marker, d[p + 4:p + 2 + n], p + 4
        p += 2 + n


def _mpf_second_image(d: bytes) -> bytes | None:
    """The second image an MPF index lists (CIPA DC-007), its offset counted from the MPF's TIFF header."""
    for marker, body, at in _jpeg_segments(d):
        if marker != 0xE2 or body[:4] != b"MPF\0":
            continue
        tiff, start = body[4:], at + 4
        end = "<" if tiff[:2] == b"II" else ">"
        ifd = struct.unpack(end + "I", tiff[4:8])[0]
        for i in range(struct.unpack(end + "H", tiff[ifd:ifd + 2])[0]):
            tag, _, count, value = struct.unpack(end + "HHII", tiff[ifd + 2 + 12 * i:ifd + 14 + 12 * i])
            if tag == 0xB002 and count >= 32:
                _, size, offset, _, _ = struct.unpack(end + "IIIHH", tiff[value + 16:value + 32])
                second = d[start + offset:start + offset + size]
                return second if second[:3] == b"\xff\xd8\xff" else None
    return None


def _jpeg(d: bytes) -> list[GainMap]:
    second = _mpf_second_image(d)
    if second is None:
        return []
    iso = hdrgm = None
    for marker, body, _ in _jpeg_segments(second):
        if marker == 0xE2 and body.startswith(ISO_URN):
            iso = parse_iso(body[len(ISO_URN):])
        elif marker == 0xE1 and body.startswith(XMP_URN):
            hdrgm = parse_hdrgm(body)
    if iso is None and hdrgm is None:
        return []
    pixels = _unit(ic.jpeg8_decode(second), 255.0)
    return [GainMap(form, m, pixels) for form, m in (("ISO 21496-1", iso), ("Ultra HDR", hdrgm)) if m is not None]


# ----- HEIF and AVIF: the tmap item; Apple's auxiliary image -----

def _items(d: bytes) -> dict:
    """A HEIF's meta box: the primary item, item types, dimg references, item locations and the idat's start."""
    meta = next(((a, b) for t, a, b in _boxes(d, 0, len(d)) if t == b"meta"), None)
    out = {"primary": None, "types": {}, "dimg": {}, "locations": {}, "idat": None}
    if meta is None:
        return out
    for t, a, b in _boxes(d, meta[0] + 4, meta[1]):              # meta is a full box: version and flags first
        if t == b"pitm":
            out["primary"] = int.from_bytes(d[a + 4:a + (6 if d[a] == 0 else 8)], "big")
            out["pitm_at"] = (a + 4, 2 if d[a] == 0 else 4)
        elif t == b"iinf":
            for t2, a2, b2 in _boxes(d, a + 4 + (2 if d[a] == 0 else 4), b):
                if t2 == b"infe" and d[a2] >= 2:
                    wide = 2 if d[a2] == 2 else 4
                    item = int.from_bytes(d[a2 + 4:a2 + 4 + wide], "big")
                    out["types"][item] = d[a2 + 6 + wide:a2 + 10 + wide]
        elif t == b"iref":
            wide = 2 if d[a] == 0 else 4
            for t2, a2, _ in _boxes(d, a + 4, b):
                if t2 == b"dimg":
                    n = struct.unpack(">H", d[a2 + wide:a2 + wide + 2])[0]
                    to = a2 + wide + 2
                    out["dimg"][int.from_bytes(d[a2:a2 + wide], "big")] = [
                        int.from_bytes(d[to + i * wide:to + (i + 1) * wide], "big") for i in range(n)]
        elif t == b"iloc":
            out["locations"] = _iloc(d, a)
        elif t == b"idat":
            out["idat"] = a
    return out


def _iloc(d: bytes, a: int) -> dict[int, tuple[int, list[tuple[int, int]]]]:
    """ISO/IEC 14496-12's iloc: item -> (construction method, [(offset, length)])."""
    version, p = d[a], a + 4
    offset_size, length_size, base_size = d[p] >> 4, d[p] & 15, d[p + 1] >> 4
    index_size = d[p + 1] & 15 if version in (1, 2) else 0
    wide = 2 if version < 2 else 4
    p += 2
    count = int.from_bytes(d[p:p + wide], "big")
    p += wide
    out = {}
    for _ in range(count):
        item = int.from_bytes(d[p:p + wide], "big")
        p += wide
        method = 0
        if version in (1, 2):
            method = struct.unpack(">H", d[p:p + 2])[0] & 15
            p += 2
        p += 2                                                     # data_reference_index
        base = int.from_bytes(d[p:p + base_size], "big")
        p += base_size
        extents = []
        count_extents = struct.unpack(">H", d[p:p + 2])[0]
        p += 2
        for _ in range(count_extents):
            p += index_size
            offset = int.from_bytes(d[p:p + offset_size], "big")
            length = int.from_bytes(d[p + offset_size:p + offset_size + length_size], "big")
            p += offset_size + length_size
            extents.append((base + offset, length))
        out[item] = (method, extents)
    return out


def _item_data(d: bytes, items: dict, item: int) -> bytes:
    method, extents = items["locations"].get(item, (0, []))
    if method == 1 and items["idat"] is not None:
        return b"".join(d[items["idat"] + o:items["idat"] + o + n] for o, n in extents)
    return b"".join(d[o:o + n] for o, n in extents) if method == 0 else b""


def _heif(d: bytes) -> list[GainMap]:
    items = _items(d)
    found = []
    for item, kind in items["types"].items():
        refs = items["dimg"].get(item, [])
        if kind == b"tmap" and len(refs) == 2 and refs[0] == items["primary"]:
            payload = _item_data(d, items, item)
            metadata = parse_iso(payload[1:]) if payload[:1] == b"\0" else None
            pixels = _map_item(d, items, refs[1]) if metadata else None
            if pixels is not None:
                found.append(GainMap("ISO 21496-1", metadata, pixels))
                break
    apple = _apple(d)
    return found + ([apple] if apple else [])


def _map_item(d: bytes, items: dict, item: int) -> np.ndarray | None:
    """A HEIF's or AVIF's map item decoded: an AVIF's through a copy whose pitm names it, a HEIC's through
    pillow-heif's auxiliary images (libheif will not take a pitm that names one)."""
    if b"avif" in d[8:32] or b"avis" in d[8:32]:
        if "pitm_at" not in items:                 # no primary item to repoint: a broken file
            return None
        at, wide = items["pitm_at"]
        copy = d[:at] + item.to_bytes(wide, "big") + d[at + wide:]
        pixels = ic.avif_decode(copy)
        depth = 8 if pixels.dtype == np.uint8 else primary_bit_depth(copy)
        return None if depth is None else _unit(pixels, float(2**depth - 1))
    return _heic_aux(d, lambda ids: item in ids)


def _heic_aux(d: bytes, wanted, kind: str = "") -> np.ndarray | None:
    import pillow_heif
    heif = pillow_heif.open_heif(io.BytesIO(d), convert_hdr_to_8bit=False)
    for urn, ids in heif.info.get("aux", {}).items():
        if (not kind or urn == kind) and wanted(ids):
            image = heif.get_aux_image(next(i for i in ids if wanted([i])))
            pixels = np.asarray(image)
            return _unit(pixels, 255.0) if pixels.dtype == np.uint8 else None
    return None


HEADROOM = re.compile(rb'HDRGainMapHeadroom(?:="|>)\s*([0-9.]+)')


def _apple(d: bytes) -> GainMap | None:
    """Apple's auxiliary gain map with its headroom (Apple's "Applying Apple HDR effect to your photos")."""
    found = HEADROOM.search(d)
    if found is None or float(found.group(1)) <= 1:
        return None
    pixels = _heic_aux(d, lambda ids: bool(ids), APPLE_AUX)
    if pixels is None:
        return None
    one = (0.0,)
    return GainMap("Apple", Metadata(one, one, (1.0,), one, one, 0.0, float(np.log2(float(found.group(1)))),
                                     apple_headroom=float(found.group(1))), pixels)


# ----- the map's pixels -----

def _unit(pixels: np.ndarray, white: float) -> np.ndarray:
    a = pixels.astype(np.float32) / white
    return a[..., None] if a.ndim == 2 else a[..., :3] if a.shape[2] >= 3 else a[..., :1]


def _stretched(pixels: np.ndarray, h: int, w: int) -> np.ndarray:
    """The map at (h, w), bilinear with half-pixel centres (Pillow's), each channel on its own."""
    if pixels.shape[:2] == (h, w):
        return pixels.copy()
    return np.dstack([np.asarray(Image.fromarray(np.ascontiguousarray(pixels[..., c])).resize((w, h),
                                                                                             Image.BILINEAR))
                      for c in range(pixels.shape[2])])
