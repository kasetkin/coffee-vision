"""Canonicalization (GLOSSARY.md; ADR 0033; ticket ML-19): any photo the service or training reads becomes its
canonical photo, and the models read only canonical photos.

    canonicalize(data, name)  -> bytes        an original's bytes to its canonical photo (lossy JPEG XL)
    canonicalize_file(path)   -> bytes
    load_canonical(data)      -> np.ndarray   a canonical photo to 8-bit sRGB, H x W x 3 uint8 (refuses anything else)
    is_canonical(data)        -> bool         and canonical_problems(data) -> list[str], why not

The canonical photo (D1-D7, D13, D16, D20, D21, D27, D28, D36, D50):
- the original's main image only, at its decoder's own bit depth (RAW developed by rawpy at 16 bits into linear
  Rec.2020 with the camera's white balance);
- turned upright: HEIC and RAW as their decoders turn them, every other format by its EXIF orientation 1-8, any other
  value as stored;
- over 60,000,000 pixels: area-averaged (OpenCV INTER_AREA) to at most 50,000,000; at or under 60 MP: as it is;
- transparent pixels over the segmenter's fill grey;
- from its colour declaration (ML-29, ADR 0040, coffeecv.colour_declaration): its ICC profile, else its coded tag or
  JPEG XL header enumeration, else a JPEG's R03 (Adobe RGB), else sRGB; a declaration that cannot be converted, PQ
  and HLG included until a tone-mapping operator is chosen, is refused. Through LittleCMS on float32, relative
  colorimetric, into Rec.2020 (D65, BT.709 transfer; the profile libjxl itself generates,
  coffeecv/data/rec2020_bt709.icc), clipped after the conversion (a float JPEG XL's values outside 0-1 included);
- stored as lossy JPEG XL, quality 95, effort 7, 16 bits, with D7's encoding as an enumeration and nothing else: no
  EXIF, no XMP, no ICC, no orientation, no alpha, bare codestream.
Read back for the models: decoded at 16 bits, LittleCMS without its lossy optimization to 8-bit sRGB, clipped; on
every CPU the process may use (ML-21), with the same pixels as on one.

Those are the store's settings (STORE_SETTINGS), every function's default: the store and training use nothing else.
The front makes its own, cheaper canonical photo (FRONT_SETTINGS, OPS-22 D1, ADR 0036): kept up to 18,000,000 pixels
and over that area-averaged to at most 15,000,000; LittleCMS's optimized integer transform from the source's own
samples (a source with transparency, or whose profile is built from a declaration, keeps the float path: OPS-22 D4,
ML-29 D26); quality 90, effort 3. Everything else, the format the
inference host checks included, is the same, and canonicalize() and its pass-through stay the store's.

No torch: the front imports this module (D35). Every library that decodes, converts or encodes a photo is held at one
version across the three locks (coffeecv.lock_check), and a version change is a re-canonicalization. libjxl's
output depends on the CPU (D51): canonical photos are made only on the store's machine and on the front.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv import colour_declaration as cd
from coffeecv import tone_map as tm
from coffeecv.jxl_encode import encode_rec2020
from coffeecv.jxl_header import Header, HeaderError, read_header

# D44: the one library behind both colour conversions and the JPEG XL coding. A new version moves LittleCMS and libjxl:
# it is a new canonical store and a retrain (ADR 0033), so this module refuses any other. tests/test_canonical.py holds
# it to uv.lock's; coffeecv.lock_check holds the three locks to one version.
IMAGECODECS = "2026.10.10"
if ic.__version__ != IMAGECODECS:
    raise ImportError(f"coffeecv.canonical is written for imagecodecs {IMAGECODECS}, found {ic.__version__}")
PROFILE_PATH = Path(__file__).resolve().parent / "data" / "rec2020_bt709.icc"
PROFILE_SHA256 = "877c5b36808954f322af187b56c3b3ebfd1cc8bad569f27f67f1a0c450d11b9c"   # D46
REC2020 = PROFILE_PATH.read_bytes()
if hashlib.sha256(REC2020).hexdigest() != PROFILE_SHA256:
    raise ImportError(f"{PROFILE_PATH} is not the committed Rec.2020 profile (ML-19 D46)")
SRGB = ic.cms_profile("srgb")            # LittleCMS's own sRGB: the read-back's target, and an untagged photo's source
# RAW: rawpy's linear Rec.2020 (gamma 1), D65; LittleCMS then applies the BT.709 curve
LINEAR_REC2020 = ic.cms_profile("rgb", whitepoint=(0.3127, 0.3290),
                                primaries=(0.708, 0.292, 0.170, 0.797, 0.131, 0.046), gamma=1.0)
RELATIVE = ic.CMS.INTENT.RELATIVE_COLORIMETRIC
NOOPTIMIZE = ic.CMS.FLAGS.NOOPTIMIZE

KEEP_MAX_PIXELS = 60_000_000             # D27: at or under this, kept at its size
CAP_PIXELS = 50_000_000                  # D27: over KEEP_MAX_PIXELS, area-averaged to at most this
QUALITY = 95                             # D7: distance 0.55
EFFORT = 7                               # D50
THREADS = 2                              # the front's two vCPU; libjxl's output does not depend on it (P1)
FILL_SRGB = (124, 116, 104)              # D13: coffeecv.segment_beans.SegmentConfig.fill_rgb
STRIP_ROWS = 256                         # rows converted at a time (D38)


@dataclass(frozen=True)
class Settings:
    """How a canonical photo is made from a decoded original (OPS-22 D1): the colour path, the encoder's effort and
    quality, and the size thresholds (at or under `keep_max_pixels` kept; over, area-averaged to at most
    `cap_pixels`)."""
    name: str
    colour: str            # "float": LittleCMS on float32 (D31); "integer": its optimized transform on the source's own
    #                        samples (OPS-22 D4)
    effort: int
    quality: int
    keep_max_pixels: int
    cap_pixels: int

    def __post_init__(self):
        if self.colour not in ("float", "integer"):
            raise ValueError(f"colour is float or integer, not {self.colour!r}")

    @property
    def label(self) -> str:
        """What the front's log line names it by: "front: integer e3 q90 18/15 MP"."""
        return (f"{self.name}: {self.colour} e{self.effort} q{self.quality} "
                f"{self.keep_max_pixels / 1e6:g}/{self.cap_pixels / 1e6:g} MP")


# The store's and training's (ADR 0033), every function's default; the front's own (OPS-22 D1, ADR 0036).
STORE_SETTINGS = Settings("store", "float", EFFORT, QUALITY, KEEP_MAX_PIXELS, CAP_PIXELS)
FRONT_SETTINGS = Settings("front", "integer", 3, 90, 18_000_000, 15_000_000)


# ----- the accepted formats (ADR 0034; ticket OPS-18 D11, D13-D18) -----

@dataclass(frozen=True)
class AcceptedFormat:
    """One row of coffeecv/data/accepted_formats.csv."""
    id: str
    label: str           # its name on the page's chip, in llms.txt, openapi.json and the refusal message
    group: str           # the name its rows are gathered under (camera RAW), or ""
    extensions: tuple[str, ...]
    mime_types: tuple[str, ...]
    decoder: str         # what decode opens it with: pillow.<FORMAT> is Pillow with that one plugin

    @property
    def is_raw(self) -> bool:
        return self.decoder == "rawpy"

    @property
    def turns_itself(self) -> bool:
        """Its decoder turns it upright (D28); any other is turned by its EXIF orientation."""
        return self.decoder in ("rawpy", "pillow_heif", "imagecodecs.jpegxl")

    @property
    def pillow_plugin(self) -> str:
        """The one Pillow plugin that may open it, for its pixels or, when imagecodecs decodes it, for the ICC profile
        and EXIF imagecodecs does not read (D4). KeyError for a format Pillow never opens."""
        if self.decoder.startswith("pillow."):
            return self.decoder.removeprefix("pillow.")
        return {"imagecodecs.jpeg8": "JPEG", "imagecodecs.png": "PNG", "imagecodecs.avif": "AVIF"}[self.decoder]


FORMATS_PATH = Path(__file__).resolve().parent / "data" / "accepted_formats.csv"


def _read_formats(path: Path) -> dict[str, AcceptedFormat]:
    lines = [line for line in path.read_text().splitlines() if not line.startswith("#")]
    return {r["id"]: AcceptedFormat(r["id"], r["label"], r["group"], tuple(r["extensions"].split()),
                                    tuple(r["mime_types"].split()), r["decoder"])
            for r in csv.DictReader(lines)}


ACCEPTED_FORMATS = _read_formats(FORMATS_PATH)   # in the file's order, which the page and the texts keep
_RAW_BY_EXTENSION = {e: f.id for f in ACCEPTED_FORMATS.values() if f.is_raw for e in f.extensions}
RAW_EXTENSIONS = frozenset(_RAW_BY_EXTENSION)
PHOTO_EXTENSIONS = frozenset(e for f in ACCEPTED_FORMATS.values() for e in f.extensions)
HEAD_BYTES = 64   # what recognise reads: every rule below, an ftyp box's compatible brands included (a phone's HEIC: 44)

# HEIF (HEIC included) and AVIF image brands, ISO/IEC 23008-12 and the AVIF spec; mif1/msf1 are HEIF's generic ones.
HEIF_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"hevm", b"hevs", b"mif1", b"msf1"})
AVIF_BRANDS = frozenset({b"avif", b"avis"})
JXL_SIGNATURES = (b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")   # bare codestream, ISO BMFF container
TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*")   # little- and big-endian: most RAW families are TIFF-based
# A BMP's info header sizes: BITMAPCOREHEADER, BITMAPINFOHEADER, its V2 and V3 extensions, OS/2 2.x, V4, V5 (R4)
BMP_INFO_SIZES = frozenset({12, 40, 52, 56, 64, 108, 124})


def _ftyp_brands(head: bytes) -> tuple[bytes, set[bytes]]:
    """The major brand and the compatible brands of an ISO-BMFF file's ftyp box, as far as `head` holds them."""
    if len(head) < 12 or head[4:8] != b"ftyp":
        return b"", set()
    end = min(int.from_bytes(head[:4], "big"), len(head))
    return head[8:12], {head[i:i + 4] for i in range(16, end - 3, 4)}


def _all_brands(head: bytes) -> set[bytes]:
    major, compatible = _ftyp_brands(head)
    return {major, *compatible} - {b""}


def _tiff_based(head: bytes) -> bool:
    return head.startswith(TIFF_SIGNATURES)


def _image_brand(head: bytes) -> str | None:
    """D18: "avif" or "heif" by the major brand if it is one of theirs, else by a compatible brand, AVIF first."""
    major, compatible = _ftyp_brands(head)
    for brands in ({major}, compatible):
        if brands & AVIF_BRANDS:
            return "avif"
        if brands & HEIF_BRANDS:
            return "heif"
    return None


# Each row's rule, by id (D14): does this file's start (HEAD_BYTES of it) look like that format? A RAW row's rule is
# its family's signature, as LibRaw identifies it: TIFF-based for most, and the four that are not.
RULES = {
    "jpeg": lambda h: h.startswith(b"\xff\xd8\xff"),
    "png": lambda h: h.startswith(b"\x89PNG\r\n\x1a\n"),
    "webp": lambda h: h[:4] == b"RIFF" and h[8:12] == b"WEBP",
    "heif": lambda h: _image_brand(h) == "heif",
    "avif": lambda h: _image_brand(h) == "avif",
    "jxl": lambda h: h.startswith(JXL_SIGNATURES),
    "gif": lambda h: h.startswith((b"GIF87a", b"GIF89a")),
    "bmp": lambda h: h[:2] == b"BM" and len(h) >= 18 and int.from_bytes(h[14:18], "little") in BMP_INFO_SIZES,
    "dng": _tiff_based, "cr2": _tiff_based, "nef": _tiff_based, "arw": _tiff_based, "pef": _tiff_based,
    "srw": _tiff_based,
    "orf": lambda h: h.startswith((b"IIRO", b"IIRS", b"MMOR")),
    "rw2": lambda h: h.startswith(b"IIU\x00"),
    "raf": lambda h: h.startswith(b"FUJIFILMCCD-RAW"),
    "cr3": lambda h: b"crx " in _all_brands(h),
}


def recognise(head: bytes, name: str = "") -> str | None:
    """The id of the accepted format a file is in, from its first HEAD_BYTES bytes and its name; None if it is in none
    (D17). A name with a RAW extension is that RAW format's or none, since rawpy reads whatever it is given; any other
    file is told by its content alone, whatever its name."""
    head = head[:HEAD_BYTES]
    raw = _RAW_BY_EXTENSION.get(Path(name).suffix.lower())
    if raw is not None:
        return raw if RULES[raw](head) else None
    return next((i for i, f in ACCEPTED_FORMATS.items() if not f.is_raw and RULES[i](head)), None)


# What the public texts say is read (llms.txt, openapi.json; ML-29 P2), held to them by tests/test_accepted_formats_named.py
COLOUR_TEXT = ("Colour is read from the file's ICC profile (a PNG's cICP colour tag comes first), else its colour tag "
               "(HEIC/HEIF and AVIF nclx) or JPEG XL header, else a JPEG's Adobe RGB EXIF tag, else as sRGB; an HDR "
               "photo (PQ or HLG) is tone-mapped into standard range; a JPEG XL with an embedded ICC profile is "
               "refused.")


class Unreadable(OSError):
    """A file that is not a photo canonicalization can read. An OSError, like every other decode failure the service
    and training already handle (photo_io.load_rgb_image's RAW errors, /analyze's 400)."""


class TooLarge(Unreadable):
    """A photo over the pixel ceiling it was given (D38), or over Pillow's decompression-bomb limit, told from its
    header before any pixel is decoded."""


class NotCanonical(ValueError):
    """A file that is not a canonical photo, given where only canonical photos are read (D12, D35)."""


# ----- decoding an original -----

class Source:
    """An original's main image as decoded: `pixels` (H x W x C, C 1-4, or H x W; uint8, uint16 or a JPEG XL's float32;
    maybe a turned view), the sample value that means 1.0, the profile it is converted from (None: sRGB), the curve
    undone before it (`linearize`, for a profile built from its declaration: colour_declaration.SourceColour), whether
    light above white is tone-mapped (`tone_map`, ML-29 D33), and its decoder's name."""

    def __init__(self, pixels: np.ndarray, full: float, profile: bytes | None, decoder: str,
                 linearize: Callable[[np.ndarray], None] | None = None, tone_map: bool = False):
        self.pixels, self.full, self.profile, self.decoder = pixels, full, profile, decoder
        self.linearize, self.tone_map = linearize, tone_map
        self._peak: float | None = None

    @classmethod
    def declared(cls, pixels: np.ndarray, full: float, declaration: cd.Declaration, decoder: str) -> Source:
        """A source converted as its colour declaration says (ADR 0040); Unreadable if it cannot be."""
        with _refusal():
            colour = cd.conversion(declaration, float_samples=pixels.dtype.kind == "f")
        return cls(pixels, full, colour.profile, decoder, colour.linearize, colour.tone_map)

    def peak(self) -> float | None:
        """D33's mastering peak of the whole picture, relative to white, for a tone-mapped source (measured once:
        libjxl asks for the picture in rectangles, and every one must take the same curve); None for any other."""
        if self.tone_map and self._peak is None:
            self._peak = measure_peak(self.pixels, self.full, self.profile, self.linearize)
        return self._peak


@contextmanager
def _refusal():
    """A colour declaration that cannot be converted (colour_declaration.Refused) is an Unreadable photo, its message
    the visitor's answer."""
    try:
        yield
    except cd.Refused as exc:
        raise Unreadable(str(exc)) from exc


def accepted_format(data: bytes, name: str = "") -> AcceptedFormat:
    """The row of the accepted format `data` is in (recognise); Unreadable if none, so nothing else is decoded."""
    fid = recognise(data[:HEAD_BYTES], name)
    if fid is None:
        raise Unreadable("not a photo in an accepted format")
    return ACCEPTED_FORMATS[fid]


def orient(a: np.ndarray, orientation: int) -> np.ndarray:
    """EXIF orientation 1-8 applied as a view, not a copy; any other value as stored (D28)."""
    t = a.swapaxes(0, 1)
    views = {2: lambda: a[:, ::-1], 3: lambda: a[::-1, ::-1], 4: lambda: a[::-1], 5: lambda: t,
             6: lambda: t[:, ::-1], 7: lambda: t[::-1, ::-1], 8: lambda: t[::-1]}
    return views[orientation]() if orientation in views else a


def capped_size(width: int, height: int, settings: Settings = STORE_SETTINGS) -> tuple[int, int]:
    """D27: the size a photo is stored at. Over `settings.keep_max_pixels` (the store's 60,000,000), the largest size
    of the same shape (each side scaled by one factor, rounded down) of at most `settings.cap_pixels` (50,000,000);
    otherwise its own."""
    if width * height <= settings.keep_max_pixels:
        return width, height
    s = (settings.cap_pixels / (width * height)) ** 0.5
    w, h = max(1, int(width * s)), max(1, int(height * s))
    while w * h > settings.cap_pixels:
        w, h = w - 1, h - 1
    return w, h


def _cap(a: np.ndarray, settings: Settings) -> np.ndarray:
    """D27, D43: over the threshold, area-averaged at the capped size, before turning (the pixel count does not depend
    on it)."""
    h, w = a.shape[:2]
    nw, nh = capped_size(w, h, settings)
    if (nw, nh) == (w, h):
        return a
    import cv2   # only for a photo over the threshold: it costs the front 15 MB to import
    return cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)


def _pillow_metadata(data: bytes, plugin: str) -> tuple[bytes | None, int, str, tuple[int, int]]:
    """Profile, EXIF orientation, mode and size, read by Pillow's `plugin` alone without decoding the pixels. Pillow's
    own decompression-bomb limit (Image.MAX_IMAGE_PIXELS) applies: the dataset tools raise it, the web service keeps
    Pillow's default."""
    try:
        with Image.open(io.BytesIO(data), formats=(plugin,)) as im:
            return im.info.get("icc_profile") or None, im.getexif().get(0x0112, 1), im.mode, im.size
    except Image.DecompressionBombError as exc:
        raise TooLarge(str(exc)) from exc


def _check_size(width: int, height: int, max_pixels: int | None) -> None:
    if max_pixels is not None and width * height > max_pixels:
        raise TooLarge(f"{width} x {height} is over {max_pixels:,} pixels")


def applied_orientation(data: bytes, name: str = "") -> int:
    """The EXIF orientation canonicalization turns this original by (D28): 1 for HEIC, RAW and JPEG XL, whose decoders
    turn them, and for a value outside 1-8."""
    fmt = accepted_format(data, name)
    if fmt.turns_itself:
        return 1
    try:
        orientation = _pillow_metadata(data, fmt.pillow_plugin)[1]
    except (OSError, ValueError, SyntaxError, EOFError) as exc:
        raise Unreadable(f"unreadable photo ({type(exc).__name__})") from exc
    return orientation if 1 <= orientation <= 8 else 1


def decode(data: bytes, name: str = "", max_pixels: int | None = None, settings: Settings = STORE_SETTINGS) -> Source:
    """An original's main image at its decoder's depth, capped by `settings` and upright (D20, D27, D28, D36), opened
    with its accepted format's decoder and no other (ADR 0034): a file in none, or one its decoder cannot read, is
    Unreadable. A photo over `max_pixels` (the front's ceiling, D38) raises TooLarge before its pixels are decoded."""
    fmt = accepted_format(data, name)
    try:
        if fmt.is_raw:
            import rawpy
            try:
                with rawpy.imread(io.BytesIO(data)) as raw:
                    _check_size(raw.sizes.width, raw.sizes.height, max_pixels)
                    rgb = raw.postprocess(output_bps=16, use_camera_wb=True, output_color=rawpy.ColorSpace.Rec2020,
                                          gamma=(1, 1))
            except rawpy.LibRawError as exc:   # not an OSError; LibRaw's message is a bytes repr
                raise Unreadable(f"unreadable RAW file ({type(exc).__name__})") from exc
            return Source(_cap(rgb, settings), 65535.0, LINEAR_REC2020, "rawpy16")
        if fmt.decoder == "pillow_heif":
            import pillow_heif
            heif = pillow_heif.open_heif(io.BytesIO(data), convert_hdr_to_8bit=False)
            profile, depth = heif.info.get("icc_profile") or None, heif.info.get("bit_depth", 8)
            _check_size(*heif.size, max_pixels)
            declaration = _declare(data, fmt.id, profile)
            rgb = np.asarray(heif)
            del heif
            # pillow-heif shifts a 10- or 12-bit sample into the top of 16 bits: 10-bit white is 1023 << 6
            full = float((2**depth - 1) << (16 - depth)) if rgb.dtype == np.uint16 else 255.0
            return Source.declared(_cap(rgb, settings), full, declaration, f"heif{depth}")
        if fmt.decoder == "imagecodecs.jpegxl":
            return _decode_jxl(data, max_pixels, settings)
        plugin = fmt.pillow_plugin
        profile, orientation, mode, size = _pillow_metadata(data, plugin)
        _check_size(*size, max_pixels)
        if mode == "CMYK":
            raise Unreadable("a CMYK photo")
        declaration = _declare(data, fmt.id, profile)
        full = None
        if fmt.decoder == "imagecodecs.jpeg8":
            rgb = ic.jpeg8_decode(data)          # the same pixels as Pillow's libjpeg-turbo decode (P1)
        elif fmt.decoder == "imagecodecs.png":
            rgb = ic.png_decode(data)            # keeps 16 bits, which Pillow would cut to 8
        elif fmt.decoder == "imagecodecs.avif":
            rgb, full = _decode_avif(data)
        else:
            with Image.open(io.BytesIO(data), formats=(plugin,)) as im:
                im.seek(0)                       # an animated GIF's first frame (OPS-18 D1)
                rgb = np.asarray(im.convert("RGBA" if im.has_transparency_data else "RGB"))
        return Source.declared(orient(_cap(rgb, settings), orientation), full or float(np.iinfo(rgb.dtype).max),
                               declaration, fmt.id)
    except Unreadable:
        raise
    except (OSError, ValueError, RuntimeError, SyntaxError, EOFError) as exc:
        raise Unreadable(f"unreadable photo ({type(exc).__name__})") from exc


def _declare(data: bytes, format_id: str, icc: bytes | None) -> cd.Declaration:
    with _refusal():
        return cd.declare(data, format_id, icc)


def _decode_avif(data: bytes) -> tuple[np.ndarray, float]:
    """An AVIF's primary image at its own depth (ML-29 D9): libavif through imagecodecs, which, like Pillow's plugin,
    leaves the turning (irot, imir) to the orientation Pillow reports. A 10- or 12-bit image comes as a uint16 with
    white at 1023 or 4095, its depth read from the pixi property."""
    rgb = ic.avif_decode(data)
    if rgb.dtype == np.uint8:
        return rgb, 255.0
    depth = cd.primary_bit_depth(data)
    if depth is None or not 8 < depth <= 16:
        raise Unreadable("an AVIF whose bit depth is not stated")
    return rgb, float(2**depth - 1)


# A JPEG XL's extra channels beside the picture, which do not change how it looks (ML-29 D36); libjxl leaves them out
# of the colour it returns. Black (CMYK's K), a spot colour and CFA are part of the colour, and libjxl leaves them out
# too, so a photo with one would read wrong: refused, as is any type the spec reserves or a reader must not ignore.
SKIPPED_EXTRA_CHANNELS = frozenset({"Depth", "Thermal", "SelectionMask", "Optional"})
REFUSED_EXTRA_CHANNELS = {"Black": "black (CMYK)", "SpotColour": "spot colour", "CFA": "CFA (raw sensor)"}


def _skipped_extra_channels(types: list[str]) -> list[str]:
    """The extra channels to skip, or Unreadable naming those the colour needs (D36)."""
    if types.count("Alpha") > 1:
        raise Unreadable("a JPEG XL with more than one alpha channel")
    refused = [REFUSED_EXTRA_CHANNELS.get(t, t) for t in types if t != "Alpha" and t not in SKIPPED_EXTRA_CHANNELS]
    if refused:
        raise Unreadable(f"a JPEG XL whose colour needs its {', '.join(refused)} extra channel"
                         f"{'s' if len(refused) > 1 else ''}")
    return [t for t in types if t in SKIPPED_EXTRA_CHANNELS]


def _decode_jxl(data: bytes, max_pixels: int | None, settings: Settings) -> Source:
    """A visitor's JPEG XL that is not canonical: its colour declared by its header's enumeration, any of them
    (ML-29 D5); an embedded ICC profile, which imagecodecs does not return, is refused (D22). Float samples are
    converted unclipped and clipped after the conversion (D21). Animation is refused. Of the extra channels, one
    alpha is laid over the fill; one beside the picture is skipped; one the colour needs, or a second alpha, refuses
    the photo (D36, _skipped_extra_channels)."""
    header = read_header(data)
    _check_size(header.width, header.height, max_pixels)
    skipped = _skipped_extra_channels(header.extra_channel_types)
    if header.have_animation:
        raise Unreadable("an animated JPEG XL")
    with _refusal():
        declaration = cd.jxl_declaration(header.colour)
    rgb = ic.jpegxl_decode(data)                 # imagecodecs applies the file's orientation
    if skipped and "Alpha" not in header.extra_channel_types and header.colour.colour_space == "Grey":
        rgb = rgb[0]   # imagecodecs gives grey and its skipped channels as planes (1 + n, h, w); libjxl drops them for RGB
    if rgb.dtype == np.float32:
        white = 1.0
    else:
        # White is the file's own bit depth, not the dtype's: imagecodecs gives a 10- or 12-bit JPEG XL's samples
        # unscaled in a uint16 (ticket ML-28: 12-bit ones read as 16 came out about 16 times too dark).
        white = float(min(2**header.bits_per_sample, np.iinfo(rgb.dtype).max + 1) - 1)
    kind = "f" if header.float_sample else ""
    return Source.declared(_cap(rgb, settings), white, declaration, f"jxl{header.bits_per_sample}{kind}")


# ----- the colour conversion and the encode -----

def to_rec2020(pixels: np.ndarray, full: float, profile: bytes | None, strip: int = STRIP_ROWS,
               settings: Settings = STORE_SETTINGS, linearize: Callable[[np.ndarray], None] | None = None,
               peak: float | None = None) -> np.ndarray:
    """Source values to Rec.2020 uint16, relative colorimetric. The store's float path (D31): `linearize` (the curve
    of a declaration whose profile is built, ML-29 D20) undone in place, then LittleCMS on float32, clipped to [0, 1]
    after the conversion, rounded; transparent pixels over the fill (D13). With a `peak` above white (Source.peak,
    the whole picture's: ML-29 D33), LittleCMS goes to linear Rec.2020 first, the tone map rolls light above white
    off there, and a second transform takes it to the canonical profile. `pixels` may be a turned view, grey (2-D, or 1-2
    channels) or RGB(A), integer or float; each strip is copied out of it once and every later step works in place.
    With settings.colour "integer" (the front's, OPS-22 D4), an integer source without transparency and without a
    built profile goes the integer path instead (_to_rec2020_integer); any other stays on this one (D4, ML-29 D26)."""
    h, w = pixels.shape[:2]
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    alpha_at = {2: 1, 4: 3}.get(channels)
    if settings.colour == "integer" and alpha_at is None and linearize is None and pixels.dtype.kind == "u":
        return _to_rec2020_integer(pixels, full, profile)
    out = np.empty((h, w, 3), np.uint16)
    colour = 1 if channels <= 2 else 3           # grey is spread over R, G and B by broadcasting
    source_profile = profile or SRGB
    fill = None
    if alpha_at is not None:
        fill = ic.cms_transform(np.array([[FILL_SRGB]], np.float32) / 255, SRGB, REC2020, intent=RELATIVE,
                                outdtype=np.float32)[0, 0]
    for y in range(0, h, strip):
        part, f = _light(pixels[y:y + strip], full, linearize, colour)
        if peak is None or peak <= 1:            # nothing above white: the conversion every other photo takes
            c = ic.cms_transform(f, source_profile, REC2020, intent=RELATIVE, outdtype=np.float32)
        else:
            c = ic.cms_transform(f, source_profile, LINEAR_REC2020, intent=RELATIVE, outdtype=np.float32)
            tm.eetf_in_place(c, peak)
            c = ic.cms_transform(c, LINEAR_REC2020, REC2020, intent=RELATIVE, outdtype=np.float32)
        if part.dtype.kind == "f":               # a float JPEG XL's NaN, which no clip removes, is black
            np.nan_to_num(c, copy=False, nan=0.0)
        np.clip(c, 0, 1, out=c)
        if fill is not None:                     # c * alpha + fill * (1 - alpha)
            f[...] = part[..., alpha_at:alpha_at + 1]
            f /= full
            c *= f
            np.subtract(1, f, out=f)
            f *= fill
            c += f
        c *= 65535
        np.rint(c, out=c)
        out[y:y + strip] = c
    return out


def _light(part: np.ndarray, full: float, linearize: Callable[[np.ndarray], None] | None,
           colour: int) -> tuple[np.ndarray, np.ndarray]:
    """A strip of source pixels (as 3-D) and its colour as float32, 1.0 white, its curve undone if `linearize`."""
    if part.ndim == 2:
        part = part[..., None]
    f = np.empty(part.shape[:2] + (3,), np.float32)
    f[...] = part[..., :colour]
    f /= full
    if linearize is not None:
        linearize(f)
    return part, f


def measure_peak(pixels: np.ndarray, full: float, profile: bytes | None,
                 linearize: Callable[[np.ndarray], None] | None, strip: int = STRIP_ROWS) -> float:
    """ML-29 D33: the picture's mastering peak, relative to white, as tone_map.picture_peak takes it, from max(R, G, B)
    of its light in linear Rec.2020 (the space the tone map works in), weighted by its alpha: light laid over the fill
    shows only that much of itself. Strip by strip, one float32 per pixel kept."""
    h, w = pixels.shape[:2]
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    colour, alpha_at = (1 if channels <= 2 else 3), {2: 1, 4: 3}.get(channels)
    maxima = np.empty(h * w, np.float32)
    for y in range(0, h, strip):
        part, f = _light(pixels[y:y + strip], full, linearize, colour)
        c = ic.cms_transform(f, profile or SRGB, LINEAR_REC2020, intent=RELATIVE, outdtype=np.float32)
        top = c.max(axis=-1)
        if alpha_at is not None:
            top *= part[..., alpha_at] / np.float32(full)
        maxima[y * w:y * w + c.shape[0] * w] = top.ravel()
    return tm.picture_peak(maxima)


def _to_rec2020_integer(pixels: np.ndarray, full: float, profile: bytes | None) -> np.ndarray:
    """OPS-22 D4: the front's colour path for a source without transparency. LittleCMS's default (optimized) transform,
    relative colorimetric, from the source's own 8- or 16-bit samples, grey spread over R, G and B, straight into
    Rec.2020 uint16: about 8x less time than the float path, within one 8-bit level of it once read back
    (tests/test_canonical.py, IntegerColour). One call for the whole rectangle libjxl asks for (at most 2048 x 2048)."""
    a = pixels if pixels.ndim == 3 else pixels[..., None]
    if a.shape[2] == 1:
        a = np.broadcast_to(a, a.shape[:2] + (3,))
    top = np.iinfo(a.dtype).max
    if full != top:   # pillow-heif's 10- and 12-bit samples sit in the top of 16 bits: scaled so `full` is the top
        f = int(full)
        a = np.minimum((a.astype(np.uint32) * top + f // 2) // f, top).astype(a.dtype)
    return ic.cms_transform(np.ascontiguousarray(a), profile or SRGB, REC2020, intent=RELATIVE, outdtype=np.uint16)


def encode(source: Source, *, settings: Settings = STORE_SETTINGS, numthreads: int = THREADS) -> bytes:
    """A decoded original as its canonical JPEG XL, each rectangle libjxl asks for converted when it asks (D52), with
    `settings`' colour path, quality and effort."""
    h, w = source.pixels.shape[:2]
    peak = source.peak()                         # before libjxl's threads ask for rectangles: measured once
    return encode_rec2020(h, w, lambda y, x, ys, xs: to_rec2020(source.pixels[y:y + ys, x:x + xs], source.full,
                                                                source.profile, settings=settings,
                                                                linearize=source.linearize, peak=peak),
                          level=settings.quality, effort=settings.effort, numthreads=numthreads)


@dataclass(frozen=True)
class Made:
    """A canonical photo and what it was made from: the decoded size and depth (the front's log line, ML-19 D38)."""
    canonical: bytes
    width: int
    height: int
    bits: int


def canonicalize_with(data: bytes, name: str = "", settings: Settings = STORE_SETTINGS,
                      max_pixels: int | None = None) -> Made:
    """An original's canonical photo made with `settings`, with no pass-through: an already canonical photo is
    re-encoded too. The front's one path (OPS-22 D1, D3), and what the deploy's checks of it call, so they make the
    bytes the front sends. Raises Unreadable, or TooLarge over `max_pixels`."""
    source = decode(data, name, max_pixels, settings=settings)
    (h, w), bits = source.pixels.shape[:2], source.pixels.dtype.itemsize * 8
    return Made(encode(source, settings=settings), w, h, bits)


def canonicalize(data: bytes, name: str = "", max_pixels: int | None = None) -> bytes:
    """An original photo's bytes to its canonical photo's. `name` is the file's name, if known: a RAW photo is told
    by its extension. A canonical photo comes back as it is (D17). Raises Unreadable, or TooLarge over
    `max_pixels`."""
    if is_canonical(data):
        return data
    return encode(decode(data, name, max_pixels))


def canonicalize_file(path: Path, max_pixels: int | None = None) -> bytes:
    return canonicalize(Path(path).read_bytes(), Path(path).name, max_pixels)


# ----- the canonical store (D18, D33) -----

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "dataset"
STORE = REPO / "data" / "canonical"


def is_photo(path: Path) -> bool:
    return path.suffix.lower() in PHOTO_EXTENSIONS


def canonical_path(original: Path, dataset: Path = DATASET, store: Path = STORE) -> Path:
    """Where an original's canonical photo is kept: data/canonical/<its path under dataset/>.jxl, the original's
    file name kept whole (PXL_1.jpg and PXL_1.dng are two photos). Records stay keyed by the original's path and
    sha256 (D33): this is the one mapping from it."""
    rel = Path(original).resolve().relative_to(Path(dataset).resolve())
    return Path(store) / rel.parent / f"{rel.name}.jxl"


# ----- what a canonical photo is, and reading one -----

def header_problems(h: Header) -> list[str]:
    """What keeps a JPEG XL header from being a canonical photo's (D35); empty if nothing."""
    c = h.colour
    problems = []
    if h.boxes:
        problems.append(f"a container ({', '.join(h.boxes)}), not a bare codestream")
    if (c.want_icc, c.colour_space, c.white_point, c.primaries, c.transfer, c.rendering_intent) != \
            (False, "RGB", "D65", "BT2100", "BT709", "Relative"):
        what = "an ICC profile" if c.want_icc else \
            f"{c.colour_space} {c.white_point} {c.primaries} {c.transfer} {c.rendering_intent}"
        problems.append(f"colour encoding {what}, not D7's RGB D65 BT2100 BT709 Relative")
    if (h.bits_per_sample, h.float_sample, h.xyb_encoded) != (16, False, True):
        problems.append(f"{h.bits_per_sample}-bit{' float' if h.float_sample else ''}"
                        f"{'' if h.xyb_encoded else ' lossless'}, not 16-bit lossy")
    if h.num_extra_channels:
        problems.append(f"extra channels ({', '.join(h.extra_channel_types)})")
    if h.orientation != 1:
        problems.append(f"orientation {h.orientation}")
    if h.have_preview or h.have_animation:
        problems.append("a preview or an animation")
    if h.width * h.height > KEEP_MAX_PIXELS:
        problems.append(f"{h.width} x {h.height}, over {KEEP_MAX_PIXELS:,} pixels")
    return problems


def canonical_problems(data: bytes) -> list[str]:
    try:
        return header_problems(read_header(data))
    except HeaderError as exc:
        return [str(exc)]


def is_canonical(data: bytes) -> bool:
    """D35: the one check behind canonicalize()'s pass-through (D17; the front's own until OPS-22 D3) and D12's
    refusal at the host."""
    return not canonical_problems(data)


def usable_cpus() -> int:
    """The CPUs this process may run on: its affinity, which taskset and systemd's CPUAffinity= narrow."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:   # not Linux
        return os.cpu_count() or 1


def load_canonical(data: bytes, strip: int = 128, numthreads: int | None = None) -> np.ndarray:
    """A canonical photo's 8-bit sRGB, H x W x 3 uint8 (D7): decoded at 16 bits, LittleCMS without its lossy
    optimization (the same as float32's output, P1), relative colorimetric, clipped and rounded. Anything else is
    refused with NotCanonical; a canonical header over a broken codestream raises Unreadable.

    ML-21: libjxl decodes on `numthreads` threads and the strips are converted on as many (None: every CPU this
    process may use). Neither libjxl's decode nor a strip's conversion depends on the thread count, so the pixels
    do not either; imagecodecs releases the GIL in both."""
    problems = canonical_problems(data)
    if problems:
        raise NotCanonical("not a canonical photo: " + "; ".join(problems))
    threads = numthreads or usable_cpus()
    try:
        rec = ic.jpegxl_decode(data, numthreads=threads)
    except (ic.JpegxlError, ValueError, RuntimeError) as exc:   # a canonical header over a broken codestream
        raise Unreadable(f"unreadable canonical photo ({type(exc).__name__})") from exc
    out = np.empty(rec.shape, np.uint8)

    def convert(y: int) -> None:
        ic.cms_transform(rec[y:y + strip], REC2020, SRGB, intent=RELATIVE, flags=NOOPTIMIZE, outdtype=np.uint8,
                         out=out[y:y + strip])

    rows = range(0, rec.shape[0], strip)
    if threads == 1:
        for y in rows:
            convert(y)
    else:
        with ThreadPoolExecutor(threads) as pool:   # per call: no threads left over across gunicorn's fork
            list(pool.map(convert, rows))
    return out


__all__ = ["CAP_PIXELS", "FRONT_SETTINGS", "KEEP_MAX_PIXELS", "NotCanonical", "STORE_SETTINGS", "Settings",
           "TooLarge", "Unreadable", "applied_orientation",
           "canonical_path",
           "canonical_problems", "canonicalize", "canonicalize_file", "canonicalize_with", "Made", "capped_size", "decode", "encode",
           "header_problems", "is_canonical", "is_photo", "load_canonical", "orient", "to_rec2020"]
