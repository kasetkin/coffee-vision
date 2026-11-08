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
- from its own ICC profile, or sRGB when it has none, through LittleCMS on float32, relative colorimetric, into
  Rec.2020 (D65, BT.709 transfer; the profile libjxl itself generates, coffeecv/data/rec2020_bt709.icc), clipped;
- stored as lossy JPEG XL, quality 95, effort 7, 16 bits, with D7's encoding as an enumeration and nothing else: no
  EXIF, no XMP, no ICC, no orientation, no alpha, bare codestream.
Read back for the models: decoded at 16 bits, LittleCMS without its lossy optimization to 8-bit sRGB, clipped.

No torch: the front imports this module (D35). Every library that decodes, converts or encodes a photo is held at one
version across the three locks (coffeecv.lock_check), and a version change is a re-canonicalization. libjxl's
output depends on the CPU (D51): canonical photos are made only on the store's machine and on the front.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

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
RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}
HEIF_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1")
JXL_SIGNATURES = (b"\xff\x0a", b"\x00\x00\x00\x0cJXL \r\n\x87\n")


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
    """An original's main image as decoded: `pixels` (H x W x C, C 1-4, or H x W; uint8 or uint16; maybe a turned
    view), the sample value that means 1.0, its ICC profile (None: untagged, read as sRGB) and its decoder's name."""

    def __init__(self, pixels: np.ndarray, full: float, profile: bytes | None, decoder: str):
        self.pixels, self.full, self.profile, self.decoder = pixels, full, profile, decoder


def _sniff(data: bytes, name: str) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data.startswith(JXL_SIGNATURES):
        return "jxl"
    if data[4:8] == b"ftyp" and data[8:12] in HEIF_BRANDS:
        return "heif"
    if Path(name).suffix.lower() in RAW_EXTENSIONS:
        return "raw"
    return "pillow"


def orient(a: np.ndarray, orientation: int) -> np.ndarray:
    """EXIF orientation 1-8 applied as a view, not a copy; any other value as stored (D28)."""
    t = a.swapaxes(0, 1)
    views = {2: lambda: a[:, ::-1], 3: lambda: a[::-1, ::-1], 4: lambda: a[::-1], 5: lambda: t,
             6: lambda: t[:, ::-1], 7: lambda: t[::-1, ::-1], 8: lambda: t[::-1]}
    return views[orientation]() if orientation in views else a


def capped_size(width: int, height: int) -> tuple[int, int]:
    """D27: the size a photo is stored at. Over 60,000,000 pixels, the largest size of the same shape (each side
    scaled by one factor, rounded down) of at most 50,000,000; otherwise its own."""
    if width * height <= KEEP_MAX_PIXELS:
        return width, height
    s = (CAP_PIXELS / (width * height)) ** 0.5
    w, h = max(1, int(width * s)), max(1, int(height * s))
    while w * h > CAP_PIXELS:
        w, h = w - 1, h - 1
    return w, h


def _cap(a: np.ndarray) -> np.ndarray:
    """D27, D43: over 60 MP, area-averaged at the capped size, before turning (the pixel count does not depend on it)."""
    h, w = a.shape[:2]
    nw, nh = capped_size(w, h)
    if (nw, nh) == (w, h):
        return a
    import cv2   # only for the few photos over 60 MP: it costs the front 15 MB to import
    return cv2.resize(a, (nw, nh), interpolation=cv2.INTER_AREA)


def _pillow_metadata(data: bytes) -> tuple[bytes | None, int, str, tuple[int, int]]:
    """Profile, EXIF orientation, mode and size, read without decoding the pixels. Pillow's own decompression-bomb
    limit (Image.MAX_IMAGE_PIXELS) applies: the dataset tools raise it, the web service keeps Pillow's default."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return im.info.get("icc_profile") or None, im.getexif().get(0x0112, 1), im.mode, im.size
    except Image.DecompressionBombError as exc:
        raise TooLarge(str(exc)) from exc


def usable_profile(profile: bytes | None) -> bytes | None:
    """The photo's ICC profile if LittleCMS can convert from it as RGB; else None, so the photo is read as sRGB, as an
    untagged one is (D16): a grey or broken profile, which a phone does not write, refuses no photo."""
    if not profile or profile[16:20] != b"RGB ":
        return None
    try:
        ic.cms_profile_validate(profile)
    except Exception:   # noqa: BLE001 -- imagecodecs raises its own CmsError, or ValueError on garbage
        return None
    return profile


def _check_size(width: int, height: int, max_pixels: int | None) -> None:
    if max_pixels is not None and width * height > max_pixels:
        raise TooLarge(f"{width} x {height} is over {max_pixels:,} pixels")


def applied_orientation(data: bytes, name: str = "") -> int:
    """The EXIF orientation canonicalization turns this original by (D28): 1 for HEIC, RAW and JPEG XL, whose decoders
    turn them, and for a value outside 1-8."""
    if _sniff(data, name) in ("raw", "heif", "jxl"):
        return 1
    orientation = _pillow_metadata(data)[1]
    return orientation if 1 <= orientation <= 8 else 1


def decode(data: bytes, name: str = "", max_pixels: int | None = None) -> Source:
    """An original's main image at its decoder's depth, capped and upright (D20, D27, D28, D36). A photo over
    `max_pixels` (the front's ceiling, D38) raises TooLarge before its pixels are decoded."""
    kind = _sniff(data, name)
    try:
        if kind == "raw":
            import rawpy
            try:
                with rawpy.imread(io.BytesIO(data)) as raw:
                    _check_size(raw.sizes.width, raw.sizes.height, max_pixels)
                    rgb = raw.postprocess(output_bps=16, use_camera_wb=True, output_color=rawpy.ColorSpace.Rec2020,
                                          gamma=(1, 1))
            except rawpy.LibRawError as exc:   # not an OSError; LibRaw's message is a bytes repr
                raise Unreadable(f"unreadable RAW file ({type(exc).__name__})") from exc
            return Source(_cap(rgb), 65535.0, LINEAR_REC2020, "rawpy16")
        if kind == "heif":
            import pillow_heif
            heif = pillow_heif.open_heif(io.BytesIO(data), convert_hdr_to_8bit=False)
            profile, depth = heif.info.get("icc_profile") or None, heif.info.get("bit_depth", 8)
            _check_size(*heif.size, max_pixels)
            rgb = np.asarray(heif)
            del heif
            # pillow-heif shifts a 10- or 12-bit sample into the top of 16 bits: 10-bit white is 1023 << 6
            full = float((2**depth - 1) << (16 - depth)) if rgb.dtype == np.uint16 else 255.0
            return Source(_cap(rgb), full, usable_profile(profile), f"heif{depth}")
        if kind == "jxl":
            return _decode_jxl(data, max_pixels)
        profile, orientation, mode, size = _pillow_metadata(data)
        _check_size(*size, max_pixels)
        if mode == "CMYK":
            raise Unreadable("a CMYK photo")
        if kind == "jpeg":
            rgb = ic.jpeg8_decode(data)          # the same pixels as Pillow's libjpeg-turbo decode (P1)
        elif kind == "png":
            rgb = ic.png_decode(data)            # keeps 16 bits, which Pillow would cut to 8
        else:
            with Image.open(io.BytesIO(data)) as im:
                im.seek(0)
                rgb = np.asarray(im.convert("RGBA" if im.has_transparency_data else "RGB"))
        return Source(orient(_cap(rgb), orientation), float(np.iinfo(rgb.dtype).max), usable_profile(profile), kind)
    except Unreadable:
        raise
    except (OSError, ValueError, RuntimeError, SyntaxError, EOFError) as exc:
        raise Unreadable(f"unreadable photo ({type(exc).__name__})") from exc


def _decode_jxl(data: bytes, max_pixels: int | None) -> Source:
    """A visitor's JPEG XL that is not canonical: decodable if its colour encoding is sRGB or D7's, as an enumeration.
    (imagecodecs gives no ICC profile for any other.)"""
    header = read_header(data)
    _check_size(header.width, header.height, max_pixels)
    c = header.colour
    if c.want_icc or c.colour_space != "RGB" or c.white_point != "D65" or header.num_extra_channels > 1:
        raise Unreadable("a JPEG XL in a colour encoding canonicalization does not read")
    if (c.primaries, c.transfer) == ("sRGB", "sRGB"):
        profile = SRGB
    elif (c.primaries, c.transfer) == ("BT2100", "BT709"):
        profile = REC2020
    else:
        raise Unreadable(f"a JPEG XL in {c.primaries} primaries with the {c.transfer} transfer")
    rgb = ic.jpegxl_decode(data)                 # imagecodecs applies the file's orientation
    if rgb.dtype == np.float32:
        raise Unreadable("a floating-point JPEG XL")
    return Source(_cap(rgb), float(np.iinfo(rgb.dtype).max), profile, f"jxl{rgb.dtype.itemsize * 8}")


# ----- the colour conversion and the encode -----

def to_rec2020(pixels: np.ndarray, full: float, profile: bytes | None, strip: int = STRIP_ROWS) -> np.ndarray:
    """Source values to Rec.2020 uint16 (D31): LittleCMS on float32, relative colorimetric, clipped to [0, 1],
    rounded; transparent pixels over the fill (D13). `pixels` may be a turned view, grey (2-D, or 1-2 channels) or
    RGB(A); each strip is copied out of it once and every later step works in place."""
    h, w = pixels.shape[:2]
    out = np.empty((h, w, 3), np.uint16)
    channels = 1 if pixels.ndim == 2 else pixels.shape[2]
    colour = 1 if channels <= 2 else 3           # grey is spread over R, G and B by broadcasting
    alpha_at = {2: 1, 4: 3}.get(channels)
    source_profile = profile or SRGB
    fill = None
    if alpha_at is not None:
        fill = ic.cms_transform(np.array([[FILL_SRGB]], np.float32) / 255, SRGB, REC2020, intent=RELATIVE,
                                outdtype=np.float32)[0, 0]
    for y in range(0, h, strip):
        part = pixels[y:y + strip]
        if part.ndim == 2:
            part = part[..., None]
        f = np.empty(part.shape[:2] + (3,), np.float32)
        f[...] = part[..., :colour]
        f /= full
        c = ic.cms_transform(f, source_profile, REC2020, intent=RELATIVE, outdtype=np.float32)
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


def encode(source: Source, *, effort: int = EFFORT, numthreads: int = THREADS) -> bytes:
    """A decoded original as its canonical JPEG XL, each rectangle libjxl asks for converted when it asks (D52)."""
    h, w = source.pixels.shape[:2]
    return encode_rec2020(h, w, lambda y, x, ys, xs: to_rec2020(source.pixels[y:y + ys, x:x + xs], source.full,
                                                                source.profile),
                          level=QUALITY, effort=effort, numthreads=numthreads)


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
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".hif", ".webp", ".avif", ".jxl", ".tif", ".tiff",
                    ".gif", ".bmp"} | RAW_EXTENSIONS


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
    """D35: the one check behind D17's pass-through at the front and D12's refusal at the host."""
    return not canonical_problems(data)


def load_canonical(data: bytes, strip: int = 128) -> np.ndarray:
    """A canonical photo's 8-bit sRGB, H x W x 3 uint8 (D7): decoded at 16 bits, LittleCMS without its lossy
    optimization (the same as float32's output, P1), relative colorimetric, clipped and rounded. Anything else is
    refused with NotCanonical; a canonical header over a broken codestream raises Unreadable."""
    problems = canonical_problems(data)
    if problems:
        raise NotCanonical("not a canonical photo: " + "; ".join(problems))
    try:
        rec = ic.jpegxl_decode(data)
    except (ic.JpegxlError, ValueError, RuntimeError) as exc:   # a canonical header over a broken codestream
        raise Unreadable(f"unreadable canonical photo ({type(exc).__name__})") from exc
    out = np.empty(rec.shape, np.uint8)
    for y in range(0, rec.shape[0], strip):
        ic.cms_transform(rec[y:y + strip], REC2020, SRGB, intent=RELATIVE, flags=NOOPTIMIZE, outdtype=np.uint8,
                         out=out[y:y + strip])
    return out


__all__ = ["CAP_PIXELS", "KEEP_MAX_PIXELS", "NotCanonical", "TooLarge", "Unreadable", "applied_orientation",
           "canonical_path",
           "canonical_problems", "canonicalize", "canonicalize_file", "capped_size", "decode", "encode",
           "header_problems", "is_canonical", "is_photo", "load_canonical", "orient", "to_rec2020"]
