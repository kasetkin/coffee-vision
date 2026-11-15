"""coffeecv.colour_declaration (ticket ML-29 P1; ADR 0040): each declaration kind read from files made here, the order
of precedence, refusals with their message, the legacy fields ignored, and the profile builder against LittleCMS's own
sRGB and libjxl's Display P3 (R2).

    python -m unittest discover -s tests -p 'test_colour_declaration.py'
"""
from __future__ import annotations

import io
import struct
import unittest
import zlib
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image, ImageCms

from coffeecv import colour_declaration as cd
from coffeecv.jxl_header import read_header

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DISPLAY_P3 = (FIXTURES / "display_p3_libjxl.icc").read_bytes()   # libjxl 0.12.0's DisplayP3 (D65, sRGB curve)
SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
RNG = np.random.default_rng(29)
PIXELS = (RNG.random((16, 24, 3)) * 255).astype(np.uint8)


# ----- files made here -----

def avif(primaries: int, transfer: int) -> bytes:
    """An AVIF whose primary item carries one nclx box with these H.273 values, written by libavif or, for values it
    will not write, patched into its box."""
    data = ic.avif_encode(PIXELS, level=90, primaries=1, transfer=13)
    at = data.index(b"colrnclx") + 8
    return data[:at] + struct.pack(">HH", primaries, transfer) + data[at + 4:]


def heif_branded(data: bytes) -> bytes:
    """The same boxes with a HEIF major brand: the reader is the one ISO BMFF reader for both."""
    assert data[4:8] == b"ftyp"
    return data[:8] + b"mif1" + data[12:]


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def png(*chunks: tuple[bytes, bytes]) -> bytes:
    """A PNG with `chunks` placed after IHDR."""
    buf = io.BytesIO()
    Image.fromarray(PIXELS).save(buf, "PNG")
    data = buf.getvalue()
    end = 8 + 12 + 13
    return data[:end] + b"".join(png_chunk(k, b) for k, b in chunks) + data[end:]


def exif_with_interop(index: str) -> bytes:
    """EXIF (little-endian TIFF) whose Exif IFD says ColorSpace Uncalibrated, as DCF's Adobe RGB files do, and points to
    an interoperability IFD with InteropIndex `index`: Pillow does not write that IFD."""
    def ifd(entries: list[tuple[int, int, int, bytes]], at: int) -> bytes:
        out = struct.pack("<H", len(entries))
        for tag, kind, count, value in entries:
            out += struct.pack("<HHI", tag, kind, count) + value.ljust(4, b"\0")
        return out + struct.pack("<I", 0)
    ifd0_at, exif_at, interop_at = 8, 8 + 18, 8 + 18 + 30
    head = b"II*\0" + struct.pack("<I", ifd0_at)
    ifd0 = ifd([(0x8769, 4, 1, struct.pack("<I", exif_at))], ifd0_at)
    exif_ifd = ifd([(0xA001, 3, 1, struct.pack("<H", 0xFFFF)), (0xA005, 4, 1, struct.pack("<I", interop_at))], exif_at)
    interop = ifd([(0x0001, 2, 4, index.encode()[:3] + b"\0")], interop_at)
    assert len(head + ifd0) == exif_at and len(head + ifd0 + exif_ifd) == interop_at
    return b"Exif\0\0" + head + ifd0 + exif_ifd + interop


def jpeg(interop: str | None = None, icc: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {"icc_profile": icc} if icc else {}
    if interop is not None:
        kw["exif"] = exif_with_interop(interop)
    Image.fromarray(PIXELS).save(buf, "JPEG", quality=95, **kw)
    return buf.getvalue()


def bmp_v5(cs_type: bytes, profile: bytes = b"") -> bytes:
    """A 2 x 2 24-bit BMP with a BITMAPV5HEADER: `cs_type` 'MBED' with `profile` after the pixels, or another type."""
    rows = b"".join(bytes(PIXELS[y, :2, ::-1].tobytes()) + b"\0\0" for y in (1, 0))   # bottom-up, BGR, 4-byte rows
    pixels_at = 14 + 124
    profile_at = 124 + len(rows)                          # from the info header's start
    info = struct.pack("<IiiHHIIiiII", 124, 2, 2, 1, 24, 0, len(rows), 2835, 2835, 0, 0)
    info += b"\0" * 16                                    # the four masks
    info += cs_type[::-1] + b"\0" * 36 + b"\0" * 12        # bV5CSType (a DWORD, little-endian), endpoints, gammas
    info += struct.pack("<IIII", 4, profile_at if profile else 0, len(profile), 0)
    assert len(info) == 124
    body = info + rows + profile
    return b"BM" + struct.pack("<IHHI", 14 + len(body), 0, 0, pixels_at) + body


class Bitwriter:
    """JPEG XL's bit order: least significant bit of each byte first."""

    def __init__(self):
        self.bits: list[int] = []

    def u(self, value: int, n: int) -> None:
        self.bits += [(value >> i) & 1 for i in range(n)]

    def u32(self, value: int, *dists: tuple[int, int]) -> None:
        for i, (offset, n) in enumerate(dists):
            if offset <= value < offset + (1 << n) and (n or value == offset):
                self.u(i, 2)
                self.u(value - offset, n)
                return
        raise ValueError(value)

    def enum(self, value: int) -> None:
        self.u32(value, (0, 0), (1, 0), (2, 4), (18, 6))

    def customxy(self, x: float, y: float) -> None:
        for v in (round(x * 1e6), round(y * 1e6)):
            u = 2 * v if v >= 0 else -2 * v - 1          # PackSigned
            self.u32(u, (0, 19), (524288, 19), (1048576, 20), (2097152, 21))

    def bytes(self) -> bytes:
        bits = self.bits + [0] * (-len(self.bits) % 8)
        return bytes(sum(b << i for i, b in enumerate(bits[k:k + 8])) for k in range(0, len(bits), 8))


def jxl_header_bytes(white=None, primaries=None, transfer: int | None = 8, gamma: float | None = None,
                     space: int = 0, want_icc: bool = False) -> bytes:
    """A JPEG XL codestream's header up to its colour encoding (ISO/IEC 18181-1, A.6, A.8, E.2), 64 x 64, 8-bit: what
    jxl_header reads. `white` and `primaries` are enumeration values or custom xy tuples."""
    w = Bitwriter()
    w.u(1, 1)
    w.u(7, 5)                                             # small size: height (7 + 1) * 8
    w.u(1, 3)                                             # ratio 1:1
    w.u(0, 1)                                             # metadata not all_default
    w.u(0, 1)                                             # no extra_fields
    w.u(0, 1)                                             # integer samples
    w.u32(8, (8, 0), (10, 0), (12, 0), (1, 6))
    w.u(1, 1)                                             # modular_16bit_buffers
    w.u32(0, (0, 0), (1, 0), (2, 4), (1, 12))             # no extra channels
    w.u(1, 1)                                             # xyb_encoded
    w.u(0, 1)                                             # colour encoding not all_default
    w.u(int(want_icc), 1)
    w.enum(space)
    if not want_icc:
        if isinstance(white, tuple):
            w.enum(2)
            w.customxy(*white)
        else:
            w.enum(1 if white is None else white)
        if space != 1:
            if isinstance(primaries, tuple):
                w.enum(2)
                for i in range(3):
                    w.customxy(*primaries[2 * i:2 * i + 2])
            else:
                w.enum(1 if primaries is None else primaries)
        if gamma is not None:
            w.u(1, 1)
            w.u(round(gamma * 1e7), 24)
        else:
            w.u(0, 1)
            w.enum(transfer)
        w.enum(1)                                         # relative
    return b"\xff\x0a" + w.bytes() + b"\0" * 16


# ----- independent colour maths -----

def rgb_to_xyz(primaries, white) -> np.ndarray:
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in zip(primaries[::2], primaries[1::2])]).T
    wx, wy = white
    return xyz * np.linalg.solve(xyz, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))


class Precedence(unittest.TestCase):
    def test_an_icc_profile_beside_a_coded_tag_wins(self):   # D4
        d = cd.declare(avif(9, 8), "avif", icc=DISPLAY_P3)
        self.assertEqual((d.kind, d.icc), ("icc", DISPLAY_P3))

    def test_a_coded_tag_alone(self):
        for fmt, data in (("avif", avif(9, 8)), ("heif", heif_branded(avif(9, 8)))):
            with self.subTest(fmt=fmt):
                d = cd.declare(data, fmt)
                self.assertEqual((d.kind, d.primaries, d.white, d.curve), ("coded", cd.REC2020_PRIMARIES, cd.D65,
                                                                           "linear"))
                self.assertEqual(d.source, "nclx 9/8 (BT.2020, linear)")

    def test_display_p3_as_a_coded_tag(self):
        d = cd.declare(avif(12, 13), "avif")
        self.assertEqual((d.primaries, d.white, d.curve), (cd.P3_PRIMARIES, cd.D65, "sRGB"))

    def test_unspecified_declares_nothing_and_half_unspecified_is_srgbs_half(self):
        self.assertEqual(cd.declare(avif(2, 2), "avif"), cd.NONE)
        self.assertEqual(cd.declare(avif(9, 2), "avif").curve, "sRGB")
        self.assertEqual(cd.declare(avif(2, 8), "avif").primaries, cd.SRGB_PRIMARIES)

    def test_reserved_coded_values_are_refused_naming_them(self):   # R1
        for p, t, word in ((3, 13, "primaries 3"), (1, 3, "transfer characteristics 3"), (0, 13, "primaries 0"),
                           (1, 19, "transfer characteristics 19"), (10, 8, "primaries 10"), (1, 9, "9")):
            with self.subTest(p=p, t=t), self.assertRaises(cd.Refused) as e:
                cd.declare(avif(p, t), "avif")
            self.assertIn(f"nclx {p}/{t}", str(e.exception))
            self.assertIn(word, str(e.exception))

    def test_an_unusable_icc_profile_is_srgb_whatever_else_is_declared(self):   # ML-19 D16, as ADR 0040 keeps it
        grey = SRGB_ICC[:16] + b"GRAY" + SRGB_ICC[20:]
        self.assertEqual(cd.declare(avif(9, 8), "avif", icc=grey), cd.NONE)
        self.assertEqual(cd.declare(png((b"cICP", bytes([9, 8, 0, 1]))), "png", icc=b"\0" * 16 + b"RGB " + b"\0" * 200),
                         cd.NONE)
        self.assertEqual(cd.declare(jpeg("R03"), "jpeg", icc=grey), cd.NONE)

    def test_boxes_that_run_past_their_ends_are_refused_not_raised(self):   # an upload's bytes are anyone's
        good = avif(9, 8)
        meta = good.index(b"meta") - 4
        broken = [good[:meta] + b"\0\0\0\x01meta",                       # a 64-bit size with no size after it
                  good[:good.index(b"ipma") + 8] + b"\0\0\0\x09"]        # an ipma claiming nine entries
        for data in broken:
            with self.subTest(n=len(data)):
                try:
                    d = cd.declare(data, "avif")
                except cd.Refused as e:
                    self.assertIn("cannot be parsed", str(e))
                else:
                    self.assertEqual(d, cd.NONE)

    def test_r03_is_adobe_rgb_and_only_without_a_profile(self):   # D8
        d = cd.declare(jpeg("R03"), "jpeg")
        self.assertEqual((d.kind, d.primaries, d.white, d.curve, d.gamma),
                         ("r03", cd.ADOBE_RGB_PRIMARIES, cd.D65, "gamma", 563 / 256))
        self.assertEqual(cd.declare(jpeg("R03", icc=SRGB_ICC), "jpeg", icc=SRGB_ICC).kind, "icc")
        self.assertEqual(cd.declare(jpeg("R98"), "jpeg"), cd.NONE)
        self.assertEqual(cd.declare(jpeg(), "jpeg"), cd.NONE)

    def test_formats_that_declare_nothing_else(self):
        for fmt in ("webp", "gif", "dng", "cr3"):
            with self.subTest(fmt=fmt):
                self.assertEqual(cd.declare(b"anything", fmt), cd.NONE)
                self.assertEqual(cd.declare(b"anything", fmt, icc=DISPLAY_P3).icc, DISPLAY_P3)


class Png(unittest.TestCase):
    def test_cicp(self):
        d = cd.declare(png((b"cICP", bytes([9, 16, 0, 1]))), "png")
        self.assertEqual((d.kind, d.source, d.curve, d.is_hdr), ("coded", "cICP 9/16 (BT.2020, PQ)", "PQ", True))

    def test_cicp_before_the_srgb_chunk_and_iccp_before_both(self):
        both = png((b"sRGB", b"\0"), (b"cICP", bytes([12, 13, 0, 1])))
        self.assertEqual(cd.declare(both, "png").primaries, cd.P3_PRIMARIES)
        self.assertEqual(cd.declare(both, "png", icc=SRGB_ICC).kind, "icc")

    def test_the_srgb_chunk(self):
        d = cd.declare(png((b"sRGB", b"\0")), "png")
        self.assertEqual((d.primaries, d.white, d.curve), (cd.SRGB_PRIMARIES, cd.D65, "sRGB"))
        self.assertIsNone(cd.conversion(d).profile)       # read exactly as an untagged photo is

    def test_legacy_gama_and_chrm_are_ignored(self):   # D7
        gama = struct.pack(">I", 45455)
        chrm = struct.pack(">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
        self.assertEqual(cd.declare(png((b"gAMA", gama), (b"cHRM", chrm)), "png"), cd.NONE)
        self.assertEqual(cd.declare(png((b"gAMA", struct.pack(">I", 100000))), "png"), cd.NONE)

    def test_a_cicp_with_a_matrix_or_in_narrow_range_is_refused(self):   # PNG's cICP is full-range RGB to be read
        for body, word in ((bytes([1, 13, 1, 1]), "matrix"), (bytes([1, 13, 0, 0]), "narrow range")):
            with self.subTest(word), self.assertRaises(cd.Refused) as e:
                cd.declare(png((b"cICP", body)), "png")
            self.assertIn(word, str(e.exception))


class Bmp(unittest.TestCase):
    def test_pillow_reads_the_files_made_here(self):
        for data in (bmp_v5(b"MBED", DISPLAY_P3), bmp_v5(b"sRGB"), bmp_v5(b"\0\0\0\0")):
            with Image.open(io.BytesIO(data), formats=("BMP",)) as im:
                self.assertEqual(np.asarray(im.convert("RGB")).tolist(), PIXELS[:2, :2].tolist())

    def test_an_embedded_profile_is_honoured(self):   # D7
        d = cd.declare(bmp_v5(b"MBED", DISPLAY_P3), "bmp")
        self.assertEqual((d.kind, d.icc), ("icc", DISPLAY_P3))

    def test_calibrated_endpoints_and_the_other_types_are_ignored(self):   # D7
        for cs in (b"\0\0\0\0", b"sRGB", b"Win ", b"LINK"):
            with self.subTest(cs=cs):
                self.assertEqual(cd.declare(bmp_v5(cs), "bmp"), cd.NONE)

    def test_a_short_or_older_bmp_has_no_profile(self):
        buf = io.BytesIO()
        Image.fromarray(PIXELS).save(buf, "BMP")          # BITMAPINFOHEADER, 40 bytes
        self.assertIsNone(cd.bmp_profile(buf.getvalue()))
        self.assertIsNone(cd.bmp_profile(b"BM"))


class JpegXl(unittest.TestCase):
    def jxl(self, primaries: str, transfer: str) -> bytes:
        return ic.jpegxl_encode(PIXELS, level=90, primaries=primaries, transfer=transfer)

    def test_every_enumeration_imagecodecs_writes(self):   # D5
        prim = {"SRGB": cd.SRGB_PRIMARIES, "BT2100": cd.REC2020_PRIMARIES, "P3": cd.P3_PRIMARIES}
        curve = {"BT709": "BT709", "SRGB": "sRGB", "LINEAR": "linear", "PQ": "PQ", "HLG": "HLG", "DCI": "gamma"}
        for p, xy in prim.items():
            for t, name in curve.items():
                with self.subTest(primaries=p, transfer=t):
                    d = cd.declare(self.jxl(p, t), "jxl")
                    self.assertEqual((d.kind, d.primaries, d.white, d.curve), ("enumeration", xy, cd.D65, name))
                    if t == "DCI":                   # libjxl's DCI is a pure 2.6 gamma, not H.273's ST 428-1
                        self.assertEqual(d.gamma, 2.6)

    def test_custom_primaries_white_point_and_gamma(self):   # darktable's Adobe RGB, and ProPhoto's D50
        adobe = (0.639997, 0.329999, 0.209995, 0.709998, 0.150005, 0.059996)
        d = cd.declare(jxl_header_bytes(primaries=adobe, gamma=0.454707), "jxl")
        self.assertEqual((d.primaries, d.white, d.curve), (adobe, cd.D65, "gamma"))
        self.assertAlmostEqual(d.gamma, 1 / 0.454707)
        prophoto = (0.7347, 0.2653, 0.1596, 0.8404, 0.0366, 0.0001)
        d = cd.declare(jxl_header_bytes(white=(0.3457, 0.3585), primaries=prophoto, transfer=8), "jxl")
        self.assertEqual((d.primaries, d.white, d.curve), (prophoto, (0.3457, 0.3585), "linear"))

    def test_the_header_writer_agrees_with_libjxls_files(self):
        """The test's own header writer, held to what jxl_header reads from imagecodecs' file for one enumeration."""
        mine = read_header(jxl_header_bytes(white=1, primaries=11, transfer=13)).colour
        theirs = read_header(self.jxl("P3", "SRGB")).colour
        self.assertEqual((mine.white_point, mine.primaries, mine.transfer),
                         (theirs.white_point, theirs.primaries, theirs.transfer))

    def test_negative_custom_coordinates(self):
        d = cd.declare(jxl_header_bytes(primaries=(0.7347, 0.2653, 0.1596, 0.8404, 0.0366, -0.0001)), "jxl")
        self.assertEqual(d.primaries[5], -0.0001)

    def test_grey(self):
        d = cd.declare(jxl_header_bytes(space=1, transfer=13), "jxl")
        self.assertEqual((d.white, d.curve), (cd.D65, "sRGB"))

    def test_refusals(self):
        cases = {"embedded ICC profile": jxl_header_bytes(want_icc=True),
                 "XYB": jxl_header_bytes(space=2),
                 "Unknown": jxl_header_bytes(transfer=2),
                 "Reserved": jxl_header_bytes(primaries=5)}
        for word, data in cases.items():
            with self.subTest(word=word), self.assertRaises(cd.Refused) as e:
                cd.declare(data, "jxl")
            self.assertIn(word, str(e.exception))


# R2, measured 2026-11-14: a profile LittleCMS builds and libjxl's Rec.2020 (the canonical photo's) disagree by up to
# 13 of 65,535 at the saturated corners (0.05 of an 8-bit level), from their colorants' rounding. The RAW path's
# LINEAR_REC2020 is the builder's own linear Rec.2020, so every developed RAW photo already carries it.
BUILT_TOLERANCE = 14 / 65535


class Builder(unittest.TestCase):
    """D20: the profiles built in code, against LittleCMS's own and libjxl's, on every 8-bit level of each channel and
    a grid of colours (R2)."""

    @staticmethod
    def samples() -> np.ndarray:
        levels = np.arange(256, dtype=np.float32) / 255
        ramp = np.stack([levels, np.zeros(256), np.zeros(256)], 1)
        grid = np.stack(np.meshgrid(*[np.linspace(0, 1, 9)] * 3), -1).reshape(-1, 3)
        return np.concatenate([ramp, ramp[:, [1, 0, 2]], ramp[:, [1, 2, 0]], np.repeat(levels[:, None], 3, 1),
                               grid]).astype(np.float32)[None]

    def through(self, d: cd.Declaration) -> np.ndarray:
        sc = cd.conversion(d)
        f = self.samples().copy()
        if sc.linearize:
            sc.linearize(f)
        return ic.cms_transform(f, sc.profile or ic.cms_profile("srgb"), cd.REC2020_PROFILE,
                                intent=ic.CMS.INTENT.RELATIVE_COLORIMETRIC, outdtype=np.float32)

    def reference(self, profile: bytes) -> np.ndarray:
        return ic.cms_transform(self.samples(), profile, cd.REC2020_PROFILE, intent=ic.CMS.INTENT.RELATIVE_COLORIMETRIC,
                                outdtype=np.float32)

    def test_display_p3_against_libjxls_profile(self):
        built = cd.Declaration("coded", "test", primaries=cd.P3_PRIMARIES, white=cd.D65, curve="sRGB")
        self.assertIsNotNone(cd.conversion(built).linearize)
        diff = np.abs(self.through(built) - self.reference(DISPLAY_P3))
        self.assertLess(float(diff.max()), BUILT_TOLERANCE)

    def test_linear_rec2020_is_the_raw_paths_profile(self):
        from coffeecv import canonical
        a, b = cd.linear_profile(cd.REC2020_PRIMARIES, cd.D65), canonical.LINEAR_REC2020
        # The same profile but for LittleCMS's creation time (bytes 24-36) and the ID it hashes into (84-100)
        self.assertEqual(a[:24] + a[36:84] + a[100:], b[:24] + b[36:84] + b[100:])

    def test_srgb_by_the_builders_own_path_against_littlecms(self):
        """The sRGB declaration itself takes LittleCMS's own profile; the builder's path for it is checked here too,
        by declaring sRGB's primaries with a white point a hair off D65's, which forces the build."""
        built = cd.Declaration("coded", "test", primaries=cd.SRGB_PRIMARIES, white=(0.31270001, 0.32900001),
                               curve="sRGB")
        self.assertIsNotNone(cd.conversion(built).linearize)
        diff = np.abs(self.through(built) - self.reference(ic.cms_profile("srgb")))
        self.assertLess(float(diff.max()) * 65535, 1.0)

    def test_the_two_declarations_that_keep_todays_profiles(self):
        srgb = cd.Declaration("enumeration", "t", primaries=cd.SRGB_PRIMARIES, white=cd.D65, curve="sRGB")
        rec = cd.Declaration("enumeration", "t", primaries=cd.REC2020_PRIMARIES, white=cd.D65, curve="BT709")
        self.assertEqual(cd.conversion(srgb), cd.SourceColour(None))
        self.assertEqual(cd.conversion(rec), cd.SourceColour(cd.REC2020_PROFILE))
        self.assertEqual(cd.conversion(cd.NONE), cd.SourceColour(None))
        self.assertEqual(cd.conversion(cd.Declaration("icc", "ICC profile", icc=DISPLAY_P3)),
                         cd.SourceColour(DISPLAY_P3))

    def test_linear_rec2020_against_colour_maths_written_here(self):
        d = cd.Declaration("coded", "t", primaries=cd.REC2020_PRIMARIES, white=cd.D65, curve="linear")
        self.assertIsNotNone(cd.conversion(d).linearize)       # a no-op, which still sends it down the float path
        out = self.through(d)[0]
        ref = self.samples()[0].astype(np.float64)
        bt709 = np.where(ref < 0.018, 4.5 * ref, 1.099 * ref ** 0.45 - 0.099)
        self.assertLess(float(np.abs(out - bt709).max()), BUILT_TOLERANCE)

    def test_adobe_rgb_against_colour_maths_written_here(self):
        out = self.through(cd.declare(jpeg("R03"), "jpeg"))[0].astype(np.float64)
        lin = self.samples()[0].astype(np.float64) ** (563 / 256)
        m = np.linalg.inv(rgb_to_xyz(cd.REC2020_PRIMARIES, cd.D65)) @ rgb_to_xyz(cd.ADOBE_RGB_PRIMARIES, cd.D65)
        rec = np.clip(lin @ m.T, 0, 1)
        bt709 = np.where(rec < 0.018, 4.5 * rec, 1.099 * rec ** 0.45 - 0.099)
        self.assertLess(float(np.abs(np.clip(out, 0, 1) - bt709).max()), 20 / 65535)   # 16 measured: two matrices

    def test_hdr_is_refused_until_the_operator_is_chosen(self):   # P2; P5 lifts it
        for t in ("PQ", "HLG"):
            with self.subTest(t=t), self.assertRaises(cd.Refused) as e:
                cd.conversion(cd.declare(ic.jpegxl_encode(PIXELS, level=90, primaries="BT2100", transfer=t), "jxl"))
            self.assertIn(t, str(e.exception))


class Curves(unittest.TestCase):
    """Each curve undone, against its standard's formula written out here."""

    def test_each_curve(self):
        v = np.linspace(0, 1, 1001)
        expected = {
            "sRGB": np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4),
            "BT709": np.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45)),
            "SMPTE240M": np.where(v < 0.0912, v / 4, ((v + 0.1115) / 1.1115) ** (1 / 0.45)),
            "DCI": v ** 2.6 * 52.37 / 48,
        }
        for name, want in expected.items():
            with self.subTest(curve=name):
                a = v.astype(np.float32)
                cd.CURVE_DECODERS[name](a)
                np.testing.assert_allclose(a, want, atol=2e-6)

    def test_gamma_and_negative_values_keep_their_sign(self):
        a = np.array([-0.5, 0.0, 0.5, 1.0, 1.2], np.float32)
        cd.conversion(cd.Declaration("coded", "t", primaries=cd.SRGB_PRIMARIES, white=(0.31, 0.33), curve="gamma",
                                     gamma=2.2)).linearize(a)
        np.testing.assert_allclose(a, [-(0.5 ** 2.2), 0, 0.5 ** 2.2, 1, 1.2 ** 2.2], rtol=1e-6)
        b = np.array([-0.5, 0.5], np.float32)
        cd.CURVE_DECODERS["sRGB"](b)
        self.assertAlmostEqual(float(b[0]), -float(b[1]))


if __name__ == "__main__":
    unittest.main()
