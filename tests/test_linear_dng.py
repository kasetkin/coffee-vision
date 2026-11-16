"""coffeecv.linear_dng (ticket ML-31 D1): a DNG whose main image is LinearRaw compressed with JPEG XL, which LibRaw
0.22 cannot open, read and developed as LibRaw develops a linear DNG.

The DNGs are written here (`dng`, a little-endian TIFF from the DNG 1.7 tags), their main image a JPEG XL codestream
per strip or tile, so the reader is not checked against itself.

    python -m unittest tests.test_linear_dng
"""
from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn
from coffeecv import linear_dng as ld
from tests._tiers import real_data

# XYZ -> camera for a camera whose native space is linear Rec.2020: the inverse of Rec.2020's RGB -> XYZ (D65).
REC2020_TO_XYZ = np.array([[0.636958, 0.144617, 0.168881], [0.262700, 0.677998, 0.059302],
                           [0.000000, 0.028073, 1.060985]])
XYZ_TO_REC2020_CAMERA = np.linalg.inv(REC2020_TO_XYZ)

SHORT, LONG, RATIONAL, SRATIONAL, UNDEFINED = 3, 4, 5, 10, 7


def dng(pixels: np.ndarray, *, colour_matrix=XYZ_TO_REC2020_CAMERA, neutral=(1.0, 1.0, 1.0), orientation=1,
        black=0, white=65535, tile: int | None = None, compression=52546, photometric=34892,
        illuminants=(21,), extra: dict | None = None) -> bytes:
    """A DNG 1.7 of `pixels` (h, w, 3) uint16: LinearRaw, each strip (one) or `tile`-square tile a lossless JPEG XL
    codestream; ColorMatrix1 for CalibrationIlluminant1 (and the same matrix for a second illuminant if given)."""
    h, w, _ = pixels.shape
    if tile:
        pieces = [np.ascontiguousarray(pixels[y:y + tile, x:x + tile]) for y in range(0, h, tile)
                  for x in range(0, w, tile)]
        pieces = [np.pad(p, ((0, tile - p.shape[0]), (0, tile - p.shape[1]), (0, 0)), mode="edge") for p in pieces]
    else:
        pieces = [pixels]
    coded = [ic.jpegxl_encode(p, lossless=True) if compression == 52546 else p.tobytes() for p in pieces]

    def srational(values):
        return [(round(v * 1_000_000), 1_000_000) for v in values]

    tags = {254: (LONG, [0]), 256: (LONG, [w]), 257: (LONG, [h]), 258: (SHORT, [16, 16, 16]),
            259: (SHORT, [compression]), 262: (SHORT, [photometric]), 274: (SHORT, [orientation]),
            277: (SHORT, [3]), 284: (SHORT, [1]),
            50706: ("B", [1, 7, 0, 0]), 50707: ("B", [1, 1, 0, 0]), 50714: (LONG, [black] * 3),
            50717: (LONG, [white] * 3), 50721: (SRATIONAL, srational(np.ravel(colour_matrix))),
            50728: (RATIONAL, srational(neutral)), 50778: (SHORT, [illuminants[0]])}
    if len(illuminants) > 1:
        tags[50722] = (SRATIONAL, srational(np.ravel(colour_matrix)))
        tags[50779] = (SHORT, [illuminants[1]])
    if tile:
        tags.update({322: (LONG, [tile]), 323: (LONG, [tile]), 324: (LONG, coded), 325: (LONG, [len(c) for c in coded])})
    else:
        tags.update({273: (LONG, coded), 278: (LONG, [h]), 279: (LONG, [len(c) for c in coded])})
    tags.update(extra or {})
    return _tiff(tags)


def _tiff(tags: dict) -> bytes:
    """A little-endian TIFF with one IFD; a LONG value list of bytes is data to place and point at (offsets)."""
    entries = []
    n = len(tags)
    data_at = 8 + 2 + 12 * n + 4
    out_data = b""
    for tag in sorted(tags):
        kind, values = tags[tag]
        if kind == LONG and values and isinstance(values[0], bytes):       # strip or tile offsets: the data itself
            offsets = []
            for blob in values:
                offsets.append(data_at + len(out_data))
                out_data += blob + (b"\0" if len(blob) % 2 else b"")
            kind, values = LONG, offsets
        if kind == "B":
            body, tiff_type, count = bytes(values), 1, len(values)
        elif kind in (RATIONAL, SRATIONAL):
            body = b"".join(struct.pack("<iI" if kind == SRATIONAL else "<II", a, b) for a, b in values)
            tiff_type, count = kind, len(values)
        else:
            body = struct.pack("<" + ("H" if kind == SHORT else "I") * len(values), *values)
            tiff_type, count = kind, len(values)
        entries.append((tag, tiff_type, count, body))
    ifd = struct.pack("<H", n)
    for tag, tiff_type, count, body in entries:
        if len(body) <= 4:
            ifd += struct.pack("<HHI", tag, tiff_type, count) + body.ljust(4, b"\0")
        else:
            ifd += struct.pack("<HHII", tag, tiff_type, count, data_at + len(out_data))
            out_data += body + (b"\0" if len(body) % 2 else b"")
    return b"II*\0" + struct.pack("<I", 8) + ifd + b"\0\0\0\0" + out_data


def ramp(h: int = 40, w: int = 60) -> np.ndarray:
    """Greys from dark to white-ish left to right, a red, green and blue band at the bottom."""
    a = np.zeros((h, w, 3), np.uint16)
    a[...] = (np.linspace(2000, 50000, w)[None, :, None]).astype(np.uint16)
    a[-6:, :20] = (30000, 6000, 6000)
    a[-6:, 20:40] = (6000, 30000, 6000)
    a[-6:, 40:] = (6000, 6000, 30000)
    return a


class Recognise(unittest.TestCase):
    def test_only_a_jpeg_xl_linear_raw_dng_is_this_readers(self):
        self.assertIsNotNone(ld.parse(dng(ramp())))
        self.assertIsNone(ld.parse(dng(ramp(), compression=1)))                  # uncompressed: LibRaw's
        self.assertIsNone(ld.parse(dng(ramp(), photometric=32803)))              # CFA: LibRaw's
        self.assertIsNone(ld.parse(b"\xff\xd8\xff\xe0" + b"\0" * 64))
        self.assertIsNone(ld.parse(b"II*\0" + b"\xff" * 8))                      # a broken TIFF is not ours

    def test_its_size_before_decoding(self):
        info = ld.parse(dng(ramp(40, 60), orientation=6))
        self.assertEqual((info.width, info.height, info.orientation), (60, 40, 6))


class Develop(unittest.TestCase):
    def test_a_rec2020_camera_comes_out_as_its_light_auto_brightened(self):
        """With the camera's own space linear Rec.2020 and a neutral white balance, the developed photo is the stored
        light times one factor: LibRaw's auto-bright, which puts the 99th percentile of the brightest channel at
        white (1% of the pixels clip)."""
        a = ramp()
        out = ld.develop(ld.parse(dng(a)))
        self.assertEqual((out.shape, out.dtype), (a.shape, np.dtype(np.uint16)))
        lit = (out < 65535).all(axis=2) & (a > 4000).all(axis=2)
        factor = out[lit].astype(float) / a[lit]
        self.assertLess(factor.std() / factor.mean(), 0.002)
        white = max(np.percentile(a[..., c], 99) for c in range(3))              # dcraw's, in its 8192 bins
        self.assertAlmostEqual(factor.mean(), 65535 / white, delta=0.01 * 65535 / white)
        top = np.sort(out.max(axis=2).ravel())
        self.assertEqual(top[int(0.995 * top.size)], 65535)                       # about 1% at white
        self.assertLess(top[int(0.95 * top.size)], 65535)

    def test_white_balance_makes_the_camera_neutral_grey(self):
        neutral = np.array([0.5, 1.0, 0.25])
        a = np.zeros((20, 30, 3), np.uint16)
        a[...] = (neutral * 20000).astype(np.uint16)
        a[:, :3] = (neutral * 40000).astype(np.uint16)                            # a brighter strip as the 1%
        out = ld.develop(ld.parse(dng(a, neutral=tuple(neutral)))).astype(float)
        grey = out[:, 10:]
        self.assertLess(np.abs(grey - grey.mean(axis=2, keepdims=True)).max(), 2)

    def test_black_and_white_levels(self):
        a = ramp()
        plain = ld.develop(ld.parse(dng(a)))
        raised = ld.develop(ld.parse(dng((a.astype(np.uint32) // 2 + 1000).astype(np.uint16), black=1000,
                                          white=1000 + 65535 // 2)))
        self.assertLess(np.abs(plain.astype(int) - raised).max(), 4 * 65535 / 50000 + 2)

    def test_the_colour_matrix_of_d65_is_used(self):
        """Of two illuminants, D65's (21) matrix develops the photo, as for the Fold8 Ultra's (D65 and A)."""
        a = ramp()
        one = ld.develop(ld.parse(dng(a)))
        info = ld.parse(dng(a, illuminants=(17, 21)))
        np.testing.assert_allclose(info.colour_matrix, XYZ_TO_REC2020_CAMERA, atol=1e-5)
        np.testing.assert_array_equal(ld.develop(info), one)

    def test_tiles_read_as_the_strip_does(self):
        a = ramp(40, 60)
        np.testing.assert_array_equal(ld.develop(ld.parse(dng(a, tile=32))), ld.develop(ld.parse(dng(a))))

    def test_an_unsupported_layout_is_refused(self):
        with self.assertRaises(ValueError):
            ld.develop(ld.parse(dng(ramp(), extra={50712: (SHORT, list(range(256)))})))   # a LinearizationTable


class Canonical(unittest.TestCase):
    def test_canonicalization_reads_it_turned(self):
        a = ramp(40, 60)
        src = cn.decode(dng(a, orientation=6), "x.dng")
        self.assertEqual((src.pixels.shape, src.full, src.profile), ((60, 40, 3), 65535.0, cn.LINEAR_REC2020))
        expected = cn.orient(ld.develop(ld.parse(dng(a))), 6)
        np.testing.assert_array_equal(src.pixels, expected)
        self.assertTrue(cn.is_canonical(cn.canonicalize(dng(a), "x.dng")))

    def test_the_fronts_ceiling_is_checked_before_decoding(self):
        with self.assertRaises(cn.TooLarge):
            cn.decode(dng(ramp(40, 60)), "x.dng", max_pixels=1000)

    def test_a_dng_nobody_reads_is_still_unreadable(self):
        with self.assertRaises(cn.Unreadable):
            cn.decode(dng(ramp(), photometric=32803, compression=52546), "x.dng")


class Intake(unittest.TestCase):
    """The strip's pixel check and the duplicate check read the photo through photo_io and strip_metadata."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "x.dng"
        self.path.write_bytes(dng(ramp(40, 60), orientation=6))

    def tearDown(self):
        self.dir.cleanup()

    def test_photo_io_reads_it_as_8_bit_srgb_turned(self):
        from coffeecv import photo_io
        rgb = photo_io.load_rgb_image(self.path)
        self.assertEqual((rgb.shape, rgb.dtype), ((60, 40, 3), np.dtype(np.uint8)))
        self.assertGreater(int(rgb[:, 0].mean()), int(rgb[:, -1].mean()) - 255)   # read, not black
        self.assertGreater(rgb.max(), 200)

    def test_the_strips_digest_and_memory_estimate(self):
        from coffeecv import strip_metadata as sm
        self.assertEqual(len(sm.pixel_digest(self.path)), 64)
        self.assertEqual(sm.estimate_decode_bytes(self.path), 40 * 60 * sm.DECODE_BYTES_PER_PIXEL[".dng"])


class Fold8(unittest.TestCase):
    @real_data
    def test_the_owners_fold8_ultra_dngs(self):
        """ML-31's five DNGs: DNG 1.7, JPEG XL LinearRaw (4080 x 3060 or 4000 x 3000), orientation 6."""
        root = Path(__file__).resolve().parents[1]
        paths = sorted((root / "dataset").glob("random_date_raccoon/*/20261115_*_Galaxy_Z_Fold8_Ultra_*.dng")) or \
            sorted((root / "tmp" / "ml31_intake" / "held").glob("*.dng"))
        if not paths:
            self.skipTest("the Fold8 Ultra DNGs are not here")
        data = paths[0].read_bytes()
        info = ld.parse(data)
        self.assertEqual(info.orientation, 6)
        src = cn.decode(data, paths[0].name)
        self.assertEqual(src.pixels.shape, (info.width, info.height, 3))
        self.assertGreater(float(src.pixels.mean()), 1000)


if __name__ == "__main__":
    unittest.main()
