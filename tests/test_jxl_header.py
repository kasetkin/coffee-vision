"""coffeecv.jxl_header: a JPEG XL file's size, depth, extra channels, container boxes and colour encoding, read from
files imagecodecs writes here (ISO/IEC 18181-1; ticket ML-19 finding 5). libjxl's own reading agreed on 31 files
(analysis/ml19/p1/README.md); these hold the reader to it.

    python -m unittest discover -s tests -p 'test_jxl_header.py'
"""
from __future__ import annotations

import unittest

import imagecodecs as ic
import numpy as np

from coffeecv.jxl_header import HeaderError, read_header


def jxl(shape=(37, 53, 3), dtype=np.uint16, **kw) -> bytes:
    a = np.random.default_rng(1).random(shape)
    a = a.astype(np.float32) if dtype == np.float32 else (a * np.iinfo(dtype).max).astype(dtype)
    return ic.jpegxl_encode(a, level=None if kw.get("lossless") else 95, **kw)


class ReadHeader(unittest.TestCase):
    def test_d7s_encoding(self):
        h = read_header(jxl(primaries="BT2100", transfer="BT709"))
        c = h.colour
        self.assertEqual((c.want_icc, c.colour_space, c.white_point, c.primaries, c.transfer, c.rendering_intent),
                         (False, "RGB", "D65", "BT2100", "BT709", "Relative"))
        self.assertEqual((h.width, h.height, h.bits_per_sample, h.float_sample, h.xyb_encoded, h.orientation),
                         (53, 37, 16, False, True, 1))
        self.assertEqual((h.num_extra_channels, h.boxes, h.have_preview, h.have_animation), (0, [], False, False))

    def test_every_primaries_and_transfer(self):
        names = {"SRGB": "sRGB", "BT2100": "BT2100", "P3": "P3", "BT709": "BT709", "LINEAR": "Linear", "PQ": "PQ",
                 "HLG": "HLG", "DCI": "DCI"}
        for p in ("SRGB", "BT2100", "P3"):
            for t in ("BT709", "SRGB", "LINEAR", "PQ", "HLG", "DCI"):
                with self.subTest(primaries=p, transfer=t):
                    c = read_header(jxl(primaries=p, transfer=t)).colour
                    self.assertEqual((c.primaries, c.transfer), (names[p], names[t]))

    def test_defaults_8_bit_and_float(self):
        c8 = read_header(jxl(dtype=np.uint8))
        self.assertEqual((c8.bits_per_sample, c8.colour.primaries, c8.colour.transfer), (8, "sRGB", "sRGB"))
        f = read_header(jxl(dtype=np.float32))
        self.assertEqual((f.bits_per_sample, f.float_sample, f.colour.transfer), (32, True, "Linear"))

    def test_grey_lossless_alpha_and_container(self):
        self.assertEqual(read_header(jxl(shape=(37, 53), dtype=np.uint8)).colour.colour_space, "Grey")
        self.assertFalse(read_header(jxl(dtype=np.uint8, lossless=True)).xyb_encoded)
        a = read_header(jxl(shape=(37, 53, 4), primaries="BT2100", transfer="BT709"))
        self.assertEqual((a.num_extra_channels, a.extra_channel_types), (1, ["Alpha"]))
        self.assertEqual(a.colour.primaries, "BT2100")   # read past the extra channel's info
        boxes = read_header(jxl(primaries="BT2100", transfer="BT709", usecontainer=True)).boxes
        self.assertEqual(boxes[0], "JXL ")
        self.assertIn("jxlc", boxes)

    def test_sizes_small_and_large(self):
        for shape in ((8, 8, 3), (64, 256, 3), (1, 1, 3), (3001, 17, 3)):
            with self.subTest(shape=shape):
                h = read_header(jxl(shape=shape, dtype=np.uint8))
                self.assertEqual((h.height, h.width), shape[:2])

    def test_not_jpeg_xl(self):
        for data in (b"\xff\xd8\xff\xe0 a JPEG", b"", b"\xff\x0a"):
            with self.subTest(data=data[:6]), self.assertRaises(HeaderError):
                read_header(data)


if __name__ == "__main__":
    unittest.main()
