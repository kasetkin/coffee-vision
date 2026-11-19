"""The page's three wide-gamut previews (ticket OPS-25 D1): the canonical photo's 16-bit Rec.2020 samples, shrunk as the
sRGB preview is, then a Display P3 JPEG, a 10-bit AVIF and a 16-bit JPEG XL. Plain unittest.

    python -m unittest tests.test_preview_formats -v

Synthetic photos only: a colour inside sRGB and Display P3's own green, which lies outside sRGB.
"""
from __future__ import annotations

import io
import unittest

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import preview_formats as pf
from coffeecv.jxl_encode import encode_rec2020

INSIDE_SRGB = (150, 110, 80)        # a bean's brown, sRGB
P3_GREEN = (0, 255, 0)              # Display P3's own green: outside sRGB, inside Rec.2020


def _rec2020(srgb_half: tuple, p3_half: tuple, h: int = 64, w: int = 96) -> np.ndarray:
    """An H x W x 3 uint16 Rec.2020 photo: its left half `srgb_half` (sRGB), its right half `p3_half` (Display P3)."""
    left = ic.cms_transform(np.full((h, w // 2, 3), srgb_half, np.uint8), cn.SRGB, cn.REC2020, intent=cn.RELATIVE,
                            flags=cn.NOOPTIMIZE, outdtype=np.uint16)
    right = ic.cms_transform(np.full((h, w // 2, 3), p3_half, np.uint8), pf.DISPLAY_P3, cn.REC2020,
                             intent=cn.RELATIVE, flags=cn.NOOPTIMIZE, outdtype=np.uint16)
    return np.concatenate([left, right], axis=1)


class Shrink(unittest.TestCase):
    def test_the_size_is_the_srgb_thumbnails(self):
        for h, w in ((3000, 4000), (4000, 3000), (1024, 700), (500, 300), (1031, 1)):
            rec = np.zeros((h, w, 3), np.uint16)
            thumb = Image.new("RGB", (w, h))
            thumb.thumbnail((1024, 1024))
            self.assertEqual(pf.shrink(rec, 1024).shape, (thumb.height, thumb.width, 3), (h, w))

    def test_it_is_the_srgb_thumbnails_filter(self):
        """Per channel, as Pillow shrinks an 8-bit photo: on noise whose samples are 8-bit levels x 257, the same image
        but for the 8-bit path's rounding after its reduce and after its resize (max 1.24 levels, mean 0.30; OpenCV's
        area, cubic and linear filters differ by 29 levels or more at most and 5.0 or more on average). Over 2x the size,
        so Pillow reduces first: the path a 15 MP photo takes."""
        rng = np.random.default_rng(0)
        srgb = rng.integers(0, 256, (900, 1300, 3)).astype(np.uint8)
        thumb = Image.fromarray(srgb)
        thumb.thumbnail((256, 256))
        diff = np.abs(pf.shrink(srgb.astype(np.uint16) * 257, 256) / 257 - np.asarray(thumb))
        self.assertLessEqual(diff.max(), 1.5)
        self.assertLessEqual(diff.mean(), 0.5)

    def test_a_photo_at_or_under_the_size_is_kept(self):
        rec = _rec2020(INSIDE_SRGB, P3_GREEN)
        np.testing.assert_array_equal(pf.shrink(rec, 1024), rec)


class EncodeAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = _rec2020(INSIDE_SRGB, P3_GREEN)
        cls.files = pf.encode_all(cls.rec)

    def _middle(self, a: np.ndarray, right: bool) -> np.ndarray:
        """One pixel well inside a half, away from the JPEG's blocks across the edge."""
        h, w = a.shape[:2]
        return a[h // 2, (3 * w) // 4 if right else w // 4].astype(int)

    def test_the_three_formats_in_the_pages_order(self):
        self.assertEqual(list(self.files), ["p3_jpeg", "avif10", "jxl"])
        self.assertEqual(list(pf.MIME), list(self.files))

    def test_the_p3_jpeg_is_8_bit_with_the_display_p3_profile(self):
        im = Image.open(io.BytesIO(self.files["p3_jpeg"]))
        self.assertEqual((im.format, im.mode, im.size), ("JPEG", "RGB", (96, 64)))
        self.assertEqual(im.info.get("icc_profile"), pf.DISPLAY_P3)
        a = np.asarray(im)
        self.assertLessEqual(np.abs(self._middle(a, True) - P3_GREEN).max(), 3, "P3's green is kept")
        p3_brown = ic.cms_transform(np.array([[INSIDE_SRGB]], np.uint8), cn.SRGB, pf.DISPLAY_P3, intent=cn.RELATIVE,
                                    flags=cn.NOOPTIMIZE)[0, 0].astype(int)
        self.assertLessEqual(np.abs(self._middle(a, False) - p3_brown).max(), 3)

    def test_the_display_p3_profile_reads_srgb_as_srgb(self):
        """Display P3's curve is sRGB's: sRGB's own primaries with it are LittleCMS's sRGB, level for level."""
        same = ic.cms_profile("rgb", whitepoint=pf.D65, primaries=(0.64, 0.33, 0.30, 0.60, 0.15, 0.06),
                              transferfunction=pf.SRGB_CURVE)
        a = np.arange(256, dtype=np.uint8)[:, None, None].repeat(3, axis=2)
        np.testing.assert_array_equal(ic.cms_transform(a, cn.SRGB, same), a)

    def test_the_avif_is_10_bit_rec2020_with_the_bt709_transfer(self):
        data = self.files["avif10"]
        self.assertEqual(pf.avif_cicp(data), (9, 1, 9, True))
        a = ic.avif_decode(data)
        self.assertEqual((a.shape, a.dtype, int(a.max()) <= 1023), ((64, 96, 3), np.dtype(np.uint16), True))
        for right in (False, True):
            want = self._middle(self.rec, right) / 65535 * 1023
            self.assertLessEqual(np.abs(self._middle(a, right) - want).max(), 12, right)

    def test_the_jxl_is_the_canonical_photos_colour_encoding_at_16_bits(self):
        data = self.files["jxl"]
        h = cn.read_header(data)
        self.assertEqual((h.width, h.height, h.bits_per_sample, h.colour.primaries, h.colour.transfer),
                         (96, 64, 16, "BT2100", "BT709"))
        a = ic.jpegxl_decode(data)
        for right in (False, True):
            want = self._middle(self.rec, right)
            self.assertLessEqual(np.abs(self._middle(a, right) - want).max(), 65535 * 0.01, right)

    def test_the_jxl_is_distance_1(self):
        # quality 90 is libjxl's distance 1.0 (JxlEncoderDistanceFromQuality: 0.1 + (100 - q) * 0.09)
        rec = self.rec
        self.assertEqual(self.files["jxl"], encode_rec2020(rec.shape[0], rec.shape[1],
                                                           lambda y, x, ys, xs: rec[y:y + ys, x:x + xs],
                                                           level=pf.JXL_QUALITY, effort=pf.JXL_EFFORT, numthreads=1))
        self.assertEqual(pf.JXL_QUALITY, 90)


class AvifCicp(unittest.TestCase):
    def test_none_without_an_nclx_box(self):
        self.assertIsNone(pf.avif_cicp(b"\x00" * 64))


if __name__ == "__main__":
    unittest.main()
