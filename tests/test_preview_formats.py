"""The page's three wide-gamut previews (ticket OPS-25 D1): the canonical photo's 16-bit Rec.2020 samples, shrunk as the
sRGB preview is, then a Display P3 JPEG, a 10-bit AVIF and a 16-bit JPEG XL, the last two Rec.2020 with the sRGB
curve (D8). Plain unittest.

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


class Label(unittest.TestCase):
    """D9: each preview's format, white on a black outline, in its bottom-right corner (D10: today's sRGB one too)."""

    def test_the_names_are_the_tabs(self):
        self.assertEqual(pf.LABELS, {"srgb_jpeg": "sRGB JPEG", "p3_jpeg": "P3 JPEG", "avif10": "AVIF 10-bit",
                                     "jxl": "JPEG XL"})

    def test_it_is_white_on_black_in_the_bottom_right_corner_and_nothing_else_changes(self):
        for dtype, top in ((np.uint8, 255), (np.uint16, 65535)):
            with self.subTest(dtype=dtype.__name__):
                a = np.full((768, 1024, 3), top // 3, dtype)
                out = pf.label(a, "avif10")
                self.assertIsNot(out, a)
                changed = np.any(out != a, axis=2)
                ys, xs = np.nonzero(changed)
                self.assertGreater(ys.min(), 768 * 0.8)
                self.assertGreater(xs.min(), 1024 * 0.6)
                self.assertLess((ys.max(), xs.max()), (768, 1024))
                self.assertEqual({int(v) for v in np.unique(out[changed])}, {0, top})   # pure: grey in any space
                self.assertGreater(np.count_nonzero(out[changed] == top), 100)
                np.testing.assert_array_equal(a, np.full((768, 1024, 3), top // 3, dtype))

    def test_it_scales_with_the_image_and_skips_one_too_small(self):
        big = np.count_nonzero(np.any(pf.label(np.zeros((1024, 768, 3), np.uint8), "jxl") != 0, axis=2))
        small = np.count_nonzero(np.any(pf.label(np.zeros((512, 384, 3), np.uint8), "jxl") != 0, axis=2))
        self.assertLess(small, big / 2)
        tiny = np.zeros((20, 30, 3), np.uint8)
        np.testing.assert_array_equal(pf.label(tiny, "jxl"), tiny)


class EncodeAll(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = _rec2020(INSIDE_SRGB, P3_GREEN)
        cls.files = pf.encode_all(cls.rec)

    def _middle(self, a: np.ndarray, right: bool) -> np.ndarray:
        """One pixel well inside a half, away from the JPEG's blocks across the edge."""
        h, w = a.shape[:2]
        return a[h // 2, (3 * w) // 4 if right else w // 4].astype(int)

    def test_each_file_carries_its_own_label(self):   # D9
        rec = np.full((768, 1024, 3), 20000, np.uint16)
        files = pf.encode_all(rec)
        decoded = {"p3_jpeg": np.asarray(Image.open(io.BytesIO(files["p3_jpeg"]))).astype(float) / 255,
                   "avif10": ic.avif_decode(files["avif10"]).astype(float) / 1023,
                   "jxl": ic.jpegxl_decode(files["jxl"]).astype(float) / 65535}
        for name, a in decoded.items():
            with self.subTest(name):
                want = np.any(pf.label(rec, name) != rec, axis=2)
                white = np.all(pf.label(rec, name) == 65535, axis=2)
                self.assertGreater(a[white].mean(), 0.9)   # the label's fill, white through each encoder
                # and nothing outside its box, past the 16 pixels of the lossy codecs' ringing around its edges
                ys, xs = np.nonzero(want)
                away = np.ones(want.shape, bool)
                away[max(0, ys.min() - 16):ys.max() + 17, max(0, xs.min() - 16):xs.max() + 17] = False
                self.assertLess(np.abs(a[away] - a[0, 0]).max(), 0.02)

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

    def test_the_committed_profile_is_littlecms_display_p3_but_for_its_date(self):
        """The header's creation time (bytes 24-35) is the one difference; the profile ID (84-99) is not set."""
        built = pf.p3_profile()
        self.assertEqual(len(built), len(pf.DISPLAY_P3))
        self.assertEqual(built[:24] + built[36:], pf.DISPLAY_P3[:24] + pf.DISPLAY_P3[36:])
        self.assertEqual(pf.DISPLAY_P3[84:100], bytes(16))

    def test_the_display_p3_profile_reads_srgb_as_srgb(self):
        """Display P3's curve is sRGB's: sRGB's own primaries with it are LittleCMS's sRGB, level for level."""
        same = ic.cms_profile("rgb", whitepoint=pf.D65, primaries=(0.64, 0.33, 0.30, 0.60, 0.15, 0.06),
                              transferfunction=pf.SRGB_CURVE)
        a = np.arange(256, dtype=np.uint8)[:, None, None].repeat(3, axis=2)
        np.testing.assert_array_equal(ic.cms_transform(a, cn.SRGB, same), a)

    def _as_shown(self, a: np.ndarray, full: int, right: bool) -> np.ndarray:
        """A pixel of a file tagged Rec.2020 with the sRGB curve (D8), as a colour-managed browser shows it on an sRGB
        screen: read through a profile built here, not the module's, then into sRGB unclipped (float)."""
        x = np.linspace(0.0, 1.0, 4096)
        curve = np.round(np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4) * 65535).astype(np.uint16)
        tag = ic.cms_profile("rgb", whitepoint=pf.D65, primaries=(0.708, 0.292, 0.170, 0.797, 0.131, 0.046),
                             transferfunction=curve)
        px = (self._middle(a, right) / full).astype(np.float32)[None, None]
        return ic.cms_transform(px, tag, pf.DISPLAY_P3, intent=cn.RELATIVE, outdtype=np.float32)[0, 0] * 255

    def _want_p3(self, right: bool) -> np.ndarray:
        """The photo's pixel in Display P3 levels: its P3 green, or its brown."""
        return np.array(P3_GREEN if right else ic.cms_transform(np.array([[INSIDE_SRGB]], np.uint8), cn.SRGB,
                                                                pf.DISPLAY_P3, intent=cn.RELATIVE,
                                                                flags=cn.NOOPTIMIZE)[0, 0], float)

    def test_the_avif_is_10_bit_rec2020_with_the_srgb_transfer(self):   # D8
        data = self.files["avif10"]
        self.assertEqual(pf.avif_cicp(data), (9, 13, 9, True))
        a = ic.avif_decode(data)
        self.assertEqual((a.shape, a.dtype, int(a.max()) <= 1023), ((64, 96, 3), np.dtype(np.uint16), True))
        for right in (False, True):   # the same colours the P3 JPEG shows, so the tabs differ in nothing else
            self.assertLessEqual(np.abs(self._as_shown(a, 1023, right) - self._want_p3(right)).max(), 3, right)

    def test_the_jxl_is_16_bit_lossy_rec2020_with_the_srgb_transfer(self):   # D8
        data = self.files["jxl"]
        h = cn.read_header(data)
        self.assertEqual((h.width, h.height, h.bits_per_sample, h.xyb_encoded, h.colour.primaries, h.colour.transfer),
                         (96, 64, 16, True, "BT2100", "sRGB"))
        a = ic.jpegxl_decode(data)
        self.assertEqual(a.dtype, np.dtype(np.uint16))
        # Within 5 levels: the conversion to the sRGB curve is exact on both colours, and distance 1.0 moves P3's
        # flat, fully saturated green by 4.1 levels of red.
        for right in (False, True):
            self.assertLessEqual(np.abs(self._as_shown(a, 65535, right) - self._want_p3(right)).max(), 5, right)

    def test_the_jxl_is_distance_1(self):
        self.assertEqual(pf.JXL_DISTANCE, 1.0)


class AvifCicp(unittest.TestCase):
    def test_none_without_an_nclx_box(self):
        self.assertIsNone(pf.avif_cicp(b"\x00" * 64))


if __name__ == "__main__":
    unittest.main()
