"""coffeecv.label_geometry (ML-19 D28): a label's points, fence, strokes and mask turn as the photo's pixels turn, for
all eight EXIF orientations, and a mask turned back is the stored one to the pixel.

    python -m unittest discover -s tests -p 'test_label_geometry.py'
"""
from __future__ import annotations

import unittest

import numpy as np

from coffeecv import canonical as cn
from coffeecv import label_geometry as lg
from coffeecv.seg_review import fence_mask, paint_mask


def pixel(f: tuple[float, float], shape) -> tuple[int, int]:
    """A fraction point's pixel (row, col), as seg_review maps it."""
    h, w = shape[:2]
    return round(f[1] * (h - 1)), round(f[0] * (w - 1))


class Turn(unittest.TestCase):
    def setUp(self):
        self.photo = np.arange(7 * 11).reshape(7, 11)    # every pixel its own value

    def test_a_point_lands_on_the_same_pixel_as_the_photo_turns(self):
        for o in range(1, 9):
            up = cn.orient(self.photo, o)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (0.3, 0.5), (0.9, 1.0), (0.5, 0.5)):
                with self.subTest(orientation=o, point=(x, y)):
                    self.assertEqual(self.photo[pixel((x, y), self.photo.shape)],
                                     up[pixel(lg.turn_point(x, y, o), up.shape)])

    def test_a_mask_turned_back_is_the_stored_one(self):
        m = np.random.default_rng(0).random((7, 11)) > 0.5
        for o in range(1, 9):
            with self.subTest(orientation=o):
                c = lg.canonical_mask(m, o)
                self.assertEqual(c.shape, cn.orient(m, o).shape)
                np.testing.assert_array_equal(lg.stored_mask(c, o), m)

    def test_fences_and_strokes_draw_the_same_pixels_turned(self):
        m = np.zeros((40, 64), bool)
        fence = [0.1, 0.2, 0.6, 0.9]
        strokes = [{"mode": "fill", "r": 3.0, "path": [[0.2, 0.3], [0.7, 0.4]]},
                   {"mode": "erase", "r": 2.0, "path": [[0.5, 0.35]]}]
        drawn = fence_mask(paint_mask(m, strokes), fence)
        for o in range(1, 9):
            with self.subTest(orientation=o):
                turned = fence_mask(paint_mask(cn.orient(m, o), lg.turn_strokes(strokes, o)), lg.turn_fence(fence, o))
                np.testing.assert_array_equal(turned, cn.orient(drawn, o))

    def test_strokes_keep_their_other_fields_and_scale_their_radius(self):
        s = lg.turn_strokes([{"mode": "fill", "r": 10.0, "path": [[0.1, 0.2]]}], 6, scale=0.5)
        self.assertEqual(s, [{"mode": "fill", "r": 5.0, "path": [[0.8, 0.1]]}])


class Cap(unittest.TestCase):
    def test_a_mask_over_60_mp_is_capped_like_its_photo(self):
        m = np.zeros((6_001, 10_000), bool)
        m[1000:3000, 2000:7000] = True
        c = lg.cap_mask(m)
        self.assertEqual(c.shape[::-1], cn.capped_size(10_000, 6_001))
        self.assertAlmostEqual(c.mean(), m.mean(), places=3)
        at_60 = m[:6_000]
        self.assertIs(lg.cap_mask(at_60), at_60)            # 60 MP exactly: kept


if __name__ == "__main__":
    unittest.main()
