"""Planted defects (ticket ML-2 P1, D14, plan §4.3): the consequential metric at its boundaries, and each type
straddling it. Synthetic masks only. Plain unittest.

    python -m unittest discover -s tests -p 'test_seg_defects.py'
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from coffeecv import seg_defects as sd
from coffeecv.segment_beans import mask_sha256


def tray(h: int = 400, w: int = 300) -> np.ndarray:
    """A pile in the middle of the frame, with room on every side."""
    m = np.zeros((h, w), bool)
    m[100:300, 80:220] = True
    return m


class Consequence(unittest.TestCase):
    def test_box_iou(self):
        self.assertAlmostEqual(sd.box_iou([0, 0, 10, 10], [0, 0, 9, 10]), 0.9)
        self.assertEqual(sd.box_iou([0, 0, 10, 10], [20, 20, 5, 5]), 0.0)

    def test_crop_iou_boundary_is_inclusive(self):
        m = tray()
        for box, want in (([0, 0, 9, 10], True), ([0, 0, 9.1, 10], False)):
            with mock.patch.object(sd, "d4_box", side_effect=[[0, 0, 10, 10], box]):
                self.assertEqual(sd.consequence(m, m)[2], want, box)

    def test_added_area_boundary_is_inclusive(self):
        """A hole inside the pile counts as non-bean area when the defect fills it: 5% of the bean region is
        consequential, one pixel less is not (the crop barely moves)."""
        n_full = 200 * 140
        for hole, want in ((None, True), (-1, False)):
            clean = tray()
            k = int(np.ceil(0.05 * n_full / 1.05))           # smallest hole with k / (n_full - k) >= 5%
            if hole == -1:
                k -= 1
            ys, xs = np.divmod(np.arange(k), 40)
            clean[180 + ys, 130 + xs] = False
            iou, added, cons = sd.consequence(clean, tray())
            self.assertGreater(iou, sd.CROP_IOU_MAX)
            self.assertEqual(cons, want, (k, added))

    def test_empty_defect_is_consequential(self):
        m = tray()
        self.assertEqual(sd.consequence(m, np.zeros_like(m))[:1], (0.0,))


class Types(unittest.TestCase):
    def test_every_type_straddles_the_boundary(self):
        rng = np.random.default_rng(0)
        p = sd.Planter(tray(), rng)
        for kind in sd.TYPES:
            got = sd.variants(p, kind, 1, 1, rng)
            self.assertEqual([v["consequential"] for v in got], [True, False], kind)
            for v in got:
                self.assertFalse(np.array_equal(v["mask"], p.m), kind)

    def test_dilate_and_band_need_room_outside_the_pile(self):
        full = np.ones((200, 300), bool)
        p = sd.Planter(full, np.random.default_rng(0))
        self.assertEqual(sd.variants(p, "dilate", 1, 1, np.random.default_rng(0)), [])
        self.assertEqual(sd.variants(p, "add_band", 1, 1, np.random.default_rng(0)), [])

    def test_erode_works_from_the_frame_edge(self):
        full = np.ones((200, 300), bool)
        e = sd.Planter(full, np.random.default_rng(0))("erode", 0.2)
        self.assertFalse(e[0].any() or e[:, 0].any())
        self.assertTrue(e[100, 150])

    def test_same_seed_same_masks(self):
        shas = []
        for _ in range(2):
            rng = np.random.default_rng(7)
            p = sd.Planter(tray(), rng)
            shas.append([mask_sha256(v["mask"]) for k in sd.TYPES for v in sd.variants(p, k, 1, 1, rng)])
        self.assertEqual(shas[0], shas[1])

    def test_add_band_joins_the_pile_outside_it(self):
        p = sd.Planter(tray(), np.random.default_rng(3))
        d = p("add_band", 0.5)
        added = d & ~p.m
        self.assertTrue(added.any())
        x0, y0, x1, y1 = p.bbox
        outside = added.copy()
        outside[y0:y1 + 1, x0:x1 + 1] = False
        self.assertTrue(outside.any())                      # reaches past the pile's bounding box


if __name__ == "__main__":
    unittest.main()
