"""The seg_eval stage's statistics (ticket ML-2 P2, plan §5). Plain unittest. The heuristic box lookup's
test went with the lookup in ticket ML-3 P2b (data/cropped retired).

    python -m unittest discover -s tests -p 'test_seg_eval.py'
"""
from __future__ import annotations

import unittest

import numpy as np

from coffeecv.seg_eval import bootstrap_rate, quantiles, wilson_rate
from coffeecv.segment_beans import SegParams, mask_and_crop


class Statistics(unittest.TestCase):
    def test_bootstrap_brackets_the_rate_and_is_seeded(self):
        passed = [True] * 30 + [False] * 10
        a, b = bootstrap_rate(passed), bootstrap_rate(passed)
        self.assertEqual(a, b)
        self.assertEqual((a["k"], a["n"], a["rate"]), (30, 40, 0.75))
        self.assertLess(a["boot95"][0], 0.75)
        self.assertGreater(a["boot95"][1], 0.75)

    def test_degenerate_inputs(self):
        self.assertEqual(bootstrap_rate([True] * 5)["boot95"], [1.0, 1.0])
        self.assertIsNone(bootstrap_rate([])["rate"])
        self.assertIsNone(wilson_rate(0, 0)["rate"])
        self.assertIsNone(quantiles([None, None]))

    def test_quantiles_skip_none(self):
        q = quantiles([None, 0.0, 1.0])
        self.assertEqual((q["n"], q["q00"], q["q50"], q["q100"]), (2, 0.0, 0.5, 1.0))


class GeometryOnlyCrop(unittest.TestCase):
    def test_broadcast_blank_gives_the_same_info_as_a_photo(self):
        rng = np.random.default_rng(0)
        mask = np.zeros((60, 80), bool)
        mask[10:50, 20:70] = True
        mask[rng.random(mask.shape) < 0.05] = False
        p = SegParams(mask_select="multi3")
        photo = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
        blank = np.broadcast_to(np.zeros(3, np.uint8), (60, 80, 3))
        self.assertEqual(mask_and_crop(photo, mask, p).info, mask_and_crop(blank, mask, p).info)


if __name__ == "__main__":
    unittest.main()
