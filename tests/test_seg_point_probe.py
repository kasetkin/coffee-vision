"""The point probe (ticket ML-5 P6, D19, D23): simulated corrective clicks and the summary table. Plain unittest.

    python -m unittest tests.test_seg_point_probe -v
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from coffeecv import seg_point_probe as probe


class NextClick(unittest.TestCase):
    def test_include_at_the_deepest_point_of_a_missed_region(self):
        ref = np.zeros((101, 201), bool)
        ref[20:61, 120:161] = True                     # 41 x 41, centre (140, 40)
        kind, (fx, fy) = probe.next_click(np.zeros_like(ref), ref)
        self.assertEqual(kind, "inc")
        self.assertAlmostEqual(fx * 200, 140, delta=1)
        self.assertAlmostEqual(fy * 100, 40, delta=1)

    def test_the_largest_error_region_wins_and_a_false_one_is_excluded(self):
        ref = np.zeros((100, 100), bool)
        ref[10:20, 10:20] = True                       # 100 px missed
        pred = ref.copy()
        pred[10:20, 10:20] = False
        pred[50:90, 50:90] = True                      # 1,600 px false
        kind, (fx, fy) = probe.next_click(pred, ref)
        self.assertEqual(kind, "exc")
        self.assertTrue(50 <= fx * 99 < 90 and 50 <= fy * 99 < 90)

    def test_no_click_when_the_masks_agree(self):
        m = np.ones((30, 40), bool)
        self.assertIsNone(probe.next_click(m, m))


class Summary(unittest.TestCase):
    def test_shares_and_medians_per_set_model_and_output(self):
        def row(replay, empty, *curve):
            return {"set": "A", "model": "m", "output": "multi3", "replay_iou": replay, "replay_empty": empty,
                    **{f"click{i}": v for i, v in enumerate(curve)}}
        rows = [row(0.95, 0, 0.0, 0.5, 0.91, 0.99), row(0.0, 1, 0.92, 0.95, 0.5, 0.95),
                row(0.9, 0, 0.1, 0.2, 0.3, 0.4), row(0.5, 0, 0.0, 0.0, 0.95, 0.97)]
        line = probe.summary(rows).splitlines()[1].split()
        # n 4; replay >= .9: 2 of 4; empty 1 of 4; clicks >= .9: 1, 1, 2, 3 of 4; medians of each click
        self.assertEqual(line, ["A", "m", "multi3", "4", "50%", "25%", "|", "25%", "25%", "50%", "75%", "|",
                                "0.050", "0.350", "0.705", "0.960"])


class ReferenceSet(unittest.TestCase):
    def test_a_fenced_label_is_not_a_reference(self):
        """Ticket ML-9 P1: a fenced mask was cut to the owner's rectangle, so no model redraws it from its points."""
        rows = [{"path": f"dataset/p{k}.jpg", "mask": f"m{k}.png", "round": r, "include": [[0.5, 0.5]],
                 "exclude": [], "fence": fence}
                for k, (r, fence) in enumerate([(2, None), (2, [0.1, 0.1, 0.9, 0.9]), (1, None), (3, None)])]
        with mock.patch("coffeecv.seg_review.accepted_labels", lambda session: rows):
            refs = probe.reference_items("ml5:pass9")
        self.assertEqual([r.path for r in refs], ["dataset/p0.jpg", "dataset/p3.jpg"])

    def test_a_painted_label_is_not_a_reference(self):
        """Ticket ML-11 R3: a painted mask has pixels the owner brushed, which no model draws from its points."""
        stroke = {"mode": "erase", "r": 12.0, "path": [[0.1, 0.1], [0.2, 0.1]]}
        rows = [{"path": f"dataset/p{k}.jpg", "mask": f"m{k}.png", "round": 2, "include": [[0.5, 0.5]],
                 "exclude": [], "fence": None, "strokes": strokes} for k, strokes in enumerate([[], [stroke]])]
        rows.append({k: v for k, v in rows[0].items() if k != "strokes"} | {"path": "dataset/p2.jpg"})  # before ML-11
        with mock.patch("coffeecv.seg_review.accepted_labels", lambda session: rows):
            refs = probe.reference_items("ml5:pass9")
        self.assertEqual([r.path for r in refs], ["dataset/p0.jpg", "dataset/p2.jpg"])


if __name__ == "__main__":
    unittest.main()
