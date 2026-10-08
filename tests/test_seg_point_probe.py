"""The point probe (ticket ML-5 P6, D19, D23): simulated corrective clicks, the summary table, and a rerun of
analysis/ml5_point_probe/point_probe.py's rows on real photos and masks. Plain unittest.

    python -m unittest tests.test_seg_point_probe -v
"""
from __future__ import annotations

import csv
import unittest

import numpy as np

from coffeecv import seg_point_probe as probe
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.config import REPO_ROOT
from coffeecv.sam_loader import L0_WEIGHTS

from tests._tiers import real_data

ANALYSIS_ROWS = REPO_ROOT / "analysis" / "ml5_point_probe" / "rows.csv"


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


@real_data
@unittest.skipUnless((MODELS_PRETRAINED / L0_WEIGHTS).exists() and (REPO_ROOT / "models/seg/ft_s123.pt").exists()
                     and (REPO_ROOT / "data/seg_masks/base_points/index.csv").exists()
                     and (REPO_ROOT / "data/seg_labels/labels.csv").exists(),
                     "L0, ft_s123, data/seg_masks/base_points or data/seg_labels missing (dvc checkout)")
class AgainstTheAnalysis(unittest.TestCase):
    """A rewrite is checked against the original: the first photo of each reference set, both models and both
    outputs, give the analysis's rows (made at 4 threads in the devcontainer)."""

    def test_rows_match_analysis_rows(self):
        items = [next(i for i in probe.reference_items(refs) if i.set == refs)
                 for refs in ("ml2_base_points", "ml2_labels")]
        rows = probe.probe(["pretrained_l0", "ft_s123"], ["single", "multi3"], items)
        old = {(probe.resolve(r["path"]), r["model"], r["output"]): r for r in csv.DictReader(ANALYSIS_ROWS.read_text().splitlines())}
        self.assertEqual(len(rows), 8)
        for r in rows:
            o = old[(r["path"], r["model"], r["output"])]
            for k in ("n_points", "replay_empty"):
                self.assertEqual(int(r[k]), int(o[k]), (r["path"], k))
            for k in ("replay_iou", "click0", "click1", "click2", "click3"):
                self.assertAlmostEqual(float(r[k]), float(o[k]), places=9, msg=(r["path"], r["model"], k))


if __name__ == "__main__":
    unittest.main()
