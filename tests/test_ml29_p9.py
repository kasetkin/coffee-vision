"""ML-29 P9 (D12 step 2): analysis/ml29/p9_models.py, today's models on P8's scratch photos: where each side's crop
is, the masks' overlap in the canonical photo's frame, and how an answer moves.

    python -m unittest tests.test_ml29_p9
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

from coffeecv.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "analysis" / "ml29"))
_spec = importlib.util.spec_from_file_location("p9_models", REPO_ROOT / "analysis" / "ml29" / "p9_models.py")
p9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p9)


def answer(top1: str | None, verdict: str = "predicted") -> dict:
    return {"top1": top1, "verdict": verdict}


class Crops(unittest.TestCase):
    def test_todays_crop_is_the_segcrop_stages_and_made_here_for_a_folder_it_does_not_crop(self):
        out = Path("/scratch")
        sessions = {"random_date_raccoon"}
        labelled = p9.p8.DATASET / "random_date_raccoon" / "class_003__Colombia" / "PXL_1.jpg"
        self.assertEqual(p9.today_crop(labelled, sessions, out),
                         REPO_ROOT / "data" / "segcropped" / "random_date_raccoon" / "class_003__Colombia"
                         / "PXL_1__cropped.jxl")
        negative = p9.p8.DATASET / "ood_negatives" / "PXL_2.jpg"
        self.assertEqual(p9.today_crop(negative, sessions, out), out / "crops" / "today" / "ood_negatives"
                         / "PXL_2__cropped.jxl")
        self.assertEqual(p9.scratch_crop(labelled, out), out / "crops" / "scratch" / "random_date_raccoon"
                         / "class_003__Colombia" / "PXL_1__cropped.jxl")


class Overlap(unittest.TestCase):
    def test_a_mask_is_placed_at_its_box_and_a_fallback_covers_the_frame(self):
        m = np.array([[True, False], [True, True]])
        placed = p9.frame_mask(m, [3, 1, 2, 2], (4, 6))
        self.assertEqual(placed.sum(), 3)
        self.assertTrue(placed[1, 3] and placed[2, 3] and placed[2, 4] and not placed[1, 4])
        self.assertTrue(p9.frame_mask(None, None, (4, 6)).all())

    def test_the_frame_is_a_fallbacks_crop_or_one_both_boxes_fit_in(self):
        m = np.ones((2, 2), bool)
        self.assertEqual(p9.frame_shape([(m, [3, 1, 2, 2], (2, 2)), (m, [0, 4, 3, 1], (1, 3))]), (5, 5))
        self.assertEqual(p9.frame_shape([(m, [3, 1, 2, 2], (2, 2)), (None, None, (40, 60))]), (40, 60))

    def test_iou(self):
        a = np.zeros((4, 4), bool)
        b = np.zeros((4, 4), bool)
        a[:2] = True
        b[1:3] = True
        self.assertAlmostEqual(p9.iou(a, b), 4 / 12)
        self.assertEqual(p9.iou(a, a), 1.0)
        self.assertEqual(p9.iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)), 1.0)


class Flips(unittest.TestCase):
    def test_how_an_answer_moves_against_the_label(self):
        self.assertEqual(p9.flip(answer("BR"), answer("BR"), "BR"), "same")
        self.assertEqual(p9.flip(answer("BR"), answer("CO"), "BR"), "right->wrong")
        self.assertEqual(p9.flip(answer("CO"), answer("BR"), "BR"), "wrong->right")
        self.assertEqual(p9.flip(answer("CO"), answer("ET"), "BR"), "wrong->wrong")
        self.assertEqual(p9.flip(answer("CO"), answer("ET"), None), "changed")

    def test_a_refusal_on_either_side_is_its_own_kind(self):
        refused = answer("BR", "REFUSED (out of distribution)")
        self.assertEqual(p9.flip(answer("BR"), refused, None), "now refused")
        self.assertEqual(p9.flip(refused, answer("BR"), None), "no longer refused")
        self.assertEqual(p9.flip(refused, refused, None), "refused both")


if __name__ == "__main__":
    unittest.main()
