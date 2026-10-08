"""The serving guard of ADR 0016 (ticket ML-5 D9): `infer.classify_one`, which the web app wraps, runs the
linear probe after every classification, on the segmenter's crop and on the whole frame (the D18 fallback
and the user's full-photo override alike), and refuses at a fixed 0.5. Plain unittest.

    python -m unittest tests.test_ood_guard_serving -v

No weights and no real photo: the segmenter is a stub that returns a chosen mask (a pile in the middle, or
nothing), the bean pitch is fixed, and the classifier is a three-number pooling layer under a linear head,
so the probe's score is set by its bias alone.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from coffeecv import infer
from coffeecv.config import RunConfig
from coffeecv.segment_beans import seg_params

CFG = replace(RunConfig.from_params_yaml(), crop_method="segment")
H, W = 900, 1200
REF = {"classes": {"a": {"centroid": [0.0, 0.0, 0.0], "spread": 1.0},
                   "b": {"centroid": [1.0, 1.0, 1.0], "spread": 1.0}}}


class TinyClassifier(torch.nn.Module):
    """Mean colour of the patch -> two-class head; the head's input is the 3-d embedding the probe reads."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.head(x.mean(dim=(2, 3)))


class StubSegmenter:
    """What `segment_and_crop` needs of a BeanSegmenter: a mask for the photo, the params, timings."""

    def __init__(self, mask: np.ndarray):
        self.mask, self.p, self.timing_ms = mask, seg_params(CFG), {}

    def predict_mask(self, rgb):
        return self.mask


def probe_scoring(score: float, threshold_in_file: float = 0.5) -> dict:
    """A probe whose P(not beans) is `score` on every patch: zero weights, bias logit(score)."""
    return {"mu": [0.0] * 3, "sd": [1.0] * 3, "w": [0.0, 0.0, 0.0, float(np.log(score / (1 - score)))],
            "threshold": threshold_in_file}


PILE = np.zeros((H, W), bool)
PILE[100:800, 150:1050] = True
EMPTY = np.zeros((H, W), bool)


class TestProbeRunsOnCropAndWholeFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.photo = Path(tmp.name) / "photo.png"
        rng = np.random.default_rng(0)
        Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(cls.photo)
        torch.manual_seed(0)
        cls.model = TinyClassifier().eval()

    def classify(self, mask: np.ndarray, probe: dict, skip_crop: bool = False) -> dict:
        with mock.patch.object(infer, "segmenter_for", return_value=StubSegmenter(mask)), \
                mock.patch.object(infer, "estimate_bean_pitch", return_value=60.0):
            return infer.classify_one(self.photo, CFG, ["a", "b"], {"a": "A", "b": "B"}, self.model,
                                      self.model.head, REF, n_patches=8, tta=False, skip_crop=skip_crop,
                                      probe=probe)

    CASES = {"crop": (PILE, False), "D18 fallback": (EMPTY, False), "full-photo override": (PILE, True)}

    def test_the_branches_are_the_ones_named(self):
        self.assertFalse(self.classify(PILE, probe_scoring(0.01))["seg_fallback"])
        self.assertTrue(self.classify(EMPTY, probe_scoring(0.01))["seg_fallback"])
        self.assertIsNone(self.classify(PILE, probe_scoring(0.01), skip_crop=True)["seg_fallback"])

    def test_an_empty_mask_is_told_from_a_tiny_one(self):
        """ood_eval reports the empty-mask rate beside the probe (D9): one pixel rounds mask_area_frac to 0
        but is not an empty mask."""
        speck = EMPTY.copy()
        speck[450, 600] = True
        got = {name: self.classify(m, probe_scoring(0.01)) for name, m in (("pile", PILE), ("speck", speck),
                                                                           ("empty", EMPTY))}
        self.assertEqual({k: (e["seg_mask_empty"], e["seg_fallback"]) for k, e in got.items()},
                         {"pile": (False, False), "speck": (False, True), "empty": (True, True)})
        self.assertEqual(got["speck"]["mask_area_frac"], 0.0)
        self.assertIsNone(self.classify(PILE, probe_scoring(0.01), skip_crop=True)["seg_mask_empty"])

    def test_probe_refuses_a_not_beans_score_on_every_branch(self):
        for name, (mask, skip) in self.CASES.items():
            with self.subTest(name):
                entry = self.classify(mask, probe_scoring(0.99), skip_crop=skip)
                self.assertEqual(entry["ood"]["method"], "linear_probe")
                self.assertAlmostEqual(entry["ood"]["probe"], 0.99, places=6)
                self.assertEqual(entry["verdict"], "REFUSED (out of distribution)")

    def test_probe_passes_a_beans_score_on_every_branch(self):
        for name, (mask, skip) in self.CASES.items():
            with self.subTest(name):
                entry = self.classify(mask, probe_scoring(0.01), skip_crop=skip)
                self.assertEqual(entry["ood"]["method"], "linear_probe")
                self.assertEqual(entry["verdict"], "predicted")

    def test_the_cutoff_is_a_fixed_half_whatever_the_file_says(self):
        refused = self.classify(PILE, probe_scoring(0.6, threshold_in_file=0.9))
        self.assertEqual(refused["verdict"], "REFUSED (out of distribution)")
        self.assertEqual(refused["ood"]["threshold"], 0.5)
        passed = self.classify(PILE, probe_scoring(0.4, threshold_in_file=0.2))
        self.assertEqual(passed["verdict"], "predicted")


if __name__ == "__main__":
    unittest.main()
