"""Ticket ML-5 P8 (D22): `seg_predict <model> --split test` stores one split of the segmenter dataset, negatives
included, in the layout the blind paired session reads (seg_review.paired_items): two models' stored test masks
pair up photo by photo. A stub segmenter stands in for the model. Plain unittest.

    python -m unittest tests.test_seg_predict_split -v
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from coffeecv import seg_predict
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.repo_files import read_csv, sha256_file
from coffeecv.seg_review import paired_items


class StubSegmenter:
    weights_sha256, decoder_sha256, pred_iou = "w" * 64, "d" * 64, 0.9

    def __init__(self, fill: float):
        self.fill = fill

    def predict_mask(self, rgb: np.ndarray) -> np.ndarray:
        mask = np.zeros(rgb.shape[:2], bool)
        mask[: int(rgb.shape[0] * self.fill)] = True
        return mask


class SplitPrediction(unittest.TestCase):
    def test_two_models_test_masks_pair_up(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            entries = []
            for i, (source, split) in enumerate((("pool", "test"), ("negative", "test"), ("pool", "train"),
                                                 ("segmenter_positive", "test"))):
                f = tmp / "dataset" / f"p{i}.png"
                f.parent.mkdir(exist_ok=True)
                Image.fromarray(np.full((40, 60, 3), 10 * i, np.uint8)).save(f)
                entries.append({"path": str(f.relative_to(REPO_ROOT)), "sha256": sha256_file(f), "source": source,
                                "split": split})
            cfg = RunConfig.from_params_yaml()
            with mock.patch.object(seg_predict, "load_seg_dataset", lambda: entries):
                got = seg_predict.split_entries("test")
                for name, fill in (("a", 0.5), ("b", 0.25)):
                    seg_predict.predict(got, StubSegmenter(fill), tmp / name, cfg)
            self.assertEqual([e["source"] for e in got], ["pool", "negative", "segmenter_positive"])
            index = read_csv(tmp / "a" / "index.csv")
            self.assertEqual([(r["source"], r["split"]) for r in index],
                             [("pool", "test"), ("negative", "test"), ("segmenter_positive", "test")])
            self.assertEqual(index[0]["area_frac"], "0.5000")
            pairs = paired_items(entries, [tmp / "a", tmp / "b"])
            self.assertEqual([p["source"] for p in pairs], ["pool", "segmenter_positive"])
            self.assertFalse(any(p["identical"] for p in pairs))
            self.assertTrue(all(m["mask"].exists() for p in pairs for m in p["masks"]))

    def test_a_split_run_names_a_registered_segmenter(self):
        with self.assertRaises(SystemExit):
            seg_predict.main(["ft_s42", "--split", "test", "--threads", "1"])


if __name__ == "__main__":
    unittest.main()
