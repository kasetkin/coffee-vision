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


class ModelArgument(unittest.TestCase):
    """Ticket ML-9 P2: a decoder not yet in NAMED_SEGMENTERS is passed as WEIGHTS:DECODER, as named_params takes
    it, so a candidate is predicted without editing segment_beans.py, a dep of every segcrop stage."""

    UNREGISTERED = "efficientvit_sam/efficientvit_sam_xl0.pt:models/seg/xl0_v3.pt"

    def run_main(self, argv: list[str]) -> list[tuple]:
        calls = []
        with mock.patch.object(seg_predict, "run_split", lambda *a: calls.append(a)), \
                mock.patch.object(seg_predict.torch, "set_num_threads"):
            seg_predict.main(argv)
        return calls

    def test_a_registered_name_writes_under_its_name_by_default(self):
        self.assertEqual(self.run_main(["xl0_v3", "--split", "test", "--threads", "1"]),
                         [("xl0_v3", "test", seg_predict.ML5_MASK_ROOT / "xl0_v3")])

    def test_out_moves_the_masks(self):
        out = REPO_ROOT / "data" / "seg_masks_ml9" / "xl0_v3"
        self.assertEqual(self.run_main(["xl0_v3", "--split", "test", "--threads", "1", "--out", str(out)]),
                         [("xl0_v3", "test", out)])

    def test_weights_and_decoder_need_out(self):
        with self.assertRaises(SystemExit):
            self.run_main([self.UNREGISTERED, "--split", "test", "--threads", "1"])
        out = REPO_ROOT / "data" / "seg_masks_ml9" / "candidate"
        self.assertEqual(self.run_main([self.UNREGISTERED, "--split", "test", "--threads", "1", "--out", str(out)]),
                         [(self.UNREGISTERED, "test", out)])

    def test_unknown_weights_are_refused(self):
        with self.assertRaises(SystemExit):
            self.run_main(["nonsense.pt:models/seg/xl0_v3.pt", "--split", "test", "--threads", "1", "--out", "x"])


if __name__ == "__main__":
    unittest.main()
