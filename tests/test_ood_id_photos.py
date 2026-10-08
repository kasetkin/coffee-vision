"""ood_eval.id_photos (ticket ML-3): the OOD probe's positives are the checkpoint's own val/test photos, so
id_photos must return exactly the photos MultiPhotoPatchDataset puts in that split -- for a country pooled
from two class folders too -- and refuse a class with no photos instead of dropping it. Plain unittest, on
synthetic crops.

    python -m unittest discover -s tests -p 'test_ood_id_photos.py'
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from coffeecv import ood_eval
from coffeecv.class_list import load_classes
from coffeecv.config import RunConfig
from coffeecv.dataset import MultiPhotoPatchDataset, resolve_captures

# Country B is folders 002 and 003; 004 (C) is declared and has no photos anywhere.
CLASSES = "001;A,One;\n002;B,Two;\n003;B,Three;x\n"
COUNTS = {"capA": {"001": 14, "002": 12, "003": 8}, "capB": {"001": 6, "002": 5}}


class TestIdPhotos(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        cls.raw = {}
        for cap, counts in COUNTS.items():
            for fid, n in counts.items():
                d = cls.tmp / cap / f"class_{fid}__X"
                d.mkdir(parents=True)
                for i in range(n):
                    stem = f"{cap}_{fid}_{i:03d}"
                    Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
                        d / f"{stem}__cropped.jpg")
                    cls.raw[stem] = Path("dataset") / cap / f"{stem}.jpg"      # stands in for the raw photo
        (cls.tmp / "classes.txt").write_text(CLASSES)
        cls.cfg = replace(RunConfig.from_params_yaml(), seed=7,
                          train_capture_dirs=(str(cls.tmp / "capA"), str(cls.tmp / "capB")),
                          classes_file=str(cls.tmp / "classes.txt"))
        cls.classes = load_classes(cls.tmp / "classes.txt")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def id_stems(self, classes, split):
        with mock.patch.object(ood_eval, "raw_photo_index", return_value=self.raw):
            return {p.stem for p in ood_eval.id_photos(self.cfg, classes, split)}

    def test_matches_the_dataset_split(self):
        self.assertEqual(self.classes.keys, ("A", "B"))
        captures = resolve_captures([Path(d) for d in self.cfg.train_capture_dirs])
        for split in ("val", "test"):
            with self.subTest(split=split):
                # A budget above any capture's photo count in the split, so every photo gets a patch.
                ds = MultiPhotoPatchDataset(
                    captures=captures, classes=self.classes, split=split, seed=self.cfg.seed, crop_size=32,
                    resize=32, safety_margin=0.97, patches_per_class={"train": 1, "val": 50, "test": 50},
                    photo_frac={"train": self.cfg.train_photo_frac, "val": self.cfg.val_photo_frac,
                                "test": self.cfg.test_photo_frac})
                in_ds = {m.photo_name.removesuffix("__cropped.jpg") for m in ds._meta}
                got = self.id_stems(self.classes, split)
                self.assertEqual(got, in_ds)
                self.assertEqual(len(got), round(0.15 * 20) + round(0.15 * 25))   # A: 20 photos; B: 25, pooled

    def test_a_class_with_no_photos_raises(self):
        (self.tmp / "classes4.txt").write_text(CLASSES + "004;C,Four;\n")
        with self.assertRaisesRegex(ValueError, "class C"):
            self.id_stems(load_classes(self.tmp / "classes4.txt"), "val")



class TestPreMl1CheckpointIsRefused(unittest.TestCase):
    """id_photos recomputes today's pooled split; a checkpoint trained before ticket ML-1 was split per camera,
    so its "train" and "test" photos would be the wrong ones. ood_eval (trained / not trained) and fit_ood_probe
    (its bean photos) read the checkpoint's config through split_config, which refuses such a card."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)

    def card(self, training_config: dict) -> Path:
        ckpt = self.tmp / "m.pt"
        ckpt.with_suffix(".json").write_text(json.dumps({"training_config": training_config}))
        return ckpt

    def test_a_card_with_train_rigs_is_refused(self):
        with self.assertRaisesRegex(ValueError, "before ticket ML-1"):
            ood_eval.split_config(self.card({"train_rigs": ["data/cropped/cam_sony"]}), None)

    def test_a_card_from_ml1_on_is_read(self):
        cfg, _ = ood_eval.split_config(self.card({"train_capture_dirs": ["data/cropped/cam_sony"], "seed": 3}), None)
        self.assertEqual((cfg.train_capture_dirs, cfg.seed), (("data/cropped/cam_sony",), 3))

if __name__ == "__main__":
    unittest.main()
