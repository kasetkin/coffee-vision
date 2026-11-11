"""coffeecv.seg_predict's incremental run (ticket ML-26), with a stand-in segmenter: after a recorded run only photos
new to the split or changed are predicted, a kept photo's index row is the recorded one, a photo gone from the split
loses its mask, and the folder is the same bytes as a full run's. Plain unittest.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from coffeecv import seg_predict
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.repo_files import sha256_file

from tests import _dvc_lock as dvc_lock
from tests.test_seg_predict_split import StubSegmenter, _toy_load

PRED_YAML = """\
stages:
  pred:
    foreach:
      - m
    do:
      cmd: python -m coffeecv.seg_predict m --split test --threads 4 --out masks/${item}
      deps:
        - labels/ml5/seg_dataset.yaml
        - data/canonical
        - code.py
      params:
        - params.yaml:
            - seg_mask_select
      outs:
        - masks/${item}:
            persist: true
"""


class Incremental(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(dir=REPO_ROOT))   # the toy photos are read by their path under the repo
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "dvc.yaml").write_text(PRED_YAML)
        shutil.copy(REPO_ROOT / "params.yaml", self.root / "params.yaml")
        (self.root / "code.py").write_text("x = 1\n")
        (self.root / "data" / "canonical").mkdir(parents=True)
        (self.root / "data" / "canonical" / "c.txt").write_text("store 1\n")
        (self.root / "labels" / "ml5").mkdir(parents=True)
        (self.root / "labels" / "ml5" / "seg_dataset.yaml").write_text("photos 1\n")
        self.out = self.root / "masks" / "m"
        self.entries = [self.photo(i) for i in range(3)]
        self.cfg = RunConfig.from_params_yaml()
        self.calls = self.predicted = 0
        p = mock.patch.object(seg_predict, "load_for_record", _toy_load)
        p.start()
        self.addCleanup(p.stop)

    def photo(self, i: int, shade: int | None = None) -> dict:
        f = self.root / "dataset" / f"p{i}.png"
        f.parent.mkdir(exist_ok=True)
        Image.fromarray(np.full((40, 60, 3), 10 * i if shade is None else shade, np.uint8)).save(f)
        rel = f.relative_to(REPO_ROOT).as_posix()
        return {"id": f"p{i}", "source": "pool", "split": "test", "path": rel, "sha256": sha256_file(f)}

    def make_seg(self):
        self.calls += 1
        return CountingSegmenter(self)

    def run_stage(self, full: bool = False) -> None:
        record, rows, how, clean = seg_predict.recorded_rows(self.out, full, self.root)
        seg_predict.predict(self.entries, self.make_seg, self.out, self.cfg, record, rows, how, clean)

    def masks(self) -> dict[str, bytes]:
        return {p.name: p.read_bytes() for p in sorted(self.out.iterdir())}

    def commit(self) -> None:
        dvc_lock.commit(self.root, "pred@m", "python -m coffeecv.seg_predict m --split test --threads 4 --out masks/m",
                        ["labels/ml5/seg_dataset.yaml", "data/canonical", "code.py"], ["masks/m"],
                        {"seg_mask_select": self.cfg.seg_mask_select})

    def test_only_new_and_changed_photos_are_predicted_and_the_folder_is_a_full_runs(self):
        self.run_stage()
        self.assertEqual(self.predicted, 3)
        self.commit()
        self.entries = [self.entries[0], self.photo(2, shade=200), self.photo(3)]   # p1 gone, p2 changed, p3 new
        (self.root / "data" / "canonical" / "c.txt").write_text("store 2\n")
        (self.root / "labels" / "ml5" / "seg_dataset.yaml").write_text("photos 2\n")
        self.predicted = 0
        self.run_stage()
        self.assertEqual(self.predicted, 2)
        incremental = self.masks()
        self.assertNotIn("p1.png", incremental)
        self.predicted = 0
        self.run_stage(full=True)
        self.assertEqual(self.predicted, 3)
        self.assertEqual(incremental, self.masks())

    def test_nothing_to_predict_loads_no_segmenter(self):
        self.run_stage()
        self.commit()
        self.calls = 0
        self.run_stage()
        self.assertEqual(self.calls, 0)

    def test_a_changed_code_file_predicts_everything(self):
        self.run_stage()
        self.commit()
        (self.root / "code.py").write_text("x = 2\n")
        self.predicted = 0
        self.run_stage()
        self.assertEqual(self.predicted, 3)

    def test_a_folder_no_stage_writes_is_never_cleaned(self):
        other = self.root / "elsewhere"
        other.mkdir()
        (other / "keep.png").write_bytes(b"not ours")
        record, rows, how, clean = seg_predict.recorded_rows(other, False, self.root)
        self.assertEqual((record, clean), (None, False))
        seg_predict.predict(self.entries, self.make_seg, other, self.cfg, record, rows, how, clean)
        self.assertTrue((other / "keep.png").exists())


class CountingSegmenter(StubSegmenter):
    def __init__(self, test: Incremental):
        super().__init__(0.5)
        self.test = test

    def predict_mask(self, rgb):
        self.test.predicted += 1
        return super().predict_mask(rgb) ^ (rgb[..., 0] > 100)   # depends on the photo


if __name__ == "__main__":
    unittest.main()
