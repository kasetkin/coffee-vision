"""Ticket ML-29 P10: `fit_frozen_head --pool DIR --label TEXT` fits on a scratch pool outside DVC in place of
data/segcropped/pool (the gain-map photos' crops swapped in, the rest of the pool as it is), so the paired
comparison's two arms differ only in those crops. The split keys on the capture dir's basename, so the scratch pool
must be named `pool` too; the resume fingerprint takes its version from its files, which dvc.lock does not know; the
run is named and described apart. Plain unittest.

    python -m unittest tests.test_fit_frozen_head_pool -v
"""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coffeecv import fit_frozen_head as ffh
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import CAPTURES


class Pool(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.pool = self.tmp / "scratch" / "pool"
        (self.pool / "class_001__X").mkdir(parents=True)
        (self.pool / "class_001__X" / "a__cropped.jxl").write_bytes(b"one")
        self.rel = str(self.pool.relative_to(REPO_ROOT))

    def test_the_config_reads_the_pool_given_and_the_tracked_pool_without_one(self):
        self.assertEqual(ffh.frozen_config("dinov3_vitb16", 42, smoke=False).train_capture_dirs, tuple(CAPTURES))
        self.assertEqual(ffh.frozen_config("dinov3_vitb16", 42, smoke=False, pool=self.rel).train_capture_dirs,
                         (self.rel,))

    def test_a_pool_dvc_lock_does_not_know_is_versioned_by_its_files(self):
        cfg = ffh.frozen_config("dinov3_vitb16", 42, smoke=False, pool=self.rel)
        first = ffh.crops_version(cfg)[self.rel]
        self.assertTrue(first.startswith("files sha256 "), first)
        (self.pool / "class_001__X" / "a__cropped.jxl").write_bytes(b"two")
        self.assertNotEqual(ffh.crops_version(cfg)[self.rel], first)
        tracked = ffh.crops_version(ffh.frozen_config("dinov3_vitb16", 42, smoke=False))
        self.assertTrue(tracked[CAPTURES[0]].endswith(".dir"), tracked)    # dvc.lock's hash, as before

    def test_a_pool_not_named_pool_or_without_a_label_is_refused(self):
        other = self.tmp / "scratch" / "pool2"
        other.mkdir()
        with self.assertRaisesRegex(SystemExit, "named 'pool'"):
            ffh.main(["--allow-dirty", "--smoke", "--pool", str(other.relative_to(REPO_ROOT)), "--label", "x"])
        with self.assertRaisesRegex(SystemExit, "--label"):
            ffh.main(["--allow-dirty", "--smoke", "--pool", self.rel])
        with self.assertRaisesRegex(SystemExit, "no directory"):
            ffh.main(["--allow-dirty", "--smoke", "--pool", self.rel + "_missing/pool", "--label", "x"])

    def test_the_run_is_named_and_described_with_its_pool(self):
        archived = {}

        def fake_fit(bb, readout, seed, C, C_rule, out, smoke, pool=None):
            run_dir = self.tmp / "run"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps({"dino": {}, "head_augment": {"views": 8},
                                                             "train_capture_dirs": [pool]}))
            archived["pool"] = pool
            return run_dir

        with (mock.patch.object(ffh, "build_backbone", lambda name: mock.Mock(name=name)),
              mock.patch.object(ffh, "feature_fingerprint", lambda *a, **k: {}),
              mock.patch.object(ffh, "clear_feature_leftovers"),
              mock.patch.object(ffh, "fit_one_seed", fake_fit),
              mock.patch.object(ffh, "EXPERIMENTS_DIR", self.tmp / "experiments"),
              mock.patch.object(ffh, "archive", lambda exp, slug, note, src_dir: archived.update(slug=slug, note=note)),
              contextlib.redirect_stdout(io.StringIO())):
            ffh.main(["--allow-dirty", "--seeds", "42", "--start-exp", "9001", "--pool", self.rel,
                      "--label", "ml29gm"])
        self.assertEqual(archived["pool"], self.rel)
        self.assertTrue(archived["slug"].endswith("_s42_ml29gm"), archived["slug"])
        self.assertIn(self.rel, archived["note"])
        self.assertIn("ml29gm", archived["note"])


if __name__ == "__main__":
    unittest.main()
