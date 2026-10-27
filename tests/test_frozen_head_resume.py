"""Ticket ML-13 D7, D8 (plan P3): the frozen head fit saves its computed features as it goes and resumes from
them. Each split's embeddings when the split is done and the training views in chunks of whole EMBED_BATCHes,
under <out>/features/<fingerprint>/; a rerun of the same command loads what matches and embeds the rest, and
gives what an uninterrupted run gives. The files go once the seed's run directory is written; a run deletes
other runs' features before it embeds anything, a --smoke run only under its own smoke/ directory. Through
main(), on test_head_augment's fake backbone and fake patches. Plain unittest.

    python -m unittest tests.test_frozen_head_resume -v
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from coffeecv import fit_frozen_head as ffh
from coffeecv.config import REPO_ROOT, RunConfig

from tests.test_head_augment import FakeBackbone, fake_folds


class Interrupted(Exception):
    """A host reboot, as the test sees it."""


class InterruptingBackbone(FakeBackbone):
    """The fake backbone, but its `stop_at`-th batch never returns."""

    def __init__(self, stop_at: int | None = None):
        super().__init__()
        self.calls, self.stop_at = 0, stop_at

    def features(self, x):
        self.calls += 1
        if self.calls == self.stop_at:
            raise Interrupted
        return super().features(x)


def straight_seen() -> int:
    """What a smoke run straight through embeds: 12 training patches, their 96 views, 6 + 6 val and test patches,
    and the 12 training patches through each transform of head_augment alone (transform_shifts)."""
    n_alone = len(ffh.single_transforms(ffh.HeadAugment.from_config(RunConfig.from_params_yaml())))
    return 12 + 96 + 6 + 6 + 12 * n_alone


class Resume(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def run_main(self, out: Path, bb: FakeBackbone, *, cfg_edit: dict | None = None, smoke: bool = True) -> str:
        """`fit_frozen_head --smoke --allow-dirty --out <out>` on the fakes (without --smoke: a real run of seed 42,
        --start-exp 1), the view chunk lowered to one EMBED_BATCH (12 training patches x 8 views = 3 chunks); the
        log."""
        read = RunConfig.from_params_yaml
        log = io.StringIO()
        with (mock.patch.object(ffh, "build_backbone", lambda name: bb),
              mock.patch.object(ffh, "build_fold_datasets", fake_folds),
              mock.patch.object(ffh, "assert_input_size"),
              mock.patch.object(ffh, "save_frozen_checkpoint", lambda path, *a, **k: Path(path).write_bytes(b"pt")),
              mock.patch.object(ffh, "VIEW_CHUNK_BATCHES", 1),
              mock.patch.object(ffh.RunConfig, "from_params_yaml", lambda: replace(read(), **(cfg_edit or {}))),
              contextlib.redirect_stdout(log)):
            try:
                ffh.main(["--allow-dirty", "--seeds", "42", "--out", str(out),
                          *(["--smoke"] if smoke else ["--start-exp", "1"])])
            except Interrupted:
                pass
        return log.getvalue()

    @staticmethod
    def outputs(run_dir: Path) -> dict:
        """What a run produced, timings left out."""
        metrics = json.loads((run_dir / "metrics.json").read_text())
        metrics.pop("created_at")
        for arm in metrics["arms"].values():
            arm.pop("fit_seconds")
        return {"metrics": metrics, **{f: (run_dir / f).read_text() for f in ("predictions_val.csv",
                                                                             "predictions_test.csv")}}

    def test_a_run_stopped_in_its_views_and_run_again_gives_the_uninterrupted_fit(self):
        straight = InterruptingBackbone()
        self.run_main(self.tmp / "a", straight)
        self.assertEqual(straight.seen, straight_seen())
        stopped = InterruptingBackbone(stop_at=4)          # the training patches, two view chunks, then a reboot
        self.run_main(self.tmp / "b", stopped)
        self.assertFalse((self.tmp / "b" / "smoke" / "s42" / "metrics.json").exists())
        resumed = InterruptingBackbone()
        log = self.run_main(self.tmp / "b", resumed)
        self.assertIn("train 12 patches loaded", log)
        self.assertIn("2 of 3 view chunks loaded", log)
        self.assertEqual(self.outputs(self.tmp / "b" / "smoke" / "s42"), self.outputs(self.tmp / "a" / "smoke" / "s42"))
        self.assertEqual(stopped.seen + resumed.seen, straight.seen)   # nothing embedded twice
        self.assertFalse((self.tmp / "b" / "smoke" / "features").exists())

    def test_saved_features_of_other_inputs_are_deleted_and_the_run_starts_fresh(self):
        exclude = self.tmp / "pool_exclude.csv"
        inputs = tuple(str(exclude.relative_to(REPO_ROOT)) if p == "labels/ml3/pool_exclude.csv" else p
                       for p in ffh.FEATURE_INPUTS)
        for change, differs in ((lambda: exclude.write_text("path\nsome/photo.jpg\n"), "sha256 " + inputs[-1]),
                                (lambda: None, "config")):
            with self.subTest(differs), mock.patch.object(ffh, "FEATURE_INPUTS", inputs):
                exclude.write_text("path\n")
                out = self.tmp / ("exclude" if differs != "config" else "config")
                self.run_main(out, InterruptingBackbone(stop_at=4))
                self.assertEqual(len(list((out / "smoke" / "features").iterdir())), 1)
                change()
                fresh = InterruptingBackbone()
                log = self.run_main(out, fresh, cfg_edit={"seg_min_area_frac": 0.5} if differs == "config" else None)
                self.assertRegex(log, rf"deleting another run's saved features \S+ \([^)]*{re.escape(differs)}[^)]*differ\)")
                self.assertNotIn("loaded", log)
                self.assertEqual(fresh.seen, straight_seen())
                self.assertFalse((out / "smoke" / "features").exists())

    def test_a_smoke_run_leaves_a_real_runs_features_and_a_real_run_deletes_them(self):
        out = self.tmp / "out"
        real = out / "features" / "0123456789abcdef"
        real.mkdir(parents=True)
        (real / "fingerprint.json").write_text("{}")
        self.run_main(out, InterruptingBackbone())
        self.assertTrue(real.exists())
        with mock.patch.object(ffh, "archive"), mock.patch.object(ffh, "EXPERIMENTS_DIR", self.tmp / "experiments"):
            log = self.run_main(out, InterruptingBackbone(), smoke=False)
        self.assertIn(f"deleting another run's saved features {real}", log)
        self.assertFalse((out / "features").exists())


if __name__ == "__main__":
    unittest.main()
