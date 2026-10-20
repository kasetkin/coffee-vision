"""Ticket ML-8 P6 (D4, ADR 0025): the frozen head is fitted on augmented views of its training patches, drawn per
run, held in memory as embeddings and never stored; validation and test are not augmented, and the augmented arm
of ADR 0002's paired print carries the unaugmented arm's total sample weight. Plain unittest, on synthetic patches
and a stand-in backbone; no weights and no photos.

    python -m unittest tests.test_head_augment -v
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from coffeecv import fit_frozen_head as ffh
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import PatchMeta
from coffeecv.geometry import Region


def patch(seed: int = 0, side: int = 448) -> np.ndarray:
    """A textured patch with no pixel at 0 or 255, so a constant fill would show."""
    return np.random.default_rng(seed).integers(20, 236, (side, side, 3), dtype=np.uint8)


def one(transform: str, **args) -> list[dict]:
    return [{"transform": transform, "p": 1.0, "args": args}]


class Views(unittest.TestCase):
    def setUp(self):
        self.aug = ffh.HeadAugment.from_config(RunConfig.from_params_yaml())

    def test_params_yaml_asks_for_eight_views_at_the_models_input_size(self):
        self.assertEqual(self.aug.views, 8)
        v = self.aug.view(patch(), seed=42, index=0, k=0)
        self.assertEqual((v.shape, v.dtype), ((224, 224, 3), np.uint8))

    def test_one_seed_repeats_its_views_and_another_seed_or_patch_or_view_differs(self):
        p = patch()
        a = self.aug.view(p, seed=42, index=3, k=1)
        np.testing.assert_array_equal(a, self.aug.view(p, seed=42, index=3, k=1))
        for other in (dict(seed=123, index=3, k=1), dict(seed=42, index=4, k=1), dict(seed=42, index=3, k=2)):
            with self.subTest(**other):
                self.assertFalse(np.array_equal(a, self.aug.view(p, **other)))

    def test_a_view_does_not_depend_on_the_views_drawn_before_it(self):
        p = patch()
        alone = self.aug.view(p, seed=7, index=5, k=4)
        for k in range(4):
            self.aug.view(p, seed=7, index=5, k=k)
        np.testing.assert_array_equal(alone, self.aug.view(p, seed=7, index=5, k=4))

    def test_a_shrunk_affine_view_has_no_constant_filled_border(self):
        spec = [t for t in RunConfig.from_params_yaml().head_augment["patch"] if t["transform"] == "Affine"]
        [affine] = spec
        aug = ffh.HeadAugment(views=1, patch=[{**affine, "p": 1.0, "args": {**affine["args"], "scale": [0.9, 0.9]}}],
                              input=[], resize=448)
        v = aug.view(patch(), seed=0, index=0, k=0)
        border = np.concatenate([v[0], v[-1], v[:, 0], v[:, -1]])
        self.assertFalse((border == 0).all(axis=1).any(), "a black border pixel: the Affine fill is not reflected")

    def test_coarse_dropout_fills_with_the_imagenet_mean(self):
        [t] = [t for t in RunConfig.from_params_yaml().head_augment["patch"] if t["transform"] == "CoarseDropout"]
        self.assertEqual(t["args"]["fill"], [124, 116, 104])

    def test_the_camera_family_acts_on_the_resized_patch(self):
        # Downscale at its strongest, alone: at 224 it must lose detail the plain resize keeps.
        aug = ffh.HeadAugment(views=1, patch=[], input=one("Downscale", scale_range=[0.5, 0.5],
                                                         interpolation_pair={"downscale": 3, "upscale": 1}),
                              resize=224)
        plain = ffh.HeadAugment(views=1, patch=[], input=[], resize=224)
        p = patch()
        self.assertGreater(np.abs(aug.view(p, 0, 0, 0).astype(int) - plain.view(p, 0, 0, 0).astype(int)).mean(), 5)

    def test_no_pipeline_is_the_eval_resize_alone(self):
        from PIL import Image
        plain = ffh.HeadAugment(views=1, patch=[], input=[], resize=224)
        p = patch()
        np.testing.assert_array_equal(plain.view(p, 0, 0, 0),
                                      np.asarray(Image.fromarray(p).resize((224, 224), Image.BILINEAR)))

    def test_the_resize_is_servings(self):
        # Train/inference parity (CODING_STANDARDS.md): one resize, model_io's, which eval_patch also calls.
        from coffeecv import model_io
        plain = ffh.HeadAugment(views=1, patch=[], input=[], resize=224)
        with mock.patch.object(model_io, "eval_resize", wraps=model_io.eval_resize) as resize:
            plain.view(patch(), 0, 0, 0)
        resize.assert_called_once()

    def test_an_unknown_key_or_argument_is_refused(self):
        block = RunConfig.from_params_yaml().head_augment
        for bad in ({**block, "view": 8}, {**block, "views": 0},
                    {**block, "input": one("ColorJitter", brightness_range=[0.5, 0.6])}):
            with self.subTest(bad=sorted(bad)), self.assertRaises(ValueError):
                ffh.HeadAugment.from_config(replace(RunConfig.from_params_yaml(), head_augment=bad))

    def test_an_empty_block_is_no_augmentation(self):
        self.assertIsNone(ffh.HeadAugment.from_config(RunConfig()))


class Rows(unittest.TestCase):
    def test_the_augmented_arm_carries_the_unaugmented_arms_total_weight(self):
        n, k, d = 10, 8, 4
        X, y = np.ones((n, d), np.float32), np.arange(n) % 3
        X_all, y_all, w = ffh.augmented_rows(X, np.zeros((n, k, d), np.float32), y)
        self.assertEqual((X_all.shape, y_all.shape, w.shape), ((n * (k + 1), d), (n * (k + 1),), (n * (k + 1),)))
        self.assertAlmostEqual(float(w.sum()), float(n))
        np.testing.assert_allclose(w, 1 / (k + 1))
        np.testing.assert_array_equal(X_all[:n], X)                      # the originals first, then the views
        np.testing.assert_array_equal(y_all, np.concatenate([y] + [y] * k))


class FakeBackbone:
    """Two features per image: its mean and its first pixel, so different pixels give different rows."""
    name, spec, weights_sha256 = "dinov3_vitb16", SimpleNamespace(weights="fake.pth"), "0" * 64

    def __init__(self):
        self.seen = 0

    def features(self, x):
        self.seen += len(x)
        f = torch.stack([x.mean(dim=(1, 2, 3)), x[:, 0, 0, 0]], dim=1)
        return {"cls_mean": f}


CLASSES = ["A", "B", "C"]      # three: sklearn fits a two-class problem as one binary column


class FakeSplit:
    """What fit_one_seed reads of a MultiPhotoPatchDataset: patches, provenance and labels."""
    bean_share_rule = None

    def __init__(self, split: str, n_per_class: int):
        rng = np.random.default_rng(len(split))
        self._patches = [rng.integers(0, 256, (448, 448, 3), dtype=np.uint8) for _ in range(3 * n_per_class)]
        self._meta = [PatchMeta(c, "pool", f"{split}{i}__cropped.jpg", Region(0, 1, 0, 1), 0.0, 448, c)
                      for i, c in enumerate(CLASSES * n_per_class)]

    def __len__(self):
        return len(self._patches)

    def __getitem__(self, i):
        from PIL import Image
        from coffeecv.transforms import build_eval_transform
        return build_eval_transform(224)(Image.fromarray(self._patches[i])), CLASSES.index(self._meta[i].class_id)


def fake_folds(cfg, train_tf, eval_tf, only):
    [split] = only
    ds = FakeSplit(split, {"train": 4, "val": 2, "test": 2}[split])
    return SimpleNamespace(**{s: ds if s == split else None for s in ("train", "val", "test")},
                           class_ids=CLASSES, class_labels={c: c for c in CLASSES},
                           captures=[SimpleNamespace(name="pool")])


class OneSeed(unittest.TestCase):
    """fit_one_seed on three classes of fake patches (12 training, 6 val, 6 test) and a fake backbone."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.bb = FakeBackbone()
        views = []
        real_view = ffh.HeadAugment.view

        def counting_view(self, p, seed, index, k):
            if self.views == 8:                       # the run's pipeline, not one transform alone (the shift)
                views.append((index, k))
            return real_view(self, p, seed, index, k)
        cls.before = sorted(p for p in (REPO_ROOT / "outputs").rglob("*")) if (REPO_ROOT / "outputs").exists() else []
        with mock.patch.object(ffh, "build_fold_datasets", fake_folds), \
                mock.patch.object(ffh, "assert_input_size"), \
                mock.patch.object(ffh, "save_frozen_checkpoint", lambda path, *a, **k: Path(path).write_bytes(b"pt")), \
                mock.patch.object(ffh.HeadAugment, "view", counting_view):
            cls.run_dir = ffh.fit_one_seed(cls.bb, "cls_mean", 42, 0.1, "test", Path(cls.tmp.name), smoke=True)
        cls.views = views
        import json
        cls.config = json.loads((cls.run_dir / "config.json").read_text())
        cls.metrics = json.loads((cls.run_dir / "metrics.json").read_text())

    def test_only_training_patches_are_augmented_eight_times_each(self):
        self.assertEqual(sorted(self.views), sorted((i, k) for i in range(12) for k in range(8)))
        n_alone = len(ffh.single_transforms(ffh.HeadAugment.from_config(RunConfig.from_params_yaml())))
        self.assertEqual(self.bb.seen, 12 * 9 + 6 + 6 + self.config["dino"]["augment"]["shift_patches"] * n_alone)

    def test_both_arms_are_recorded_at_equal_total_weight(self):
        arms = self.metrics["arms"]
        self.assertEqual(set(arms), {"augmented", "unaugmented"})
        self.assertEqual(arms["augmented"]["train_rows"], 12 * 9)
        self.assertEqual(arms["unaugmented"]["train_rows"], 12)
        self.assertAlmostEqual(arms["augmented"]["total_sample_weight"], arms["unaugmented"]["total_sample_weight"])
        for arm in arms.values():
            self.assertIn("diagnostic_val_selected_C", arm)
            self.assertEqual(set(arm["splits"]), {"val", "test"})
        self.assertEqual(self.metrics["splits"]["val"], arms["augmented"]["splits"]["val"])   # the run's head

    def test_the_config_records_the_pipeline_views_and_library(self):
        a = self.config["dino"]["augment"]
        self.assertEqual((a["views"], a["albumentations"]), (8, "2.0.8"))
        self.assertEqual((a["patch"], a["input"]),
                         (RunConfig.from_params_yaml().head_augment["patch"],
                          RunConfig.from_params_yaml().head_augment["input"]))
        self.assertEqual(self.config["head_augment"], RunConfig.from_params_yaml().head_augment)
        self.assertIn("train_views", self.config["dino"]["embed_ms_per_img"])

    def test_each_transform_alone_reports_its_embedding_shift(self):
        shift = self.config["dino"]["augment"]["shift_by_transform"]
        self.assertEqual(list(shift), [name for name, _ in ffh.single_transforms(
            ffh.HeadAugment.from_config(RunConfig.from_params_yaml()))])
        for v in shift.values():
            self.assertTrue(-1.0 <= v <= 1.0)

    def test_nothing_is_written_outside_the_run_dir(self):
        written = sorted(p.relative_to(self.tmp.name) for p in Path(self.tmp.name).rglob("*") if p.is_file())
        self.assertEqual({p.parts[0] for p in written}, {"s42"})
        self.assertEqual(sorted(p.name for p in self.run_dir.iterdir()),
                         ["config.json", "metrics.json", "model.pt", "model_unaugmented.pt", "predictions_test.csv",
                          "predictions_val.csv"])
        after = sorted(p for p in (REPO_ROOT / "outputs").rglob("*")) if (REPO_ROOT / "outputs").exists() else []
        self.assertEqual(after, self.before)


class Names(unittest.TestCase):
    def test_new_runs_drop_allrigs(self):
        self.assertEqual(ffh.run_slug("dinov3_vitb16", "cls_mean", 8, 42),
                         "dino3b16_frozen_clsmean_segcrop_country_aug8_s42")
        self.assertEqual(ffh.run_slug("dinov3_vitb16", "cls_mean", 0, 7), "dino3b16_frozen_clsmean_segcrop_country_s7")


class ResNetEraKnobs(unittest.TestCase):
    def test_the_frozen_path_refuses_them_set(self):
        for knob, value in (("brightness_jitter_strength", 0.1), ("illum_gradient_strength", 0.2),
                            ("zoom_scale_min", 0.8), ("patch_scale_frac_max", 0.3), ("rotation_jitter_degrees", 3.0)):
            with self.subTest(knob=knob), mock.patch.object(
                    ffh.RunConfig, "from_params_yaml", return_value=replace(RunConfig.from_params_yaml(), **{knob: value})), \
                    self.assertRaisesRegex(ValueError, knob):
                ffh.frozen_config("dinov3_vitb16", 42, smoke=False)

    def test_params_yamls_resting_values_pass(self):
        cfg = ffh.frozen_config("dinov3_vitb16", 42, smoke=False)
        self.assertEqual((cfg.color_jitter_strength, cfg.random_erasing_p, cfg.train_patches_per_class), (0.0, 0.0, 600))


if __name__ == "__main__":
    unittest.main()
