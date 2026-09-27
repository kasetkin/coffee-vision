"""The frozen-DINO checkpoint: what it stores, what loading it refuses, and that it reaches the rest of the
pipeline through the same `infer.load_model` a ResNet18 checkpoint does. Plain unittest.

    python -m unittest discover -s tests -p 'test_frozen_checkpoint.py'

On real inputs: the ViT-B/16 weights from models_pretrained/ and real bean patches from
`coffeecv_dino.reference`; the head is fitted on their real features by the same `fit_head_at` the
shipping fit uses. Weights and crops are not in git, so these skip with a reason when absent.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from coffeecv.backbones import MODELS_PRETRAINED, SPECS, build_backbone
from coffeecv.config import RunConfig
from coffeecv.dino_classifier import (CHECKPOINT_FORMAT, DinoClassifier, is_frozen_model,
                                      load_frozen_checkpoint, save_frozen_checkpoint)
from coffeecv.fold_data import build_fold_datasets
from coffeecv.infer import embedding_dim_of, forward_with_embeddings, inference_tta_for, load_model
from coffeecv.linear_head import fit_head_at, predict
from coffeecv.model import FROZEN_MODELS, build_model
from coffeecv.transforms import build_eval_transform
from coffeecv_dino.reference import have_reference_data, reference_patches

NAME = "dinov3_vitb16"
HAVE_WEIGHTS = (MODELS_PRETRAINED / SPECS[NAME].weights).exists()


@unittest.skipUnless(HAVE_WEIGHTS and have_reference_data(), "ViT-B/16 weights or real crops not present")
class TestFrozenCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bb = build_backbone(NAME)
        cls.x, y, class_ids, _ = reference_patches(per_class=3)
        with torch.no_grad():
            feats = cls.bb.features(cls.x)["cls_mean"].numpy()
        cls.feats = feats
        cls.head, _ = fit_head_at(feats, y, C=0.1, n_classes=len(class_ids))
        cls.class_ids = class_ids
        cls.tmp = Path(tempfile.mkdtemp())
        cls.ckpt = cls.tmp / "head.pt"
        save_frozen_checkpoint(cls.ckpt, cls.bb, "cls_mean", cls.head, class_ids, C=0.1)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_checkpoint_holds_the_head_not_the_backbone(self):
        ck = torch.load(self.ckpt, weights_only=True)
        self.assertEqual(ck["format"], CHECKPOINT_FORMAT)
        self.assertEqual(ck["weights_sha256"], self.bb.weights_sha256)
        self.assertEqual(set(ck["head"]), {"weight", "bias"})
        self.assertLess(self.ckpt.stat().st_size, 1 << 20)          # 10 x 1536 floats, not 343 MB

    def test_load_model_scores_exactly_what_was_fitted(self):
        model, head = load_model(self.ckpt, NAME, len(self.class_ids), dropout=0.0)
        self.assertIsInstance(model, DinoClassifier)
        self.assertEqual(embedding_dim_of(head), 1536)
        probs, embeds = forward_with_embeddings(model, head, self.x)
        self.assertEqual(embeds.shape, (len(self.x), 1536))         # the pre-hook saw the readout
        np.testing.assert_allclose(embeds, self.feats, rtol=0, atol=1e-5)
        _, want, _ = predict(self.head, self.feats)
        np.testing.assert_allclose(probs, want, rtol=0, atol=1e-5)

    def test_refuses_a_mismatched_pairing(self):
        with self.assertRaisesRegex(ValueError, "model_name"):
            load_frozen_checkpoint(self.ckpt, "dinov3_vits16", len(self.class_ids))
        with self.assertRaisesRegex(ValueError, "classes"):
            load_frozen_checkpoint(self.ckpt, NAME, len(self.class_ids) + 1)
        ck = torch.load(self.ckpt, weights_only=True)
        for key, value, pattern in (("weights_sha256", "0" * 64, "backbone weights"),
                                    ("timm_version", "0.0.0", "timm"),
                                    ("format", "something/else", "not a")):
            with self.subTest(key=key):
                bad = self.tmp / f"bad_{key}.pt"
                torch.save({**ck, key: value}, bad)
                with self.assertRaisesRegex(ValueError, pattern):
                    load_frozen_checkpoint(bad, NAME, len(self.class_ids))

    def test_tta_follows_the_card(self):
        self.assertFalse(inference_tta_for(self.ckpt, NAME))         # no card: frozen -> off
        self.assertTrue(inference_tta_for(self.tmp / "none.pt", "resnet18"))
        card = self.ckpt.with_suffix(".json")
        card.write_text(json.dumps({"inference_defaults": {"tta": True}}))
        try:
            self.assertTrue(inference_tta_for(self.ckpt, NAME))
        finally:
            card.unlink()


class TestFrozenModelsStayOutOfTheSgdLoop(unittest.TestCase):
    def test_build_model_refuses(self):
        for name in FROZEN_MODELS:
            self.assertTrue(is_frozen_model(name))
            with self.assertRaisesRegex(ValueError, "fit_frozen_head"):
                build_model(name, num_classes=10, freeze_mode="full")
        self.assertFalse(is_frozen_model("resnet18"))


@unittest.skipUnless(have_reference_data(), "real crops not present")
class TestSplitsBuiltAloneAreIdentical(unittest.TestCase):
    """fit_frozen_head builds one split at a time to bound memory; that must not change a single box."""

    def test_val_alone_equals_val_with_test(self):
        cfg = replace(RunConfig.from_params_yaml(), seed=42, train_rigs=("data/cropped/cam_sony",),
                      heldout_rig="", val_patches_per_class=2, test_patches_per_class=2)
        tf = build_eval_transform(cfg.patch_resize)
        alone = build_fold_datasets(cfg, tf, tf, only=("val",))
        both = build_fold_datasets(cfg, tf, tf, only=("val", "test"))
        self.assertIsNone(alone.train)
        self.assertIsNone(alone.test)
        self.assertEqual(len(alone.val), len(both.val))
        for i in range(len(alone.val)):
            self.assertTrue(torch.equal(alone.val[i][0], both.val[i][0]))


if __name__ == "__main__":
    unittest.main()
