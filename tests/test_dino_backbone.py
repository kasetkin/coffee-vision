"""Tests for coffeecv_dino's backbone loading and its seam with coffeecv. Plain unittest.

Run from the repo root:  python -m unittest discover -s tests -p "test_dino*" -v

Every model input here is real: bean patches drawn from the DVC-tracked cam_iphone crops by
coffeecv_dino.reference. The weights and the crops are not in git, so tests that need them skip (with a
pointer) on a clone that has not obtained them -- see models_pretrained/README.md.
"""
import re
import unittest
from pathlib import Path

import numpy as np
import torch

from coffeecv.backbones import MODELS_PRETRAINED, SPECS, assert_input_size, build_backbone
from coffeecv_dino.reference import have_reference_data, reference_patches

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "dinov3_vits16_reference.npz"
HAVE_V3 = (MODELS_PRETRAINED / SPECS["dinov3_vits16"].weights).exists() and have_reference_data()
SKIP_V3 = "DINOv3 weights or the cam_iphone crops are missing -- run models_pretrained/verify.py / dvc pull"


@unittest.skipUnless(HAVE_V3, SKIP_V3)
class TestDinov3Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bb = build_backbone("dinov3_vits16")
        cls.sd = torch.load(MODELS_PRETRAINED / SPECS["dinov3_vits16"].weights, map_location="cpu",
                            weights_only=True)
        cls.ref = np.load(FIXTURE)
        cls.x = reference_patches(int(cls.ref["per_class"]))[0]      # the fixture's own real patches

    def test_matches_meta_reference(self):
        """timm + the copied bf16 RoPE periods reproduce facebookresearch/dinov3's output.

        atol 1e-4, not bit-exactness: the same weights differ by up to 3.2e-5 between the workstation
        (AVX2) and the compute box (AVX-512) because float sums reorder (plan §2.4). Leaving timm's own
        fp32 periods in place differs by ~1.2e-3, which this still catches with >10x margin."""
        with torch.no_grad():
            f = self.bb.features(self.x)
        for readout in ("cls", "cls_mean"):
            np.testing.assert_allclose(f[readout].numpy(), self.ref[readout], atol=1e-4, rtol=0,
                                       err_msg=f"{readout} drifted from Meta's reference")

    def test_rope_periods_are_the_checkpoints(self):
        rope = self.bb.model.rope
        self.assertIsNone(rope.feat_shape, "timm caches a RoPE table; the periods copy would be ignored")
        self.assertTrue(torch.equal(rope.periods, self.sd["rope_embed.periods"].float()))

    def test_checkpoint_qkv_bias_is_zero(self):
        """The precondition that makes timm's no-bias variant a lossless load."""
        for k, v in self.sd.items():
            if k.endswith("qkv.bias"):
                self.assertFalse(v.any(), k)

    def test_readout_shapes_and_register_tokens(self):
        self.assertEqual(self.bb.model.num_prefix_tokens, 5)      # CLS + 4 registers
        with torch.no_grad():
            f = self.bb.features(self.x[:2])
        self.assertEqual(tuple(f["cls"].shape), (2, 384))
        self.assertEqual(tuple(f["cls_mean"].shape), (2, 768))

    def test_stays_frozen_and_in_eval(self):
        self.bb.train()                                          # must be refused
        self.assertFalse(self.bb.training or self.bb.model.training)
        self.assertFalse(any(p.requires_grad for p in self.bb.parameters()))

    def test_input_size_guard(self):
        assert_input_size(self.bb, 224)
        with self.assertRaisesRegex(ValueError, "224 or 240"):
            assert_input_size(self.bb, 230)

    def test_head_pre_hook_captures_the_readout(self):
        """forward_with_embeddings must capture the readout vector the head is fed -- the embedding
        contract the OOD guard relies on -- which only holds if the head is a called submodule."""
        from coffeecv.infer import forward_with_embeddings
        from coffeecv.dino_classifier import DinoClassifier

        model = DinoClassifier(self.bb, "cls_mean", torch.nn.Linear(768, 10)).eval()
        probs, embeds = forward_with_embeddings(model, model.head, self.x[:3])
        self.assertEqual(probs.shape, (3, 10))
        self.assertEqual(embeds.shape, (3, 768))


class TestNoPrivateCopies(unittest.TestCase):
    """coffeecv_dino owns only its screen drivers -- never a second copy of behaviour that must stay
    identical between arms or between the screen and the shipped model. analysis/bean_scale grew such a copy twice and it drifted
    silently both times (plan §4.2)."""

    BANNED = ("load_rgb_image", "estimate_bean_pitch", "sample_bean_unit_patch_boxes",
              "sample_bean_unit_centers", "build_eval_transform", "build_train_transform",
              "compute_split_metrics", "build_fold_datasets", "split_photos_by_class", "run_photowise",
              "pool_photos", "archive", "patches_for_photo", "forward_with_embeddings", "id_photos",
              "negatives_from", "linear_probe_scores", "_fit_logistic", "auroc", "detection_at_fpr",
              # Moved into coffeecv on 2026-09-27 (plan §8.2): the shipped model and the screen must
              # build the backbone and fit the head with one implementation.
              "build_backbone", "verify_weights", "fit_head", "fit_head_at", "export_linear",
              "DinoClassifier", "FrozenBackbone")

    def test_package_defines_none_of_them(self):
        for f in (REPO / "coffeecv_dino").rglob("*.py"):
            text = f.read_text()
            for name in self.BANNED:
                self.assertIsNone(re.search(rf"^\s*(def|class) {name}\b", text, re.M),
                                  f"{f.name} defines its own {name} -- import it from coffeecv")


if __name__ == "__main__":
    unittest.main()
