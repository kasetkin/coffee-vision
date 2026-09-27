"""The embedding contract of every supported architecture, and the checked OOD-artifact loaders.
Plain unittest.

    python -m unittest discover -s tests -p 'test_model_and_ood_loading.py'

On real artifacts: the deployed checkpoint `models/allrigs_cam_s123.pt` with its own OOD reference and
probe, and real bean patches from `coffeecv_dino.reference`. The checkpoint is DVC-tracked, not in git,
so the tests that need it skip with a reason when it is absent.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch

from coffeecv.config import REPO_ROOT
from coffeecv.infer import (embedding_dim_of, load_model, load_ood_probe, load_ood_reference,
                            probe_path_for, reference_path_for)
from coffeecv.model import SUPPORTED_MODELS, build_model
from coffeecv_dino.reference import have_reference_data, reference_patches

DEPLOYED = REPO_ROOT / "models" / "allrigs_cam_s123.pt"
HAVE_DEPLOYED = DEPLOYED.exists() and reference_path_for(DEPLOYED).exists() and probe_path_for(DEPLOYED).exists()


class TestEmbeddingContract(unittest.TestCase):
    """A future arm cannot land without proving its head's pre-hook fires and captures
    embedding_dim_of(head) features -- the vector every OOD artifact is built in (plan §8.0)."""

    @unittest.skipUnless(have_reference_data(), "real crops (data/cropped/cam_iphone) not present")
    def test_pre_hook_captures_the_embedding(self):
        x, _, _, _ = reference_patches(per_class=1)
        for name in SUPPORTED_MODELS:
            with self.subTest(name=name):
                model, head = build_model(name, num_classes=10, freeze_mode="full", dropout=0.0)
                captured = []
                handle = head.register_forward_pre_hook(lambda _m, inp: captured.append(inp[0]))
                with torch.inference_mode():
                    out = model.eval()(x)
                handle.remove()
                self.assertEqual(out.shape, (len(x), 10))
                self.assertEqual(len(captured), 1, "the head must be called by forward, exactly once")
                self.assertEqual(captured[0].shape, (len(x), embedding_dim_of(head)))

    def test_pruned_architectures_fail_loudly(self):
        for name in ("mobilenet_v3_small", "efficientnet_b0"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "removed on 2026-09-26"):
                build_model(name, num_classes=10, freeze_mode="full")


@unittest.skipUnless(HAVE_DEPLOYED, f"{DEPLOYED} and its OOD sidecars are not present (dvc pull)")
class TestOodLoaders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cls.head = load_model(DEPLOYED, "resnet18", 10, 0.2)

    def _copy_with(self, tmp: Path, suffix: str, **overrides) -> Path:
        ckpt = tmp / DEPLOYED.name
        if not ckpt.exists():
            shutil.copy2(DEPLOYED, ckpt)
        src = DEPLOYED.with_suffix(suffix)
        data = json.loads(src.read_text())
        data.update(overrides)
        (tmp / src.name).write_text(json.dumps(data))
        return ckpt

    def test_deployed_pairing_loads(self):
        self.assertEqual(embedding_dim_of(self.head), 512)
        self.assertEqual(load_ood_reference(DEPLOYED, self.head)["embedding_dim"], 512)
        self.assertEqual(load_ood_probe(DEPLOYED, head=self.head)["embedding_dim"], 512)

    def test_width_mismatch_refused(self):
        # What a ResNet18 reference beside a DINOv3 ViT-B/16 cls_mean model would look like, seen
        # from the other side: the artifact's width is not what this head is fed.
        with tempfile.TemporaryDirectory() as d:
            ckpt = self._copy_with(Path(d), ".ood_reference.json", embedding_dim=1536)
            with self.assertRaisesRegex(SystemExit, "1536-d but this model emits 512-d"):
                load_ood_reference(ckpt, self.head)
            ckpt = self._copy_with(Path(d), ".ood_probe.json", embedding_dim=1536)
            with self.assertRaisesRegex(SystemExit, "1536-d but this model emits 512-d"):
                load_ood_probe(ckpt, head=self.head)

    def test_other_checkpoint_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = self._copy_with(Path(d), ".ood_reference.json", checkpoint_sha="0000000000000000")
            with self.assertRaisesRegex(SystemExit, "different checkpoint"):
                load_ood_reference(ckpt, self.head)

    def test_absent_reference_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            ckpt = Path(d) / "x.pt"
            ckpt.write_bytes(b"")
            self.assertIsNone(load_ood_reference(ckpt, self.head))


@unittest.skipUnless(DEPLOYED.exists(), f"{DEPLOYED} is not present (dvc pull)")
class TestInferenceNeedsNoImageNetWeights(unittest.TestCase):
    """load_model builds ResNet18 with weights=None (docs/ops1_release_isolation_plan.html §4.3): the
    service runs under ProtectHome=true, where ~/.cache/torch does not exist. Safe only because the
    strict load_state_dict overwrites every parameter and buffer -- which these tests prove."""

    def test_loads_without_touching_the_hub(self):
        def refuse(*args, **kwargs):
            raise AssertionError("load_model tried to fetch pretrained weights")
        with mock.patch("torchvision.models._api.load_state_dict_from_url", refuse):
            load_model(DEPLOYED, "resnet18", 10, 0.2)

    @unittest.skipUnless(have_reference_data(), "real crops (data/cropped/cam_iphone) not present")
    def test_logits_bit_identical_to_imagenet_init(self):
        x, _, _, _ = reference_patches(per_class=2)
        new, _ = load_model(DEPLOYED, "resnet18", 10, 0.2)
        old, _ = build_model("resnet18", num_classes=10, freeze_mode="none", dropout=0.2)  # the old path
        old.load_state_dict(torch.load(DEPLOYED, map_location="cpu"), strict=True)
        with torch.inference_mode():
            self.assertTrue(torch.equal(new(x), old.eval()(x)))


if __name__ == "__main__":
    unittest.main()
