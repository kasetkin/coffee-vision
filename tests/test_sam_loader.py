"""Tests for coffeecv.sam_loader: the vendored EfficientViT-SAM-L0 builds and loads correctly. Plain unittest.

Run from the repo root:  python -m unittest tests.test_sam_loader -v
Regenerate the logit reference (only after a deliberate weights or upstream change):
                         python -m tests.test_sam_loader --regen

The weights are not in git, so the tests that need them skip on a clone that has not obtained them
(models_pretrained/verify.py, dvc pull).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from coffeecv import backbones
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.sam_loader import L0_WEIGHTS, build_sam_l0

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "sam_l0_reference.npz"
HAVE_L0 = (MODELS_PRETRAINED / L0_WEIGHTS).exists()
SKIP_L0 = "EfficientViT-SAM-L0 weights missing -- run models_pretrained/verify.py / dvc pull"


def reference_input() -> np.ndarray:
    """A fixed 480x640 RGB image with structure (blurred seeded noise), not a photo from the dataset,
    so the reference needs nothing but the weights."""
    rng = np.random.default_rng(0)
    x = rng.random((60, 80, 3))
    x = np.kron(x, np.ones((8, 8, 1)))            # 480x640 blocks
    return (x * 255).astype(np.uint8)


def reference_outputs(predictor) -> dict[str, np.ndarray]:
    """Low-res logits (every 8th pixel) and predicted IoUs for the whole-image box, both decoder modes."""
    rgb = reference_input()
    h, w = rgb.shape[:2]
    predictor.set_image(rgb)
    box = np.array([0, 0, w - 1, h - 1], float)
    out = {}
    for name, multimask in (("single", False), ("multi", True)):
        _, iou, low = predictor.predict(box=box, multimask_output=multimask, return_logits=True)
        out[f"{name}_iou"] = iou
        out[f"{name}_logits"] = low[:, ::8, ::8]
    return out


class TestImportChain(unittest.TestCase):
    def test_builds_without_triton_omegaconf_onnx(self):
        """In a fresh interpreter with the three patched-out modules blocked (PATCHES.md), so an
        upstream import creeping back fails here, not in the deploy."""
        code = ("import sys\n"
                "for m in ('triton', 'omegaconf', 'onnx', 'onnxsim'): sys.modules[m] = None\n"
                "from coffeecv.sam_loader import _import_efficientvit_sam\n"
                "create, _ = _import_efficientvit_sam()\n"
                "m = create('efficientvit-sam-l0', pretrained=False)\n"
                "print(sum(p.numel() for p in m.parameters()))\n")
        r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertEqual(int(r.stdout.split()[-1]), 34_792_784)     # L0, 34.8M (upstream README)


class TestWeightsRefusal(unittest.TestCase):
    def test_refuses_a_file_not_in_the_manifest(self):
        with self.assertRaisesRegex(ValueError, "not in models_pretrained/manifest.json"):
            build_sam_l0("efficientvit_sam/not_listed.pt")

    def test_refuses_a_sha256_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "efficientvit_sam").mkdir()
            (d / L0_WEIGHTS).write_bytes(b"not the weights")
            (d / "manifest.json").write_text(json.dumps([{"path": L0_WEIGHTS, "sha256": "0" * 64}]))
            with mock.patch.object(backbones, "MODELS_PRETRAINED", d):
                with self.assertRaisesRegex(ValueError, "does not match the manifest"):
                    build_sam_l0()


@unittest.skipUnless(HAVE_L0, SKIP_L0)
class TestL0Weights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor, cls.sha = build_sam_l0()

    def test_sha256_is_the_manifests(self):
        manifest = {e["path"]: e for e in json.loads((MODELS_PRETRAINED / "manifest.json").read_text())}
        self.assertEqual(self.sha, manifest[L0_WEIGHTS]["sha256"])

    def test_frozen_and_eval(self):
        m = self.predictor.model
        self.assertFalse(m.training)
        self.assertFalse(any(p.requires_grad for p in m.parameters()))

    def test_matches_stored_reference(self):
        """The loaded model reproduces the stored logits. atol 1e-3: float sums reorder between CPUs
        (AVX2 vs AVX-512), as tests/test_dino_backbone.py found; a wrong or half-loaded checkpoint is
        off by whole units."""
        ref = np.load(FIXTURE)
        with torch.no_grad():
            out = reference_outputs(self.predictor)
        for k, v in out.items():
            np.testing.assert_allclose(v, ref[k], atol=1e-3, rtol=0, err_msg=k)


if __name__ == "__main__":
    if "--regen" in sys.argv:
        predictor, _ = build_sam_l0()
        with torch.no_grad():
            np.savez_compressed(FIXTURE, **reference_outputs(predictor))
        print(f"wrote {FIXTURE}")
    else:
        unittest.main()
