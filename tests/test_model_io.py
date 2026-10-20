"""Tests for coffeecv.model_io: each port against the torch code it replaces (ticket ML-7 D5 (b), plan P8).
Plain unittest.

Run from the repo root:  python -m unittest tests.test_model_io -v

The resizes, normalisations, pad and box must be bit-identical. The mask's upsampling and the head round
differently from torch, so they are held to tolerances: the mask to ML-7 D7's IoU bar, the head to float rounding.
"""
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv import model_io
from coffeecv.config import REPO_ROOT
from coffeecv.transforms import build_eval_transform

from tests._tiers import real_data

EFFICIENTVIT_DIR = REPO_ROOT / "third_party" / "efficientvit"


def _efficientvit_sam():
    if str(EFFICIENTVIT_DIR) not in sys.path:
        sys.path.insert(0, str(EFFICIENTVIT_DIR))
    from efficientvit.models.efficientvit import sam
    return sam


def _sam_transform(side: int):
    """EfficientViTSam.transform, as its __init__ builds it (sam.py), without building the model."""
    import torchvision.transforms as T
    sam = _efficientvit_sam()
    return T.Compose([sam.SamResize(side), T.ToTensor(),
                      T.Normalize(mean=[123.675 / 255, 116.28 / 255, 103.53 / 255],
                                  std=[58.395 / 255, 57.12 / 255, 57.375 / 255]),
                      sam.SamPad(side)])


def _smooth_rgb(h: int, w: int, seed: int) -> np.ndarray:
    """Seeded noise blurred by upscaling, so a resize has structure to interpolate."""
    rng = np.random.default_rng(seed)
    small = (rng.random((max(h // 16, 2), max(w // 16, 2), 3)) * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(small).resize((w, h), Image.BICUBIC))


def _low_logits(seed: int) -> np.ndarray:
    """A [256, 256] logit map shaped like a decoder's: a smooth field crossing 0 along curves."""
    rng = np.random.default_rng(seed)
    coarse = rng.normal(0, 4, (16, 16)).astype(np.float32)
    return np.asarray(F.interpolate(torch.from_numpy(coarse)[None, None], (256, 256), mode="bicubic",
                                    align_corners=False)[0, 0])


def _torch_upsample(low: np.ndarray, input_size, original_size) -> np.ndarray:
    """EfficientViTSam.postprocess_masks, called unbound on a stand-in self, and the threshold as _upsample does."""
    sam = _efficientvit_sam()
    self = types.SimpleNamespace(image_size=(1024, 1024))
    with torch.inference_mode():
        out = sam.EfficientViTSam.postprocess_masks(self, torch.from_numpy(low)[None, None], input_size, original_size)
    return (out[0, 0] > 0.0).numpy()


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


class TestEvalPatch(unittest.TestCase):
    def test_bit_identical_to_torchvision(self):
        """Against the torchvision pipeline build_eval_transform held before ML-7 P8."""
        import torchvision.transforms as T
        tf = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                        T.Normalize(mean=model_io.IMAGENET_MEAN, std=model_io.IMAGENET_STD)])
        for seed, side in enumerate((900, 448, 224, 97)):
            patch = Image.fromarray(_smooth_rgb(side, side, seed))
            with self.subTest(side=side):
                ours = model_io.eval_patch(patch, 224)
                self.assertEqual(ours.dtype, np.float32)
                np.testing.assert_array_equal(ours, tf(patch).numpy())
                np.testing.assert_array_equal(build_eval_transform(224)(patch).numpy(), ours)

    def test_is_the_shared_resize_then_normalise(self):
        patch = Image.fromarray(_smooth_rgb(448, 448, 0))
        with mock.patch.object(model_io, "eval_resize", wraps=model_io.eval_resize) as resize:
            model_io.eval_patch(patch, 224)
        resize.assert_called_once_with(patch, 224)
        sized = patch.resize((224, 224))
        self.assertIs(model_io.eval_resize(sized, 224), sized)   # a patch already at size is not resampled


class TestSamInput(unittest.TestCase):
    def test_bit_identical_to_efficientvit_transform(self):
        tf = _sam_transform(1024)
        for seed, (h, w) in enumerate(((3000, 4000), (4032, 3024), (1024, 768), (500, 700), (1024, 1024))):
            rgb = _smooth_rgb(h, w, seed)
            with self.subTest(hw=(h, w)):
                x, input_size = model_io.sam_input(rgb, 1024)
                ref = tf(rgb).unsqueeze(0).numpy()
                self.assertEqual(x.shape, (1, 3, 1024, 1024))
                np.testing.assert_array_equal(x, ref)
                self.assertEqual(input_size, model_io.sam_preprocess_shape(h, w, 1024))

    def test_sam_resize_bit_identical_to_samresize(self):
        sam = _efficientvit_sam()
        for seed, (h, w, side) in enumerate(((3000, 4000, 512), (4032, 3024, 1024), (512, 384, 512))):
            rgb = _smooth_rgb(h, w, seed)
            with self.subTest(hw=(h, w), side=side):
                np.testing.assert_array_equal(model_io.sam_resize(rgb, side), sam.SamResize(side)(rgb))

    def test_input_size_matches_segment_anything(self):
        from segment_anything.utils.transforms import ResizeLongestSide
        for h, w in ((3000, 4000), (4031, 3023), (777, 1555), (1024, 1024)):
            self.assertEqual(model_io.sam_preprocess_shape(h, w, 1024),
                             ResizeLongestSide.get_preprocess_shape(h, w, 1024))


class TestSamBox(unittest.TestCase):
    def test_bit_identical_to_apply_boxes_torch(self):
        sam = _efficientvit_sam()
        for h, w in ((3000, 4000), (4032, 3024), (1023, 767)):
            input_size = model_io.sam_preprocess_shape(h, w, 1024)
            self_ = types.SimpleNamespace(original_size=(h, w), input_size=input_size)
            self_.apply_coords_torch = types.MethodType(sam.EfficientViTSamPredictor.apply_coords_torch, self_)
            box = [0, 0, w - 1, h - 1]
            ref = sam.EfficientViTSamPredictor.apply_boxes_torch(self_, torch.tensor([box], dtype=torch.float))
            with self.subTest(hw=(h, w)):
                np.testing.assert_array_equal(model_io.sam_box(box, (h, w), input_size), ref.numpy())


class TestUpsampleMask(unittest.TestCase):
    def test_within_d7_of_postprocess_masks(self):
        """D7's fp32 bar for masks is IoU >= 0.999; float rounding may move a pixel at the threshold, no more."""
        for seed, (h, w) in enumerate(((3000, 4000), (4032, 3024), (1024, 768), (500, 700))):
            low = _low_logits(seed)
            input_size = model_io.sam_preprocess_shape(h, w, 1024)
            ours = model_io.upsample_mask(low, input_size, (h, w))
            ref = _torch_upsample(low, input_size, (h, w))
            with self.subTest(hw=(h, w)):
                self.assertEqual(ours.shape, (h, w))
                self.assertGreaterEqual(_iou(ours, ref), 0.999)
                self.assertLessEqual(np.count_nonzero(ours != ref), 1e-4 * h * w)


class TestHead(unittest.TestCase):
    def test_matches_torch_linear_and_softmax(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, (40, 1536)).astype(np.float32)
        lin = torch.nn.Linear(1536, 10)
        with torch.no_grad():
            ref_logits = lin(torch.from_numpy(x))
            ref_probs = F.softmax(ref_logits, dim=1).numpy()
        logits = model_io.head_logits(x, lin.weight.detach().numpy(), lin.bias.detach().numpy())
        np.testing.assert_allclose(logits, ref_logits.numpy(), rtol=0, atol=1e-5)
        np.testing.assert_allclose(model_io.softmax(logits), ref_probs, rtol=0, atol=1e-6)


@real_data
class TestOnFixturePhotos(unittest.TestCase):
    """The deploy's fixture photos (webapp/deploy/fixtures.txt) through the input ports: real sizes, real pixels."""

    @classmethod
    def setUpClass(cls):
        from coffeecv.dataset import load_rgb_image
        lines = (REPO_ROOT / "webapp" / "deploy" / "fixtures.txt").read_text().splitlines()
        paths = [REPO_ROOT / ln.split()[2] for ln in lines if ln.strip() and not ln.startswith("#")]
        paths = [p for p in paths if p.exists()]
        if not paths:
            raise unittest.SkipTest("fixture photos missing -- dvc pull dataset")
        cls.photos = [(Path(p).name, load_rgb_image(p)) for p in paths]

    def test_sam_input_bit_identical(self):
        tf = _sam_transform(1024)
        for name, rgb in self.photos:
            with self.subTest(photo=name):
                x, _ = model_io.sam_input(rgb, 1024)
                np.testing.assert_array_equal(x, tf(rgb).unsqueeze(0).numpy())


if __name__ == "__main__":
    unittest.main()
