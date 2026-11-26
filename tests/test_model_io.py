"""Tests for coffeecv.model_io: each port against the torch code it replaces (ticket ML-7 D5 (b), plan P8).
Plain unittest.

Run from the repo root:  python -m unittest tests.test_model_io -v

The normalisations, pad and box must be bit-identical. The resizes give float since ML-37 D11 (b): they are held to
torch's antialiased bilinear within 0.01 of a level (the oracle D11's research named), and to today's 8-bit resize
within one level (one rounding). The mask's upsampling and the head round differently from torch, so they are held
to tolerances: the mask to ML-7 D7's IoU bar, the head to float rounding.
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


LEVEL = 1 / 255
ORACLE_LEVELS = 0.01      # the float resize against torch's antialiased bilinear (ML-37 D11's research: <= 0.006)


def _torch_resize(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """The oracle (ML-37 D11's research): torch's antialiased bilinear, which its documentation says matches Pillow's
    downsampling, on the 8-bit picture / 255, HxWx3 float32."""
    x = torch.from_numpy(rgb.astype(np.float32) / 255).permute(2, 0, 1)[None]
    return F.interpolate(x, (h, w), mode="bilinear", align_corners=False, antialias=True)[0].permute(1, 2, 0).numpy()


class TestEvalPatch(unittest.TestCase):
    """ML-37 D11 (b): the stored 8-bit patch to the model's input with no rounding after the resize."""

    def test_eval_resize_against_torchs_antialiased_bilinear(self):
        for seed, side in enumerate((900, 448, 97)):
            rgb = _smooth_rgb(side, side, seed)
            with self.subTest(side=side):
                ours = model_io.eval_resize(Image.fromarray(rgb), 224)
                self.assertEqual((ours.dtype, ours.shape), (np.float32, (224, 224, 3)))
                self.assertLessEqual(float(np.abs(ours - _torch_resize(rgb, 224, 224)).max()), ORACLE_LEVELS * LEVEL)
                np.testing.assert_array_equal(model_io.eval_resize(rgb, 224), ours)   # an array is the same patch

    def test_eval_resize_within_one_level_of_todays_8_bit_resize(self):
        for seed, side in enumerate((900, 448, 97)):
            patch = Image.fromarray(_smooth_rgb(side, side, seed))
            today = np.asarray(patch.resize((224, 224), Image.BILINEAR), dtype=np.float32)
            with self.subTest(side=side):
                self.assertLessEqual(float(np.abs(model_io.eval_resize(patch, 224) * 255 - today).max()), 1.0)

    def test_a_patch_already_at_size_is_not_resampled(self):
        rgb = _smooth_rgb(224, 224, 0)
        np.testing.assert_array_equal(model_io.eval_resize(rgb, 224), rgb.astype(np.float32) / np.float32(255))

    def test_eval_patch_is_torchvisions_eval_transform_on_the_float_resize(self):
        import torchvision.transforms as T
        norm = T.Normalize(mean=model_io.IMAGENET_MEAN, std=model_io.IMAGENET_STD)
        for seed, side in enumerate((900, 448, 224, 97)):
            rgb = _smooth_rgb(side, side, seed)
            patch = Image.fromarray(rgb)
            with self.subTest(side=side):
                ours = model_io.eval_patch(patch, 224)
                self.assertEqual((ours.dtype, ours.shape), (np.float32, (3, 224, 224)))
                ref = norm(torch.from_numpy(_torch_resize(rgb, 224, 224)).permute(2, 0, 1)).numpy()
                self.assertLessEqual(float(np.abs(ours - ref).max()), _normalised_bound(model_io.IMAGENET_STD))
                np.testing.assert_array_equal(build_eval_transform(224)(patch).numpy(), ours)

    def test_a_patch_at_size_is_todays_input_to_the_bit(self):
        import torchvision.transforms as T
        patch = Image.fromarray(_smooth_rgb(224, 224, 0))
        tf = T.Compose([T.ToTensor(), T.Normalize(mean=model_io.IMAGENET_MEAN, std=model_io.IMAGENET_STD)])
        np.testing.assert_array_equal(model_io.eval_patch(patch, 224), tf(patch).numpy())

    def test_is_the_shared_resize_then_the_input_normalisation(self):
        patch = Image.fromarray(_smooth_rgb(448, 448, 0))
        with mock.patch.object(model_io, "eval_resize", wraps=model_io.eval_resize) as resize, \
                mock.patch.object(model_io, "eval_input", wraps=model_io.eval_input) as to_input:
            out = model_io.eval_patch(patch, 224)
        resize.assert_called_once_with(patch, 224)
        to_input.assert_called_once()
        np.testing.assert_array_equal(out, model_io.eval_input(model_io.eval_resize(patch, 224)))


class TestSamResize(unittest.TestCase):
    """ML-37 D11 (b): the 8-bit photo in, float32 in [0, 1] out, by PIL's bilinear with no rounding after it."""

    def test_against_torchs_antialiased_bilinear(self):
        for seed, (h, w, side) in enumerate(((3000, 4000, 1024), (4032, 3024, 1024), (512, 384, 1024),
                                             (700, 500, 512))):
            rgb = _smooth_rgb(h, w, seed)
            nh, nw = model_io.sam_preprocess_shape(h, w, side)
            with self.subTest(hw=(h, w), side=side):
                ours = model_io.sam_resize(rgb, side)
                self.assertEqual((ours.dtype, ours.shape), (np.float32, (nh, nw, 3)))
                self.assertLessEqual(float(ours.max()), 1.0)
                self.assertGreaterEqual(float(ours.min()), 0.0)
                self.assertLessEqual(float(np.abs(ours - _torch_resize(rgb, nh, nw)).max()), ORACLE_LEVELS * LEVEL)

    def test_a_photo_already_at_size_is_only_scaled_to_one(self):
        rgb = _smooth_rgb(768, 1024, 3)
        np.testing.assert_array_equal(model_io.sam_resize(rgb, 1024), rgb.astype(np.float32) / np.float32(255))

    def test_within_one_level_of_todays_8_bit_samresize(self):
        sam = _efficientvit_sam()
        for seed, (h, w, side) in enumerate(((3000, 4000, 512), (4032, 3024, 1024), (512, 384, 1024))):
            rgb = _smooth_rgb(h, w, seed)
            with self.subTest(hw=(h, w), side=side):
                today = sam.SamResize(side)(rgb).astype(np.float32)
                self.assertLessEqual(float(np.abs(model_io.sam_resize(rgb, side) * 255 - today).max()), 1.0)

    def test_a_picture_not_8_bit_is_refused(self):
        with self.assertRaisesRegex(TypeError, "uint8"):
            model_io.sam_resize(np.zeros((300, 400, 3), np.float32), 1024)


def _sam_oracle(rgb: np.ndarray, side: int) -> np.ndarray:
    """EfficientViT-SAM's transform with ML-37 D11's float resize: torch's antialiased bilinear in SamResize's place,
    then its ToTensor-less Normalize and SamPad, [1, 3, side, side]."""
    import torchvision.transforms as T
    sam = _efficientvit_sam()
    nh, nw = model_io.sam_preprocess_shape(*rgb.shape[:2], side)
    x = torch.from_numpy(_torch_resize(rgb, nh, nw)).permute(2, 0, 1)
    x = T.Normalize(mean=model_io.SAM_MEAN, std=model_io.SAM_STD)(x)
    return sam.SamPad(side)(x).unsqueeze(0).numpy()


def _normalised_bound(std: list[float]) -> float:
    """ORACLE_LEVELS after Normalize divides by the smallest std, with float32's own rounding beside it."""
    return ORACLE_LEVELS * LEVEL / min(std) + 1e-6


class TestSamInput(unittest.TestCase):
    def test_efficientvit_transform_with_the_float_resize(self):
        for seed, (h, w) in enumerate(((3000, 4000), (4032, 3024), (1024, 768), (500, 700), (1024, 1024))):
            rgb = _smooth_rgb(h, w, seed)
            with self.subTest(hw=(h, w)):
                x, input_size = model_io.sam_input(rgb, 1024)
                self.assertEqual((x.dtype, x.shape), (np.float32, (1, 3, 1024, 1024)))
                self.assertLessEqual(float(np.abs(x - _sam_oracle(rgb, 1024)).max()),
                                     _normalised_bound(model_io.SAM_STD))
                self.assertEqual(input_size, model_io.sam_preprocess_shape(h, w, 1024))

    def test_a_photo_at_the_frame_is_todays_input_to_the_bit(self):
        rgb = _smooth_rgb(768, 1024, 0)
        np.testing.assert_array_equal(model_io.sam_input(rgb, 1024)[0], _sam_transform(1024)(rgb).unsqueeze(0).numpy())

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

    def test_sam_input_against_the_float_oracle(self):
        for name, rgb in self.photos:
            with self.subTest(photo=name):
                x, _ = model_io.sam_input(rgb, 1024)
                self.assertLessEqual(float(np.abs(x - _sam_oracle(rgb, 1024)).max()),
                                     _normalised_bound(model_io.SAM_STD))

    def test_sam_input_within_one_level_of_todays(self):
        """Today's 8-bit SamResize against the float resize, in levels: one rounding, no filter change."""
        tf = _sam_transform(1024)
        scale = np.asarray(model_io.SAM_STD, np.float32)[None, :, None, None] * 255
        for name, rgb in self.photos:
            with self.subTest(photo=name):
                d = (model_io.sam_input(rgb, 1024)[0] - tf(rgb).unsqueeze(0).numpy()) * scale
                self.assertLessEqual(float(np.abs(d).max()), 1.0 + 1e-3)


if __name__ == "__main__":
    unittest.main()
