"""Tests for coffeecv.segment_beans, the shared bean-region step of ticket ML-2 (plan §10). Plain unittest.

    python -m unittest tests.test_segment_beans -v

The tests that build L0 skip on a clone without its weights (models_pretrained/verify.py, dvc pull).
"""
import unittest
from dataclasses import replace

import numpy as np
import torch

from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.sam_loader import L0_WEIGHTS
from coffeecv.segment_beans import BeanSegmenter, SegParams, d4_box, mask_and_crop, seg_params

HAVE_L0 = (MODELS_PRETRAINED / L0_WEIGHTS).exists()
SKIP_L0 = "EfficientViT-SAM-L0 weights missing -- run models_pretrained/verify.py / dvc pull"
P = SegParams(mask_select="multi3", min_area_frac=0.083)


def random_blobs(rng, h, w) -> np.ndarray:
    yy, xx = np.mgrid[:h, :w]
    m = np.zeros((h, w), bool)
    for _ in range(int(rng.integers(1, 5))):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        ry, rx = rng.uniform(0.05, 0.5) * h, rng.uniform(0.05, 0.5) * w
        m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    return m


class TestMaskAndCrop(unittest.TestCase):
    def test_quantile_crop_keeps_keep_frac(self):
        rng = np.random.default_rng(0)
        for _ in range(60):
            h, w = int(rng.integers(40, 300)), int(rng.integers(40, 300))
            m = random_blobs(rng, h, w)
            if not m.any():
                continue
            keep = float(rng.uniform(0.8, 0.99))
            x0, y0, bw, bh = d4_box(m, keep)
            kept = m[y0:y0 + bh, x0:x0 + bw].sum()
            self.assertGreaterEqual(kept / m.sum(), keep - 1e-12)
            # and it is a trim of the mask's bounding box, never larger
            ys, xs = np.nonzero(m)
            self.assertTrue(xs.min() <= x0 and x0 + bw - 1 <= xs.max() and ys.min() <= y0 and y0 + bh - 1 <= ys.max())

    def test_fill_crop_and_diagnostics(self):
        rgb = np.full((100, 120, 3), 200, np.uint8)
        m = np.zeros((100, 120), bool)
        m[20:80, 30:90] = True
        m[50, 50] = False                                     # a hole inside the region gets filled too
        c = mask_and_crop(rgb, m, P)
        x0, y0, bw, bh = c.info["box"]
        self.assertEqual(c.rgb.shape[:2], (bh, bw))
        self.assertTrue(np.array_equal(c.mask, m[y0:y0 + bh, x0:x0 + bw]))
        self.assertTrue((c.rgb[~c.mask] == P.fill_rgb).all())
        self.assertTrue((c.rgb[c.mask] == 200).all())
        self.assertEqual(set(c.info), {"mask_area_frac", "mask_sha256", "fallback", "box", "bean_frac_in_crop",
                                       "retained_frac"})
        self.assertFalse(c.info["fallback"])
        self.assertAlmostEqual(c.info["retained_frac"], c.mask.sum() / m.sum())

    def test_fill_is_imagenet_mean_rounded(self):
        mean = np.array([0.485, 0.456, 0.406]) * 255
        self.assertEqual(P.fill_rgb, tuple(int(round(v)) for v in mean))

    def test_empty_and_tiny_masks_fall_back(self):
        rgb = np.zeros((100, 100, 3), np.uint8)
        tiny = np.zeros((100, 100), bool)
        tiny[:8, :] = True                                    # 0.08 < 0.083
        for m in (np.zeros((100, 100), bool), tiny):
            c = mask_and_crop(rgb, m, P)
            self.assertIs(c.rgb, rgb)
            self.assertIsNone(c.mask)
            self.assertTrue(c.info["fallback"])
            self.assertIsNone(c.info["box"])
        tiny[:9, :] = True                                    # 0.09: not tiny
        self.assertFalse(mask_and_crop(rgb, tiny, P).info["fallback"])

    def test_shape_mismatch_refused(self):
        with self.assertRaises(ValueError):
            mask_and_crop(np.zeros((10, 12, 3), np.uint8), np.ones((12, 10), bool), P)


class TestSegParams(unittest.TestCase):
    def test_from_params_yaml(self):
        cfg = RunConfig.from_params_yaml()
        p = seg_params(cfg)
        self.assertEqual((p.weights, p.decoder, p.prompt, p.mask_select, p.keep_frac, p.min_area_frac),
                         (cfg.seg_weights, cfg.seg_decoder, cfg.seg_prompt, cfg.seg_mask_select, cfg.seg_keep_frac,
                          cfg.seg_min_area_frac))
        self.assertEqual((p.weights_sha256, p.decoder_sha256), (cfg.seg_weights_sha256, cfg.seg_decoder_sha256))

    def test_pretrained_decoder_and_unpinned_decoder(self):
        cfg = RunConfig.from_params_yaml()
        self.assertIsNone(seg_params(replace(cfg, seg_decoder="", seg_decoder_sha256="")).decoder)
        with self.assertRaises(ValueError):
            seg_params(replace(cfg, seg_decoder_sha256=""))


@unittest.skipUnless(HAVE_L0, SKIP_L0)
class TestSegmenter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seg = BeanSegmenter(P)

    def test_refuses_unexpected_weights_and_decoder(self):
        with self.assertRaisesRegex(ValueError, "not the"):
            BeanSegmenter(replace(P, weights_sha256="0" * 64))
        cfg = RunConfig.from_params_yaml()
        if (REPO_ROOT / cfg.seg_decoder).exists():
            with self.assertRaisesRegex(ValueError, "not the"):
                BeanSegmenter(replace(seg_params(cfg), decoder_sha256="0" * 64))

    def test_upsample_drops_the_padding_on_a_non_square_photo(self):
        """480x640 photo: SAM resizes to 768x1024 and pads the bottom to 1024x1024, so the low-res frame's
        rows 0-191 are the photo and 192-255 padding. A logit map positive on the photo's left half only
        must come back as exactly the photo's left half, not shifted or stretched by the padding."""
        pr = self.seg.predictor
        pr.original_size, pr.input_size = (480, 640), (768, 1024)
        low = torch.full((256, 256), -10.0)
        low[:192, :128] = 10.0
        low[192:, :] = 10.0                                    # padding: must not leak into the photo
        m = self.seg._upsample(low)
        self.assertEqual(m.shape, (480, 640))
        cols = np.nonzero(m.all(axis=0))[0]
        self.assertTrue(m[:, :318].all() and not m[:, 322:].any(), cols.max())
        self.assertTrue(m[-1, :318].all())                     # the last photo row is still the photo's


if __name__ == "__main__":
    unittest.main()
