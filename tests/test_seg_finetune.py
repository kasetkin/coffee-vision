"""The P5 fine-tune's geometry and loss (ticket ML-2 D7, plan §8): the training path (cached fp16 embedding ->
decoder -> logits in the encoder's frame) gives the mask serving gives, the box prompt matches serving's, labels sit
in the frame the way SamPad pads the image, the loss and IoU behave on empty negatives, the params block is checked
key by key, and a fine-tuned decoder refuses other base weights. ML-5 P4: the frame follows the base weights'
variant (512 for L0, 1024 for XL0), in the cache and in training. Plain unittest.

    python -m unittest tests.test_seg_finetune -v

The tests that need L0 weights skip on a clone without them (models_pretrained/verify.py, dvc pull).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from coffeecv import seg_finetune as ft
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.sam_loader import L0_WEIGHTS, weights_for

from tests._tiers import real_data

HAVE_L0 = (MODELS_PRETRAINED / L0_WEIGHTS).exists()
SKIP_L0 = "EfficientViT-SAM-L0 weights missing -- run models_pretrained/verify.py / dvc pull"
XL0_WEIGHTS = weights_for("xl0")
HAVE_XL0 = (MODELS_PRETRAINED / XL0_WEIGHTS).exists()
SKIP_XL0 = "EfficientViT-SAM-XL0 weights missing -- dvc pull"
F = 512                                   # a frame for the loss tests; any size works


def pile_photo(h: int = 600, w: int = 900) -> np.ndarray:
    """A non-square image with a textured blob on a flat background: something SAM segments."""
    rng = np.random.default_rng(1)
    img = np.full((h, w, 3), (200, 205, 210), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    blob = ((yy - h * 0.55) / (h * 0.3)) ** 2 + ((xx - w * 0.45) / (w * 0.25)) ** 2 < 1
    tex = np.kron(rng.integers(40, 140, (h // 12 + 1, w // 12 + 1, 1)), np.ones((12, 12, 3)))[:h, :w]
    img[blob] = tex[blob].astype(np.uint8)
    return img


class Geometry(unittest.TestCase):
    def test_label_frame_pads_bottom_right(self):
        mask = np.ones((300, 600), bool)
        for frame, rows in ((512, 256), (1024, 512)):
            lab = ft.label_in_frame(mask, frame)
            self.assertEqual(lab.shape, (frame, frame))
            self.assertTrue((lab[:rows, :] == 1).all())
            self.assertTrue((lab[rows:, :] == ft.IGNORE).all())

    def test_views_are_dihedral_and_invertible_in_shape(self):
        a = np.arange(6).reshape(2, 3)
        for name, f in ft.VIEWS.items():
            self.assertEqual(sorted(f(a).ravel()), list(range(6)), name)
        self.assertEqual(len({ft.VIEWS[v](a).tobytes() + bytes(ft.VIEWS[v](a).shape) for v in ft.VIEWS}), 8)


class Loss(unittest.TestCase):
    P = ft.FtParams()

    def test_empty_label_and_empty_prediction_have_iou_one(self):
        self.assertEqual(float(ft.hard_iou(torch.zeros(5, dtype=bool), torch.zeros(5, dtype=bool))), 1.0)

    def test_padding_is_ignored(self):
        lab = torch.full((F, F), ft.IGNORE, dtype=torch.uint8)
        lab[:100] = 0
        good = torch.full((F, F), -20.0)
        good[100:] = 20.0                                    # wrong only on the padding
        ls = ft.losses(good, torch.tensor(1.0), lab, self.P)
        self.assertLess(float(ls["total"]), 1e-3)

    def test_wrong_prediction_costs_more(self):
        lab = torch.zeros((F, F), dtype=torch.uint8)
        lab[:200] = 1
        right = torch.where(lab == 1, 10.0, -10.0)
        self.assertLess(float(ft.losses(right, torch.tensor(1.0), lab, self.P)["total"]),
                        float(ft.losses(-right, torch.tensor(0.0), lab, self.P)["total"]))


class Params(unittest.TestCase):
    def test_unknown_key_is_an_error(self):
        with self.assertRaises(ValueError):
            ft.FtParams.from_config(RunConfig(seg_ft={"lr": 1e-4, "lrr": 1}))
        with self.assertRaises(ValueError):
            ft.FtParams.from_config(RunConfig(seg_ft={"views": ["hflip", "id"]}))

    def test_base_weights_default_to_l0_and_name_a_variant(self):
        self.assertEqual(ft.FtParams.from_config(RunConfig(seg_ft={})).weights, L0_WEIGHTS)
        self.assertEqual(ft.FtParams.from_config(RunConfig(seg_ft={"weights": XL0_WEIGHTS})).weights, XL0_WEIGHTS)
        with self.assertRaisesRegex(ValueError, "variant"):
            ft.FtParams.from_config(RunConfig(seg_ft={"weights": "efficientvit_sam/other.pt"}))

    def test_output_index(self):
        self.assertEqual([ft.output_index(s) for s in ("single", "multi1", "multi3")], [0, 1, 3])
        with self.assertRaises(ValueError):
            ft.output_index("best_iou")


@unittest.skipUnless(HAVE_L0, SKIP_L0)
class ServingParity(unittest.TestCase):
    WEIGHTS = L0_WEIGHTS

    @classmethod
    def setUpClass(cls):
        from coffeecv.segment_beans import BeanSegmenter, SegParams
        cls.seg = BeanSegmenter(SegParams(mask_select="multi3", prompt="box", weights=cls.WEIGHTS))
        cls.rgb = pile_photo()

    def test_box_matches_the_serving_prompt(self):
        pr = self.seg.predictor
        pr.original_size = self.rgb.shape[:2]
        pr.input_size = ft.frame_size(*self.rgb.shape[:2], ft.PROMPT_FRAME)
        h, w = self.rgb.shape[:2]
        serving = pr.apply_boxes_torch(torch.tensor([[0, 0, w - 1, h - 1]], dtype=torch.float))[0].numpy()
        np.testing.assert_allclose(ft.whole_box(h, w), serving, atol=1e-4)

    def test_training_path_gives_the_serving_mask(self):
        model = self.seg.model
        serving = self.seg.predict_mask(self.rgb)
        with torch.no_grad():
            emb = model.image_encoder(model.transform(self.rgb).unsqueeze(0)).half().float()
            logits, _ = ft.decode(model, emb, torch.from_numpy(ft.whole_box(*self.rgb.shape[:2]))[None], 3)
        frame = ft.encoder_frame(model)
        self.assertEqual(tuple(logits.shape), (frame, frame))
        lab = torch.from_numpy(ft.label_in_frame(serving, frame))
        valid = lab != ft.IGNORE
        self.assertGreater(float(ft.hard_iou(logits[valid] > 0, lab[valid] == 1)), 0.98)

    def test_decoder_from_other_weights_is_refused(self):
        from coffeecv.segment_beans import load_decoder
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            f = Path(tmp) / "d.pt"
            torch.save({"mask_decoder": self.seg.model.mask_decoder.state_dict(), "base_weights_sha256": "0" * 64}, f)
            with self.assertRaises(ValueError):
                load_decoder(self.seg.model, str(f.relative_to(REPO_ROOT)), self.seg.weights_sha256)



@real_data
@unittest.skipUnless(HAVE_XL0, SKIP_XL0)
class ServingParityXL0(ServingParity):
    """The same three checks on XL0, whose encoder (and so the training frame) is 1024 px."""
    WEIGHTS = XL0_WEIGHTS

    def test_frame_is_1024(self):
        self.assertEqual(ft.encoder_frame(self.seg.model), 1024)


@real_data
@unittest.skipUnless(HAVE_XL0, SKIP_XL0)
class CacheAndTrainFollowTheVariant(unittest.TestCase):
    """build_cache and train, the two DVC stages' entry points, on a toy label set of three synthetic photos with
    XL0 as the base: labels are cached in its 1024 frame, one epoch trains, and the card names XL0."""

    def test_xl0_cache_and_one_epoch(self):
        from coffeecv.seg_lists import sha256_file
        p = ft.FtParams(weights=XL0_WEIGHTS, views=("id", "hflip"), epochs=1, batch_size=2, neg_share=0.5)
        cfg = RunConfig(seg_mask_select="multi3", seg_prompt="box", seg_min_area_frac=0.083)
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            rows = []
            for i, (role, h, w) in enumerate((("train", 600, 900), ("val", 450, 300), ("neg_train", 500, 400))):
                rgb = pile_photo(h, w)
                mask = (rgb != rgb[0, 0]).any(axis=2) if role != "neg_train" else np.zeros((h, w), bool)
                Image.fromarray(rgb).save(tmp / f"{i}.png")
                Image.fromarray(mask).save(tmp / f"{i}_mask.png")
                rows.append({"id": str(i), "list": "toy", "role": role,
                             "path": str((tmp / f"{i}.png").relative_to(REPO_ROOT)),
                             "photo_sha256": sha256_file(tmp / f"{i}.png"),
                             "mask": str((tmp / f"{i}_mask.png").relative_to(REPO_ROOT)), "mask_sha256": "toy"})
            with (mock.patch.object(ft, "CACHE_ROOT", tmp / "cache"),
                  mock.patch.object(ft, "label_rows", lambda c, q: rows)):
                ft.build_cache(cfg, p)
                cache = ft.Cache()
                self.assertEqual(cache.frame, 1024)
                self.assertEqual(cache.label["hflip"].shape, (3, 1024, 1024))
                self.assertEqual(cache.info["weights"], XL0_WEIGHTS)
                res = ft.train(42, cfg, p)
                self.assertEqual(len(res["history"]), 2)
                with mock.patch.object(ft, "MODEL_DIR", tmp / "models"):
                    card = ft.write_model(res, cfg, p, cache).with_suffix(".json").read_text()
                self.assertIn(XL0_WEIGHTS, card)


if __name__ == "__main__":
    unittest.main()
