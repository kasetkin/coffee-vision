"""The P5 fine-tune's geometry and loss (ticket ML-2 D7, plan §8): the training path (cached fp16 embedding ->
decoder -> logits in the 512 frame) gives the mask serving gives, the box prompt matches serving's, labels sit in
the frame the way SamPad pads the image, the loss and IoU behave on empty negatives, the params block is checked
key by key, and a fine-tuned decoder refuses other base weights. Plain unittest.

    python -m unittest tests.test_seg_finetune -v

The tests that need L0 weights skip on a clone without them (models_pretrained/verify.py, dvc pull).
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from coffeecv import seg_finetune as ft
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.sam_loader import L0_WEIGHTS

HAVE_L0 = (MODELS_PRETRAINED / L0_WEIGHTS).exists()
SKIP_L0 = "EfficientViT-SAM-L0 weights missing -- run models_pretrained/verify.py / dvc pull"


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
        lab = ft.label_in_frame(mask)
        self.assertEqual(lab.shape, (ft.FRAME, ft.FRAME))
        self.assertTrue((lab[:256, :] == 1).all())
        self.assertTrue((lab[256:, :] == ft.IGNORE).all())

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
        lab = torch.full((ft.FRAME, ft.FRAME), ft.IGNORE, dtype=torch.uint8)
        lab[:100] = 0
        good = torch.full((ft.FRAME, ft.FRAME), -20.0)
        good[100:] = 20.0                                    # wrong only on the padding
        ls = ft.losses(good, torch.tensor(1.0), lab, self.P)
        self.assertLess(float(ls["total"]), 1e-3)

    def test_wrong_prediction_costs_more(self):
        lab = torch.zeros((ft.FRAME, ft.FRAME), dtype=torch.uint8)
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

    def test_output_index(self):
        self.assertEqual([ft.output_index(s) for s in ("single", "multi1", "multi3")], [0, 1, 3])
        with self.assertRaises(ValueError):
            ft.output_index("best_iou")


@unittest.skipUnless(HAVE_L0, SKIP_L0)
class ServingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from coffeecv.segment_beans import BeanSegmenter, SegParams
        cls.seg = BeanSegmenter(SegParams(mask_select="multi3", prompt="box"))
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
        lab = torch.from_numpy(ft.label_in_frame(serving))
        valid = lab != ft.IGNORE
        self.assertGreater(float(ft.hard_iou(logits[valid] > 0, lab[valid] == 1)), 0.98)

    def test_decoder_from_other_weights_is_refused(self):
        from coffeecv.segment_beans import load_decoder
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            f = Path(tmp) / "d.pt"
            torch.save({"mask_decoder": self.seg.model.mask_decoder.state_dict(), "base_weights_sha256": "0" * 64}, f)
            with self.assertRaises(ValueError):
                load_decoder(self.seg.model, str(f.relative_to(REPO_ROOT)), self.seg.weights_sha256)


if __name__ == "__main__":
    unittest.main()
