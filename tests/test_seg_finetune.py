"""The P5 fine-tune's geometry and loss (ticket ML-2 D7, plan §8): the training path (cached fp16 embedding ->
decoder -> logits in the encoder's frame) gives the mask serving gives, the box prompt matches serving's, labels sit
in the frame the way SamPad pads the image, the loss and IoU behave on empty negatives, the params block is checked
key by key, and a fine-tuned decoder refuses other base weights. ML-5 P4: the frame follows the base weights'
variant (512 for L0, 1024 for XL0), in the cache and in training. ML-5 P8: roles come from a labelling session's
splits, and point prompts drawn from the error region decode as the review page's redraw does. Plain unittest.

    python -m unittest tests.test_seg_finetune -v

The tests that need L0 weights skip on a clone without them (models_pretrained/verify.py, dvc pull).
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml
from PIL import Image

from coffeecv import seg_finetune as ft
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.config import PARAMS_FILE, REPO_ROOT, RunConfig
from coffeecv.repo_files import sha256_file
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
    P = ft.FtParams(weights=L0_WEIGHTS)

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


def label_csv(rows: list[tuple[str, str]]) -> str:
    """A session's labels.csv with one photo per (status, split)."""
    head = "id,path,photo_sha256,source,split,status,round,mask,mask_sha256,include,exclude,reason\n"
    return head + "".join(f"p{i},dataset/x/p{i}.jpg,{i:064x},pool,{split},{status},,m{i}.png,{i:064x},,,\n"
                          for i, (status, split) in enumerate(rows))


class Roles(unittest.TestCase):
    """ML-5 D8, D14: training positives train, validation positives and negatives only choose the checkpoint."""

    def rows(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "s").mkdir()
            (Path(tmp) / "s" / "labels.csv").write_text(label_csv(rows))
            with mock.patch.object(ft, "LABELS_ROOT", Path(tmp)):
                return ft.label_rows("s")

    def test_roles_follow_status_and_split_and_drops_are_left_out(self):
        got = self.rows([("accepted", "train"), ("accepted", "validation"), ("negative", "train"),
                         ("negative", "validation"), ("dropped", "train")])
        self.assertEqual([r["role"] for r in got], ["train", "val", "neg_train", "neg_select"])

    def test_pending_or_test_photos_are_refused(self):
        with self.assertRaisesRegex(ValueError, "pending"):
            self.rows([("accepted", "train"), ("pending", "train")])
        with self.assertRaisesRegex(ValueError, "outside training and validation"):
            self.rows([("accepted", "test")])


class Points(unittest.TestCase):
    """D15: points come from the error region of the box-only prediction, labelled by the label there."""

    def test_points_sit_on_errors_with_the_labels_class(self):
        lab = torch.full((F, F), ft.IGNORE, dtype=torch.uint8)
        lab[:256] = 0
        lab[:100, :100] = 1                                   # bean the prediction misses
        logits = torch.full((F, F), -5.0)
        logits[200:300, 300:400] = 5.0                        # predicted bean: 200-255 wrong, 256+ is padding
        coords, labels = ft.error_points(logits, lab, 200, np.random.default_rng(0))
        self.assertEqual((tuple(coords.shape), tuple(labels.shape)), ((1, 200, 2), (1, 200)))
        xy = (coords[0].numpy() * F / ft.PROMPT_FRAME - 0.5).round().astype(int)
        for (x, y), c in zip(xy, labels[0].tolist()):
            if c == 1:
                self.assertTrue(x < 100 and y < 100)
            else:
                self.assertTrue(200 <= y < 256 and 300 <= x < 400)
        self.assertEqual(set(labels[0].tolist()), {0, 1})

    def test_no_error_no_points(self):
        lab = torch.zeros((F, F), dtype=torch.uint8)
        self.assertIsNone(ft.error_points(torch.full((F, F), -5.0), lab, 3, np.random.default_rng(0)))

    def test_frame_to_prompt_scales_pixel_centres(self):
        np.testing.assert_allclose(ft.frame_to_prompt([[0, 511]], 512), [[1.0, 1023.0]])
        np.testing.assert_allclose(ft.frame_to_prompt([[0, 1023]], 1024), [[0.5, 1023.5]])


class Params(unittest.TestCase):
    def test_unknown_key_is_an_error(self):
        with self.assertRaises(ValueError):
            ft.FtParams.from_config(RunConfig(seg_ft={"lr": 1e-4, "lrr": 1}))
        with self.assertRaises(ValueError):
            ft.FtParams.from_config(RunConfig(seg_ft={"views": ["hflip", "id"]}))
        with self.assertRaisesRegex(ValueError, "point_share"):
            ft.FtParams.from_config(RunConfig(seg_ft={"weights": XL0_WEIGHTS, "point_share": 1.5}))

    def test_base_weights_are_params_yamls_and_name_a_variant(self):
        with self.assertRaisesRegex(ValueError, "seg_ft.weights"):
            ft.FtParams.from_config(RunConfig(seg_ft={}))
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

    def test_points_decode_as_the_review_pages_redraw(self):
        """Box plus an include and an exclude point, through multi3 (D23): the training path's mask is the one
        segment_beans.decode_points (the review page's redraw) draws from the same points."""
        model, h, w = self.seg.model, *self.rgb.shape[:2]
        frame = ft.encoder_frame(model)
        fh, fw = ft.frame_size(h, w, frame)
        in_xy, ex_xy = (0.45 * fw, 0.55 * fh), (0.05 * fw, 0.05 * fh)
        to_frac = [((x + 0.5) * max(h, w) / frame / (w - 1), (y + 0.5) * max(h, w) / frame / (h - 1))
                   for x, y in (in_xy, ex_xy)]
        self.seg.predict_mask(self.rgb)
        serving, _ = self.seg.decode_points([to_frac[0]], [to_frac[1]], "multi3")
        with torch.no_grad():
            emb = model.image_encoder(model.transform(self.rgb).unsqueeze(0)).half().float()
            points = (torch.from_numpy(ft.frame_to_prompt([in_xy, ex_xy], frame))[None], torch.tensor([[1, 0]]))
            logits, _ = ft.decode(model, emb, torch.from_numpy(ft.whole_box(h, w))[None], 3, points)
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


class DvcStagesReadTheBaseWeights(unittest.TestCase):
    """dvc.yaml's seg_ml5_cache@<session> and seg_ml5_finetune@<name> depend on models_pretrained/${seg_ft.weights},
    the file seg_finetune builds from, resolved by DVC itself (ticket ML-5 P4, P8); so does seg_ml5_predict@<name>,
    a decoder fine-tuned over it, so that the fine-tune's base is one edit of params.yaml."""

    def test_the_interpolated_dependency_is_the_file_seg_finetune_builds_from(self):
        from dvc.repo import Repo
        weights = ft.FtParams.from_config(RunConfig.from_params_yaml()).weights
        with Repo(str(REPO_ROOT)) as repo:
            for name in ("seg_ml5_cache@pass1", "seg_ml5_cache@pass2", "seg_ml5_finetune@xl0_v1",
                         "seg_ml5_finetune@xl0_v2", "seg_ml5_predict@xl0_v1", "seg_ml5_predict@xl0_v2"):
                with self.subTest(name):
                    stage = repo.stage.collect(name)[0]
                    pretrained = [d.def_path for d in stage.deps if d.def_path.startswith("models_pretrained/")]
                    self.assertEqual(pretrained, [f"models_pretrained/{weights}"])
                    module = "seg_predict" if "predict" in name else "seg_finetune"
                    self.assertTrue(stage.cmd.startswith(f"python -m coffeecv.{module} "))


class EntryPointAtToyScale:
    """`python -m coffeecv.seg_finetune cache --labels toy` then `train --labels toy --seed 42 --name toy_s42`, the
    two DVC stages' commands, through
    main() and a params.yaml that is the repo's with seg_ft at toy scale and its base weights set to WEIGHTS
    (CODING_STANDARDS "Run the real entry point"). The label set is three synthetic photos: labels are cached in
    the variant's frame, one epoch trains with every batch adding points, and the card names the base weights."""

    WEIGHTS = FRAME = None

    def test_cache_then_train_through_main(self):
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as tmp:
            tmp = Path(tmp)
            raw = yaml.safe_load(PARAMS_FILE.read_text())
            raw["seg_ft"] = {**raw["seg_ft"], "weights": self.WEIGHTS, "views": ["id", "hflip"], "epochs": 1,
                             "batch_size": 2, "point_share": 1.0}
            params = tmp / "params.yaml"
            params.write_text(yaml.safe_dump(raw, sort_keys=False))
            rows = []
            for i, (role, h, w) in enumerate((("train", 600, 900), ("val", 450, 300), ("neg_train", 500, 400))):
                rgb = pile_photo(h, w)
                mask = (rgb != rgb[0, 0]).any(axis=2) if role != "neg_train" else np.zeros((h, w), bool)
                Image.fromarray(rgb).save(tmp / f"{i}.png")
                Image.fromarray(mask).save(tmp / f"{i}_mask.png")
                rows.append({"id": str(i), "source": "toy", "split": "toy", "role": role,
                             "path": str((tmp / f"{i}.png").relative_to(REPO_ROOT)),
                             "photo_sha256": sha256_file(tmp / f"{i}.png"),
                             "mask": str((tmp / f"{i}_mask.png").relative_to(REPO_ROOT)), "mask_sha256": "toy"})
            read = RunConfig.from_params_yaml
            with (mock.patch.object(RunConfig, "from_params_yaml", lambda: read(params)),
                  mock.patch.object(ft, "CACHE_ROOT", tmp / "cache"), mock.patch.object(ft, "MODEL_DIR", tmp / "m"),
                  mock.patch.object(ft, "label_rows", lambda session: rows)):
                ft.main(["cache", "--labels", "toy", "--threads", "4"])
                ft.main(["train", "--labels", "toy", "--seed", "42", "--name", "toy_s42", "--threads", "4"])
                cache = ft.Cache("toy")
            self.assertEqual((cache.info["weights"], cache.frame), (self.WEIGHTS, self.FRAME))
            self.assertEqual(cache.label["hflip"].shape, (3, self.FRAME, self.FRAME))
            card = json.loads((tmp / "m" / "toy_s42.json").read_text())
            self.assertEqual((card["name"], card["labels"], card["base_weights"], card["seg_ft"]["epochs"],
                              len(card["history"])), ("toy_s42", "toy", self.WEIGHTS, 1, 2))
            self.assertEqual(card["history"][1]["point_steps"], 1)


@real_data
@unittest.skipUnless(HAVE_L0, SKIP_L0)
class EntryPointL0(EntryPointAtToyScale, unittest.TestCase):
    WEIGHTS, FRAME = L0_WEIGHTS, 512


@real_data
@unittest.skipUnless(HAVE_XL0, SKIP_XL0)
class EntryPointXL0(EntryPointAtToyScale, unittest.TestCase):
    WEIGHTS, FRAME = XL0_WEIGHTS, 1024


if __name__ == "__main__":
    unittest.main()
