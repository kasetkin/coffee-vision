"""Train/serve parity of ticket ML-2's "segment" crop_method (plan §6, §10). Plain unittest.

    python -m unittest tests.test_seg_parity -v

The segcrop stage (training) and infer.patches_for_photo (serving) must cut the same patches from the same
photo: the stage's arrays, before encoding, equal the serving path's in-memory ones, and
MultiPhotoPatchDataset reading those arrays places the same boxes as serving for the same RNG key. The crop
is written losslessly here, so a difference is the code's, not the JPEG q95 re-encode training pools get
(an existing train/serve difference this ticket does not widen).

Needs the L0 weights, the seed-123 decoder and a deploy fixture photo; skips without them.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from coffeecv import infer
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.bean_scale import pitch_kwargs
from coffeecv.class_list import folder_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import (CAPTURES, SPLIT_SEED_COMPONENT, Capture, MultiPhotoPatchDataset, bean_mask_path,
                              bean_share_rule, load_rgb_image)
from coffeecv.merge_rig import merge_rig
from coffeecv.repro_utils import CAPTURE_STAGE_OVERRIDES, _stage_key, stale_crop_stages
from coffeecv.segcrop_session import crop_photo, write_crop

CFG = replace(RunConfig.from_params_yaml(), crop_method="segment")
# A tray photo (iPhone, HEIC) from webapp/deploy/fixtures.txt: the mask is the pile, so fill and the D17
# rule both matter.
PHOTO = REPO_ROOT / "dataset/2026-08-25__iphone/class_003__Colombia_PinkBourbon/IMG_6258D.HEIC"
HAVE = all(p.exists() for p in (MODELS_PRETRAINED / CFG.seg_weights, REPO_ROOT / CFG.seg_decoder, PHOTO))
SKIP = "needs the L0 weights, models/seg/ft_s123.pt and the fixture photo (dvc pull)"
N = 40


def dataset_patches(cap_dir: Path, cfg: RunConfig) -> MultiPhotoPatchDataset:
    """`cap_dir`/class_003__X/*__cropped.jpg as a one-photo, split="all" capture: RNG key
    [seed, 0, 0, 0, SPLIT_SEED_COMPONENT["all"]]."""
    classes = cap_dir.parent / "classes.txt"
    classes.write_text("003;Colombia,PinkBourbon\n")
    budget = {"train": N, "val": N, "test": N, "all": N}
    return MultiPhotoPatchDataset(
        captures=[Capture("cap", cap_dir)], classes=folder_classes(classes), split="all", seed=cfg.seed,
        crop_size=cfg.patch_crop_size, resize=cfg.patch_resize, safety_margin=cfg.safety_margin,
        patches_per_class=budget, photo_frac={"train": 0.7, "val": 0.15, "test": 0.15},
        patch_store_size=cfg.patch_store_size, patch_beans=(cfg.patch_beans_min, cfg.patch_beans_max),
        pitch_geometry=pitch_kwargs(cfg), bean_share_rule=bean_share_rule(cfg))


@unittest.skipUnless(HAVE, SKIP)
class TestSegmentParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stage = crop_photo(PHOTO, infer.segmenter_for(CFG))
        cls.key = [CFG.seed, 0, 0, 0, SPLIT_SEED_COMPONENT["all"]]

    def write_capture(self, tmp: Path, rgb: np.ndarray, crop=None) -> Path:
        cap = tmp / "cap"
        out = cap / "class_003__Colombia_PinkBourbon" / f"{PHOTO.stem}__cropped.jpg"
        out.parent.mkdir(parents=True)
        if crop is not None:
            write_crop(crop, out)                              # the stage's own mask encoding
        Image.fromarray(rgb).save(out, format="PNG")          # lossless, under the stage's name
        return cap

    def test_stage_arrays_equal_serving(self):
        rgb, mask, crop_info, diag = infer.segment_bean_region(load_rgb_image(PHOTO), CFG)
        self.assertFalse(self.stage.info["fallback"])
        self.assertTrue(np.array_equal(rgb, self.stage.rgb))
        self.assertTrue(np.array_equal(mask, self.stage.mask))
        self.assertEqual(crop_info["box"], self.stage.info["box"])
        self.assertEqual(diag["seg_fallback"], False)
        self.assertLess(self.stage.info["mask_area_frac"], 0.9)  # a tray photo: the pile, not the frame

    def test_patches_equal_for_the_same_key(self):
        # The real crop is ~99.8% bean region: at 0.80 every box is accepted, and even most boxes are 100%
        # beans. A bar above 1 rejects every candidate, so all 40 come from the D17 top-up (the best of
        # 2 x 40 rejected, highest share first) -- on both sides.
        strict = replace(CFG, patch_min_bean_share=1.01, patch_max_attempts_factor=2)
        for cfg, below in ((CFG, 0), (strict, N)):
            with tempfile.TemporaryDirectory() as tmp:
                ds = dataset_patches(self.write_capture(Path(tmp), self.stage.rgb, self.stage), cfg)
                self.assertEqual(ds.fallback_photos, [])
            patches, diag = infer.patches_for_photo(PHOTO, cfg, N, self.key)
            self.assertEqual(len(ds), N)
            for a, b in zip(ds._patches, patches):
                self.assertTrue(np.array_equal(a, np.asarray(b)))
            self.assertEqual(diag["patches_below_share"], ds.n_below_share)
            self.assertEqual(diag["patches_below_share"], below)

    def test_fallback_photo_equals_skip_crop(self):
        """No mask beside a crop (D18) and the user's skip_crop both mean: whole original photo, unfilled,
        every patch accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = dataset_patches(self.write_capture(Path(tmp), load_rgb_image(PHOTO)), CFG)
            self.assertEqual(ds.fallback_photos, [f"{PHOTO.stem}__cropped.jpg"])
        patches, diag = infer.patches_for_photo(PHOTO, CFG, N, self.key, skip_crop=True)
        for a, b in zip(ds._patches, patches):
            self.assertTrue(np.array_equal(a, np.asarray(b)))
        self.assertEqual((diag["patches_below_share"], ds.n_below_share, diag["seg_fallback"]), (0, 0, None))

    def test_mask_round_trips_through_its_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "x__cropped.jpg"
            write_crop(self.stage, out)
            back = np.array(Image.open(bean_mask_path(out)).convert("1"), dtype=bool)
        self.assertTrue(np.array_equal(back, self.stage.mask))


class TestMergeCarriesMasks(unittest.TestCase):
    def test_mask_follows_its_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for session, stems in (("s1", ["a", "b"]), ("s2", ["c"])):
                d = root / session / "class_001__X"
                d.mkdir(parents=True)
                for s in stems:
                    (d / f"{s}__cropped.jpg").write_bytes(b"jpg" + s.encode())
                    if s != "b":                               # b is a D18 fallback: no mask
                        (d / f"{s}__beanmask.png").write_bytes(b"png" + s.encode())
            merge_rig("cam_x", ["s1", "s2"], root)
            out = root / "cam_x" / "class_001__X"
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["a__beanmask.png", "a__cropped.jpg", "b__cropped.jpg", "c__beanmask.png",
                              "c__cropped.jpg"])
            self.assertEqual((out / "c__beanmask.png").read_bytes(), b"pngc")


class TestSegPoolsWiring(unittest.TestCase):
    def test_the_pools_are_the_segmenters(self):
        """Ticket ML-3 P2b: the tray heuristic's pools are retired; training reads the segmenter's only."""
        self.assertEqual({Path(p).parent for p in CAPTURES}, {Path("data/segcropped")})
        self.assertEqual(RunConfig.from_params_yaml().crop_method, "segment")

    def test_pools_and_crop_method_must_agree(self):
        """A config naming neither field gets the segmenter's pools and the tray heuristic (old cards need that
        default): sampled so, the pools would skip their masks' patch rule (D17). It is refused when read."""
        with self.assertRaisesRegex(ValueError, "crop_method"):
            RunConfig().resolve_paths()
        with self.assertRaisesRegex(ValueError, "crop_method"):
            replace(RunConfig(), train_capture_dirs=("data/cropped/cam_sony",), crop_method="segment").resolve_paths()
        replace(RunConfig(), crop_method="segment").resolve_paths()
        replace(RunConfig(), train_capture_dirs=("data/cropped/cam_sony",)).resolve_paths()     # an old card

    def test_staleness_check_names_real_stages(self):
        """Every stage the fit's staleness check asks dvc about for the segmenter's pools is in dvc.yaml,
        and every segcrop session feeds one of them."""
        stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"]
        names = set(stages) | {f"segcrop@{s}" for s in stages["segcrop"]["foreach"]}
        named = [st for p in CAPTURES for st in CAPTURE_STAGE_OVERRIDES[_stage_key(p)]]
        self.assertLessEqual(set(named), names)
        self.assertEqual({st for st in named if st.startswith("segcrop@")},
                         {f"segcrop@{s}" for s in stages["segcrop"]["foreach"]})
        for p in CAPTURES:                                              # and per pool, the sessions it merges
            merge, *segcrops = CAPTURE_STAGE_OVERRIDES[_stage_key(p)]
            cmd = stages[merge]["cmd"].split()
            self.assertEqual(cmd[cmd.index("--name") + 1], Path(p).name)
            self.assertEqual([f"segcrop@{s}" for s in cmd[cmd.index("--sessions") + 1:]], segcrops, p)
        with self.assertRaisesRegex(ValueError, "segmenter"):           # a retired pool: nothing tracks it
            stale_crop_stages(["data/cropped/cam_pixel"])


if __name__ == "__main__":
    unittest.main()
