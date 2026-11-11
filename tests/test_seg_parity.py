"""Train/serve parity of ticket ML-2's "segment" crop_method (plan §6, §10). Plain unittest.

    python -m unittest tests.test_seg_parity -v

The segcrop stage (training) and infer.patches_for_photo (serving) must cut the same patches from the same
photo: the stage's arrays, before encoding, equal the serving path's in-memory ones, and
MultiPhotoPatchDataset reading those arrays places the same boxes as serving for the same RNG key. Since ticket
ML-19 P6 the photo is the fixture's canonical photo, which both sides read, and the stage stores its crop
losslessly (D32): the crop training reads back is the array serving computes, to the pixel.

Needs the L0 weights, the seed-123 decoder and a deploy fixture photo; skips without them.
"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import crop_io
from coffeecv import infer
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.bean_scale import pitch_kwargs
from coffeecv.class_list import folder_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import (CAPTURES, SPLIT_SEED_COMPONENT, Capture, MultiPhotoPatchDataset, bean_mask_path,
                              bean_share_rule, load_rgb_image)
from coffeecv.merge_rig import merge_cmd_args, merge_rig, read_exclusions
from coffeecv.repro_utils import POOL_MERGE_STAGE, _stage_key, capture_stage_overrides, stale_crop_stages
from coffeecv.segcrop_session import crop_photo, write_crop

from tests._tiers import real_data

CFG = replace(RunConfig.from_params_yaml(), crop_method="segment")
# A tray photo (iPhone, HEIC) from webapp/deploy/fixtures.txt: the mask is the pile, so fill and the D17
# rule both matter.
PHOTO = REPO_ROOT / "dataset/2026-08-25__iphone/class_003__Colombia_PinkBourbon/IMG_6258D.HEIC"
HAVE = all(p.exists() for p in (MODELS_PRETRAINED / CFG.seg_weights, REPO_ROOT / CFG.seg_decoder, PHOTO))
SKIP = "needs the segmenter weights and decoder params.yaml names, and the fixture photo (dvc pull)"
N = 40


def dataset_patches(cap_dir: Path, cfg: RunConfig) -> MultiPhotoPatchDataset:
    """`cap_dir`/class_003__X/*__cropped.jxl as a one-photo, split="all" capture: RNG key
    [seed, 0, 0, SPLIT_SEED_COMPONENT["all"]] (seed, class, photo, split; ticket ML-8 took the capture out)."""
    classes = cap_dir.parent / "classes.txt"
    classes.write_text("003;Colombia,PinkBourbon\n")
    budget = {"train": N, "val": N, "test": N, "all": N}
    return MultiPhotoPatchDataset(
        captures=[Capture("cap", cap_dir)], classes=folder_classes(classes), split="all", seed=cfg.seed,
        crop_size=cfg.patch_crop_size, resize=cfg.patch_resize, safety_margin=cfg.safety_margin,
        patches_per_class=budget, photo_frac={"train": 0.7, "val": 0.15, "test": 0.15},
        patch_store_size=cfg.patch_store_size, patch_beans=(cfg.patch_beans_min, cfg.patch_beans_max),
        pitch_geometry=pitch_kwargs(cfg), bean_share_rule=bean_share_rule(cfg))


@real_data
@unittest.skipUnless(HAVE, SKIP)
class TestSegmentParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.canon = Path(cls.tmp.name) / f"{PHOTO.name}.jxl"          # as the store names it
        cls.canon.write_bytes(cn.canonicalize_file(PHOTO))
        cls.stage = crop_photo(cls.canon, infer.segmenter_for(CFG))
        cls.key = [CFG.seed, 0, 0, SPLIT_SEED_COMPONENT["all"]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def write_capture(self, tmp: Path, rgb: np.ndarray, crop=None) -> Path:
        cap = tmp / "cap"
        out = cap / "class_003__Colombia_PinkBourbon" / crop_io.crop_name(PHOTO.stem)
        out.parent.mkdir(parents=True)
        if crop is not None:
            write_crop(crop, out)                              # the stage's own encodings, crop and mask
        crop_io.write_crop(rgb, out)
        return cap

    def test_the_stored_crop_is_the_array_serving_computes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / crop_io.crop_name("x")
            write_crop(self.stage, out)
            rgb, _, _, _ = infer.segment_bean_region(load_rgb_image(self.canon), CFG)
            self.assertTrue(np.array_equal(crop_io.load_crop(out), rgb))

    def test_stage_arrays_equal_serving(self):
        rgb, mask, crop_info, diag = infer.segment_bean_region(load_rgb_image(self.canon), CFG)
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
            patches, diag = infer.patches_for_photo(self.canon, cfg, N, self.key)
            self.assertEqual(len(ds), N)
            for a, b in zip(ds.patches, patches):
                self.assertTrue(np.array_equal(a, np.asarray(b)))
            self.assertEqual(diag["patches_below_share"], ds.n_below_share)
            self.assertEqual(diag["patches_below_share"], below)

    def test_fallback_photo_equals_skip_crop(self):
        """No mask beside a crop (D18) and the user's skip_crop both mean: whole original photo, unfilled,
        every patch accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            ds = dataset_patches(self.write_capture(Path(tmp), load_rgb_image(self.canon)), CFG)
            self.assertEqual(ds.fallback_photos, [crop_io.crop_name(PHOTO.stem)])
        patches, diag = infer.patches_for_photo(self.canon, CFG, N, self.key, skip_crop=True)
        for a, b in zip(ds.patches, patches):
            self.assertTrue(np.array_equal(a, np.asarray(b)))
        self.assertEqual((diag["patches_below_share"], ds.n_below_share, diag["seg_fallback"]), (0, 0, None))

    def test_mask_round_trips_through_its_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / crop_io.crop_name("x")
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
                    (d / f"{s}__cropped.jxl").write_bytes(b"jxl" + s.encode())
                    if s != "b":                               # b is a D18 fallback: no mask
                        (d / f"{s}__beanmask.png").write_bytes(b"png" + s.encode())
            merge_rig("cam_x", ["s1", "s2"], root)
            out = root / "cam_x" / "class_001__X"
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["a__beanmask.png", "a__cropped.jxl", "b__cropped.jxl", "c__beanmask.png",
                              "c__cropped.jxl"])
            self.assertEqual((out / "c__beanmask.png").read_bytes(), b"pngc")

    def test_a_session_with_two_folders_of_one_class_counts_both(self):
        """random_date_raccoon holds class_013__Brazil_SulDeMinas and class_013__Brazil_Decaff (ACCEPTED_DRIFT):
        the session's count is both folders', less its exclusions; until 2026-11-10 the second folder's count
        replaced the first's in the manifest (the images were all merged)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, stems in (("class_013__A", ["a", "b", "c"]), ("class_013__B", ["d", "e"])):
                d = root / "s1" / folder
                d.mkdir(parents=True)
                for s in stems:
                    (d / f"{s}__cropped.jxl").write_bytes(b"jxl" + s.encode())
            manifest = merge_rig("pool", ["s1"], root, {("s1", "class_013__A", "a.jpg")})
            self.assertEqual(manifest["images"], 4)
            self.assertEqual(manifest["per_class_counts"], {"013": {"s1": 4}})

    def test_excluded_photos_leave_the_pool_with_their_masks(self):
        """Ticket ML-3 P5: a declined crop stays in its session and leaves the pool; a row naming one of the
        merge's sessions but no crop there stops the merge before anything is written."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = root / "s1" / "class_001__X"
            d.mkdir(parents=True)
            for s in ("a", "b"):
                (d / f"{s}__cropped.jxl").write_bytes(b"jxl")
                (d / f"{s}__beanmask.png").write_bytes(b"png")
            manifest = merge_rig("cam_x", ["s1"], root, {("s1", "class_001__X", "a.JPG"), ("s9", "class_001__X", "z.jpg")})
            self.assertEqual(sorted(p.name for p in (root / "cam_x" / "class_001__X").iterdir()),
                             ["b__beanmask.png", "b__cropped.jxl"])
            self.assertEqual((manifest["images"], manifest["excluded"]), (1, ["s1/class_001__X/a.JPG"]))
            self.assertEqual(sorted(p.name for p in d.iterdir())[:2], ["a__beanmask.png", "a__cropped.jxl"])
            with self.assertRaisesRegex(ValueError, "match no crop"):
                merge_rig("cam_y", ["s1"], root, {("s1", "class_001__X", "nope.jpg")})
            self.assertFalse((root / "cam_y").exists())
            self.assertNotIn("excluded", merge_rig("cam_z", ["s1"], root))      # no flag: as before

    def test_committed_exclusions_follow_the_review(self):
        """labels/ml3/pool_exclude.csv is what analysis/ml3/mask_review.py derives from the committed reviews
        (ML-3 P5's 19 photos, ML-8 P4's 6 declines of the 20 new to its re-crop, and ML-9 P5's 8 among the crops of
        XL0-v4, less the 2 of ML-8's it readmits, less the 4 the owner moved to dataset/segmenter_positives on
        2026-10-24, and less the 18 of the remaining 27 the owner's re-review on XL0-v4 masks readmitted the same day;
        then the 6 declines of the 18 Brazil photos new to random_date_raccoon's crops on 2026-10-25, which the owner
        moved to dataset/segmenter_positives the same day, as were the 6 of the file's other photos that the
        segmenter dataset did not hold; then ML-19 P6's 10 declines of the photos new to the crops of canonical
        photos and the fallbacks, 2026-11-04, and 1 of the owner's 10 new Brazil photos; then ML-16's, on the re-crop with
        xl0_v7r_s123: 11 of the 87 photos added to random_date_raccoon on 2026-11-06 and the 6 fallbacks it made in
        2026-10-21__new_green, 2026-11-07). The decisions hold for their photos across re-crops: XL0-v4's changed
        nearly every mask."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("mask_review", REPO_ROOT / "analysis/ml3/mask_review.py")
        mask_review = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mask_review)
        want = {(r["session"], r["class_folder"], r["file"]) for r in mask_review.exclusions()}
        self.assertEqual(read_exclusions(mask_review.POOL_EXCLUDE), want)
        self.assertEqual(len(want), 31)


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
        replace(RunConfig(), crop_method="segment").resolve_paths()
        replace(RunConfig(), train_capture_dirs=("data/cropped/cam_sony",)).resolve_paths()     # an old card

    def test_staleness_check_names_real_stages(self):
        """Every stage the fit's staleness check asks dvc about for the segmenter's pools is in dvc.yaml,
        and every segcrop session feeds one of them."""
        stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"]
        names = set(stages) | {f"segcrop@{s}" for s in stages["segcrop"]["foreach"]}
        named = [st for p in CAPTURES for st in capture_stage_overrides()[_stage_key(p)]]
        self.assertLessEqual(set(named), names)
        self.assertEqual({st for st in named if st.startswith("segcrop@")},
                         {f"segcrop@{s}" for s in stages["segcrop"]["foreach"]})
        for p in CAPTURES:                                              # and per pool, the sessions it merges
            merge, *segcrops = capture_stage_overrides()[_stage_key(p)]
            name, sessions = merge_cmd_args(stages[merge]["cmd"])
            self.assertEqual(name, Path(p).name)
            self.assertEqual([f"segcrop@{s}" for s in sessions], segcrops, p)
        with self.assertRaisesRegex(ValueError, "segmenter"):           # a retired pool: nothing tracks it
            stale_crop_stages(["data/cropped/cam_pixel"])

    def test_the_merge_stage_depends_on_the_sessions_it_merges(self):
        """dvc.yaml writes merge_segpool's sessions twice, in its cmd and its deps: a session only in the cmd is
        merged but does not invalidate the pool, one only in the deps invalidates it but is not merged."""
        stage = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"][POOL_MERGE_STAGE]
        _, sessions = merge_cmd_args(stage["cmd"])
        crops = [d.removeprefix("data/segcropped/") for d in stage["deps"] if d.startswith("data/segcropped/")]
        self.assertEqual(crops, sessions)


if __name__ == "__main__":
    unittest.main()
