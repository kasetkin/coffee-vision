"""ML-29 P8 (D12 step 1): analysis/ml29/p8_scratch.py, the gain-map photos canonicalized with the setting on into a
scratch store and the change from today's canonical photos, on photos and maps made in code.

    python -m unittest tests.test_ml29_p8
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import imagecodecs as ic
import numpy as np

from coffeecv import canonical as cn
from coffeecv.config import REPO_ROOT
from tests import test_gain_map as tg

_spec = importlib.util.spec_from_file_location("p8_scratch", REPO_ROOT / "analysis" / "ml29" / "p8_scratch.py")
p8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p8)


def ultra_hdr() -> bytes:
    """An Ultra HDR JPEG with a one-channel map, as every dataset photo's is (D38), its right half boosted 2x."""
    gain = np.zeros((24, 32), np.uint8)
    gain[:, 16:] = 255
    return tg.ultra_hdr(np.full((48, 64, 3), 100, np.uint8), gain, b'<x hdrgm:Version="1.0" hdrgm:GainMapMax="1" '
                        b'hdrgm:HDRCapacityMax="1" hdrgm:OffsetSDR="0" hdrgm:OffsetHDR="0"/>')


def libultrahdr() -> bytes:
    """An Ultra HDR JPEG from libultrahdr, whose map of a colour photo has three channels: a grey ramp whose right
    half is 4x brighter than white."""
    light = np.full((32, 48, 4), 0.3, np.float16)
    light[:, 24:, :3] = 1.2
    return ic.ultrahdr_encode(light)

WITHOUT_MAPS = dataclasses.replace(cn.STORE_SETTINGS, gain_maps=False)   # the store before P11


def row(path: str, gain_map: str = "Ultra HDR (hdrgm)", make: str = "Google", model: str = "Pixel 9 Pro") -> dict:
    return {"file": path, "gain_map": gain_map, "make": make, "model": model}


class Targets(unittest.TestCase):
    def test_the_stores_folders_are_dvc_yamls_canonicalize_stages(self):
        folders = p8.store_folders(REPO_ROOT / "dvc.yaml")
        self.assertIn("random_date_raccoon", folders)
        self.assertIn("ood_negatives", folders)
        self.assertNotIn("classes_labels_only", folders)

    def test_gain_map_photos_in_the_store_and_the_rest_counted_by_folder(self):
        ds = Path("/d")
        rows = [row("/d/a/x/1.jpg"), row("/d/a/x/2.jpg", gain_map=""), row("/d/b/3.heic"), row("/d/c/4.jpg")]
        chosen, left_out = p8.targets(rows, ["a", "b"], ds)
        self.assertEqual([r["file"] for r in chosen], ["/d/a/x/1.jpg", "/d/b/3.heic"])
        self.assertEqual(dict(left_out), {"c": 1})

    def test_the_camera_is_make_and_model(self):
        self.assertEqual(p8.camera(row("x", make="samsung", model="SM-S928B")), "samsung SM-S928B")
        self.assertEqual(p8.camera(row("x", make="", model="")), "unknown")


class Canonicalize(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_a_photo_with_a_map_is_canonicalized_with_it_applied(self):
        original = self.root / "dataset" / "s" / "x.jpg"
        original.parent.mkdir(parents=True)
        original.write_bytes(ultra_hdr())
        target = self.root / "scratch" / "s" / "x.jpg.jxl"
        done = p8.canonicalize_one((original, target, 1))
        self.assertEqual(done["error"], "")
        self.assertIn("+", done["decoder"])                       # e.g. "pillow8+ISO 21496-1"
        made = target.read_bytes()
        self.assertTrue(cn.is_canonical(made))
        today = cn.encode(cn.decode(original.read_bytes(), "x.jpg", settings=WITHOUT_MAPS), numthreads=1)
        self.assertNotEqual(made, today)
        brighter = cn.load_canonical(made).astype(int) - cn.load_canonical(today)
        self.assertGreater(brighter[:, -8:].mean(), 5)              # the boosted half comes out brighter
        self.assertEqual(done["map_channels"], 1)

    def test_a_three_channel_map_is_left_unapplied(self):
        """D38: libultrahdr's map of a colour photo is applied in no primaries here, so the photo is today's."""
        original = self.root / "x.jpg"
        original.write_bytes(libultrahdr())
        target = self.root / "scratch" / "x.jpg.jxl"
        done = p8.canonicalize_one((original, target, 1))
        self.assertEqual((done["error"], done["decoder"], done["map_channels"]), ("", "jpeg", 0))
        self.assertEqual(target.read_bytes(),
                         cn.encode(cn.decode(original.read_bytes(), "x.jpg", settings=WITHOUT_MAPS), numthreads=1))

    def test_a_photo_without_a_map_keeps_todays_bytes(self):
        original = self.root / "y.png"
        original.write_bytes(ic.png_encode(np.full((16, 24, 3), 120, np.uint8)))
        target = self.root / "scratch" / "y.png.jxl"
        done = p8.canonicalize_one((original, target, 1))
        self.assertNotIn("+", done["decoder"])
        self.assertEqual(done["map_channels"], 0)
        self.assertEqual(target.read_bytes(), cn.encode(cn.decode(original.read_bytes(), "y.png"), numthreads=1))

    def test_an_unreadable_photo_is_an_error_not_a_crash(self):
        original = self.root / "z.jpg"
        original.write_bytes(b"\xff\xd8\xff\xe0broken")
        done = p8.canonicalize_one((original, self.root / "z.jpg.jxl", 1))
        self.assertTrue(done["error"])
        self.assertFalse((self.root / "z.jpg.jxl").exists())

    def test_the_log_resumes(self):
        log = self.root / "canonical.jsonl"
        self.assertEqual(p8.done_paths(log), set())
        with log.open("a") as f:
            f.write(json.dumps({"path": "a.jpg", "error": ""}) + "\n")
            f.write(json.dumps({"path": "b.jpg", "error": "unreadable"}) + "\n")
        self.assertEqual(p8.done_paths(log), {"a.jpg"})              # an error is tried again


class ThreeChannelMaps(unittest.TestCase):
    def test_a_map_of_more_than_one_channel_is_named(self):
        """ML-29 D38: a three-channel map applied in linear Rec.2020 is not the map's own maths, so check names it."""
        log = {"a.jpg": {"error": "", "decoder": "jpeg+Ultra HDR", "map_channels": 1},
               "b.jpg": {"error": "", "decoder": "jpeg+ISO 21496-1", "map_channels": 3},
               "c.jpg": {"error": "", "decoder": "jpeg", "map_channels": 0},
               "d.jpg": {"error": "", "decoder": "jpeg+Ultra HDR"}}           # logged before the count: unknown
        self.assertEqual(p8.map_channel_problems(log), {"three channels": ["b.jpg"], "not counted": ["d.jpg"]})


class Change(unittest.TestCase):
    def test_identical_photos_change_by_nothing(self):
        a = np.random.default_rng(0).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        s = p8.change(a, a.copy(), None)
        self.assertEqual((s["all_mean_abs"], s["all_mean"], s["all_p99_abs"]), (0.0, 0.0, 0.0))
        self.assertTrue(s["under_a_level"])
        self.assertEqual(s["bean_mean_abs"], "")                    # no mask: no bean region

    def test_the_bean_region_and_the_rest_apart(self):
        today = np.full((20, 30, 3), 100, np.uint8)
        scratch = today.copy()
        mask = np.zeros((20, 30), bool)
        mask[5:15, 5:20] = True
        scratch[mask] += 6
        s = p8.change(today, scratch, mask)
        self.assertEqual((s["bean_mean"], s["bean_mean_abs"], s["rest_mean_abs"]), (6.0, 6.0, 0.0))
        self.assertAlmostEqual(s["all_mean"], 6 * mask.mean())
        self.assertAlmostEqual(s["bean_frac"], mask.mean())
        self.assertFalse(s["under_a_level"])

    def test_a_darker_photo_has_a_negative_mean(self):
        today = np.full((10, 10, 3), 100, np.uint8)
        self.assertEqual(p8.change(today, today - 3, None)["all_mean"], -3.0)

    def test_a_mask_of_another_size_is_fitted_to_the_photo(self):
        mask = np.zeros((40, 60), bool)
        mask[:20] = True                                            # the top half, at twice the size
        fitted = p8.fit_mask(mask, (20, 30))
        self.assertEqual(fitted.shape, (20, 30))
        self.assertEqual(int(fitted.sum()), 10 * 30)


class Summary(unittest.TestCase):
    def test_per_camera(self):
        rows = [{"camera": "A", "all_mean_abs": 1.0, "all_mean": 1.0, "bean_mean_abs": 2.0, "bean_mean": 2.0,
                 "rest_mean_abs": 0.5, "under_a_level": False},
                {"camera": "A", "all_mean_abs": 0.2, "all_mean": 0.2, "bean_mean_abs": "", "bean_mean": "",
                 "rest_mean_abs": "", "under_a_level": True},
                {"camera": "B", "all_mean_abs": 3.0, "all_mean": -3.0, "bean_mean_abs": 4.0, "bean_mean": -4.0,
                 "rest_mean_abs": 2.0, "under_a_level": False}]
        by = {s["camera"]: s for s in p8.summary(rows)}
        self.assertEqual((by["A"]["photos"], by["A"]["with_mask"], by["A"]["under_a_level"]), (2, 1, 1))
        self.assertAlmostEqual(by["A"]["all_mean_abs"], 0.6)        # the median photo
        self.assertEqual(by["A"]["bean_mean_abs"], 2.0)
        self.assertEqual(by["B"]["bean_mean"], -4.0)


class Crop(unittest.TestCase):
    def test_a_crop_centred_on_the_bean_region_inside_the_photo(self):
        mask = np.zeros((100, 200), bool)
        mask[10:30, 150:190] = True
        y0, y1, x0, x1 = p8.crop_box(mask, (100, 200), 64)
        self.assertEqual((y1 - y0, x1 - x0), (64, 64))
        self.assertTrue(0 <= y0 and y1 <= 100 and 0 <= x0 and x1 <= 200)
        self.assertTrue(y0 <= 20 < y1 and x0 <= 170 < x1)             # the region's centre is inside

    def test_without_a_mask_the_centre(self):
        self.assertEqual(p8.crop_box(None, (100, 200), 64), (18, 82, 68, 132))

    def test_a_photo_smaller_than_the_crop_is_whole(self):
        self.assertEqual(p8.crop_box(None, (30, 40), 64), (0, 30, 0, 40))


if __name__ == "__main__":
    unittest.main()
