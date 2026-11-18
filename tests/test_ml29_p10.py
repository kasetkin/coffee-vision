"""ML-29 P10 (D12 step 3): analysis/ml29/p10_pool.py, the scratch pool (the tracked pool with the gain-map photos'
crops swapped in, every other file a link) and the per-camera comparison's grouping.

    python -m unittest tests.test_ml29_p10
"""
from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from coffeecv.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "analysis" / "ml29"))
_spec = importlib.util.spec_from_file_location("p10_pool", REPO_ROOT / "analysis" / "ml29" / "p10_pool.py")
p10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p10)


class Build(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(dir=REPO_ROOT))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tracked = self.tmp / "segcropped" / "pool"
        c = self.tracked / "class_001__Ethiopia"
        c.mkdir(parents=True)
        for stem, mask in (("A", True), ("B", True), ("C", False)):
            (c / f"{stem}__cropped.jxl").write_bytes(f"today {stem}".encode())
            if mask:
                (c / f"{stem}__beanmask.png").write_bytes(f"mask {stem}".encode())
        (self.tracked / "merge_manifest.json").write_text("{}")
        s = self.tmp / "scratch" / "sess" / "class_001__Ethiopia"
        s.mkdir(parents=True)
        (s / "A__cropped.jxl").write_bytes(b"scratch A")
        (s / "A__beanmask.png").write_bytes(b"scratch mask A")
        (s / "B__cropped.jxl").write_bytes(b"scratch B")      # a D18 fallback now: no mask
        (s / "D__cropped.jxl").write_bytes(b"scratch D")      # declined: not in the pool
        self.scratch = p10.scratch_index(self.tmp / "scratch", {"sess"})

    def test_the_gain_map_crops_are_swapped_in_and_the_rest_linked(self):
        pool = self.tmp / "out" / "pool"
        with mock.patch.object(p10, "REPO", self.tmp):
            m = p10.build_pool(self.tracked, self.scratch, pool)
        c = pool / "class_001__Ethiopia"
        self.assertEqual((c / "A__cropped.jxl").read_bytes(), b"scratch A")
        self.assertEqual((c / "A__beanmask.png").read_bytes(), b"scratch mask A")
        self.assertFalse((c / "A__cropped.jxl").is_symlink())
        self.assertTrue((c / "A__cropped.jxl").samefile(self.scratch[("class_001", "A__cropped.jxl")]))  # no copy
        self.assertEqual((c / "B__cropped.jxl").read_bytes(), b"scratch B")
        self.assertFalse((c / "B__beanmask.png").exists())
        self.assertTrue((c / "C__cropped.jxl").is_symlink())
        self.assertEqual((c / "C__cropped.jxl").read_bytes(), b"today C")
        self.assertEqual((pool / "merge_manifest.json").read_text(), "{}")
        self.assertFalse((c / "D__cropped.jxl").exists())
        self.assertEqual((m["swapped"], m["linked_files"]), (2, 2))
        self.assertEqual(m["unused_scratch_crops"], ["class_001/D__cropped.jxl"])
        self.assertEqual(m["fallbacks_changed"], ["class_001__Ethiopia/B__cropped.jxl"])

    def test_a_session_naming_the_class_otherwise_is_matched_by_class_id(self):
        # merge_rig names the pool's class dir after the first session holding the class; the others' photos go
        # into it under their own file names (class_013__Brazil_SulDeMinas into class_013__Brazil_Decaff).
        other = self.tmp / "scratch" / "sess2" / "class_001__Sidamo"
        other.mkdir(parents=True)
        (other / "C__cropped.jxl").write_bytes(b"scratch C")
        (other / "C__beanmask.png").write_bytes(b"scratch mask C")
        scratch = p10.scratch_index(self.tmp / "scratch", {"sess", "sess2"})
        pool = self.tmp / "out" / "pool"
        with mock.patch.object(p10, "REPO", self.tmp):
            m = p10.build_pool(self.tracked, scratch, pool)
        c = pool / "class_001__Ethiopia"
        self.assertEqual((c / "C__cropped.jxl").read_bytes(), b"scratch C")
        self.assertEqual((c / "C__beanmask.png").read_bytes(), b"scratch mask C")
        self.assertEqual(m["swapped"], 3)
        self.assertEqual(m["unused_scratch_crops"], ["class_001/D__cropped.jxl"])

    def test_a_pool_not_named_as_the_tracked_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, "named 'pool'"):
            p10.build_pool(self.tracked, self.scratch, self.tmp / "out" / "pool2")


class Cameras(unittest.TestCase):
    def test_patches_are_grouped_by_their_photos_camera_and_by_gain_maps(self):
        scan = [{"file": str(p10.p8.DATASET / "s" / "class_001__Ethiopia" / "PXL_1.jpg"), "make": "Google",
                 "model": "Pixel 9 Pro", "gain_map": "Ultra HDR (hdrgm)"},
                {"file": str(p10.p8.DATASET / "s" / "class_001__Ethiopia" / "DSC_2.JPG"), "make": "Sony",
                 "model": "XQ", "gain_map": ""}]
        cams = p10.camera_of_crop(scan)
        self.assertEqual(cams[("001", "PXL_1__cropped.jxl")], "Google Pixel 9 Pro")
        rows = [{"folder_id": "001", "photo": "PXL_1__cropped.jxl", "true_label": "E", "pred_label": "E"},
                {"folder_id": "001", "photo": "PXL_1__cropped.jxl", "true_label": "E", "pred_label": "K"},
                {"folder_id": "001", "photo": "DSC_2__cropped.jxl", "true_label": "E", "pred_label": "E"}]
        got = p10.by_camera(rows, cams, {"Google Pixel 9 Pro"}, swapped={("001", "PXL_1__cropped.jxl")})
        self.assertEqual(got["Google Pixel 9 Pro"]["n"], 2)
        self.assertAlmostEqual(got["Google Pixel 9 Pro"]["acc"], 0.5)
        self.assertEqual(got["all cameras with gain maps"]["n"], 2)
        self.assertEqual(got["all cameras without"]["acc"], 1.0)
        self.assertEqual(got["all photos, gain map applied"]["n"], 2)
        self.assertEqual(got["all photos as today"]["n"], 1)


class CompareRows(unittest.TestCase):
    def test_a_camera_missing_from_a_seeds_split_keeps_its_column(self):
        def sc(n, f):
            return {"n": n, "acc": f, "macro_f1": f}

        table = {"Pixel": {289: (sc(10, 0.9), sc(10, 1.0)), 291: (sc(4, 0.5), sc(4, 0.75))}}
        (line,) = p10.camera_rows(table, [289, 290, 291])
        cells = line.split("  delta")[0].split()
        self.assertEqual(cells[-3:], ["0.9000->1.0000", "-", "0.5000->0.7500"])
        self.assertIn("n=10/-/4", line)
        self.assertIn("+0.1750 [+0.1000, +0.2500]", line)


class PerClass(unittest.TestCase):
    def test_a_class_dir_and_a_folder_id_give_one_class_id(self):
        self.assertEqual(p10.class_id("class_013__Brazil_Decaff"), "013")
        self.assertEqual(p10.class_id("class_013__Brazil_SulDeMinas"), "013")

    def test_scores_carry_each_classes_f1(self):
        rows = [{"true_label": "E", "pred_label": "E"}, {"true_label": "E", "pred_label": "K"},
                {"true_label": "K", "pred_label": "K"}]
        got = p10.scores(rows)
        self.assertAlmostEqual(got["per_class"]["E"], 2 / 3)
        self.assertAlmostEqual(got["per_class"]["K"], 2 / 3)

    def test_class_rows_list_the_cameras_with_enough_patches_in_every_seed(self):
        def sc(n, e, k):
            return {"n": n, "acc": 1.0, "macro_f1": 1.0, "per_class": {"E": e, "K": k}}

        table = {"Pixel": {1: (sc(200, 0.9, 0.8), sc(200, 1.0, 0.7)), 2: (sc(150, 0.9, 0.8), sc(150, 0.8, 0.8))},
                 "Small": {1: (sc(20, 0.5, 0.5), sc(20, 1.0, 1.0)), 2: (sc(20, 0.5, 0.5), sc(20, 1.0, 1.0))},
                 "Gone": {1: (sc(300, 0.5, 0.5), sc(300, 1.0, 1.0))}}
        lines = p10.class_rows(table, [1, 2], min_n=100)
        self.assertEqual(len(lines), 1)
        self.assertIn("Pixel", lines[0])
        self.assertIn("K -0.050", lines[0])          # (-0.1 + 0.0) / 2, the larger drop first
        self.assertIn("E +0.000", lines[0])


if __name__ == "__main__":
    unittest.main()
