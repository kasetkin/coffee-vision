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
        self.assertEqual((c / "B__cropped.jxl").read_bytes(), b"scratch B")
        self.assertFalse((c / "B__beanmask.png").exists())
        self.assertTrue((c / "C__cropped.jxl").is_symlink())
        self.assertEqual((c / "C__cropped.jxl").read_bytes(), b"today C")
        self.assertEqual((pool / "merge_manifest.json").read_text(), "{}")
        self.assertFalse((c / "D__cropped.jxl").exists())
        self.assertEqual((m["swapped"], m["linked_files"]), (2, 2))
        self.assertEqual(m["unused_scratch_crops"], ["class_001__Ethiopia/D__cropped.jxl"])
        self.assertEqual(m["fallbacks_changed"], ["class_001__Ethiopia/B__cropped.jxl"])

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
        got = p10.by_camera(rows, cams, {"Google Pixel 9 Pro"})
        self.assertEqual(got["Google Pixel 9 Pro"]["n"], 2)
        self.assertAlmostEqual(got["Google Pixel 9 Pro"]["acc"], 0.5)
        self.assertEqual(got["all cameras with gain maps"]["n"], 2)
        self.assertEqual(got["all cameras without"]["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
