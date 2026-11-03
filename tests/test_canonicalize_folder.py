"""coffeecv.canonicalize_folder (ML-19 D18, D33): every photo of a dataset folder becomes its canonical photo at
coffeecv.canonical.canonical_path, the originals untouched, files that are not photos left out, an unreadable photo a
failure by name, and a re-run the same bytes.

    python -m unittest discover -s tests -p 'test_canonicalize_folder.py'
"""
from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from tests._tiers import real_data

REPO = Path(__file__).resolve().parents[1]


def jpeg(colour) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.full((40, 56, 3), colour, np.uint8)).save(buf, "JPEG", quality=95)
    return buf.getvalue()


class PixelLimit(unittest.TestCase):
    def test_the_stores_pixel_limit_is_the_dataset_tools(self):
        from coffeecv import canonicalize_folder
        from coffeecv.photo_io import DATASET_MAX_PIXELS
        self.assertEqual(canonicalize_folder.DATASET_MAX_PIXELS, DATASET_MAX_PIXELS)


class CanonicalPath(unittest.TestCase):
    def test_the_original_name_is_kept_whole(self):
        d = Path("/x/dataset")
        self.assertEqual(cn.canonical_path(d / "s/class_001/PXL_1.jpg", d, Path("/x/store")),
                         Path("/x/store/s/class_001/PXL_1.jpg.jxl"))
        self.assertNotEqual(cn.canonical_path(d / "a/P.jpg", d, d), cn.canonical_path(d / "a/P.dng", d, d))

    def test_a_path_outside_the_dataset_is_refused(self):
        with self.assertRaises(ValueError):
            cn.canonical_path(Path("/elsewhere/P.jpg"), Path("/x/dataset"), Path("/x/store"))


@real_data   # about 4 s: the tool runs in a subprocess, its workers spawned
class Folder(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.dataset, self.store = self.root / "dataset", self.root / "store"
        (self.dataset / "sess" / "class_001").mkdir(parents=True)
        (self.dataset / "sess" / "class_001" / "a.jpg").write_bytes(jpeg((200, 120, 60)))
        (self.dataset / "sess" / "b.JPG").write_bytes(jpeg((20, 120, 160)))
        (self.dataset / "sess" / "manifest.csv").write_text("path,sha256\n")

    def run_tool(self, folder="sess"):
        return subprocess.run([sys.executable, "-m", "coffeecv.canonicalize_folder", folder, "--jobs", "2",
                               "--dataset", str(self.dataset), "--store", str(self.store)],
                              cwd=REPO, capture_output=True, text=True)

    def test_every_photo_and_nothing_else(self):
        before = {p: p.read_bytes() for p in self.dataset.rglob("*") if p.is_file()}
        r = self.run_tool()
        self.assertEqual(r.returncode, 0, r.stderr)
        got = sorted(p.relative_to(self.store).as_posix() for p in self.store.rglob("*") if p.is_file())
        self.assertEqual(got, ["sess/b.JPG.jxl", "sess/class_001/a.jpg.jxl"])
        for p in self.store.rglob("*.jxl"):
            self.assertTrue(cn.is_canonical(p.read_bytes()), p)
        self.assertEqual({p: p.read_bytes() for p in before}, before)   # the originals untouched
        first = {p: p.read_bytes() for p in self.store.rglob("*.jxl")}
        self.assertEqual(self.run_tool().returncode, 0)
        self.assertEqual({p: p.read_bytes() for p in self.store.rglob("*.jxl")}, first)   # a re-run, the same bytes

    def test_an_unreadable_photo_fails_the_run_by_name(self):
        (self.dataset / "sess" / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really")
        r = self.run_tool()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("broken.jpg", r.stderr)


if __name__ == "__main__":
    unittest.main()
