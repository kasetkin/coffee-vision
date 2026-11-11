"""coffeecv.canonicalize_folder (ML-19 D18, D33): every photo of a dataset folder becomes its canonical photo at
coffeecv.canonical.canonical_path, the originals untouched, files that are not photos left out, an unreadable photo a
failure by name, and a re-run the same bytes.

    python -m unittest discover -s tests -p 'test_canonicalize_folder.py'
"""
from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from tests import _dvc_lock as dvc_lock
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

    def test_a_file_in_no_accepted_format_is_reported_and_left_alone(self):   # OPS-18 D16
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), (120, 80, 40)).save(buf, "TIFF")
        (self.dataset / "sess" / "class_001" / "scan.tif").write_bytes(buf.getvalue())
        r = self.run_tool()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 files not in an accepted format, left alone", r.stdout)
        self.assertIn("sess/class_001/scan.tif", r.stdout)
        self.assertIn("sess/manifest.csv", r.stdout)
        self.assertFalse(any(self.store.rglob("scan.tif*")))

    def test_an_unreadable_photo_fails_the_run_by_name(self):
        (self.dataset / "sess" / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 not really")
        r = self.run_tool()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("broken.jpg", r.stderr)


CANON_YAML = """\
stages:
  canonicalize:
    foreach:
      - sess
    do:
      cmd: python -m coffeecv.canonicalize_folder ${item} --jobs 4
      deps:
        - dataset/${item}
        - code.py
      outs:
        - data/canonical/${item}:
            persist: true
"""


@real_data   # the tool runs in a subprocess, its workers spawned
class Incremental(unittest.TestCase):
    """Ticket ML-26: after a recorded run, only new and changed photos are canonicalized, a gone photo's canonical photo
    is deleted, and the store is the same bytes as a full rebuild's."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "dvc.yaml").write_text(CANON_YAML)
        (self.root / "params.yaml").write_text("{}\n")
        (self.root / "code.py").write_text("x = 1\n")
        self.sess = self.root / "dataset" / "sess" / "class_001"
        self.sess.mkdir(parents=True)
        for name, rgb in [("a.jpg", (200, 120, 60)), ("b.jpg", (20, 120, 160)), ("gone.jpg", (90, 90, 90))]:
            (self.sess / name).write_bytes(jpeg(rgb))

    def run_tool(self, *extra):
        r = subprocess.run([sys.executable, "-m", "coffeecv.canonicalize_folder", "sess", "--jobs", "2",
                            "--dataset", str(self.root / "dataset"), "--store", str(self.root / "data" / "canonical"),
                            *extra], cwd=REPO, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        return r.stdout

    def store(self):
        out = self.root / "data" / "canonical" / "sess"
        return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}

    def commit(self):
        dvc_lock.commit(self.root, "canonicalize@sess", "python -m coffeecv.canonicalize_folder sess --jobs 4",
                        ["dataset/sess", "code.py"], ["data/canonical/sess"])

    def test_only_the_new_photo_is_done_and_the_store_is_a_full_rebuilds(self):
        self.assertIn("full rebuild (not a repository's store): 3 photos to do", self.run_tool())   # no dvc.lock yet
        self.commit()
        (self.sess / "c.jpg").write_bytes(jpeg((10, 200, 10)))
        (self.sess / "gone.jpg").unlink()
        (self.sess / "b.jpg").write_bytes(jpeg((30, 30, 220)))
        out = self.run_tool()
        self.assertIn("sess: incremental: 2 photos to do, 1 kept, 1 stale removed", out)
        incremental = self.store()
        self.assertIn("full rebuild (--full): 3 photos to do", self.run_tool("--full"))
        self.assertEqual(incremental, self.store())
        self.assertEqual(sorted(incremental), ["class_001/a.jpg.jxl", "class_001/b.jpg.jxl", "class_001/c.jpg.jxl"])

    def test_a_kept_photo_that_differs_from_the_record_is_done_again(self):
        self.run_tool()
        self.commit()
        out = self.root / "data" / "canonical" / "sess" / "class_001" / "a.jpg.jxl"
        good = out.read_bytes()
        out.write_bytes(b"not as recorded")
        self.assertIn("incremental: 1 photos to do, 2 kept", self.run_tool())
        self.assertEqual(out.read_bytes(), good)

    def test_a_changed_dependency_rebuilds_in_full(self):
        self.run_tool()
        self.commit()
        (self.root / "code.py").write_text("x = 2\n")
        self.assertIn("full rebuild (code.py changed): 3 photos to do", self.run_tool())


if __name__ == "__main__":
    unittest.main()
