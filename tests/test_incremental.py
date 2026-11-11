"""coffeecv.incremental (ticket ML-26): which of a folder stage's recorded outputs still stand, from dvc.lock.

Plain unittest on a made-up repository: dvc.yaml, params.yaml, a dvc.lock entry and the cache's .dir listings, written
here as DVC writes them. No DVC runs.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from coffeecv import incremental as inc

from tests import _dvc_lock as dvc_lock

DVC_YAML = """\
stages:
  crop:
    foreach:
      - sess
    do:
      cmd: python -m tool --session ${item} --threads 4
      deps:
        - photos/${item}
        - code.py
        - lib
        - models/${model}
      params:
        - params.yaml:
            - model
            - knob.size
      outs:
        - out/${item}:
            persist: true
"""
STAGE = "crop@sess"
MASSIMO_PERU = ("20261107_101201_Pixel_9_Pro_Peru_Massimo.jpg", "1d98b7ed3da8a55e5425e40e81eb9788")


class Repo:
    """A repository with one stage, crop@sess, and a helper that records its run in dvc.lock, as `dvc commit` would."""

    def __init__(self, root: Path):
        self.root = root
        (root / "dvc.yaml").write_text(DVC_YAML)
        (root / "params.yaml").write_text("model: m.pt\nknob:\n  size: 3\n")
        (root / ".dvcignore").write_text("# bytecode\n__pycache__/\n")
        self.write("code.py", b"print(1)\n")
        self.write("lib/a.py", b"a = 1\n")
        self.write("models/m.pt", b"weights")
        self.write("photos/sess/class_1/a.jpg", b"photo a")
        self.write("photos/sess/class_1/b.jpg", b"photo b")
        self.write("out/sess/class_1/a.out", b"out a")
        self.write("out/sess/class_1/b.out", b"out b")

    def write(self, rel: str, data: bytes) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return p

    def commit(self) -> None:
        dvc_lock.commit(self.root, STAGE, "python -m tool --session sess --threads 4",
                        ["photos/sess", "code.py", "lib", "models/m.pt"], ["out/sess"],
                        {"knob.size": 3, "model": "m.pt"})

    def recorded(self):
        return inc.recorded(STAGE, "photos/sess", "out/sess", self.root)


class Recorded(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)
        self.repo = Repo(root)
        self.repo.commit()

    def test_nothing_changed_keeps_every_photo(self):
        rec, why = self.repo.recorded()
        self.assertEqual(why, "")
        self.assertEqual(sorted(rec.inputs), ["class_1/a.jpg", "class_1/b.jpg"])
        self.assertEqual(sorted(rec.outputs), ["class_1/a.out", "class_1/b.out"])

    def test_a_new_or_changed_photo_is_not_kept(self):
        self.repo.write("photos/sess/class_1/c.jpg", b"photo c")
        self.repo.write("photos/sess/class_1/b.jpg", b"photo b, edited")
        rec, _ = self.repo.recorded()
        self.assertEqual(sorted(rec.inputs), ["class_1/a.jpg"])

    def test_a_changed_code_file_rebuilds(self):
        self.repo.write("code.py", b"print(2)\n")
        self.assertEqual(self.repo.recorded(), (None, "code.py changed"))

    def test_a_changed_file_in_a_code_folder_rebuilds(self):
        self.repo.write("lib/b.py", b"b = 1\n")
        self.assertEqual(self.repo.recorded(), (None, "lib changed"))

    def test_bytecode_is_no_change(self):
        self.repo.write("lib/__pycache__/a.cpython-312.pyc", b"\x00")
        self.assertEqual(self.repo.recorded()[1], "")

    def test_a_changed_model_rebuilds(self):
        self.repo.write("models/m.pt", b"other weights")
        self.assertEqual(self.repo.recorded(), (None, "models/m.pt changed"))

    def test_a_changed_param_rebuilds(self):
        (self.repo.root / "params.yaml").write_text("model: m.pt\nknob:\n  size: 4\n")
        self.assertEqual(self.repo.recorded(), (None, "params.yaml knob.size changed"))

    def test_a_param_that_names_another_model_rebuilds(self):
        self.repo.write("models/n.pt", b"weights")
        (self.repo.root / "params.yaml").write_text("model: n.pt\nknob:\n  size: 3\n")
        self.assertIsNone(self.repo.recorded()[0])

    def test_a_changed_command_rebuilds(self):
        y = self.repo.root / "dvc.yaml"
        y.write_text(y.read_text().replace("--threads 4", "--threads 8"))
        self.assertEqual(self.repo.recorded(), (None, "its dvc.yaml stage changed since dvc.lock"))

    def test_a_new_dep_in_dvc_yaml_rebuilds(self):
        y = self.repo.root / "dvc.yaml"
        y.write_text(y.read_text().replace("        - code.py\n", "        - code.py\n        - more.py\n"))
        self.assertEqual(self.repo.recorded(), (None, "its dvc.yaml stage changed since dvc.lock"))

    def test_a_stage_never_run_rebuilds(self):
        (self.repo.root / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n")
        self.assertEqual(self.repo.recorded(), (None, f"{STAGE} is not in dvc.lock"))

    def test_a_code_folder_needs_no_listing_in_the_cache(self):
        lock = yaml.safe_load((self.repo.root / "dvc.lock").read_text())
        lib = next(d for d in lock["stages"][STAGE]["deps"] if d["path"] == "lib")["md5"]
        (self.repo.root / ".dvc/cache/files/md5" / lib[:2] / lib[2:]).unlink()
        self.assertEqual(self.repo.recorded()[1], "")

    def test_dir_md5_is_dvcs(self):
        """The md5 of a .dir object DVC wrote (dataset/massimo's, 2026-11-11) is DVC's md5 of the folder."""
        self.assertEqual(inc.dir_md5({
            "class_016__Ethiopia/20261107_101219_Pixel_9_Pro_Ethiopia_Massimo.jpg": "6a8ed6a46cd5fd52e8d679aa9e25d1cb",
            "class_017__Peru/" + MASSIMO_PERU[0]: MASSIMO_PERU[1]}), "1211c5b9ccbd2066f496fbb21bf10fab.dir")

    def test_a_listing_missing_from_the_cache_rebuilds(self):
        shutil.rmtree(self.repo.root / ".dvc/cache")
        self.assertIsNone(self.repo.recorded()[0])

    def test_keeps_only_an_output_on_disk_as_recorded(self):
        rec, _ = self.repo.recorded()
        out = self.repo.root / "out/sess"
        self.assertTrue(inc.keeps(rec, out, "class_1/a.out"))
        (out / "class_1/a.out").write_bytes(b"out a, rewritten")
        self.assertFalse(inc.keeps(rec, out, "class_1/a.out"))
        (out / "class_1/b.out").unlink()
        self.assertFalse(inc.keeps(rec, out, "class_1/b.out"))
        self.assertFalse(inc.keeps(rec, out, "class_1/never.out"))

    def test_a_dvcignore_pattern_it_cannot_read_is_refused(self):
        (self.repo.root / ".dvcignore").write_text("*.tmp\n")
        with self.assertRaisesRegex(ValueError, "cannot read"):
            self.repo.recorded()


class TheRepositorysStages(unittest.TestCase):
    """The real dvc.yaml: the three incremental stages resolve as DVC's dvc.lock records them."""

    def test_every_canonicalize_and_segcrop_stage_resolves_to_its_lock_entry(self):
        root = inc.REPO_ROOT
        lock = yaml.safe_load((root / "dvc.lock").read_text())["stages"]
        params = yaml.safe_load((root / "params.yaml").read_text())
        names = [s for s in lock if s.split("@")[0] in ("canonicalize", "segcrop", "seg_ml19_predict")]
        self.assertGreater(len(names), 30)
        for stage in names:
            spec = inc.stage_spec(root, stage, params)
            self.assertEqual(" ".join(lock[stage]["cmd"].split()), spec["cmd"], stage)
            self.assertEqual(sorted(d["path"] for d in lock[stage]["deps"]), spec["deps"], stage)
            self.assertEqual(sorted(o["path"] for o in lock[stage]["outs"]), spec["outs"], stage)
            self.assertEqual(sorted((lock[stage].get("params") or {}).get("params.yaml", {})), spec["params"], stage)


if __name__ == "__main__":
    unittest.main()
