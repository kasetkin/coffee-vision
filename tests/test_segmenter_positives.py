"""dataset/segmenter_positives' manifest (ticket ML-5 P2, D1): one row per photo in the DVC-tracked folder,
the guard-era dev/holdout split and scenario tag gone, and every copy of another dataset photo naming it.
Plain unittest.

    python -m unittest discover -s tests -p 'test_segmenter_positives.py'

The row count and columns need only git-tracked files; the rest skips unless the folder is checked out
(`dvc checkout dataset/segmenter_positives.dvc`).
"""
from __future__ import annotations

import csv
import hashlib
import unittest

import yaml

from coffeecv.config import REPO_ROOT

from tests._tiers import real_data

FOLDER = REPO_ROOT / "dataset" / "segmenter_positives"
MANIFEST = FOLDER.parent / f"{FOLDER.name}.manifest.csv"
COLUMNS = ["filename", "camera", "date", "batch", "duplicate_of", "notes"]


def rows() -> list[dict]:
    return list(csv.DictReader(MANIFEST.read_text().splitlines()))


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Manifest(unittest.TestCase):
    def test_columns_have_no_split_or_scenario(self):
        with MANIFEST.open() as f:
            self.assertEqual(next(csv.reader(f)), COLUMNS)

    def test_one_row_per_tracked_photo(self):
        nfiles = yaml.safe_load((FOLDER.parent / f"{FOLDER.name}.dvc").read_text())["outs"][0]["nfiles"]
        names = [r["filename"] for r in rows()]
        self.assertEqual(len(names), nfiles)
        self.assertEqual(len(set(names)), len(names), "a photo is listed twice")

    def test_a_copy_names_a_dataset_photo_outside_this_folder(self):
        for r in rows():
            if r["duplicate_of"]:
                self.assertTrue(r["duplicate_of"].startswith("dataset/"), r)
                self.assertFalse(r["duplicate_of"].startswith(f"dataset/{FOLDER.name}/"), r)


@unittest.skipUnless(FOLDER.is_dir(), "dataset/segmenter_positives is not checked out")
class AgainstThePhotos(unittest.TestCase):
    def test_rows_are_exactly_the_folder(self):
        self.assertEqual({r["filename"] for r in rows()}, {p.name for p in FOLDER.iterdir()})

    @real_data
    def test_each_copy_is_byte_identical_to_the_photo_it_names(self):
        checked = 0
        for r in rows():
            original = REPO_ROOT / r["duplicate_of"] if r["duplicate_of"] else None
            if original is None or not original.is_file():
                continue                      # not a copy, or its folder is not checked out
            self.assertEqual(sha256(FOLDER / r["filename"]), sha256(original), r["filename"])
            checked += 1
        if not checked:
            self.skipTest("no copied-from folder is checked out")


if __name__ == "__main__":
    unittest.main()
