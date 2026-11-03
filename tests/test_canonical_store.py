"""coffeecv.canonical_store (ML-19 P6, finding 2): a record's photo is read from its canonical photo, the original
checked against its sha256 first; a missing canonical photo is an error, never the original.

    python -m unittest discover -s tests -p 'test_canonical_store.py'
"""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import canonical_store as cs
from coffeecv.repo_files import sha256_file


class LoadForRecord(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.original = self.root / "dataset" / "sess" / "class_001__X" / "P.jpg"
        self.original.parent.mkdir(parents=True)
        buf = io.BytesIO()
        Image.fromarray(np.full((24, 40, 3), (180, 90, 40), np.uint8)).save(buf, "JPEG", quality=95)
        self.original.write_bytes(buf.getvalue())
        self.sha = sha256_file(self.original)

    def test_reads_the_canonical_photo(self):
        canonical = self.root / "data" / "canonical" / "sess" / "class_001__X" / "P.jpg.jxl"
        canonical.parent.mkdir(parents=True)
        canonical.write_bytes(cn.canonicalize_file(self.original))
        self.assertEqual(cs.canonical_for(self.original, self.sha), canonical)
        np.testing.assert_array_equal(cs.load_for_record(self.original, self.sha),
                                      cn.load_canonical(canonical.read_bytes()))

    def test_a_missing_canonical_photo_is_an_error_not_the_original(self):
        with self.assertRaisesRegex(FileNotFoundError, "canonicalize@sess"):
            cs.load_for_record(self.original, self.sha)

    def test_the_original_must_match_its_record(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            cs.canonical_for(self.original, "0" * 64)

    def test_a_photo_outside_a_dataset_folder_has_none(self):
        with self.assertRaisesRegex(ValueError, "dataset"):
            cs.canonical_for(self.root / "elsewhere.jpg")


if __name__ == "__main__":
    unittest.main()
