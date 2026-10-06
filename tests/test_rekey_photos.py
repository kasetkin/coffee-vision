"""coffeecv.rekey_photos (ticket ML-3 P3): pinned photo hashes move from before to after the strip, and a
hash the strip manifest does not know stops the re-key. Plain unittest, no data needed.

    python -m unittest discover -s tests -p 'test_rekey_photos.py'
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from coffeecv.rekey_photos import load_map, rekey_values

A, A2, B, C = ("a" * 64), ("b" * 64), ("c" * 64), ("d" * 64)


class TestRekey(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        manifest = self.tmp / "m.csv"
        # A was stripped to A2; B had nothing to strip (same hash on both sides).
        manifest.write_text(f"# header\npath,sha256_before,sha256_after,pixel_sha256,removed\n"
                            f"x.jpg,{A},{A2},p,EXIF:GPS:GPSLatitude\ny.png,{B},{B},q,\n")
        self.rekey, self.current = load_map(manifest)

    def test_map_holds_only_changed_photos(self):
        self.assertEqual(self.rekey, {A: A2})
        self.assertEqual(self.current, {A2, B})

    def test_old_hash_moves_and_current_ones_stay(self):
        todo, unknown = rekey_values([A, B, A2], self.rekey, self.current)
        self.assertEqual(todo, {A: A2})
        self.assertEqual(unknown, [])

    def test_unknown_hash_is_reported(self):
        _, unknown = rekey_values([A, C], self.rekey, self.current)
        self.assertEqual(unknown, [C])


if __name__ == "__main__":
    unittest.main()
