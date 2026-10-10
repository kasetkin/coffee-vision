"""coffeecv.rename_photos (2026-10-10): every string a record derives from a renamed photo's path moves to the
new path, on token boundaries, and nothing else changes. Plain unittest, no data needed.

    python -m unittest discover -s tests -p 'test_rename_photos.py'
"""
from __future__ import annotations

import unittest

from coffeecv.rename_photos import batch_of, replacements, rewrite

OLD = "dataset/ood_negatives/2026-09__user_realworld/real_world_negatives/real_world_negatives_001.jpg"
NEW = "dataset/ood_negatives/user_realworld/20260910_195601.jpg"
OLD10 = "dataset/ood_negatives/2026-09__user_realworld/real_world_negatives/real_world_negatives_0010.jpg"


class TestRename(unittest.TestCase):
    def setUp(self):
        self.subs, self.stems = replacements([{"old_path": OLD, "new_path": NEW, "sha256": "a" * 64}])

    def test_batch_is_the_folder_under_ood_negatives(self):
        self.assertEqual(batch_of(OLD), "2026-09__user_realworld")
        self.assertEqual(batch_of(NEW), "user_realworld")
        with self.assertRaises(ValueError):
            batch_of("dataset/segmenter_positives/PXL_1.jpg")

    def test_mask_stems_are_both_label_ids(self):
        self.assertEqual(self.stems, {
            "ood_negatives__2026-09__user_realworld__real_world_negatives__real_world_negatives_001":
                "ood_negatives__user_realworld__20260910_195601",
            "2026-09__user_realworld__real_world_negatives_001": "user_realworld__20260910_195601"})

    def test_yaml_entry_moves_path_group_and_batch(self):
        text = (f"- path: {OLD}\n  sha256: x\n  batch: 2026-09__user_realworld\n"
                f"  tag: real_world_negatives\n  group: {OLD}\n")
        out, n = rewrite(text, self.subs)
        self.assertEqual(n, 3)
        self.assertEqual(out, f"- path: {NEW}\n  sha256: x\n  batch: user_realworld\n"
                              f"  tag: real_world_negatives\n  group: {NEW}\n")

    def test_csv_row_moves_id_path_and_mask_and_keeps_crlf(self):
        old_id = "2026-09__user_realworld__real_world_negatives_001"
        out, n = rewrite(f"{old_id},neg,{OLD},h,negative,,data/seg_labels/neg/{old_id}.png,m\r\n", self.subs)
        self.assertEqual(n, 3)
        self.assertEqual(out, f"user_realworld__20260910_195601,neg,{NEW},h,negative,,"
                              f"data/seg_labels/neg/user_realworld__20260910_195601.png,m\r\n")

    def test_a_longer_name_is_not_touched(self):
        text = f"- path: {OLD10}\n  batch: 2026-09__user_realworld_extra\n"
        self.assertEqual(rewrite(text, self.subs), (text, 0))


if __name__ == "__main__":
    unittest.main()
