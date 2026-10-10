"""The segmenter's photo lists (ticket ML-2 P1, plan §4.1): disjoint, holdout never listed, near-duplicate
groups on one side, append-only against the committed version (D20). Retired by ticket ML-5 (D3): the file
is frozen, and the coverage check against today's pools went with `seg_lists check`. Plain unittest.

    python -m unittest discover -s tests -p 'test_seg_lists.py'

The grouping, allocation and append-only rules run on synthetic rows. The committed-file checks need only
git-tracked files.
"""
from __future__ import annotations

import subprocess
import unittest
from unittest import mock

import numpy as np
import yaml

from coffeecv import seg_lists
from coffeecv.config import REPO_ROOT, RunConfig


def neg(path: str, tag: str = "confusable_grain", batch: str = "b", camera: str = "sony",
        title: str = "", url: str = "") -> dict:
    return {"path": path, "batch": batch, "tag": tag, "camera": camera, "source_url": url,
            "source_title": title, "filename": path.rsplit("/", 1)[-1]}


class NearDuplicateGroups(unittest.TestCase):
    def test_chains_shots_within_the_window(self):
        rows = [neg("a", title="PIC_20260911_132408.JPG"), neg("b", title="PIC_20260911_132600.JPG"),
                neg("c", title="PIC_20260911_132759.JPG"),   # 119 s after b: same group via the chain
                neg("d", title="PIC_20260911_133000.JPG")]   # 121 s after c: new group
        groups = seg_lists.near_duplicate_groups(rows)
        self.assertEqual([[r["path"] for r in g] for g in groups], [["a", "b", "c"], ["d"]])

    def test_camera_and_tag_split_groups(self):
        rows = [neg("a", title="PXL_20260911_063355850.jpg", tag="green_legume"),
                neg("b", title="PXL_20260911_063400000.jpg", tag="confusable_grain"),
                neg("c", title="PXL_20260911_063401000.jpg", tag="confusable_grain", camera="pixel")]
        self.assertEqual(len(seg_lists.near_duplicate_groups(rows)), 3)

    def test_timestamp_from_file_name_when_title_has_none(self):
        rows = [neg("x/PIC_20260911_132408.JPG", title="renamed"), neg("x/PIC_20260911_132410.JPG")]
        self.assertEqual(len(seg_lists.near_duplicate_groups(rows)), 1)

    def test_internet_rows_group_by_url(self):
        rows = [neg("a", url="u1"), neg("b", url="u1"), neg("c", url="u2")]
        self.assertEqual(sorted(len(g) for g in seg_lists.near_duplicate_groups(rows)), [1, 2])

    def test_no_url_and_no_timestamp_is_an_error(self):
        with self.assertRaises(ValueError):
            seg_lists.near_duplicate_groups([neg("nothing.jpg", title="no time here")])


class SplitNegatives(unittest.TestCase):
    def test_groups_stay_whole_and_eval_is_near_a_third(self):
        rows = []
        for tag in ("confusable_grain", "green_legume"):
            for g in range(6):                            # 6 groups of 1-3 shots, 10 minutes apart
                for s in range(1 + g % 3):
                    rows.append(neg(f"{tag}/{g}_{s}", tag=tag, title=f"PIC_20260911_1{g}0{s}00.JPG"))
        with mock.patch.object(seg_lists, "sha256_file", return_value="0" * 64):
            out = seg_lists.split_negatives(rows, np.random.default_rng(0))
        self.assertEqual(seg_lists.structural_problems(out), [])
        for tag in ("confusable_grain", "green_legume"):
            n = sum(r["tag"] == tag for r in rows)
            n_eval = sum(e["tag"] == tag for e in out["neg_seg_eval"])
            self.assertLessEqual(abs(n_eval - round(n / 3)), 1, tag)


class SplitPositives(unittest.TestCase):
    def test_bursts_reaching_the_holdout_never_train(self):
        rows = []
        for g in range(6):                                # 6 bursts, 10 minutes apart; bursts 0 and 3 reach the holdout
            for s in range(3):
                split = "holdout" if g % 3 == 0 and s == 1 else "dev"
                r = neg(f"p/{g}_{s}", tag="user_beans_independent", title=f"PIC_20260911_1{g}0{s}00.JPG")
                rows.append({**r, "split": split})
        with mock.patch.object(seg_lists, "sha256_file", return_value="0" * 64):
            out = seg_lists.split_positives(rows, np.random.default_rng(0))
        listed = [e["path"] for v in out.values() for e in v]
        self.assertNotIn("p/0_1", listed)                  # holdout photos are never listed
        self.assertEqual(len(listed), 16)
        near = {e["path"] for e in out["pos_seg_eval"] if e["near_holdout"]}
        self.assertEqual(near, {"p/0_0", "p/0_2", "p/3_0", "p/3_2"})
        self.assertFalse(any(e["near_holdout"] for e in out["pos_seg_train"]))
        self.assertEqual(seg_lists.structural_problems(out), [])


class BaseSplit(unittest.TestCase):
    CANDS = [{"path": f"p{i}", "session": f"s{i % 3}", "roast": "roasted" if i < 8 else "green"} for i in range(60)]

    def test_only_accepted_masks_with_the_accepted_hash(self):
        decisions = [{"path": "p0", "decision": "accept", "mask_sha256": "a"},
                     {"path": "p1", "decision": "accept", "mask_sha256": "b"},
                     {"path": "p1", "decision": "decline", "mask_sha256": "b"}]     # the last write wins
        got = seg_lists.accepted_bases(self.CANDS, decisions, {"p0": "a", "p1": "b"})
        self.assertEqual([e["path"] for e in got], ["p0"])
        with self.assertRaisesRegex(ValueError, "not the one the owner accepted"):
            seg_lists.accepted_bases(self.CANDS, decisions, {"p0": "CHANGED"})

    def test_25_each_disjoint_with_roasted_on_both_sides(self):
        out = seg_lists.split_bases(self.CANDS, np.random.default_rng(0))
        dev, held = ({e["path"] for e in out[k]} for k in ("base_dev", "base_heldout"))
        self.assertEqual((len(dev), len(held)), (25, 25))
        self.assertFalse(dev & held)
        for k in ("base_dev", "base_heldout"):
            self.assertGreaterEqual(sum(e["roast"] == "roasted" for e in out[k]), 3, k)


class Allocate(unittest.TestCase):
    def test_sums_to_total_and_respects_floors(self):
        sizes = {("s1", "green"): 60, ("s2", "green"): 200, ("s3", "roasted"): 14}
        quota = seg_lists.allocate(sizes, 70, {("s3", "roasted"): 6})
        self.assertEqual(sum(quota.values()), 70)
        self.assertGreaterEqual(quota[("s3", "roasted")], 6)
        self.assertTrue(all(quota[k] <= sizes[k] for k in sizes))

    def test_proportional_without_floors(self):
        self.assertEqual(seg_lists.allocate({"a": 50, "b": 50}, 10), {"a": 5, "b": 5})


class AppendOnly(unittest.TestCase):
    OLD = {"seg_eval": [{"path": "p1", "sha256": "a"}], "seg_train": [{"path": "p2", "sha256": "b"}]}

    def test_appending_keeps_old_order(self):
        new = {"seg_eval": [{"path": "p0", "sha256": "z"}, {"path": "p1", "sha256": "a"}],
               "seg_train": [{"path": "p2", "sha256": "b"}]}
        merged = seg_lists.merge_append_only(self.OLD, new)
        self.assertEqual([e["path"] for e in merged["seg_eval"]], ["p1", "p0"])

    def test_moving_an_eval_photo_to_training_is_refused(self):
        new = {"seg_eval": [], "seg_train": [{"path": "p1", "sha256": "a"}, {"path": "p2", "sha256": "b"}]}
        with self.assertRaisesRegex(ValueError, "seg_eval: p1 would be dropped"):
            seg_lists.merge_append_only(self.OLD, new)

    def test_changed_content_is_refused(self):
        new = {"seg_eval": [{"path": "p1", "sha256": "CHANGED"}], "seg_train": [{"path": "p2", "sha256": "b"}]}
        with self.assertRaisesRegex(ValueError, "changed content"):
            seg_lists.merge_append_only(self.OLD, new)


class RetiredCommands(unittest.TestCase):
    """ML-5 D3 retires ML-2's lists: a rebuild or a coverage check against today's data would read folders
    and manifests ML-5 changed (D1), so both refuse, and the committed file is left untouched."""

    def test_build_and_check_refuse_and_leave_the_file(self):
        before = seg_lists.LISTS_FILE.read_bytes()
        for command in ("build", "check"):
            with self.subTest(command=command), self.assertRaisesRegex(SystemExit, "ML-5"):
                seg_lists.main([command])
        self.assertEqual(seg_lists.LISTS_FILE.read_bytes(), before)


@unittest.skipUnless(seg_lists.LISTS_FILE.exists(), "labels/ml2/photo_lists.yaml not built yet")
class CommittedLists(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta, cls.lists = seg_lists.load_lists()

    def test_structure(self):
        self.assertEqual(seg_lists.structural_problems(self.lists), [])

    def test_built_at_the_params_seed(self):
        self.assertEqual(self.meta["seg_split_seed"], RunConfig.from_params_yaml().seg_split_seed)

    def test_sizes(self):
        self.assertEqual(len(self.lists["base_candidates"]), seg_lists.N_BASE_CANDIDATES + seg_lists.N_POS_BASE_CANDIDATES)
        self.assertGreaterEqual(sum(e.get("roast") == "roasted" for e in self.lists["base_candidates"]),
                                seg_lists.MIN_ROASTED_CANDIDATES)
        self.assertEqual(len(self.lists["audit_sample"]), seg_lists.N_AUDIT)

    def test_append_only_against_head(self):
        rel = seg_lists.LISTS_FILE.relative_to(REPO_ROOT)
        shown = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=REPO_ROOT, capture_output=True, text=True)
        if shown.returncode != 0:
            self.skipTest("the lists are not committed yet")
        # The allowed changes: a photo's path before -> after an owner's rename (coffeecv.rename_photos, from
        # labels/photo_renames.csv), and below its hash before -> after the ML-3 metadata strip (`seg_lists
        # rekey`, from labels/ml3/strip_manifest.csv). Pixels are unchanged; any other change still fails.
        from coffeecv.rename_photos import RENAMES, load_renames, replacements, rewrite
        text = rewrite(shown.stdout, replacements(load_renames())[0])[0] if RENAMES.exists() else shown.stdout
        committed = yaml.safe_load(text)["lists"]
        from coffeecv.rekey_photos import MANIFEST, load_map
        rekey, _ = load_map() if MANIFEST.exists() else ({}, set())
        for entries in committed.values():
            for e in entries:
                if e.get("sha256") in rekey:
                    e["sha256"] = rekey[e["sha256"]]
        seg_lists.merge_append_only(committed, self.lists)       # raises on a drop, move or content change
        for name, entries in committed.items():
            self.assertEqual(self.lists[name][:len(entries)], entries, f"{name}: committed entries rewritten")

    def test_ood_positive_bases_come_from_their_eval_side(self):
        pos_eval = {e["path"] for e in self.lists["pos_seg_eval"]}
        seg_eval = {e["path"] for e in self.lists["seg_eval"]}
        pos_bases = [e for e in self.lists["base_candidates"] if e["path"] not in seg_eval]
        self.assertEqual(len(pos_bases), seg_lists.N_POS_BASE_CANDIDATES)
        self.assertTrue(all(e["path"] in pos_eval for e in pos_bases))


if __name__ == "__main__":
    unittest.main()
