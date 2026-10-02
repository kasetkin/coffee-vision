"""The P5 label loop's decisions (ticket ML-2 D6/D11, plan §8): when a photo is accepted, corrected, dropped or
pending; that corrective points accumulate over rounds; that the photo lists it reads are disjoint from every
eval list. No judge and no SAM is called. Plain unittest.

    python -m unittest discover -s tests -p 'test_seg_labels.py'
"""
from __future__ import annotations

import unittest

from coffeecv import seg_labels, seg_lists

INC = {"x": 0.5, "y": 0.5, "label": "include"}
EXC = {"x": 0.1, "y": 0.9, "label": "exclude"}


def v(verdict: str, points: list[dict] | None = None) -> dict:
    return {"verdict": verdict, "failed_rules": [1] if verdict == "decline" else [], "points_photo": points or []}


class NextStep(unittest.TestCase):
    def test_accept_in_any_round(self):
        for k in range(1, 4):
            vs = [v("decline", [INC])] * (k - 1) + [v("accept")]
            self.assertEqual(seg_labels.next_step(vs), ("accepted", ""))

    def test_decline_with_points_is_corrected_while_rounds_remain(self):
        self.assertEqual(seg_labels.next_step([v("decline", [INC])])[0], "correct")
        self.assertEqual(seg_labels.next_step([v("decline", [INC])] * 2)[0], "correct")

    def test_last_round_decline_is_dropped(self):
        self.assertEqual(seg_labels.next_step([v("decline", [INC])] * 3)[0], "dropped")

    def test_decline_without_points_is_dropped(self):
        self.assertEqual(seg_labels.next_step([v("decline")]), ("dropped", "declined with no corrective point"))

    def test_unjudged_never_counts_as_accept(self):
        self.assertEqual(seg_labels.next_step([{"verdict": "unjudged"}]), ("dropped", "unjudged"))

    def test_missing_verdict_waits(self):
        self.assertEqual(seg_labels.next_step([])[0], "pending")
        self.assertEqual(seg_labels.next_step([v("decline", [INC]), None])[0], "pending")


class Points(unittest.TestCase):
    def test_points_accumulate_over_rounds(self):
        inc, exc = seg_labels.points_so_far([v("decline", [INC, EXC]), v("decline", [{**INC, "x": 0.25}])])
        self.assertEqual(inc, [(0.5, 0.5), (0.25, 0.5)])
        self.assertEqual(exc, [(0.1, 0.9)])


class Report(unittest.TestCase):
    def test_yield_counts_corrected_masks(self):
        rows = [{"id": "d__s__c__a", "list": "seg_train", "status": "accepted", "round": "1", "reason": "", "failed_rules": ""},
                {"id": "d__s__c__b", "list": "seg_train", "status": "accepted", "round": "3", "reason": "", "failed_rules": ""},
                {"id": "d__s__c__c", "list": "seg_val", "status": "dropped", "round": "3", "reason": "declined in round 3",
                 "failed_rules": "1"},
                {"id": "b__n", "list": "neg_seg_train", "status": "negative", "round": "", "reason": "", "failed_rules": ""}]
        rep = seg_labels.report(rows)
        self.assertEqual((rep["photos"], rep["negatives"], rep["corrected_accepted"]), (3, 1, 1))
        self.assertEqual(rep["accepted_by_round"], {"r1": 1, "r2": 0, "r3": 1})
        self.assertEqual(rep["by_session"]["d__s"], {"accepted_r1": 1, "accepted_r3": 1, "dropped": 1})


class Lists(unittest.TestCase):
    def test_label_photos_never_come_from_an_eval_list(self):
        _, lists = seg_lists.load_lists()
        label = {e["path"] for e in seg_labels.entries(lists, seg_labels.POSITIVE_LISTS + seg_labels.NEGATIVE_LISTS)}
        for name in ("seg_eval", "neg_seg_eval", "pos_seg_eval", "base_candidates", "audit_sample"):
            self.assertEqual(label & {e["path"] for e in lists[name]}, set(), name)


if __name__ == "__main__":
    unittest.main()
