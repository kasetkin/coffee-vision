"""The ML-5 review page (ticket ML-5 P6, D14, D19, D22, D23): the labelling loop's rounds, g/r points and
redraws, storage, and the blind paired mode, through Flask's test client on synthetic photos with a stand-in
redraw model. Plain unittest.

    python -m unittest tests.test_seg_review -v
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from coffeecv import seg_review
from coffeecv.backbones import MODELS_PRETRAINED
from coffeecv.config import REPO_ROOT
from coffeecv.sam_loader import L0_WEIGHTS
from coffeecv.segment_beans import BeanSegmenter, mask_sha256, named_params

from tests._tiers import real_data

H, W = 120, 160


def rect(x0, y0, x1, y1) -> np.ndarray:
    """A mask over the fractions [x0, x1) x [y0, y1) of the photo."""
    m = np.zeros((H, W), bool)
    m[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
    return m


class FakeDrawer:
    """The models at their boundary: a proposer that always draws the left half, and a redraw model that draws
    the bounding box of the include points, minus a cell around each exclude point."""

    propose_output = "multi3"

    def __init__(self, name):
        self.name, self.prepared, self.calls = name, [], []

    def needs(self, item):
        return item["id"] not in self.prepared

    def info(self):
        return {"model": self.name, "weights_sha256": "w" * 64, "decoder_sha256": ""}

    def propose(self, rgb):
        return rect(0, 0, 0.5, 1), 0.9

    def prepare(self, item, rgb):
        self.prepared.append(item["id"])

    def redraw(self, item, include, exclude):
        self.calls.append((item["id"], [list(p) for p in include], [list(p) for p in exclude]))
        xs, ys = [p[0] for p in include] or [0], [p[1] for p in include] or [0]
        m = rect(min(xs) - 0.1, min(ys) - 0.1, max(xs) + 0.1, max(ys) + 0.1)
        for x, y in exclude:
            m &= ~rect(x - 0.05, y - 0.05, x + 0.05, y + 0.05)
        return m, 0.8


def v(decision, reason=None):
    return {"decision": decision, "reason": reason}


class Progress(unittest.TestCase):
    """Where a photo is in the loop, from its masks per round and the verdicts on them (D14: up to 3 rounds; ADR
    0030: a session's own number, 7 since 2026-10-24)."""

    def test_the_rounds(self):
        def P(masks, verdicts):
            return seg_review.progress(masks, verdicts, 3)
        self.assertEqual(P({}, {}).status, "pending")                                  # not prepared
        self.assertEqual(P({1: "m1"}, {}), ("judge", 1, None))
        self.assertEqual(P({1: "m1"}, {1: v("accept")}), ("accepted", 1, None))
        self.assertEqual(P({1: "m1"}, {1: v("decline", "beans missed")}), ("points", 1, "beans missed"))
        self.assertEqual(P({1: "m1", 2: "m2"}, {1: v("decline")}), ("judge", 2, None))
        self.assertEqual(P({1: "m1", 2: "m2"}, {1: v("decline"), 2: v("accept")}), ("accepted", 2, None))
        three = {1: "m1", 2: "m2", 3: "m3"}
        self.assertEqual(P(three, {1: v("decline"), 2: v("decline"), 3: v("accept")}), ("accepted", 3, None))
        self.assertEqual(P(three, {1: v("decline"), 2: v("decline"), 3: v("decline", "rim/tray included")}),
                         ("dropped", 3, "declined in round 3: rim/tray included"))
        self.assertEqual(P(three, {1: v("decline"), 2: v("decline"), 3: v("decline")}),
                         ("dropped", 3, "declined in round 3"))

    def test_seven_rounds_drop_only_after_the_seventh(self):
        masks = {k: f"m{k}" for k in range(1, 8)}
        declined = {k: v("decline") for k in range(1, 8)}
        self.assertEqual(seg_review.progress({k: masks[k] for k in (1, 2, 3)}, {k: declined[k] for k in (1, 2, 3)}, 7),
                         ("points", 3, None))
        self.assertEqual(seg_review.progress(masks, {**{k: declined[k] for k in range(1, 7)}, 7: v("accept")}, 7),
                         ("accepted", 7, None))
        self.assertEqual(seg_review.progress(masks, declined, 7), ("dropped", 7, "declined in round 7"))


class SessionFixture(unittest.TestCase):
    """Six synthetic photos (three training and validation positives, a training negative, two test photos), a
    session pass1 over them, prepared, and the page's helpers."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.entries = []
        for k, (source, split) in enumerate([("segmenter_positive", "train"), ("pool", "validation"),
                                             ("internet_positive", "train"), ("negative", "train"),
                                             ("pool", "test"), ("negative", "test")]):
            f = root / "dataset" / source / f"p{k}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((H, W, 3), 40 * k, np.uint8)).save(f)
            self.entries.append({"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                                 "source": source, "split": split, "group": str(f)})
        self.labels_dir, self.mask_root = root / "labels", root / "masks"
        self.patch = mock.patch.object(seg_review, "load_rgb_image", lambda p: np.array(Image.open(p)))
        self.patch.start()
        self.proposer, self.redrawer = FakeDrawer("prop"), FakeDrawer("redraw")
        self.session = self.open()
        self.session.prepare(self.proposer, self.redrawer)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def open(self, proposer="prop", redraw="redraw"):
        return seg_review.LabelSession("pass1", self.entries, {"proposer": proposer, "redraw": redraw, "rounds": 3},
                                       self.labels_dir, self.mask_root)

    def client(self):
        return seg_review.create_label_app(self.open(), self.redrawer, "Accept the mask only if <rule>.").test_client()

    def post(self, c, url, body, code=200):
        r = c.post(url, json=body)
        self.assertEqual(r.status_code, code, r.get_data(as_text=True))
        return r.get_json()

    def info(self, c, i):
        page = c.get(f"/item/{i}").get_data(as_text=True)
        return json.loads(page.split("const info = ", 1)[1].split(";\n", 1)[0])

    def verdict_rows(self):
        return [json.loads(line) for line in (self.labels_dir / "pass1.verdicts.jsonl").read_text().splitlines()]


class LabelSession(SessionFixture):
    def test_only_training_and_validation_positives_are_shown_negatives_get_empty_labels(self):
        c = self.client()
        self.assertEqual(self.info(c, 0)["n"], 3)
        self.assertEqual(c.get("/item/3").status_code, 404)
        self.assertEqual(self.redrawer.prepared, ["segmenter_positive__p0", "pool__p1", "internet_positive__p2"])
        neg = np.array(Image.open(self.mask_root / "pass1" / "neg" / "negative__p3.png"))
        self.assertEqual((neg.shape, int(neg.sum())), ((H, W), 0))
        self.assertFalse((self.mask_root / "pass1" / "neg" / "negative__p5.png").exists())   # test split
        rows = {r["id"]: r for r in self.session.labels()}
        self.assertEqual(rows["negative__p3"]["status"], "negative")
        self.assertNotIn("pool__p4", rows)

    def test_round_1_shows_the_proposal_and_an_accept_makes_it_the_label(self):
        c = self.client()
        info = self.info(c, 0)
        self.assertEqual((info["status"], info["round"], info["model"]), ("judge", 1, "prop"))
        r = self.post(c, "/decide/0", {"decision": "accept"})
        self.assertEqual((r["status"], r["round"], r["next"]), ("accepted", 1, 1))
        row = self.verdict_rows()[-1]
        self.assertEqual({k: row[k] for k in ("item", "round", "decision", "include", "exclude", "model")},
                         {"item": "segmenter_positive__p0", "round": 1, "decision": "accept", "include": [],
                          "exclude": [], "model": "prop"})
        self.assertEqual(row["mask_sha256"], mask_sha256(rect(0, 0, 0.5, 1)))
        label = {r["id"]: r for r in self.session.labels()}["segmenter_positive__p0"]
        self.assertEqual((label["status"], label["round"]), ("accepted", 1))
        self.assertTrue(np.array_equal(np.array(Image.open(Path(label["mask"]))) > 0, rect(0, 0, 0.5, 1)))

    def test_decline_points_redraw_and_judge_again(self):
        c = self.client()
        r = self.post(c, "/decide/1", {"decision": "decline", "reason": "beans missed"})
        self.assertEqual((r["status"], r["round"]), ("points", 1))
        r = self.post(c, "/redraw/1", {"include": [[0.5, 0.5]], "exclude": []})
        self.assertEqual((r["status"], r["round"]), ("judge", 2))
        # not judged yet: another point and Enter replace round 2's mask
        r = self.post(c, "/redraw/1", {"include": [[0.5, 0.5], [0.8, 0.8]], "exclude": [[0.6, 0.6]]})
        self.assertEqual((r["status"], r["round"]), ("judge", 2))
        self.assertEqual(self.redrawer.calls[-1], ("pool__p1", [[0.5, 0.5], [0.8, 0.8]], [[0.6, 0.6]]))
        info = self.info(c, 1)
        self.assertEqual((info["model"], info["points"]), ("redraw", {"include": [[0.5, 0.5], [0.8, 0.8]],
                                                                       "exclude": [[0.6, 0.6]]}))
        r = self.post(c, "/decide/1", {"decision": "accept"})
        self.assertEqual((r["status"], r["round"]), ("accepted", 2))
        row = self.verdict_rows()[-1]
        want = rect(0.5 - 0.1, 0.5 - 0.1, 0.8 + 0.1, 0.8 + 0.1) & ~rect(0.6 - 0.05, 0.6 - 0.05, 0.6 + 0.05, 0.6 + 0.05)
        self.assertEqual((row["round"], row["mask_sha256"], row["include"], row["exclude"], row["output"]),
                         (2, mask_sha256(want), [[0.5, 0.5], [0.8, 0.8]], [[0.6, 0.6]], "multi3"))
        index = list(csv.DictReader((self.mask_root / "pass1" / "r2" / "index.csv").read_text().splitlines()))
        self.assertEqual([(x["id"], x["mask_sha256"]) for x in index], [("pool__p1", mask_sha256(want))])
        self.assertEqual(self.info(self.client(), 1)["status"], "accepted")       # a new server reads it back

    def test_still_declined_in_round_3_is_dropped_with_its_reason(self):
        c = self.client()
        self.post(c, "/decide/2", {"decision": "decline"})
        self.post(c, "/redraw/2", {"include": [[0.3, 0.3]], "exclude": []})
        self.post(c, "/decide/2", {"decision": "decline"})
        self.assertEqual(self.info(c, 2)["points"], {"include": [[0.3, 0.3]], "exclude": []})   # points so far
        self.post(c, "/redraw/2", {"include": [[0.3, 0.3]], "exclude": [[0.9, 0.9]]})
        r = self.post(c, "/decide/2", {"decision": "decline", "reason": "rim/tray included"})
        self.assertEqual((r["status"], r["round"]), ("dropped", 3))
        self.post(c, "/redraw/2", {"include": [[0.5, 0.5]], "exclude": []}, 409)
        label = {r["id"]: r for r in self.session.labels()}["internet_positive__p2"]
        self.assertEqual((label["status"], label["reason"], label["mask"]),
                         ("dropped", "declined in round 3: rim/tray included", ""))

    def test_bad_redraws_and_verdicts_are_refused(self):
        c = self.client()
        self.post(c, "/redraw/0", {"include": [[0.5, 0.5]], "exclude": []}, 409)       # round 1 not declined
        self.post(c, "/decide/0", {"decision": "maybe"}, 400)
        self.post(c, "/decide/0", {"decision": "decline", "reason": "made up"}, 400)
        self.post(c, "/decide/0", {"decision": "decline"})
        self.post(c, "/redraw/0", {"include": [], "exclude": []}, 400)                  # no point
        self.post(c, "/redraw/0", {"include": [[1.5, 0.5]], "exclude": []}, 400)        # off the photo
        self.post(c, "/redraw/9", {"include": [[0.5, 0.5]], "exclude": []}, 404)
        self.post(c, "/redraw/0", {"include": [[0.5, 0.5]], "exclude": []})
        self.post(c, "/decide/0", {"decision": "decline"})
        self.post(c, "/redraw/0", {"include": [[0.5, 0.5]], "exclude": []}, 400)        # nothing new
        self.assertEqual(len(self.redrawer.calls), 1)

    def fenced(self, mask, fence):
        """`mask` cut to `fence`, whose corners map to pixels as points do: f * (side - 1), rounded."""
        x0, y0, x1, y1 = fence
        keep = np.zeros_like(mask)
        keep[round(y0 * (H - 1)):round(y1 * (H - 1)) + 1, round(x0 * (W - 1)):round(x1 * (W - 1)) + 1] = True
        return mask & keep

    def test_a_fence_cuts_the_redraw_to_its_rectangle_and_is_recorded(self):
        # Ticket ML-9 P1: XL0-v3 ignores a box smaller than the photo, so the owner's rectangle bounds the mask.
        c = self.client()
        self.post(c, "/decide/1", {"decision": "decline", "reason": "rim/tray included"})
        fence = [0.45, 0.45, 0.55, 0.55]
        r = self.post(c, "/redraw/1", {"include": [[0.5, 0.5]], "exclude": [], "fence": fence})
        self.assertEqual((r["status"], r["round"]), ("judge", 2))
        self.assertEqual(self.redrawer.calls[-1], ("pool__p1", [[0.5, 0.5]], []))     # the model sees no fence
        want = self.fenced(rect(0.4, 0.4, 0.6, 0.6), fence)
        info = self.info(c, 1)
        self.assertEqual((info["mask_sha256"], info["fence"]), (mask_sha256(want), fence))
        self.assertEqual(info["points"], {"include": [[0.5, 0.5]], "exclude": []})
        self.post(c, "/decide/1", {"decision": "accept"})
        self.assertEqual((self.verdict_rows()[-1]["fence"], self.verdict_rows()[-1]["mask_sha256"]),
                         (fence, mask_sha256(want)))
        index = list(csv.DictReader((self.mask_root / "pass1" / "r2" / "index.csv").read_text().splitlines()))
        self.assertEqual(json.loads(index[0]["fence"]), fence)
        self.session.write_labels()
        label = {r["id"]: r for r in seg_review.accepted_labels("pass1", self.mask_root)}["pool__p1"]
        self.assertEqual((label["fence"], label["include"]), (fence, [[0.5, 0.5]]))

    def test_a_fence_alone_redraws_and_a_new_fence_is_a_new_prompt(self):
        c = self.client()
        self.post(c, "/decide/0", {"decision": "decline"})
        self.post(c, "/redraw/0", {"include": [], "exclude": [], "fence": [0.0, 0.0, 0.05, 0.05]})
        self.assertEqual(self.info(c, 0)["round"], 2)
        self.post(c, "/decide/0", {"decision": "decline"})
        self.post(c, "/redraw/0", {"include": [], "exclude": [], "fence": [0.0, 0.0, 0.05, 0.05]}, 400)   # nothing new
        self.post(c, "/redraw/0", {"include": [], "exclude": [], "fence": [0.0, 0.0, 0.08, 0.08]})
        self.assertEqual((self.info(c, 0)["round"], self.info(c, 0)["fence"]), (3, [0.0, 0.0, 0.08, 0.08]))

    def test_bad_fences_are_refused(self):
        c = self.client()
        self.post(c, "/decide/0", {"decision": "decline"})
        for fence in ([0.6, 0.4, 0.4, 0.7], [0.1, 0.1, 1.2, 0.9], [0.1, 0.1, 0.9], "all", [0.3, 0.3, 0.3, 0.6]):
            self.post(c, "/redraw/0", {"include": [[0.5, 0.5]], "exclude": [], "fence": fence}, 400)
        self.assertEqual(self.redrawer.calls, [])

    def test_rows_from_before_the_fence_read_as_none(self):
        c = self.client()
        self.post(c, "/decide/0", {"decision": "accept"})
        self.assertIsNone(self.info(c, 0)["fence"])
        self.assertIsNone(self.verdict_rows()[-1]["fence"])
        self.session.write_labels()
        f = self.mask_root / "pass1" / "labels.csv"
        rows = list(csv.DictReader(f.read_text().splitlines()))
        with open(f, "w", newline="") as out:                       # a labels.csv written before the column
            wr = csv.DictWriter(out, fieldnames=[k for k in rows[0] if k != "fence"], extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        self.assertIsNone(seg_review.accepted_labels("pass1", self.mask_root)[0]["fence"])

    def test_home_goes_to_the_first_photo_left_to_do_and_images_render(self):
        c = self.client()
        self.post(c, "/decide/0", {"decision": "accept"})
        self.assertEqual(c.get("/").headers["Location"], "/item/1")
        page = c.get("/item/0").get_data(as_text=True)
        self.assertIn("Accept the mask only if &lt;rule&gt;.", page)
        for url in ("/img/0/view.jpg", "/img/0/view.jpg?outline=0", "/img/0/zoom.jpg?x=0.5&y=0.5"):
            r = c.get(url)
            self.assertEqual((r.status_code, r.mimetype), (200, "image/jpeg"), url)

    def test_status_writes_labels_and_reports_by_round_and_source(self):
        c = self.client()
        self.post(c, "/decide/0", {"decision": "accept"})
        self.post(c, "/decide/1", {"decision": "decline"})
        self.post(c, "/redraw/1", {"include": [[0.5, 0.5]], "exclude": []})
        self.post(c, "/decide/1", {"decision": "accept"})
        rows = self.session.write_labels()
        on_disk = list(csv.DictReader((self.mask_root / "pass1" / "labels.csv").read_text().splitlines()))
        self.assertEqual([r["status"] for r in on_disk], ["accepted", "accepted", "pending", "negative"])
        rep = seg_review.report(rows, rounds=3)
        self.assertEqual(rep["by_source"], {"internet_positive": {"pending": 1}, "pool": {"accepted_r2": 1},
                                            "segmenter_positive": {"accepted_r1": 1}})
        self.assertEqual((rep["accepted_by_round"], rep["negatives"]), ({"r1": 1, "r2": 1, "r3": 0}, 1))
        self.assertEqual([(r["path"], r["round"], r["include"]) for r in seg_review.accepted_labels(
            "pass1", self.mask_root)], [(self.entries[0]["path"], 1, []), (self.entries[1]["path"], 2, [[0.5, 0.5]])])

    def test_a_resume_with_other_models_is_refused(self):
        with self.assertRaisesRegex(ValueError, "redraw"):
            self.open(redraw="pretrained_xl0")

    def test_a_session_file_from_before_carry_resumes_without_one(self):
        meta = self.labels_dir / "pass1.session.json"
        old = {k: v for k, v in json.loads(meta.read_text()).items() if k != "carry"}   # as pass1 and pass2 are
        meta.write_text(json.dumps(old))
        self.assertIsNone(self.open().carry)
        self.assertIsNone(seg_review.LabelSession("pass1", self.entries, None, self.labels_dir, self.mask_root).carry)

    def test_a_session_file_from_before_rounds_resumes_with_three(self):
        meta = self.labels_dir / "pass1.session.json"
        old = {k: v for k, v in json.loads(meta.read_text()).items() if k != "rounds"}  # as pass1-pass4 are
        meta.write_text(json.dumps(old))
        resumed = seg_review.LabelSession("pass1", self.entries, {"proposer": "prop", "redraw": "redraw"},
                                          self.labels_dir, self.mask_root)
        self.assertEqual(resumed.rounds, 3)
        self.assertEqual(seg_review.LabelSession("pass1", self.entries, None, self.labels_dir, self.mask_root).rounds, 3)

    def test_a_new_session_gets_seven_rounds_and_keeps_them(self):
        new = seg_review.LabelSession("pass9", self.entries, {"proposer": "prop", "redraw": "redraw"},
                                      self.labels_dir, self.mask_root)
        self.assertEqual(new.rounds, seg_review.ROUNDS_DEFAULT)
        self.assertEqual(seg_review.ROUNDS_DEFAULT, 7)
        self.assertEqual(json.loads((self.labels_dir / "pass9.session.json").read_text())["rounds"], 7)
        self.assertEqual(seg_review.LabelSession("pass9", self.entries, None, self.labels_dir, self.mask_root).rounds, 7)
        with self.assertRaisesRegex(ValueError, "rounds"):
            seg_review.LabelSession("pass9", self.entries, {"proposer": "prop", "redraw": "redraw", "rounds": 3},
                                    self.labels_dir, self.mask_root)
        self.assertEqual(self.info(seg_review.create_label_app(new, self.redrawer, "rule").test_client(), 0)["rounds"], 7)


class CarriedSession(SessionFixture):
    """Ticket ML-6 D3 (a), plan P3: a session that starts from an earlier one. pass1 accepts p0 in round 1 and p1
    in round 2 and drops p2; pass2 carries pass1 and adds a new photo, p6."""

    def setUp(self):
        super().setUp()
        c = self.client()
        self.post(c, "/decide/0", {"decision": "accept"})
        self.post(c, "/decide/1", {"decision": "decline"})
        self.post(c, "/redraw/1", {"include": [[0.5, 0.5]], "exclude": []})
        self.post(c, "/decide/1", {"decision": "accept"})
        for k in range(3):
            self.post(c, "/decide/2", {"decision": "decline", "reason": "beans missed"})
            if k < 2:
                self.post(c, "/redraw/2", {"include": [[0.3, 0.3 + 0.1 * k]], "exclude": []})
        self.pass1_labels = {r["id"]: r for r in self.session.write_labels()}
        f = Path(self.tmp.name) / "dataset" / "segmenter_positive" / "p6.png"
        Image.fromarray(np.full((H, W, 3), 250, np.uint8)).save(f)
        self.entries.append({"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                             "source": "segmenter_positive", "split": "train", "group": str(f)})
        self.proposer2, self.redrawer2 = FakeDrawer("prop2"), FakeDrawer("redraw2")
        self.carried = self.open_pass2()
        self.carried.prepare(self.proposer2, self.redrawer2)

    def open_pass2(self, carry="pass1"):
        return seg_review.LabelSession("pass2", self.entries, {"proposer": "prop2", "redraw": "redraw2",
                                                               "carry": carry, "rounds": 3},
                                       self.labels_dir, self.mask_root)

    def test_an_accepted_photo_keeps_its_round_and_mask_marked_carried(self):
        for i, k in (("segmenter_positive__p0", 1), ("pool__p1", 2)):
            item = next(it for it in self.carried.items if it["id"] == i)
            self.assertEqual(self.carried.state(item), ("accepted", k, None))
            self.assertEqual(self.carried.row(item, k)["mask_sha256"], self.pass1_labels[i]["mask_sha256"])
        rows = [json.loads(x) for x in (self.labels_dir / "pass2.verdicts.jsonl").read_text().splitlines()]
        self.assertEqual({(r["item"], r["round"], r["decision"], r["carried_from"]) for r in rows},
                         {("segmenter_positive__p0", 1, "accept", "pass1"), ("pool__p1", 1, "decline", "pass1"),
                          ("pool__p1", 2, "accept", "pass1")})

    def test_the_page_shows_the_earlier_drop_and_the_new_photo_with_the_new_proposal(self):
        c = seg_review.create_label_app(self.open_pass2(), self.redrawer2, "rule").test_client()
        shown = [self.info(c, i) for i in range(2)]
        self.assertEqual([(s["item"], s["status"], s["round"], s["model"]) for s in shown],
                         [("internet_positive__p2", "judge", 1, "prop2"), ("segmenter_positive__p6", "judge", 1, "prop2")])
        self.assertEqual(shown[0]["n"], 2)
        self.assertEqual(c.get("/item/2").status_code, 404)
        self.assertEqual(self.redrawer2.prepared, ["internet_positive__p2", "segmenter_positive__p6"])

    def test_labels_of_the_carried_photos_equal_the_earlier_sessions_but_for_where_the_mask_is(self):
        rows = {r["id"]: r for r in self.carried.write_labels()}
        for i in ("segmenter_positive__p0", "pool__p1"):
            mine, theirs = rows[i], self.pass1_labels[i]
            self.assertEqual({k: v for k, v in mine.items() if k != "mask"},
                             {k: v for k, v in theirs.items() if k != "mask"})
            self.assertIn("/pass2/", mine["mask"])
            self.assertEqual(Path(mine["mask"]).read_bytes(), Path(theirs["mask"]).read_bytes())
        self.assertEqual((rows["internet_positive__p2"]["status"], rows["segmenter_positive__p6"]["status"]),
                         ("pending", "pending"))

    def test_a_second_prepare_carries_nothing_twice(self):
        before = (self.labels_dir / "pass2.verdicts.jsonl").read_text()
        self.open_pass2().prepare(self.proposer2, self.redrawer2)
        self.assertEqual((self.labels_dir / "pass2.verdicts.jsonl").read_text(), before)

    def test_the_session_file_records_the_carry_and_a_resume_with_another_is_refused(self):
        meta = json.loads((self.labels_dir / "pass2.session.json").read_text())
        self.assertEqual(meta["carry"], "pass1")
        for other in (None, "pass0"):
            with self.subTest(carry=other), self.assertRaisesRegex(ValueError, "carry"):
                self.open_pass2(carry=other)
        resumed = seg_review.LabelSession("pass2", self.entries, None, self.labels_dir, self.mask_root)
        self.assertEqual(resumed.carry, "pass1")

    def test_the_report_counts_carried_and_new_labels_apart(self):
        c = seg_review.create_label_app(self.open_pass2(), self.redrawer2, "rule").test_client()
        self.post(c, "/decide/1", {"decision": "accept"})                  # the new photo, p6
        rep = seg_review.report(self.carried.write_labels(), self.carried.carried(), rounds=3)
        self.assertEqual(rep["by_source"], {"internet_positive": {"pending": 1}, "pool": {"carried_r2": 1},
                                            "segmenter_positive": {"accepted_r1": 1, "carried_r1": 1}})
        self.assertEqual((rep["carried"], rep["accepted_by_round"]), (2, {"r1": 1, "r2": 0, "r3": 0}))


class RecheckedSession(CarriedSession):
    """The owner's re-review of every label (2026-10-25): pass3 rechecks pass1. As in CarriedSession, pass1
    accepted p0 in round 1 and p1 in round 2 and dropped p2, and p6 is new. pass3 shows every positive: p0 and p1
    with their pass1 label as the round-1 mask to judge, p2 and p6 with the proposer's."""

    def setUp(self):
        super().setUp()
        self.proposer3, self.redrawer3 = FakeDrawer("prop3"), FakeDrawer("redraw3")
        self.rechecked = self.open_pass3()
        self.rechecked.prepare(self.proposer3, self.redrawer3)

    def open_pass3(self, recheck="pass1", carry=None):
        return seg_review.LabelSession("pass3", self.entries, {"proposer": "prop3", "redraw": "redraw3",
                                                               "carry": carry, "recheck": recheck, "rounds": 3},
                                       self.labels_dir, self.mask_root)

    def item(self, i):
        return next(it for it in self.rechecked.items if it["id"] == i)

    def test_every_positive_is_shown_an_earlier_label_as_its_round_1_mask(self):
        c = seg_review.create_label_app(self.open_pass3(), self.redrawer3, "rule").test_client()
        shown = [self.info(c, i) for i in range(4)]
        self.assertEqual([(s["item"], s["status"], s["round"], s["model"]) for s in shown],
                         [("segmenter_positive__p0", "judge", 1, "prop"), ("pool__p1", "judge", 1, "redraw"),
                          ("internet_positive__p2", "judge", 1, "prop3"), ("segmenter_positive__p6", "judge", 1, "prop3")])
        for i in ("segmenter_positive__p0", "pool__p1"):
            self.assertEqual(self.rechecked.row(self.item(i), 1)["mask_sha256"], self.pass1_labels[i]["mask_sha256"])
        self.assertEqual(shown[1]["points"], {"include": [[0.5, 0.5]], "exclude": []})     # what drew p1's label
        self.assertEqual(self.rechecked.carried(), set())
        self.assertFalse((self.labels_dir / "pass3.verdicts.jsonl").exists())               # nothing judged yet
        self.assertEqual(sorted(self.redrawer3.prepared), sorted(it["id"] for it in self.rechecked.items))

    def test_an_accepted_recheck_keeps_the_earlier_label(self):
        c = seg_review.create_label_app(self.open_pass3(), self.redrawer3, "rule").test_client()
        self.post(c, "/decide/1", {"decision": "accept"})
        row = {r["id"]: r for r in self.rechecked.write_labels()}["pool__p1"]
        self.assertEqual((row["status"], row["round"], row["mask_sha256"], row["include"]),
                         ("accepted", 1, self.pass1_labels["pool__p1"]["mask_sha256"], "[[0.5, 0.5]]"))

    def test_a_declined_recheck_redraws_from_its_earlier_points_plus_new_ones(self):
        c = seg_review.create_label_app(self.open_pass3(), self.redrawer3, "rule").test_client()
        self.post(c, "/decide/1", {"decision": "decline", "reason": "beans missed"})
        self.post(c, "/redraw/1", {"include": [[0.5, 0.5]], "exclude": []}, code=400)      # no new point
        self.post(c, "/redraw/1", {"include": [[0.5, 0.5], [0.2, 0.2]], "exclude": []})
        self.assertEqual(self.redrawer3.calls, [("pool__p1", [[0.5, 0.5], [0.2, 0.2]], [])])
        self.assertEqual(self.rechecked.state(self.item("pool__p1")), ("judge", 2, None))

    def test_a_second_prepare_changes_nothing(self):
        index = (self.mask_root / "pass3" / "r1" / "index.csv").read_text()
        self.open_pass3().prepare(self.proposer3, self.redrawer3)
        self.assertEqual((self.mask_root / "pass3" / "r1" / "index.csv").read_text(), index)

    def test_the_session_file_records_the_recheck_and_a_resume_with_another_is_refused(self):
        self.assertEqual(json.loads((self.labels_dir / "pass3.session.json").read_text())["recheck"], "pass1")
        for other in (None, "pass2"):
            with self.subTest(recheck=other), self.assertRaisesRegex(ValueError, "recheck"):
                self.open_pass3(recheck=other)
        self.assertEqual(seg_review.LabelSession("pass3", self.entries, None, self.labels_dir, self.mask_root).recheck,
                         "pass1")

    def test_a_session_carries_or_rechecks_not_both(self):
        with self.assertRaisesRegex(ValueError, "carries or rechecks"):
            seg_review.LabelSession("pass4", self.entries, {"proposer": "p", "redraw": "r", "carry": "pass1",
                                                            "recheck": "pass2"}, self.labels_dir, self.mask_root)
        with self.assertRaisesRegex(ValueError, "itself"):
            seg_review.LabelSession("pass4", self.entries, {"proposer": "p", "redraw": "r", "recheck": "pass4"},
                                    self.labels_dir, self.mask_root)

    def test_a_session_file_from_before_recheck_resumes_without_one(self):
        meta = self.labels_dir / "pass2.session.json"
        meta.write_text(json.dumps({k: v for k, v in json.loads(meta.read_text()).items() if k != "recheck"}))
        self.assertIsNone(self.open_pass2().recheck)


class SessionMasksStayOutOfGit(unittest.TestCase):
    """docs/agents/dvc.md, "Keeping git and DVC apart": a session's masks are gitignored from its first click,
    not only once `dvc add data/ml5_labels/<session>` writes its own .gitignore, and that `dvc add` still works
    with the rule in place (DVC refuses to write a .dvc file that git ignores). Run in a scratch repo carrying
    this repo's .gitignore."""

    SESSION = seg_review.MASK_ROOT.relative_to(REPO_ROOT) / "pass1"

    def git(self, root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)

    def ignored(self, root: Path, path: Path) -> bool:
        return self.git(root, "check-ignore", "-q", "--no-index", str(path), check=False).returncode == 0

    def test_masks_are_ignored_before_and_after_dvc_add_and_only_the_dvc_file_shows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.git(root, "init", "-q")
            (root / ".gitignore").write_text((REPO_ROOT / ".gitignore").read_text())
            mask = self.SESSION / "r1" / "dataset__x.png"
            for f in (mask, self.SESSION / "labels.csv"):
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                (root / f).write_bytes(b"x")
                self.assertTrue(self.ignored(root, f), f)
            subprocess.run([sys.executable, "-m", "dvc", "init", "-q"], cwd=root, check=True)
            add = subprocess.run([sys.executable, "-m", "dvc", "add", "-q", str(self.SESSION)], cwd=root,
                                 capture_output=True, text=True)
            self.assertEqual(add.returncode, 0, add.stderr)
            self.assertTrue(self.ignored(root, mask))
            self.assertFalse(self.ignored(root, self.SESSION.with_suffix(".dvc")))
            untracked = self.git(root, "status", "--porcelain", "-uall").stdout.split("\n")
            self.assertEqual([x[3:] for x in untracked if x.startswith("??") and x[3:].startswith("data/")],
                             [self.SESSION.with_suffix(".dvc").as_posix()])


@real_data
@unittest.skipUnless((MODELS_PRETRAINED / L0_WEIGHTS).exists(), "EfficientViT-SAM-L0 weights missing (dvc checkout)")
class RealRedraw(unittest.TestCase):
    """The redraw model as the page runs it: encoded once before the session and stored, then a click decodes
    the stored encoding through multi3 (D23), giving the mask a full run gives."""

    def test_a_redraw_from_the_stored_encoding_is_the_full_runs_multi3_mask(self):
        yy, xx = np.mgrid[:300, :400]
        rgb = np.full((300, 400, 3), 60, np.uint8)
        rgb[(yy - 150) ** 2 + (xx - 220) ** 2 < 90 ** 2] = (200, 160, 90)
        item, inc, exc = {"id": "x", "sha256": "ab" * 32}, [[0.55, 0.5]], [[0.1, 0.1]]
        with tempfile.TemporaryDirectory() as tmp:
            d = seg_review.SegDrawer("pretrained_l0", embed_root=Path(tmp))
            self.assertTrue(d.needs(item))
            d.prepare(item, rgb)
            self.assertFalse(d.needs(item))
            d.seg.encode(np.zeros_like(rgb))                     # another photo in between
            mask, iou = d.redraw(item, inc, exc)
        want, want_iou = BeanSegmenter(named_params("pretrained_l0")).predict_with_points(rgb, inc, exc, "multi3")
        self.assertTrue(np.array_equal(mask, want))
        self.assertEqual(iou, want_iou)


class PairedSession(unittest.TestCase):
    """D22: each test positive's two masks side by side, sides at random per photo, models hidden until the end."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.entries, self.dirs = [], [root / "masks" / "xl0_v1", root / "masks" / "xl0_v2"]
        index = {d: [] for d in self.dirs}
        for k, (source, split) in enumerate([("pool", "test"), ("segmenter_positive", "test"),
                                             ("internet_positive", "test"), ("pool", "train"), ("negative", "test")]):
            f = root / "dataset" / source / f"p{k}.png"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((H, W, 3), 40 * k, np.uint8)).save(f)
            sha = hashlib.sha256(f.read_bytes()).hexdigest()
            self.entries.append({"path": str(f), "sha256": sha, "source": source, "split": split, "group": str(f)})
            for j, d in enumerate(self.dirs):
                m = rect(0, 0, 0.5, 1) if (k == 2 or j == 0) else rect(0.5, 0, 1, 1)   # p2: identical masks
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(m).convert("1").save(d / f"id{k}.png")
                index[d].append({"id": f"id{k}", "path": str(f), "photo_sha256": sha, "mask_sha256": mask_sha256(m)})
        for d, rows in index.items():
            with open(d / "index.csv", "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
        self.decisions = root / "labels" / "p10.paired.jsonl"
        self.patch = mock.patch.object(seg_review, "load_rgb_image", lambda p: np.array(Image.open(p)))
        self.patch.start()
        self.items = seg_review.paired_items(self.entries, self.dirs)
        self.c = seg_review.create_paired_app(self.items, self.decisions, 7, "the rule").test_client()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def post(self, i, body, code=200):
        r = self.c.post(f"/decide/{i}", json=body)
        self.assertEqual(r.status_code, code, r.get_data(as_text=True))
        return r.get_json()

    def test_only_test_positives_with_two_masks_each(self):
        self.assertEqual([it["source"] for it in self.items], ["pool", "segmenter_positive", "internet_positive"])
        (self.dirs[1] / "index.csv").write_text("id,path,photo_sha256,mask_sha256\n")
        with self.assertRaisesRegex(ValueError, "xl0_v2"):
            seg_review.paired_items(self.entries, self.dirs)

    def test_sides_are_drawn_per_photo_from_the_seed_and_recorded(self):
        orders = {seg_review.paired_sides(f"{k:064x}", 7) for k in range(40)}
        self.assertEqual(orders, {(0, 1), (1, 0)})                     # (left, right) as indexes: both occur
        self.assertEqual(seg_review.paired_sides("ab" * 32, 7), seg_review.paired_sides("ab" * 32, 7))
        left = seg_review.paired_sides(self.items[0]["photo_sha256"], 7)[0]
        self.post(0, {"side": "left", "decision": "accept"})
        self.post(0, {"side": "right", "decision": "decline", "reason": "beans missed"})
        rows = [json.loads(x) for x in self.decisions.read_text().splitlines()]
        self.assertEqual([(r["side"], r["models"], r["decision"]) for r in rows],
                         [("left", [self.items[0]["names"][left]], "accept"),
                          ("right", [self.items[0]["names"][1 - left]], "decline")])
        self.assertEqual(rows[0]["mask_sha256"], self.items[0]["masks"][left]["mask_sha256"])

    def test_model_names_are_hidden_until_every_mask_is_judged(self):
        for i in range(3):
            self.assertNotIn("xl0_v", self.c.get(f"/item/{i}").get_data(as_text=True))
        self.assertEqual(self.c.get("/reveal").status_code, 403)
        self.post(0, {"side": "left", "decision": "accept"})
        self.post(0, {"side": "right", "decision": "accept"})
        self.post(1, {"side": "left", "decision": "decline"})
        self.post(1, {"side": "right", "decision": "accept"})
        r = self.post(2, {"side": "both", "decision": "accept"})
        self.assertTrue(r["complete"])
        reveal = self.c.get("/reveal").get_json()
        self.assertEqual(sorted(reveal["models"]), ["xl0_v1", "xl0_v2"])
        self.assertEqual(reveal["verdicts"][self.items[2]["item"]], {"xl0_v1": "accept", "xl0_v2": "accept"})
        self.assertIn("xl0_v", self.c.get("/item/0").get_data(as_text=True))

    def test_identical_masks_are_judged_once(self):
        page = self.c.get("/item/2").get_data(as_text=True)
        self.assertIn('"identical": true', page)
        self.post(2, {"side": "left", "decision": "accept"}, 400)
        self.post(0, {"side": "both", "decision": "accept"}, 400)
        self.post(2, {"side": "both", "decision": "decline", "reason": "other"})
        row = json.loads(self.decisions.read_text().splitlines()[-1])
        self.assertEqual((row["side"], sorted(row["models"])), ("both", ["xl0_v1", "xl0_v2"]))

    def test_images(self):
        for url in ("/img/0/left.jpg", "/img/0/right.jpg?outline=0", "/img/2/both.jpg",
                    "/img/1/right/zoom.jpg?x=0.5&y=0.5"):
            r = self.c.get(url)
            self.assertEqual((r.status_code, r.mimetype), (200, "image/jpeg"), url)
        self.assertEqual(self.c.get("/img/0/both.jpg").status_code, 404)

    def judge(self, verdicts):
        """Judge every mask through the page: `verdicts` {item index: (xl0_v1 accepted, xl0_v2 accepted)}."""
        for i, (a, b) in verdicts.items():
            it = self.items[i]
            for side, ks in seg_review.shown_sides(it, 7).items():
                ok = (a, b)[ks[0]]
                self.post(i, {"side": side, "decision": "accept" if ok else "decline"})

    def test_paired_report_counts_pairs_by_model_not_by_side(self):
        with self.assertRaisesRegex(ValueError, "not judged"):
            seg_review.paired_verdicts(self.items, self.decisions, 7)
        self.judge({0: (True, False), 1: (False, True), 2: (True, True)})
        verdicts = seg_review.paired_verdicts(self.items, self.decisions, 7)
        self.assertEqual(verdicts[self.items[0]["item"]], {"xl0_v1": True, "xl0_v2": False})
        self.assertEqual(verdicts[self.items[2]["item"]], {"xl0_v1": True, "xl0_v2": True})   # one verdict, both
        empty, tiny, big = np.zeros((10, 10), bool), np.zeros((10, 10), bool), np.ones((10, 10), bool)
        tiny[0, 0] = True                                                   # 1% of the photo, under 0.083
        b1, b2 = "user_realworld", "2026-09__internet_proxy"            # by batch (ticket ML-8 D9)
        rep = seg_review.paired_report(self.items, verdicts, {"xl0_v1": [(b1, empty), (b1, tiny), (b2, big)],
                                                              "xl0_v2": [(b2, big)]}, 0.083)
        g = rep["groups"]
        self.assertEqual((g["pool"]["n"], g["pool"]["xl0_v1 only"]), (1, 1))
        self.assertEqual((g["segmenter_positive"]["n"], g["segmenter_positive"]["xl0_v2 only"],   # internet joins
                          g["segmenter_positive"]["both pass"]), (2, 1, 1))
        self.assertEqual({k: g["overall"][k] for k in ("both pass", "xl0_v1 only", "xl0_v2 only", "both fail")},
                         {"both pass": 1, "xl0_v1 only": 1, "xl0_v2 only": 1, "both fail": 0})
        self.assertEqual(g["overall"]["mcnemar_p"], 1.0)
        self.assertEqual({k: rep["negatives"]["xl0_v1"][k] for k in ("n", "empty", "empty_or_tiny")},
                         {"n": 3, "empty": 1, "empty_or_tiny": 2})
        self.assertEqual({k: rep["negatives"]["xl0_v2"][k] for k in ("n", "empty", "empty_or_tiny")},
                         {"n": 1, "empty": 0, "empty_or_tiny": 0})
        # ML-6 P7 (R1, D6): each model's test negatives per batch as well as overall
        self.assertEqual(rep["negatives"]["xl0_v1"]["by_batch"],
                         {b1: {"n": 2, "empty": 1, "empty_or_tiny": 2}, b2: {"n": 1, "empty": 0, "empty_or_tiny": 0}})
        self.assertEqual(rep["negatives"]["xl0_v2"]["by_batch"], {b2: {"n": 1, "empty": 0, "empty_or_tiny": 0}})
        self.assertEqual(json.loads(json.dumps(rep))["negatives"], rep["negatives"])      # --out writes it

    def test_tiny_threshold_is_half_the_smallest_accepted_mask_of_the_chosen_model(self):
        self.judge({0: (True, False), 1: (False, True), 2: (True, True)})
        verdicts = seg_review.paired_verdicts(self.items, self.decisions, 7)
        ids = [it["item"] for it in self.items]
        areas = {ids[0]: 0.30, ids[1]: 0.05, ids[2]: 0.20}                  # item 1's xl0_v1 mask was declined
        t = seg_review.tiny_threshold(self.items, verdicts, "xl0_v1", areas, 0.083)
        self.assertEqual((t["new"], t["from_n_accepted"]), (0.1, 2))
        self.assertEqual(t["under_old"], [self.items[1]["path"]])
        self.assertEqual(t["under_new"], [self.items[1]["path"]])

    def test_an_accepted_empty_mask_sets_no_threshold(self):
        """XL0-v5's test split (2026-10-26): the owner accepted empty masks on photos with no pile (the accept
        rule's point 4); half of 0 would let every speck crop. The smallest non-empty accepted mask sets it."""
        self.judge({0: (True, False), 1: (False, True), 2: (True, True)})
        verdicts = seg_review.paired_verdicts(self.items, self.decisions, 7)
        ids = [it["item"] for it in self.items]
        areas = {ids[0]: 0.0, ids[1]: 0.05, ids[2]: 0.20}
        t = seg_review.tiny_threshold(self.items, verdicts, "xl0_v1", areas, 0.083)
        self.assertEqual((t["new"], t["from_n_accepted"], t["accepted_empty"]), (0.1, 1, 1))
        self.assertEqual(t["under_new"], [self.items[0]["path"], self.items[1]["path"]])


class McNemar(unittest.TestCase):
    def test_exact_two_sided_p(self):
        self.assertEqual(seg_review.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(seg_review.mcnemar_exact(0, 5), 2 / 32)
        self.assertAlmostEqual(seg_review.mcnemar_exact(9, 1), 2 * 11 / 1024)
        self.assertEqual(seg_review.mcnemar_exact(3, 3), 1.0)


if __name__ == "__main__":
    unittest.main()
