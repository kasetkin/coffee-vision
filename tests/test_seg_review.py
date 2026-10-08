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
    """Where a photo is in the loop, from its masks per round and the verdicts on them (D14: up to 3 rounds)."""

    def test_the_rounds(self):
        P = seg_review.progress
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


class LabelSession(unittest.TestCase):
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
        return seg_review.LabelSession("pass1", self.entries, {"proposer": proposer, "redraw": redraw},
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
        rep = seg_review.report(rows)
        self.assertEqual(rep["by_source"], {"internet_positive": {"pending": 1}, "pool": {"accepted_r2": 1},
                                            "segmenter_positive": {"accepted_r1": 1}})
        self.assertEqual((rep["accepted_by_round"], rep["negatives"]), ({"r1": 1, "r2": 1, "r3": 0}, 1))
        self.assertEqual([(r["path"], r["round"], r["include"]) for r in seg_review.accepted_labels(
            "pass1", self.mask_root)], [(self.entries[0]["path"], 1, []), (self.entries[1]["path"], 2, [[0.5, 0.5]])])

    def test_a_resume_with_other_models_is_refused(self):
        with self.assertRaisesRegex(ValueError, "redraw"):
            self.open(redraw="pretrained_xl0")


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


if __name__ == "__main__":
    unittest.main()
