"""The judge's I/O (ticket ML-2 P1, D13/D16, plan §4.3, §7): schema parse and one retry, unjudged never counts
as accept, verdicts keyed by the mask array hash, a pass-rate reader that fails loudly naming missing keys,
and the prompt carrying the D13 rule byte for byte. No judge is called. Plain unittest.

    python -m unittest discover -s tests -p 'test_seg_judge_io.py'
"""
from __future__ import annotations

import html
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from coffeecv import seg_judge, seg_overlay
from coffeecv.config import REPO_ROOT
from coffeecv.segment_beans import mask_sha256

ACCEPT = '{"verdict": "accept", "failed_rules": [], "reason": "Covers the pile.", "points": []}'
DECLINE = ('```json\n{"verdict": "decline", "failed_rules": [1], "reason": "Rim inside on the left.", '
           '"points": [{"where": "T2", "x": 0.1, "y": 0.5, "label": "exclude"}]}\n```')


class Parse(unittest.TestCase):
    def test_valid_replies(self):
        self.assertEqual(seg_judge.parse_verdict(ACCEPT, 6)["verdict"], "accept")
        d = seg_judge.parse_verdict(DECLINE, 6)
        self.assertEqual((d["verdict"], d["failed_rules"], d["points"][0]["where"]), ("decline", [1], "T2"))

    def test_off_schema_replies_are_errors(self):
        bad = [
            "no json here",
            '{"verdict": "pass", "failed_rules": [], "reason": "x", "points": []}',
            '{"verdict": "decline", "failed_rules": [], "reason": "x", "points": []}',     # decline, no rule
            '{"verdict": "accept", "failed_rules": [2], "reason": "x", "points": []}',     # accept with a rule
            '{"verdict": "decline", "failed_rules": [5], "reason": "x", "points": []}',
            '{"verdict": "accept", "failed_rules": [], "reason": "", "points": []}',
            '{"verdict": "accept", "failed_rules": [], "reason": "x", '
            '"points": [{"where": "overview", "x": 0.5, "y": 0.5, "label": "include"}]}',  # points on accept
            '{"verdict": "decline", "failed_rules": [1], "reason": "x", '
            '"points": [{"where": "T7", "x": 0.5, "y": 0.5, "label": "include"}]}',        # no such tile
            '{"verdict": "decline", "failed_rules": [1], "reason": "x", '
            '"points": [{"where": "T1", "x": 1.5, "y": 0.5, "label": "include"}]}',
            '{"verdict": "decline", "failed_rules": [1], "reason": "x", "points": ['
            + ",".join(['{"where": "T1", "x": 0.5, "y": 0.5, "label": "include"}'] * 7) + "]}",
        ]
        for text in bad:
            with self.assertRaises((ValueError, json.JSONDecodeError), msg=text):
                seg_judge.parse_verdict(text, 6)


class Prompt(unittest.TestCase):
    def test_rule_file_is_the_ticket_d13_text(self):
        ticket = (REPO_ROOT / "docs" / "ticket_segmentation_mask.html").read_text()
        d13 = html.unescape(re.search(r"<br>(Accept the mask only if.*?Otherwise, decline\.)</td>", ticket, re.S).group(1))
        self.assertEqual(seg_judge.RULE_FILE.read_text().strip(), d13)

    def test_prompt_holds_the_rule_byte_for_byte(self):
        prompt = seg_judge.build_prompt()
        self.assertIn(seg_judge.RULE_FILE.read_text().strip().encode(), prompt.encode())
        self.assertNotIn(seg_judge.PLACEHOLDER, prompt)


class Overlay(unittest.TestCase):
    META = seg_overlay.OverlayMeta((1001, 2001), 0.5, [(100, 200, 768, 768)], False)

    def test_points_map_back_to_photo_fractions(self):
        self.assertEqual(seg_overlay.to_photo_xy("overview", 0.25, 0.5, self.META), (0.25, 0.5))
        x, y = seg_overlay.to_photo_xy("T1", 0.0, 1.0, self.META)
        self.assertAlmostEqual(x * 2000, 100)
        self.assertAlmostEqual(y * 1000, 200 + 767)
        with self.assertRaises(ValueError):
            seg_overlay.to_photo_xy("T2", 0.5, 0.5, self.META)

    def test_png_carries_no_metadata(self):
        rgb = np.full((300, 400, 3), 120, np.uint8)
        mask = np.zeros((300, 400), bool)
        mask[50:250, 100:300] = True
        with tempfile.TemporaryDirectory() as tmp:
            paths, meta, _ = seg_overlay.write(rgb, mask, Path(tmp))
            self.assertEqual([p.stem for p in paths], ["overview", *(f"T{k}" for k in range(1, 7))])
            for p in paths:
                with Image.open(p) as im:
                    self.assertFalse(im.getexif())
                    self.assertFalse({k for k in im.info if k not in ("dpi",)}, p.name)


class Loop(unittest.TestCase):
    """judge() with a fake backend on a tiny synthetic item."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rgb = np.full((200, 300, 3), 90, np.uint8)
        self.mask = np.zeros((200, 300), bool)
        self.mask[40:160, 60:240] = True
        Image.fromarray(rgb).save(root / "photo.png")
        Image.fromarray(self.mask).convert("1").save(root / "mask.png")
        import hashlib
        self.item = {"item": "x1", "path": str(root / "photo.png"), "mask": str(root / "mask.png"),
                     "photo_sha256": hashlib.sha256((root / "photo.png").read_bytes()).hexdigest(),
                     "mask_sha256": mask_sha256(self.mask), "trial": 0}
        self.verdicts = root / "verdicts.jsonl"
        self.patches = [mock.patch.object(seg_judge, "VERDICTS_FILE", self.verdicts),
                        mock.patch.object(seg_judge, "OVERLAY_DIR", root / "overlays"),
                        mock.patch.object(seg_judge, "load_rgb_image", lambda p: np.array(Image.open(p)))]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_with(self, replies: list[str], **kw) -> mock.Mock:
        fake = mock.Mock(side_effect=[{"text": t, "usage": {"input_tokens": 10, "output_tokens": 5},
                                       "cost_usd": 0.01, "wall_s": 1.0} for t in replies])
        with mock.patch.object(seg_judge, "call_cli", fake):
            seg_judge.judge([self.item], "m", "cli", False, kw.get("rejudge", False), False, 1, "claude", None)
        return fake

    def test_bad_reply_is_retried_once(self):
        fake = self.run_with(["garbage", DECLINE])
        self.assertEqual(fake.call_count, 2)
        row = seg_judge.read_verdicts()[0]
        self.assertEqual((row["verdict"], row["attempts"]), ("decline", 2))
        self.assertEqual(row["mask_sha256"], mask_sha256(self.mask))
        self.assertEqual(row["usage"]["input_tokens"], 20)
        self.assertEqual(len(row["points_photo"]), 1)

    def test_two_bad_replies_are_unjudged_never_accept(self):
        self.run_with(["garbage", "still garbage"])
        row = seg_judge.read_verdicts()[0]
        self.assertEqual(row["verdict"], "unjudged")
        self.assertEqual(row["raw"], ["garbage", "still garbage"])

    def test_failed_call_is_not_a_verdict(self):
        fake = mock.Mock(side_effect=RuntimeError("claude -p exited 1: limit"))
        with mock.patch.object(seg_judge, "call_cli", fake), mock.patch.object(seg_judge, "RETRY_WAIT_S", 0):
            seg_judge.judge([self.item], "m", "cli", False, False, False, 1, "claude", None)
        self.assertEqual((fake.call_count, seg_judge.read_verdicts()), (2, []))
        self.assertEqual(self.run_with([ACCEPT]).call_count, 1)          # judged on the next run

    def test_judged_item_is_skipped_unless_rejudge(self):
        self.run_with([ACCEPT])
        self.assertEqual(self.run_with([]).call_count, 0)
        self.assertEqual(self.run_with([ACCEPT], rejudge=True).call_count, 1)
        self.assertEqual(len(seg_judge.read_verdicts()), 2)

    def test_a_different_mask_file_is_refused(self):
        other = self.mask.copy()
        other[0, 0] = True
        Image.fromarray(other).convert("1").save(self.item["mask"])
        with self.assertRaisesRegex(ValueError, "not the mask the manifest pins"):
            self.run_with([ACCEPT])

    def test_require_verdicts_names_what_is_missing(self):
        self.run_with([ACCEPT])
        psha = seg_judge.prompt_sha256(seg_judge.build_prompt())
        key = (self.item["photo_sha256"], self.item["mask_sha256"])
        self.assertEqual(seg_judge.require_verdicts([key], "m", psha)[key]["verdict"], "accept")
        with self.assertRaisesRegex(LookupError, "1 masks have no verdict.*" + "f" * 12):
            seg_judge.require_verdicts([key, ("f" * 64, "f" * 64)], "m", psha)
        with self.assertRaises(LookupError):
            seg_judge.require_verdicts([key], "other-model", psha)


class Wilson(unittest.TestCase):
    def test_known_values(self):
        lo, hi = seg_judge.wilson(45, 50)
        self.assertAlmostEqual(lo, 0.7864, places=3)
        self.assertAlmostEqual(hi, 0.9565, places=3)


if __name__ == "__main__":
    unittest.main()
