"""The owner's review page (ticket ML-2 D12, plan §4.2, §7): write, overwrite, unreviewed and blind mode,
through Flask's test client on a synthetic item. Plain unittest.

    python -m unittest discover -s tests -p 'test_review_masks.py'
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from coffeecv import review_masks
from coffeecv.segment_beans import mask_sha256


class Review(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.items = []
        for k in range(3):
            mask = np.zeros((300, 400), bool)
            mask[50:250, 60 + k:300] = True
            Image.fromarray(np.full((300, 400, 3), 100 + k, np.uint8)).save(root / f"p{k}.png")
            Image.fromarray(mask).convert("1").save(root / f"m{k}.png")
            self.items.append({"item": f"it{k}", "path": str(root / f"p{k}.png"), "mask": str(root / f"m{k}.png"),
                               "mask_sha256": mask_sha256(mask), "photo_sha256": f"ph{k}"})
        self.decisions = root / "d.jsonl"
        self.patch = mock.patch.object(review_masks, "load_rgb_image", lambda p: np.array(Image.open(p)))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def client(self, verdicts=None):
        return review_masks.create_app(self.items, self.decisions, "Accept the mask only if <rule>.", verdicts).test_client()

    def decide(self, c, i, decision, reason=None):
        r = c.post(f"/decide/{i}", json={"decision": decision, "reason": reason})
        self.assertEqual(r.status_code, 200)
        return r.get_json()

    def test_write_appends_one_row_and_moves_on(self):
        c = self.client()
        r = self.decide(c, 0, "decline", "beans missed")
        self.assertEqual((r["reviewed"], r["decline"], r["next"]), (1, 1, 1))
        rows = [json.loads(l) for l in self.decisions.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual({k: rows[0][k] for k in ("item", "decision", "reason", "mask_sha256")},
                         {"item": "it0", "decision": "decline", "reason": "beans missed",
                          "mask_sha256": self.items[0]["mask_sha256"]})

    def test_overwrite_last_click_wins(self):
        c = self.client()
        self.decide(c, 0, "decline", "other")
        r = self.decide(c, 0, "accept", "other")               # a reason on accept is dropped
        self.assertEqual((r["reviewed"], r["accept"], r["decline"]), (1, 1, 0))
        dec = review_masks.current_decisions(self.decisions, self.items)
        self.assertEqual((dec["it0"]["decision"], dec["it0"]["reason"]), ("accept", None))

    def test_unreviewed_items_and_changed_masks(self):
        c = self.client()
        self.decide(c, 0, "accept")
        self.assertEqual(c.get("/").headers["Location"], "/item/1")     # home goes to the first unreviewed
        self.items[0]["mask_sha256"] = "0" * 64                          # the mask was redrawn
        self.assertNotIn("it0", review_masks.current_decisions(self.decisions, self.items))

    def test_bad_input_is_refused(self):
        c = self.client()
        self.assertEqual(c.post("/decide/0", json={"decision": "maybe"}).status_code, 400)
        self.assertEqual(c.post("/decide/0", json={"decision": "decline", "reason": "made up"}).status_code, 400)
        self.assertEqual(c.post("/decide/9", json={"decision": "accept"}).status_code, 404)
        self.assertFalse(self.decisions.exists())

    def test_blind_reveals_the_verdict_only_after_the_click(self):
        c = self.client({"it0": {"verdict": "decline", "failed_rules": [1], "reason": "JUDGE-SAYS", "model": "m"}})
        page = c.get("/item/0").get_data(as_text=True)
        self.assertNotIn("JUDGE-SAYS", page)
        self.assertEqual(self.decide(c, 0, "accept")["verdict"]["reason"], "JUDGE-SAYS")
        self.assertIn("JUDGE-SAYS", c.get("/item/0").get_data(as_text=True))   # after the click, shown
        self.assertIsNone(self.client().post("/decide/1", json={"decision": "accept"}).get_json()["verdict"])

    def test_page_and_images(self):
        c = self.client()
        page = c.get("/item/2").get_data(as_text=True)
        self.assertIn("Accept the mask only if &lt;rule&gt;.", page)
        for url in ("/img/2/view.jpg", "/img/2/view.jpg?outline=0", "/img/2/zoom.jpg?x=0.5&y=0.5"):
            r = c.get(url)
            self.assertEqual((r.status_code, r.mimetype), (200, "image/jpeg"), url)


if __name__ == "__main__":
    unittest.main()
