"""The timeline's schema (ticket OPS-3 P3, D7, D17's layer 2). Plain unittest.

    python -m unittest tests.test_timeline_schema

coffeecv/timeline_schema.py is what the check service accepts on POST /timeline and what the reading script
(coffeecv/timelines.py) reads: known keys only, fixed sets of values, whole milliseconds in a range, IDs in nginx's
form, an example named in examples.json. These tests hold each refusal, and the page's sets to the schema's.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from coffeecv import timeline_schema as ts

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = ts.example_names(REPO / "webapp" / "static" / "examples" / "examples.json")
RID = "0123456789abcdef0123456789abcdef"
RID2 = "fedcba9876543210fedcba9876543210"


def timeline(**changes) -> dict:
    """A real-looking timeline: an own photo from Choose Photo, retried once while busy, then answered."""
    base = {"v": 1, "request_id": RID, "control": "choose_photo", "outcome": "answered",
            "pick_ms": 5200, "size_check_ms": 0, "to_preview_ms": 9100, "preview_drawn_ms": 40,
            "to_answer_ms": 11800, "answer_shown_ms": 12, "total_ms": 30400,
            "attempts": [{"request_id": RID2, "status": 503, "connect_ms": 80, "upload_ms": 900, "headers_ms": 20,
                          "wait_ms": 3000},
                         {"request_id": RID, "status": 200, "connect_ms": 0, "upload_ms": 950, "headers_ms": 8800}]}
    base.update(changes)
    return {k: v for k, v in base.items() if v is not None}


class Accepts(unittest.TestCase):
    def test_a_real_timeline(self):
        self.assertEqual(ts.check(timeline(), EXAMPLES), timeline())

    def test_an_example(self):
        t = timeline(control="example", example="brazil-grinder", download_ms=400, pick_ms=None)
        self.assertEqual(ts.check(t, EXAMPLES), t)

    def test_a_photo_refused_on_the_page_has_no_id(self):
        t = {"v": 1, "request_id": None, "control": "camera", "outcome": "refused_on_page", "size_check_ms": 1,
             "total_ms": 3000, "attempts": []}
        self.assertEqual(ts.check(t, EXAMPLES), t)

    def test_a_rerun_names_the_first(self):   # "Use full photo instead"
        t = timeline(rerun_of=RID2, pick_ms=None)
        self.assertEqual(ts.check(t, EXAMPLES), t)

    def test_the_probe_marker(self):
        self.assertEqual(ts.check(timeline(probe="verify"), EXAMPLES)["probe"], "verify")

    def test_every_control_and_outcome(self):
        for control in ts.CONTROLS:
            for outcome in ts.OUTCOMES:
                extra = {"example": "cappuccino"} if control == "example" else {}
                ts.check(timeline(control=control, outcome=outcome, **extra), EXAMPLES)


class Refuses(unittest.TestCase):
    def refused(self, data, why: str = ""):
        with self.assertRaises(ts.Invalid) as caught:
            ts.check(data, EXAMPLES)
        self.assertIn(why, str(caught.exception))
        return str(caught.exception)

    def test_not_an_object(self):
        for data in ([], "x", 3, None):
            with self.subTest(data=data):
                self.refused(data)

    def test_an_unknown_key_without_echoing_it(self):
        why = self.refused(timeline(**{"<script>": 1}), "unknown key")
        self.assertNotIn("script", why)
        attempt = {**timeline()["attempts"][0], "ip": "203.0.113.9"}
        self.assertNotIn("203.0.113.9", self.refused(timeline(attempts=[attempt]), "unknown key"))

    def test_a_missing_key(self):
        for key in ("v", "control", "outcome", "request_id", "attempts"):
            with self.subTest(key=key):
                data = timeline()
                del data[key]
                self.refused(data, key)

    def test_a_value_outside_its_set(self):
        self.refused(timeline(v=2), "v")
        self.refused(timeline(control="Choose Photo"), "control")
        self.refused(timeline(outcome="answered\n"), "outcome")
        self.refused(timeline(probe="yes"), "probe")
        self.assertNotIn("\n", self.refused(timeline(outcome="x\ny"), "outcome"))

    def test_an_example_only_from_the_list_and_only_with_its_control(self):
        self.refused(timeline(control="example", example="nope"), "example")
        self.refused(timeline(control="example"), "example")
        self.refused(timeline(example="cappuccino"), "example")

    def test_times_are_whole_milliseconds_from_0_to_10_minutes(self):
        for bad in (-1, 600_001, 1.5, True, "5", None, float("nan")):
            with self.subTest(bad=bad):
                self.refused({**timeline(), "total_ms": bad}, "total_ms")
                attempt = {**timeline()["attempts"][0], "wait_ms": bad}
                self.refused(timeline(attempts=[attempt]), "wait_ms")
        ts.check(timeline(total_ms=600_000, pick_ms=0), EXAMPLES)

    def test_ids_in_nginxs_form_only(self):
        for bad in ("abcd1234", RID.upper(), RID + "0", 7):
            with self.subTest(bad=bad):
                self.refused(timeline(request_id=bad), "request_id")
                self.refused(timeline(rerun_of=bad), "rerun_of")
                self.refused(timeline(attempts=[{"request_id": bad, "status": 200}]), "request_id")

    def test_attempts(self):
        self.refused(timeline(attempts={}), "attempts")
        self.refused(timeline(attempts=[{"status": 200}] * (ts.MAX_ATTEMPTS + 1)), "attempts")
        self.refused(timeline(attempts=[3]), "attempt")
        for bad in (99, 600, "200", True):
            with self.subTest(status=bad):
                self.refused(timeline(attempts=[{"request_id": RID, "status": bad}]), "status")
        ts.check(timeline(attempts=[{"request_id": None, "status": 0}]), EXAMPLES)   # a network failure: no status


class Parse(unittest.TestCase):
    def test_a_body_is_utf8_json_with_no_constants_and_no_duplicate_keys(self):
        self.assertEqual(ts.parse(json.dumps(timeline()).encode(), EXAMPLES), timeline())
        for body in (b"\xff", b"{", b'{"v": NaN}', b'{"v": Infinity}', b'{"v": 1, "v": 1}'):
            with self.subTest(body=body), self.assertRaises(ts.Invalid):
                ts.parse(body, EXAMPLES)

    def test_a_real_timeline_is_well_under_the_body_cap(self):
        fat = timeline(attempts=[{"request_id": RID, "status": 503, **{k: ts.MAX_MS for k in ts.ATTEMPT_TIMES}}]
                       * ts.MAX_ATTEMPTS, rerun_of=RID2, probe="verify",
                       **{k: ts.MAX_MS for k in ts.TIMES})
        ts.check(fat, EXAMPLES)
        self.assertLess(len(json.dumps(fat)), ts.MAX_BODY)


class ExampleList(unittest.TestCase):
    def test_the_example_list_is_read(self):
        self.assertEqual(EXAMPLES, {e["name"] for e in json.loads(
            (REPO / "webapp" / "static" / "examples" / "examples.json").read_text())["examples"]})

    def test_a_list_that_cannot_be_read_names_no_example(self):
        self.assertEqual(ts.example_names(REPO / "no-such-examples.json"), frozenset())


if __name__ == "__main__":
    unittest.main()
