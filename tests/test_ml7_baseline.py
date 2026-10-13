"""ML-7 P0's baseline instrument (analysis/ml7_gpu/baseline_stages.py): the live log's aggregates, the stage clock,
the 10% sum check and the stop rule. Plain unittest; no model is loaded.

    python -m unittest tests.test_ml7_baseline -v
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from coffeecv.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("baseline_stages",
                                               REPO_ROOT / "analysis" / "ml7_gpu" / "baseline_stages.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def _req(endpoint, latency, status=200, **fields):
    return {"msg": "request", "endpoint": endpoint, "status": status, "latency_ms": latency, **fields}


class Quantiles(unittest.TestCase):
    def test_median_and_p90_by_linear_interpolation(self):
        q = bs.quantiles(list(range(1, 11)))           # 1..10
        self.assertEqual(q, {"n": 10, "min": 1, "median": 5.5, "p90": 9.1, "max": 10})

    def test_no_values(self):
        self.assertEqual(bs.quantiles([]), {"n": 0, "min": None, "median": None, "p90": None, "max": None})


class SummariseRequests(unittest.TestCase):
    def test_per_endpoint_successful_requests_only(self):
        records = [
            _req("/classify", 20000, decode_ms=300, crop_detect_ms=9000, inference_ms=8000),
            _req("/classify", 22000, decode_ms=310, crop_detect_ms=9100, inference_ms=8100),
            _req("/classify", 500, status=400),                      # a refusal: not a timing sample
            _req("/crop", 9500, decode_ms=290),
            _req("/preview", 400),                                   # not a model endpoint
            {"msg": "decode failure", "endpoint": "/crop"},
        ]
        s = bs.summarise_requests(records)
        self.assertEqual(set(s), {"/classify", "/crop"})
        self.assertEqual(s["/classify"]["n"], 2)
        self.assertEqual(s["/classify"]["latency_ms"]["median"], 21000)
        self.assertEqual(s["/classify"]["inference_ms"]["median"], 8050)
        self.assertEqual(s["/crop"]["n"], 1)
        self.assertEqual(s["/crop"]["decode_ms"]["n"], 1)
        # /crop never logs these; their absence is a count of 0, not an error
        self.assertEqual(s["/crop"]["inference_ms"]["n"], 0)

    def test_only_the_named_model_and_what_else_the_log_held(self):
        records = [_req("/crop", 1000, model="new.pt", build={"git": "a"}),
                   _req("/crop", 9000, model="old.pt", build={"git": "b"}),
                   _req("/crop", 1200, model="new.pt", build={"git": "a"})]
        s = bs.summarise_requests(records, model="new.pt")
        self.assertEqual(s["/crop"]["latency_ms"]["max"], 1200)
        self.assertEqual(bs.models_seen(records), {"new.pt": 2, "old.pt": 1})

    def test_read_log_takes_rotated_and_gzipped_files_and_counts_bad_lines(self):
        with tempfile.TemporaryDirectory() as d:
            plain, packed = Path(d, "app.jsonl"), Path(d, "app.jsonl.2.gz")
            plain.write_text(json.dumps(_req("/crop", 1)) + "\nnot json\n")
            with gzip.open(packed, "wt") as f:
                f.write(json.dumps(_req("/crop", 2)) + "\n")
            bad = []
            got = list(bs.read_log([plain, packed], bad))
        self.assertEqual([r["latency_ms"] for r in got], [1, 2])
        self.assertEqual(len(bad), 1)


class StageClock(unittest.TestCase):
    def test_wrap_passes_calls_through_and_accumulates_per_stage(self):
        ticks = iter([0.0, 0.25, 1.0, 1.5])            # seconds: two calls of 250 and 500 ms
        clock = bs.StageClock(now=lambda: next(ticks))
        double = clock.wrap(lambda x, k=1: x * 2 * k, "decode")
        self.assertEqual(double(3), 6)
        self.assertEqual(double(3, k=2), 12)
        self.assertEqual(clock.take(), {"decode": 750.0})
        self.assertEqual(clock.take(), {})             # take() resets

    def test_a_raising_call_is_still_timed_and_reraised(self):
        ticks = iter([0.0, 0.1])
        clock = bs.StageClock(now=lambda: next(ticks))

        def boom():
            raise ValueError("bad photo")
        with self.assertRaises(ValueError):
            clock.wrap(boom, "decode")()
        self.assertAlmostEqual(clock.take()["decode"], 100.0)

    def test_add_sums_into_a_stage(self):
        clock = bs.StageClock()
        clock.add("encoder", 2000.0)
        clock.add("encoder", 100.0)
        self.assertEqual(clock.take(), {"encoder": 2100.0})


class StageCheck(unittest.TestCase):
    def test_within_ten_percent(self):
        c = bs.stage_check(1000.0, {"decode": 300.0, "encoder": 620.0})
        self.assertEqual(c["sum_ms"], 920.0)
        self.assertEqual(c["unexplained_ms"], 80.0)
        self.assertAlmostEqual(c["unexplained_frac"], 0.08)
        self.assertTrue(c["ok"])

    def test_over_ten_percent_fails(self):
        self.assertFalse(bs.stage_check(1000.0, {"decode": 850.0})["ok"])

    def test_stages_summing_past_the_whole_fail_too(self):
        # overlapping wrappers would count a stage twice
        self.assertFalse(bs.stage_check(1000.0, {"a": 700.0, "b": 450.0})["ok"])


class ModelShare(unittest.TestCase):
    @staticmethod
    def _photo(crop_whole, classify_whole, encoder, decoder, dino):
        seg = {"encoder": encoder, "decoder": decoder, "resize": 50.0}
        return {"crop": {"whole_ms": crop_whole, "stages": dict(seg)},
                "classify": {"whole_ms": classify_whole,
                             "stages": {**seg, "dino_b32": dino * 0.8, "dino_b8": dino * 0.2, "decode": 300.0}}}

    def test_both_encoder_passes_and_dino_count_as_model_time(self):
        # model time: 2 x (2000 + 150) + 6000 = 10300 of 3000 + 10000
        s = bs.model_share([self._photo(3000.0, 10000.0, 2000.0, 150.0, 6000.0)])
        self.assertAlmostEqual(s["photos"][0]["model_ms"], 10300.0)
        self.assertAlmostEqual(s["photos"][0]["share"], 10300.0 / 13000.0)
        self.assertFalse(s["stop"])

    def test_each_model_stage_has_its_own_share(self):
        s = bs.model_share([self._photo(3000.0, 10000.0, 2000.0, 150.0, 6000.0)])
        self.assertAlmostEqual(s["stage_shares"]["encoder"], 4000.0 / 13000.0)
        self.assertAlmostEqual(s["stage_shares"]["decoder"], 300.0 / 13000.0)
        self.assertAlmostEqual(s["stage_shares"]["dinov3"], 6000.0 / 13000.0)

    def test_stop_when_the_models_are_under_half_of_the_photo_time(self):
        s = bs.model_share([self._photo(6000.0, 16000.0, 2000.0, 150.0, 6000.0),
                            self._photo(6000.0, 16000.0, 2000.0, 150.0, 6000.0)])
        self.assertAlmostEqual(s["median_share"], 10300.0 / 22000.0)
        self.assertTrue(s["stop"])


class CompareLive(unittest.TestCase):
    def test_fixture_median_against_the_live_range(self):
        summary = {"crop": {"whole_ms": bs.quantiles([5000.0])}, "classify": {"whole_ms": bs.quantiles([30000.0])}}
        live = {"endpoints": bs.summarise_requests([_req("/crop", 4000), _req("/crop", 7000),
                                                    _req("/classify", 20000), _req("/classify", 26000)])}
        c = bs.compare_live(summary, live)
        self.assertTrue(c["crop"]["within_range"])
        self.assertFalse(c["classify"]["within_range"])
        self.assertEqual(c["classify"]["live"]["max"], 26000)

    def test_an_endpoint_with_no_live_requests(self):
        summary = {"crop": {"whole_ms": bs.quantiles([5000.0])}, "classify": {"whole_ms": bs.quantiles([1.0])}}
        c = bs.compare_live(summary, {"endpoints": bs.summarise_requests([_req("/crop", 4000)])})
        self.assertIsNone(c["classify"]["within_range"])


class ReadFixtures(unittest.TestCase):
    def test_every_role_found_flat_by_basename_and_checked(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d, "a.jpg"), Path(d, "b.JPG")
            a.write_bytes(b"one")
            b.write_bytes(b"two")
            lst = Path(d, "fixtures.txt")
            lst.write_text("# comment\n"
                           f"smoke {hashlib.sha256(b'one').hexdigest()} dataset/x/a.jpg\n\n"
                           f"compare {hashlib.sha256(b'two').hexdigest()} dataset/y/b.JPG\n")
            self.assertEqual(bs.read_fixtures(lst, Path(d)), [a, b])
            b.write_bytes(b"changed")
            with self.assertRaises(SystemExit):
                bs.read_fixtures(lst, Path(d))


if __name__ == "__main__":
    unittest.main()
