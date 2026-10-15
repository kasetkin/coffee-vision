"""ML-7 P7: the fp16 answer check's comparison (analysis/ml7_gpu/p7_answers.py): one photo's answers under two
readout sets, and the summary over photos (plan P7). Plain unittest, numpy only, fake readouts and answers.

    python -m unittest tests.test_ml7_answers -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from coffeecv.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("p7_answers", REPO_ROOT / "analysis" / "ml7_gpu" / "p7_answers.py")
pa = importlib.util.module_from_spec(_spec)
sys.modules["p7_answers"] = pa
_spec.loader.exec_module(pa)


def _ans(probs, score=None, threshold=0.5):
    """What parity._answers returns: probs, top1 and, with a guard, its score, threshold and verdict."""
    probs = np.asarray(probs, dtype=np.float64)
    a = {"probs": probs, "top1": int(probs.argmax())}
    if score is not None:
        a.update(method="linear_probe", score=score, threshold=threshold, refused=score > threshold)
    return a


def _readouts(seed=0, n=40, d=16):
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


def _row(path, role, measures: dict, sets=None):
    return {"path": path, "role": role, "sets": sets or {}, "pairs": measures}


def _m(top1_same=True, verdict_same=True, cos=0.9999, ref_margin=0.5, got_margin=0.5, top2_same=True,
       distance=-0.3, change=1e-4, prob=1e-3, nonfinite=False):
    return {"least_cosine": cos, "nonfinite": nonfinite, "top1_same": top1_same, "top2_same": top2_same,
            "prob_max_change": prob, "ref_margin": ref_margin, "got_margin": got_margin,
            "ref_score": 0.5 + distance, "distance_from_threshold": distance, "score_change": change,
            "verdict_same": verdict_same}


class AnswerView(unittest.TestCase):
    def test_top1_top2_and_margin_come_from_the_mean_probabilities(self):
        v = pa.answer_view(_ans([0.1, 0.6, 0.3]))
        self.assertEqual((v["top1"], v["top2"]), (1, 2))
        self.assertAlmostEqual(v["margin"], 0.3)
        self.assertNotIn("score", v)

    def test_the_guard_is_kept_when_there_is_one(self):
        v = pa.answer_view(_ans([0.5, 0.5], score=0.7))
        self.assertEqual(v["top1"], 0)                           # a tie goes to the first, as argmax does
        self.assertEqual((v["score"], v["threshold"], v["refused"]), (0.7, 0.5, True))


class PairMeasures(unittest.TestCase):
    def test_identical_readouts_change_nothing(self):
        r = _readouts()
        m = pa.pair_measures(r, r.copy(), _ans([0.2, 0.8], 0.3), _ans([0.2, 0.8], 0.3))
        self.assertAlmostEqual(m["least_cosine"], 1.0, places=12)
        self.assertTrue(m["top1_same"] and m["top2_same"] and m["verdict_same"])
        self.assertEqual((m["prob_max_change"], m["score_change"]), (0.0, 0.0))
        self.assertAlmostEqual(m["distance_from_threshold"], -0.2)
        self.assertFalse(m["nonfinite"])

    def test_a_flip_of_top1_and_of_the_verdict_is_seen(self):
        r = _readouts()
        m = pa.pair_measures(r, r, _ans([0.45, 0.55], 0.49), _ans([0.55, 0.45], 0.51))
        self.assertFalse(m["top1_same"])
        self.assertFalse(m["verdict_same"])
        self.assertAlmostEqual(m["score_change"], 0.02)
        self.assertAlmostEqual(m["prob_max_change"], 0.1)
        self.assertAlmostEqual(m["ref_margin"], 0.1)

    def test_second_place_changing_is_an_order_change_not_a_flip(self):
        r = _readouts()
        m = pa.pair_measures(r, r, _ans([0.5, 0.3, 0.2]), _ans([0.5, 0.2, 0.3]))
        self.assertTrue(m["top1_same"])
        self.assertFalse(m["top2_same"])
        self.assertNotIn("verdict_same", m)                     # no guard, nothing to compare

    def test_least_cosine_is_the_worst_patch(self):
        ref = _readouts()
        got = ref.copy()
        got[7] = -got[7]
        self.assertAlmostEqual(pa.pair_measures(ref, got, _ans([1, 0]), _ans([1, 0]))["least_cosine"], -1.0)

    def test_nan_in_the_readouts_is_flagged(self):
        got = _readouts()
        got[3, 2] = np.nan
        self.assertTrue(pa.pair_measures(_readouts(), got, _ans([1, 0]), _ans([1, 0]))["nonfinite"])


class Pairs(unittest.TestCase):
    def test_each_set_against_the_base_and_against_torch(self):
        got = pa.pairs_for(["litert_gpu", "litert_gpu_fp16", "torch"], "litert_gpu")
        self.assertEqual(got, [("litert_gpu", "litert_gpu_fp16"), ("torch", "litert_gpu"),
                               ("torch", "litert_gpu_fp16")])

    def test_without_torch_only_the_base(self):
        self.assertEqual(pa.pairs_for(["litert_gpu", "litert_gpu_fp16", "litert_gpu_split"], "litert_gpu"),
                         [("litert_gpu", "litert_gpu_fp16"), ("litert_gpu", "litert_gpu_split")])

    def test_a_missing_base_pairs_nothing_but_torch(self):
        self.assertEqual(pa.pairs_for(["litert_gpu_fp16", "torch"], "litert_gpu"), [("torch", "litert_gpu_fp16")])

    def test_the_key_names_the_set_tested_then_its_reference(self):
        self.assertEqual(pa.pair_key("litert_gpu", "litert_gpu_fp16"), "litert_gpu_fp16_vs_litert_gpu")


class SummarisePair(unittest.TestCase):
    K = "b_vs_a"

    def test_flips_are_counted_by_role_and_listed(self):
        rows = [_row("n1", "neg_ood", {self.K: _m(verdict_same=False)}),
                _row("n2", "neg_ood", {self.K: _m()}),
                _row("p1", "segpos", {self.K: _m(top1_same=False, top2_same=False)}),
                _row("p2", "segpos", {})]                       # not in this pair
        s = pa.summarise_pair(rows, self.K)
        self.assertEqual(s["photos"], 3)
        self.assertEqual((s["top1_flips"], s["verdict_flips"]), (["p1"], ["n1"]))
        self.assertEqual(s["by_role"]["neg_ood"], {"photos": 2, "top1_flips": 0, "verdict_flips": 1,
                                                    "top2_order_changes": 0})
        self.assertEqual(s["by_role"]["segpos"]["top1_flips"], 1)
        self.assertNotIn("p2", [n["photo"] for n in s["near_threshold"] + s["smallest_margin"]])

    def test_least_cosine_distribution_leaves_nan_out_and_counts_it(self):
        rows = [_row(f"x{i}", "neg_ood", {self.K: _m(cos=c)}) for i, c in enumerate([0.99, 0.999, 0.9999])]
        rows.append(_row("bad", "neg_ood", {self.K: _m(cos=float("nan"), nonfinite=True)}))
        s = pa.summarise_pair(rows, self.K)
        self.assertEqual(s["least_cosine"]["min"], 0.99)
        self.assertEqual(s["least_cosine"]["p50"], 0.999)
        self.assertEqual(s["nonfinite"], ["bad"])

    def test_nearest_the_threshold_first_with_how_far_the_score_moved(self):
        rows = [_row("far", "neg_ood", {self.K: _m(distance=0.4, change=0.001)}),
                _row("near", "neg_ood", {self.K: _m(distance=-0.01, change=0.002)}),
                _row("mid", "segpos", {self.K: _m(distance=0.1, change=-0.05)})]
        near = pa.summarise_pair(rows, self.K, k=2)["near_threshold"]
        self.assertEqual([n["photo"] for n in near], ["near", "mid"])
        self.assertAlmostEqual(near[0]["moved_of_distance"], 0.2)   # 0.002 of the 0.01 to the threshold
        self.assertAlmostEqual(near[1]["moved_of_distance"], 0.5)

    def test_smallest_margin_first_with_whether_the_order_held(self):
        rows = [_row("wide", "segpos", {self.K: _m(ref_margin=0.6)}),
                _row("tight", "segpos", {self.K: _m(ref_margin=0.01, got_margin=-0.002, top1_same=False,
                                                    top2_same=False)}),
                _row("mid", "segpos", {self.K: _m(ref_margin=0.2)})]
        small = pa.summarise_pair(rows, self.K, k=2)["smallest_margin"]
        self.assertEqual([s["photo"] for s in small], ["tight", "mid"])
        self.assertFalse(small[0]["top1_same"])
        self.assertTrue(small[1]["top2_same"])

    def test_without_a_guard_no_verdict_measures(self):
        m = _m()
        for key in ("ref_score", "distance_from_threshold", "score_change", "verdict_same"):
            del m[key]
        s = pa.summarise_pair([_row("x", "segpos", {self.K: m})], self.K)
        self.assertEqual((s["verdict_flips"], s["near_threshold"], s["score_change_max_abs"]), ([], [], None))

    def test_the_summary_is_plain_json(self):
        rows = [_row("x", "neg_ood", {self.K: {**_m(), "top1_same": np.bool_(True), "least_cosine": np.float64(1)}})]
        json.dumps(pa.summarise_pair(rows, self.K))


class SummariseSets(unittest.TestCase):
    def test_refusals_per_set_by_role(self):
        rows = [_row("n1", "neg_ood", {}, {"a": {"refused": True}, "b": {"refused": False}}),
                _row("n2", "neg_ood", {}, {"a": {"refused": True}}),
                _row("p1", "segpos", {}, {"a": {"refused": False}, "b": {"refused": False}})]
        s = pa.summarise_sets(rows)
        self.assertEqual(s["a"]["neg_ood"], {"photos": 2, "refused": 2})
        self.assertEqual(s["b"]["neg_ood"], {"photos": 1, "refused": 0})
        self.assertEqual(s["b"]["segpos"], {"photos": 1, "refused": 0})


class Record(unittest.TestCase):
    def test_a_photo_keeps_its_flags_and_cosine_whole_and_drops_what_the_sets_give(self):
        row = _row("x", "neg_ood", {"b_vs_a": _m(cos=0.999876543219, prob=1.23456789e-4)},
                   {"a": {"top1": "Brazil", "top2": "Peru", "margin": 0.123456789, "score": 0.98765432101,
                          "refused": True}})
        r = pa.photo_record(row)
        m = r["pairs"]["b_vs_a"]
        self.assertEqual(m["least_cosine"], 0.999876543219)
        self.assertEqual(m["prob_max_change"], 1.23457e-4)
        self.assertNotIn("ref_margin", m)
        self.assertTrue(m["verdict_same"])
        self.assertEqual(r["sets"]["a"]["margin"], 0.123457)
        self.assertEqual(r["sets"]["a"]["score"], 0.98765432101)

    def test_the_json_has_one_line_per_photo_and_reads_back(self):
        out = {"photos": 2, "per_photo": [{"path": "a", "pairs": {}}, {"path": "b", "pairs": {}}]}
        text = pa.dumps(out)
        self.assertEqual(json.loads(text), out)
        self.assertEqual(sum(line.lstrip().startswith('{"path"') for line in text.splitlines()), 2)
        self.assertEqual(json.loads(pa.dumps({"per_photo": []})), {"per_photo": []})


class AnswerChange(unittest.TestCase):
    def test_the_reference_top1s_lead_is_followed_through_a_flip(self):
        c = pa.answer_change(_ans([0.5, 0.3, 0.2], 0.2), _ans([0.4, 0.45, 0.15], 0.6))
        self.assertFalse(c["top1_same"] or c["verdict_same"])
        self.assertAlmostEqual(c["ref_margin"], 0.2)
        self.assertAlmostEqual(c["got_margin"], 0.05)            # the got set's own top-1 (class 1) over class 0
        self.assertAlmostEqual(c["margin_change"], -0.25)        # class 0's lead went from +0.2 to -0.05
        self.assertAlmostEqual(c["prob_max_change"], 0.15)

    def test_without_a_flip_the_margin_change_is_the_margins_difference(self):
        c = pa.answer_change(_ans([0.6, 0.3, 0.1]), _ans([0.55, 0.35, 0.1]))
        self.assertTrue(c["top1_same"])
        self.assertAlmostEqual(c["margin_change"], c["got_margin"] - c["ref_margin"])
        self.assertNotIn("verdict_same", c)

    def test_pair_measures_carries_the_answer_change(self):
        r = _readouts()
        ref, got = _ans([0.5, 0.3, 0.2], 0.2), _ans([0.4, 0.45, 0.15], 0.6)
        self.assertEqual({k: v for k, v in pa.pair_measures(r, r, ref, got).items()
                          if k not in ("least_cosine", "nonfinite")}, pa.answer_change(ref, got))


class Wilson(unittest.TestCase):
    def test_no_events_reaches_zero_and_the_rule_of_three_region(self):
        lo, hi = pa.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1.959964 ** 2 / (10 + 1.959964 ** 2), places=6)

    def test_half_is_symmetric(self):
        lo, hi = pa.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.236593, places=5)
        self.assertAlmostEqual(lo + hi, 1.0)

    def test_one_in_157(self):
        lo, hi = pa.wilson(1, 157)
        self.assertTrue(0.001 < lo < 0.0013 and 0.034 < hi < 0.036)

    def test_nothing_counted_is_no_interval(self):
        self.assertIsNone(pa.wilson(0, 0))


def _fake_runs(probs_by_run: dict, scores: dict | None = None) -> dict:
    """{run: {photo: answer}} from {run: {photo: probs}}, a guard score per (run, photo) if given."""
    scores = scores or {}
    return {run: {p: _ans(v, scores.get((run, p))) for p, v in by_photo.items()}
            for run, by_photo in probs_by_run.items()}


class SeedPairs(unittest.TestCase):
    def test_every_pair_once_in_the_given_order(self):
        self.assertEqual(pa.seed_pairs(["s0", "s1", "s2"]), [("s0", "s1"), ("s0", "s2"), ("s1", "s2")])

    def test_one_run_has_no_pair(self):
        self.assertEqual(pa.seed_pairs(["s0"]), [])


class SummariseChanges(unittest.TestCase):
    def test_counts_rate_interval_and_distributions(self):
        ms = [pa.answer_change(_ans([0.6, 0.4], 0.2), _ans([0.6, 0.4], 0.2)),
              pa.answer_change(_ans([0.6, 0.4], 0.2), _ans([0.4, 0.6], 0.7)),
              pa.answer_change(_ans([0.7, 0.3], 0.2), _ans([0.8, 0.2], 0.2)),
              pa.answer_change(_ans([0.9, 0.1], 0.2), _ans([0.9, 0.1], 0.2))]
        s = pa.summarise_changes(ms)
        self.assertEqual((s["photos"], s["top1_flips"], s["verdict_flips"]), (4, 1, 1))
        self.assertEqual(s["top1_flip_rate"], 0.25)
        self.assertEqual(s["top1_flip_wilson95"], pa.wilson(1, 4))
        self.assertAlmostEqual(s["prob_max_change"]["max"], 0.2)
        self.assertAlmostEqual(s["prob_max_change"]["p50"], 0.05)
        self.assertIn("p95", s["prob_max_change"])
        self.assertAlmostEqual(s["margin_change_abs"]["max"], 0.4)  # +0.2 lead to -0.2
        json.dumps(s)

    def test_nothing_is_an_empty_summary(self):
        s = pa.summarise_changes([])
        self.assertEqual((s["photos"], s["top1_flips"], s["top1_flip_rate"], s["prob_max_change"]), (0, 0, None, None))


class SamplingSpread(unittest.TestCase):
    RUNS = {"s0": {"a": [0.6, 0.4], "b": [0.9, 0.1], "c": [0.52, 0.48]},
            "s1": {"a": [0.6, 0.4], "b": [0.85, 0.15], "c": [0.47, 0.53]},
            "s2": {"a": [0.65, 0.35], "b": [0.9, 0.1], "c": [0.51, 0.49]}}

    def test_every_pair_and_pooled(self):
        sp = pa.sampling_spread(_fake_runs(self.RUNS), ["a", "b", "c"])
        self.assertEqual(list(sp["pairs"]), ["s1_vs_s0", "s2_vs_s0", "s2_vs_s1"])
        self.assertEqual([sp["pairs"][k]["top1_flips"] for k in sp["pairs"]], [1, 0, 1])
        self.assertEqual((sp["pooled"]["photos"], sp["pooled"]["top1_flips"]), (9, 2))
        self.assertEqual(sp["photos_top1_not_unanimous"], 1)        # only c
        self.assertFalse(sp["per_photo"]["c"]["s1_vs_s0"]["top1_same"])
        self.assertAlmostEqual(sp["per_photo"]["b"]["s1_vs_s0"]["prob_max_change"], 0.05)

    def test_only_the_photos_given(self):
        sp = pa.sampling_spread(_fake_runs(self.RUNS), ["a", "b"])
        self.assertEqual((sp["pooled"]["photos"], sp["pooled"]["top1_flips"]), (6, 0))
        self.assertNotIn("c", sp["per_photo"])


class PrecisionDrift(unittest.TestCase):
    def test_per_seed_and_pooled(self):
        fp32 = _fake_runs({"s0": {"a": [0.6, 0.4], "c": [0.52, 0.48]}, "s1": {"a": [0.6, 0.4], "c": [0.47, 0.53]}})
        fp16 = _fake_runs({"s0": {"a": [0.59, 0.41], "c": [0.49, 0.51]}, "s1": {"a": [0.6, 0.4], "c": [0.48, 0.52]}})
        d = pa.precision_drift(fp32, fp16, ["a", "c"])
        self.assertEqual(list(d["per_seed"]), ["s0", "s1"])
        self.assertEqual((d["per_seed"]["s0"]["top1_flips"], d["per_seed"]["s1"]["top1_flips"]), (1, 0))
        self.assertEqual((d["pooled"]["photos"], d["pooled"]["top1_flips"]), (4, 1))
        self.assertFalse(d["per_photo"]["c"]["s0"]["top1_same"])

    def test_a_run_without_the_fp16_set_is_left_out(self):
        fp32 = _fake_runs({"s0": {"a": [0.6, 0.4]}, "s1": {"a": [0.6, 0.4]}})
        d = pa.precision_drift(fp32, _fake_runs({"s0": {"a": [0.6, 0.4]}}), ["a"])
        self.assertEqual(list(d["per_seed"]), ["s0"])


class Comparison(unittest.TestCase):
    def test_fp16s_flips_beside_samplings_and_whether_sampling_flips_them_too(self):
        fp32 = _fake_runs({"s0": {"a": [0.6, 0.4], "c": [0.52, 0.48], "d": [0.55, 0.45]},
                           "s1": {"a": [0.6, 0.4], "c": [0.47, 0.53], "d": [0.56, 0.44]}})
        fp16 = _fake_runs({"s0": {"a": [0.6, 0.4], "c": [0.49, 0.51], "d": [0.49, 0.51]},
                           "s1": {"a": [0.6, 0.4], "c": [0.47, 0.53], "d": [0.55, 0.45]}})
        photos = ["a", "c", "d"]
        sp, d = pa.sampling_spread(fp32, photos), pa.precision_drift(fp32, fp16, photos)
        c = pa.comparison(sp, d, fp32, fp16, photos, ids=["Brazil", "Peru"])
        self.assertEqual((c["sampling"]["top1_flips"], c["sampling"]["photos"]), (1, 3))
        self.assertEqual((c["fp16"]["top1_flips"], c["fp16"]["photos"]), (2, 6))
        self.assertEqual(c["fp16"]["top1_flip_wilson95"], pa.wilson(2, 6))
        self.assertEqual(set(c["prob_max_change"]), {"sampling", "fp16"})
        flipped = {f["photo"]: f for f in c["fp16_flipped"]}
        self.assertEqual(list(flipped), ["c", "d"])
        self.assertTrue(flipped["c"]["sampling_flips"])          # c's top-1 also changes between seeds
        self.assertEqual(flipped["c"]["sampling_pairs_flipped"], [1, 1])
        self.assertFalse(flipped["d"]["sampling_flips"])
        self.assertEqual(flipped["d"]["fp32_top1"], {"s0": "Brazil", "s1": "Brazil"})
        self.assertEqual(flipped["d"]["fp16_flipped_in"], ["s0"])
        self.assertAlmostEqual(flipped["d"]["fp32_margin"]["s0"], 0.1)
        self.assertAlmostEqual(flipped["d"]["fp16_margin_change"]["s0"], -0.12)
        self.assertEqual(c["fp16_flipped_photos_sampling_also_flips"], [1, 2])
        json.dumps(c)


class Files(unittest.TestCase):
    def test_the_last_row_per_photo_wins(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.jsonl"
            p.write_text('{"sha256": "a", "error": "E"}\n{"sha256": "b", "s": 1}\n{"sha256": "a", "s": 2}\n'
                         '{"sha256": "c", "s"')                  # a line cut off by a kill is skipped
            rows = pa.last_rows(p)
        self.assertEqual(rows, {"a": {"sha256": "a", "s": 2}, "b": {"sha256": "b", "s": 1}})

    def test_a_missing_file_is_no_rows(self):
        self.assertEqual(pa.last_rows(Path("/nonexistent/x.jsonl")), {})

    def test_an_array_round_trips_with_the_hash_of_its_bytes(self):
        a = _readouts()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.npy"
            sha = pa.save_array(p, a)
            got, sha2 = pa.load_array(p)
            self.assertEqual(sorted(x.name for x in Path(d).iterdir()), ["a.npy"])   # no temporary left
        np.testing.assert_array_equal(got, a)
        self.assertEqual(sha, sha2)

    def test_the_photo_list_skips_comments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "l.txt"
            p.write_text("# head\n\nneg_ood abc dataset/x y.JPG\nsegpos def dataset/z.jpg\n")
            self.assertEqual(pa.read_list(p), [("neg_ood", "abc", "dataset/x y.JPG"),
                                               ("segpos", "def", "dataset/z.jpg")])


if __name__ == "__main__":
    unittest.main()
