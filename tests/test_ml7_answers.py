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
