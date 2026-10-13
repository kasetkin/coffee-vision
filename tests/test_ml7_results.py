"""ML-7 P4: the comparison run's records summarised (analysis/ml7_gpu/p4_results.py): across rounds, per cell and
graph, and per cell's photo replay (plan P4). Plain unittest, synthetic records.

    python -m unittest tests.test_ml7_results -v
"""
from __future__ import annotations

import importlib.util
import sys
import unittest

from coffeecv.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("p4_results", REPO_ROOT / "analysis" / "ml7_gpu" / "p4_results.py")
pr = importlib.util.module_from_spec(_spec)
sys.modules["p4_results"] = pr
_spec.loader.exec_module(pr)


def _graph(median, p90=None, cold=5.0, proof=None, nonfinite=0, pl1=False, per_photo=1):
    r = {"timing": {"cold_s": cold, "warm": {"n": 20, "median": median, "p90": p90 or median * 1.1}},
         "cpu_over_wall": 3.9, "peak_rss_mb": 600, "mem_available_drop_mb": 400.0, "nonfinite_calls": nonfinite,
         "counters": {"cpu_mhz_median": 1400, "gpu_mhz_median": 0, "pl1_throttled": pl1},
         "calls_per_photo": per_photo}
    if proof is not None:
        r["gpu_proof"] = {"ok": proof}
    return r


class Graphs(unittest.TestCase):
    def test_rounds_are_summarised_by_their_median_and_spread(self):
        s = pr.summarise_graph({1: _graph(2.0), 2: _graph(2.2), 3: _graph(1.8)})
        self.assertEqual(s["warm_median_s"], 2.0)
        self.assertAlmostEqual(s["round_spread"], 0.2)          # (2.2 - 1.8) / 2.0
        self.assertEqual(s["rounds"], 3)

    def test_per_photo_time_scales_by_the_calls_a_photo_needs(self):
        s = pr.summarise_graph({1: _graph(2.0, per_photo=5)})
        self.assertEqual(s["per_photo_s"], 10.0)

    def test_a_gpu_cell_counts_only_if_every_round_proved_the_gpu(self):
        self.assertTrue(pr.summarise_graph({1: _graph(1.0, proof=True), 2: _graph(1.0, proof=True)})["gpu_proven"])
        self.assertFalse(pr.summarise_graph({1: _graph(1.0, proof=True), 2: _graph(1.0, proof=False)})["gpu_proven"])
        self.assertIsNone(pr.summarise_graph({1: _graph(1.0)})["gpu_proven"])

    def test_nan_and_throttling_in_any_round_are_kept(self):
        s = pr.summarise_graph({1: _graph(1.0), 2: _graph(1.0, nonfinite=2, pl1=True)})
        self.assertEqual(s["nonfinite_calls"], 2)
        self.assertTrue(s["pl1_throttled"])

    def test_a_missing_round_is_a_failed_one(self):
        s = pr.summarise_graph({1: _graph(1.0), 2: None})
        self.assertEqual(s["failed_rounds"], [2])
        self.assertEqual(s["warm_median_s"], 1.0)


class Replay(unittest.TestCase):
    @staticmethod
    def _replay(median, first=20.0, proof=None):
        r = {"photo": {"n": 25, "median": median, "p90": median * 1.2}, "first_answer_s": first,
             "process_to_first_answer_s": first + 2, "peak_rss_mb": 2000, "mem_available_drop_mb": 1500.0,
             "nonfinite_photos": 0, "batch": 8, "counters": {"pl1_throttled": False}}
        if proof is not None:
            r["gpu_proof"] = {"ok": proof}
        return r

    def test_the_median_photo_and_its_spread_between_rounds(self):
        s = pr.summarise_replay({1: self._replay(20.0), 2: self._replay(22.0), 3: self._replay(21.0)})
        self.assertEqual(s["photo_median_s"], 21.0)
        self.assertAlmostEqual(s["round_spread"], 2.0 / 21.0)
        self.assertEqual(s["batch"], [8])


if __name__ == "__main__":
    unittest.main()
