"""ML-37 P7: analysis/ml37/p7_compare.py, D13's comparison and its photo bootstrap (new code).

On the archived live family, exp289-291, read from experiments/ (tracked in git): against themselves every change
and every bootstrap bound is exactly 0; a planted per-photo loss moves the observed change as an independent F1
(sklearn's) says, and every resample with it; photos one arm lacks are drawn apart. Also D13's OOD floor, read from
ML-29 P11's ood_eval JSON.

    python -m unittest tests.test_p7_compare
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest

import numpy as np
from sklearn.metrics import f1_score

from coffeecv.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("p7_compare", REPO_ROOT / "analysis" / "ml37" / "p7_compare.py")
p7 = importlib.util.module_from_spec(_spec)
sys.modules["p7_compare"] = p7      # dataclasses look their module up
_spec.loader.exec_module(p7)

BASE = [289, 290, 291]
SEEDS = [42, 123, 7]
N_BOOT = 400


def _test_preds() -> list:
    return [p7.load_preds(e, "test") for e in BASE]


def _plant(pr, true_class: str, to_class: str, n_photos: int):
    """A copy of pr with every patch of the first n_photos photos of true_class that pr gets fully right predicted
    as to_class: a whole-photo loss."""
    out = copy.deepcopy(pr)
    c, t = pr.classes.index(true_class), pr.classes.index(to_class)
    keys = list(dict.fromkeys(k for k, y in zip(pr.photos, pr.y) if y == c))
    right = [k for k in keys if all(pr.p[i] == c for i, kk in enumerate(pr.photos) if kk == k)]
    chosen = set(right[:n_photos])
    assert len(chosen) == n_photos, "not enough fully right photos to plant on"
    for i, k in enumerate(pr.photos):
        if k in chosen:
            out.p[i] = t
    return out


class SelfComparison(unittest.TestCase):
    """exp289-291 against themselves: the change and the bound are 0, exactly."""

    @classmethod
    def setUpClass(cls):
        cls.preds = _test_preds()
        cls.pairs = [p7.make_pair(s, pr, pr) for s, pr in zip(SEEDS, cls.preds)]

    def test_every_photo_is_shared(self):
        for pr in self.pairs:
            self.assertTrue(all(len(b) == 0 and len(n) == 0 for _, b, n in pr.strata))
            self.assertEqual(sum(len(s) for s, _, _ in pr.strata), pr.n_photos)

    def test_change_and_bound_are_exactly_zero(self):
        s = p7.summarize(self.pairs, n_boot=N_BOOT)
        self.assertEqual(s.mean, 0.0)
        self.assertEqual((s.lo, s.hi, s.se), (0.0, 0.0, 0.0))
        self.assertTrue(np.all(s.class_mean == 0.0))
        self.assertTrue(np.all(s.class_lo == 0.0) and np.all(s.class_hi == 0.0))

    def test_recomputed_f1_is_the_record(self):
        for e, pr in zip(BASE, self.preds):
            m = json.loads((p7.exp_dir(e) / "metrics.json").read_text())["splits"]["test"]
            self.assertAlmostEqual(p7.class_f1(pr.y, pr.p, len(pr.classes)).mean(), m["macro_f1"], places=9)

    def test_the_report_says_met_with_the_probe_pending(self):
        text = p7.run(BASE, BASE, "t", None, N_BOOT, p7.RNG_SEED)
        self.assertIn("lower bound +0.0000 > -0.010: MET", text)
        self.assertIn("part 3, OOD probe: PENDING", text)
        self.assertIn("in-distribution parts MET", text)


class PlantedLoss(unittest.TestCase):
    """Two Kenya photos of seed 42 turned wholly to Ethiopia in the candidate; seeds 123 and 7 unchanged."""

    @classmethod
    def setUpClass(cls):
        cls.preds = _test_preds()
        cls.planted = _plant(cls.preds[0], "Kenya", "Ethiopia", 2)
        news = [cls.planted] + cls.preds[1:]
        cls.pairs = [p7.make_pair(s, b, n) for s, b, n in zip(SEEDS, cls.preds, news)]

    def test_observed_change_is_sklearns(self):
        pr, pl = self.preds[0], self.planted
        labels = list(range(len(pr.classes)))
        want = (f1_score(pl.y, pl.p, labels=labels, average=None)
                - f1_score(pr.y, pr.p, labels=labels, average=None))
        obs = p7.observed(self.pairs)
        np.testing.assert_allclose(obs[0], want, atol=1e-12)
        self.assertTrue(np.all(obs[1:] == 0.0))
        s = p7.summarize(self.pairs, n_boot=N_BOOT)
        self.assertAlmostEqual(s.mean, want.mean() / 3, places=12)

    def test_only_the_two_classes_move_and_both_down(self):
        s = p7.summarize(self.pairs, n_boot=N_BOOT)
        classes = self.preds[0].classes
        moved = {classes[c] for c in np.flatnonzero(s.class_mean != 0)}
        self.assertEqual(moved, {"Kenya", "Ethiopia"})
        self.assertLess(s.class_mean[classes.index("Kenya")], s.class_mean[classes.index("Ethiopia")])
        self.assertLess(s.class_mean[classes.index("Ethiopia")], 0)

    def test_every_resample_loses_and_the_bound_sits_below_the_change(self):
        boot = p7.bootstrap(self.pairs, n_boot=N_BOOT)
        self.assertLessEqual(boot.mean(axis=2).max(), 0.0)
        s = p7.summarize(self.pairs, n_boot=N_BOOT)
        self.assertLess(s.lo, s.mean)
        self.assertLess(s.mean, 0)
        self.assertLessEqual(s.hi, 0.0)

    def test_a_larger_loss_misses_the_class_margin(self):
        news = [_plant(pr, "Kenya", "Ethiopia", 4) for pr in self.preds]
        pairs = [p7.make_pair(s, b, n) for s, b, n in zip(SEEDS, self.preds, news)]
        s = p7.summarize(pairs, n_boot=N_BOOT)
        self.assertLess(s.class_mean[self.preds[0].classes.index("Kenya")], p7.CLASS_MARGIN)

    def test_same_rng_seed_same_bound(self):
        a = p7.summarize(self.pairs, n_boot=N_BOOT, rng_seed=5)
        b = p7.summarize(self.pairs, n_boot=N_BOOT, rng_seed=5)
        self.assertEqual((a.lo, a.hi), (b.lo, b.hi))


class UnsharedPhotos(unittest.TestCase):
    """A photo one arm has and the other lacks is drawn on that arm alone."""

    def _preds(self, photos: list[str], ys: list[int], ps: list[int]):
        classes = ["A", "B"]
        return p7.Preds(classes=classes, y=np.array(ys), p=np.array(ps), prob=np.eye(2)[ps],
                        photos=[(classes[y], "001", ph) for y, ph in zip(ys, photos)])

    def test_strata_and_draws(self):
        base = self._preds(["a1", "a1", "a2", "b1"], [0, 0, 0, 1], [0, 0, 1, 1])
        new = self._preds(["a1", "a1", "a3", "b1"], [0, 0, 0, 1], [0, 0, 0, 1])
        pr = p7.make_pair(42, base, new)
        shared, base_only, new_only = pr.strata[0]
        self.assertEqual(len(shared), 1)
        self.assertEqual(len(base_only), 1)
        self.assertEqual(len(new_only), 1)
        self.assertEqual([len(x) for x in pr.strata[1]], [1, 0, 0])
        boot = p7.bootstrap([pr], n_boot=200)
        self.assertGreater(boot.std(), 0)               # a2 and a3 drawn apart: the change varies
        self.assertGreaterEqual(boot.min(), 0)          # the candidate never does worse here

    def test_disjoint_identical_arms_are_not_exactly_zero(self):
        base = self._preds(["a1", "a2", "b1"], [0, 0, 1], [0, 1, 1])
        new = self._preds(["a3", "a4", "b2"], [0, 0, 1], [0, 1, 1])
        boot = p7.bootstrap([p7.make_pair(42, base, new)], n_boot=200)
        self.assertGreater(boot.std(), 0)


class OodFloor(unittest.TestCase):
    """D13's OOD part on ML-29 P11's measurement: 164/164 caught, 2/295 refused."""

    def setUp(self):
        self.report = json.loads((REPO_ROOT / "analysis" / "ml29" / "p11_ood_eval.json").read_text())

    def test_p11_meets_it(self):
        c = p7.ood_counts(self.report)
        self.assertEqual((c["neg_caught"], c["neg_n"], c["bean_refused"], c["bean_n"]), (164, 164, 2, 295))
        self.assertTrue(p7.ood_met(c))

    def test_five_refused_or_one_negative_missed_misses_it(self):
        r = copy.deepcopy(self.report)
        r["positives"]["all"]["not_trained"]["refused"] = 5
        self.assertFalse(p7.ood_met(p7.ood_counts(r)))
        r = copy.deepcopy(self.report)
        r["negatives"]["all"]["caught"] = 163
        self.assertFalse(p7.ood_met(p7.ood_counts(r)))
        r["negatives"]["all"]["caught"] = 164
        r["positives"]["all"]["not_trained"]["refused"] = 4
        self.assertTrue(p7.ood_met(p7.ood_counts(r)))


if __name__ == "__main__":
    unittest.main()
