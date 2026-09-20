"""Tests for coffeecv.lr_schedules. Plain unittest -- the repo has no pytest.

Run from the repo root:  python -m unittest discover -s tests -v
"""
import json
import math
import random
import unittest
import warnings
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from coffeecv.config import RunConfig
from coffeecv.lr_schedules import CosineLR, PlateauLR, build_scheduler

REPO = Path(__file__).resolve().parent.parent

# The adopted recipe: head 1e-3, backbone 1e-5, eta_min 1e-5.
KW = dict(eta_min=1e-5, smooth=5, threshold=0.003, patience=6, cooldown=3, factor=0.3, floor_epochs=15)


def two_group_optimizer(head_lr=1e-3, backbone_lr=1e-5):
    head, backbone = torch.nn.Linear(4, 2), torch.nn.Linear(4, 4)
    return torch.optim.AdamW(
        [{"params": head.parameters(), "lr": head_lr}, {"params": backbone.parameters(), "lr": backbone_lr}],
        weight_decay=1e-4,
    )


def one_group_optimizer(lr=1e-3):
    return torch.optim.AdamW(torch.nn.Linear(4, 2).parameters(), lr=lr, weight_decay=1e-4)


def trailing_mean(xs, w):
    return [sum(xs[max(0, i - w + 1): i + 1]) / len(xs[max(0, i - w + 1): i + 1]) for i in range(len(xs))]


def drive(sched, curve):
    """Feed a val-F1 curve; return (LRs *used during* each epoch, events)."""
    used, events = [], []
    for epoch, v in enumerate(curve, start=1):
        used.append(sched.lrs())
        events.append(sched.step(epoch, v))
    return used, events


def oracle_lrs(curve, opt, *, eta_min, smooth, threshold, patience, cooldown, factor, **_):
    """Independent reference: hand-smoothed input into a bare torch ReduceLROnPlateau."""
    base = [g["lr"] for g in opt.param_groups]
    sched = ReduceLROnPlateau(
        opt, mode="max", factor=factor, patience=patience, threshold=threshold, threshold_mode="abs",
        cooldown=cooldown, min_lr=[min(eta_min, b) for b in base],
    )
    used = []
    for s in trailing_mean(curve, smooth):
        used.append([g["lr"] for g in opt.param_groups])
        sched.step(s)
    return used


def reference_first_trigger(metric, patience, threshold):
    """Pure-Python statement of when ReduceLROnPlateau(mode=max, threshold_mode=abs) first fires:
    the 1-based epoch on which num_bad_epochs exceeds patience. Independent of torch."""
    best, bad = -math.inf, 0
    for i, m in enumerate(metric):
        if m > best + threshold:
            best, bad = m, 0
        else:
            bad += 1
        if bad > patience:
            return i + 1
    return None


def synthetic_curve(seed, n=200):
    rng = random.Random(seed)
    return [min(0.93, 0.6 + 0.02 * e) + rng.gauss(0, 0.005) for e in range(n)]


def recorded_curves():
    """Val macro-F1 of a few archived runs; empty if experiments/ is not on this machine."""
    out = {}
    for n in (200, 164, 209):
        for d in (REPO / "experiments").glob(f"exp{n}__*"):
            f = d / "history.json"
            if f.exists():
                out[n] = [r["val_macro_f1"] for r in json.loads(f.read_text())]
    return out


class TestCosineLR(unittest.TestCase):
    def test_matches_raw_scheduler_and_pins_the_flat_backbone(self):
        # Regression test for the 2026-09-20 finding: with eta_min == backbone_lr (the adopted
        # recipe) CosineAnnealingLR is a no-op on the backbone group, which stays at 1e-5.
        ours, raw_opt = two_group_optimizer(), two_group_optimizer()
        with warnings.catch_warnings():
            # torch warns when scheduler.step() runs without an optimizer.step() in between; this
            # test drives schedulers alone, real training always steps the optimizer.
            warnings.simplefilter("ignore", UserWarning)
            cos = CosineLR(ours, epochs=100, eta_min=1e-5)
            raw = CosineAnnealingLR(raw_opt, T_max=100, eta_min=1e-5)
            used, _ = drive(cos, [0.5] * 100)
            for epoch_lrs in used:
                self.assertEqual(epoch_lrs, [g["lr"] for g in raw_opt.param_groups])
                raw.step()
        self.assertTrue(all(lrs[1] == 1e-5 for lrs in used), "backbone LR must stay flat at 1e-5")
        self.assertEqual(used[0][0], 1e-3)
        self.assertAlmostEqual(used[-1][0], 1.0245e-5, delta=2e-8)

    def test_never_owns_stopping(self):
        cos = CosineLR(two_group_optimizer(), epochs=10, eta_min=1e-5)
        self.assertFalse(cos.owns_stopping)
        self.assertFalse(cos.should_stop())


class TestPlateauLR(unittest.TestCase):
    def test_matches_oracle_on_synthetic_curves(self):
        for seed in range(6):
            curve = synthetic_curve(seed)
            used, _ = drive(PlateauLR(two_group_optimizer(), **KW), curve)
            self.assertEqual(used, oracle_lrs(curve, two_group_optimizer(), **KW), f"seed {seed}")

    def test_matches_oracle_on_recorded_curves(self):
        curves = recorded_curves()
        if not curves:
            self.skipTest("experiments/ not present on this machine")
        for n, curve in curves.items():
            used, _ = drive(PlateauLR(two_group_optimizer(), **KW), curve)
            self.assertEqual(used, oracle_lrs(curve, two_group_optimizer(), **KW), f"exp{n}")

    def test_first_drop_matches_pure_python_reference(self):
        curves = {**recorded_curves(), **{f"syn{s}": synthetic_curve(s) for s in range(4)}}
        for name, curve in curves.items():
            sched = PlateauLR(two_group_optimizer(), **KW)
            _, events = drive(sched, curve)
            first = next((i + 1 for i, e in enumerate(events) if e), None)
            ref = reference_first_trigger(trailing_mean(curve, KW["smooth"]), KW["patience"], KW["threshold"])
            self.assertEqual(first, ref, f"{name}: wrapper first drop {first} vs reference {ref}")

    def test_floor_clamps_head_and_leaves_backbone_alone(self):
        sched = PlateauLR(two_group_optimizer(), **KW)
        used, events = drive(sched, [0.9] * 120)  # flat: plateaus immediately, drops as fast as allowed
        self.assertEqual(sched.min_lrs, [1e-5, 1e-5])
        self.assertTrue(all(lrs[1] == 1e-5 for lrs in used), "backbone must never move")
        self.assertTrue(all(lrs[0] >= 1e-5 for lrs in used), "head must never go below the floor")
        self.assertEqual(used[-1][0], 1e-5)
        self.assertEqual(events.count("floor"), 1)
        self.assertIsNotNone(sched.floor_epoch)

    def test_single_group_optimizer_builds_and_runs(self):
        # freeze_mode=full gives one param group; a fixed two-element min_lr list would raise.
        sched = PlateauLR(one_group_optimizer(), **KW)
        self.assertEqual(sched.min_lrs, [1e-5])
        drive(sched, synthetic_curve(0))

    def test_forced_drops_then_stop_exactly_floor_epochs_after_the_floor(self):
        # patience 0 + threshold 0.9: every epoch after the first is "not better", so the LR drops
        # after epochs 2, 3, 4, 5 and lands on the floor at epoch 5.
        kw = {**KW, "patience": 0, "threshold": 0.9, "cooldown": 0, "floor_epochs": 2}
        sched = PlateauLR(two_group_optimizer(), **kw)
        used, events, stops = [], [], []
        for epoch in range(1, 10):
            used.append(sched.lrs())
            events.append(sched.step(epoch, 0.5))
            stops.append(sched.should_stop())
        self.assertEqual(events[:6], [None, "drop", "drop", "drop", "floor", None])
        for got, want in zip([u[0] for u in used[:7]], [1e-3, 1e-3, 3e-4, 9e-5, 2.7e-5, 1e-5, 1e-5]):
            self.assertAlmostEqual(got, want, delta=1e-12)
        self.assertEqual(sched.floor_epoch, 5)
        self.assertEqual(stops, [False] * 6 + [True] * 3)   # False through epoch 6, True from epoch 7
        self.assertTrue(all(u[1] == 1e-5 for u in used))

    def test_never_stops_while_still_improving(self):
        sched = PlateauLR(two_group_optimizer(), **KW)
        curve = [0.5 + 0.01 * e for e in range(100)]  # gains of 0.05 per smoothing window: never a plateau
        _, events = drive(sched, curve)
        self.assertEqual(events, [None] * 100)
        self.assertIsNone(sched.floor_epoch)
        self.assertFalse(sched.should_stop())

    def test_smoothed_value_is_the_trailing_mean(self):
        sched = PlateauLR(two_group_optimizer(), **KW)
        curve = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.4]
        for e, v in enumerate(curve, start=1):
            sched.step(e, v)
        self.assertAlmostEqual(sched.smoothed, sum(curve[-5:]) / 5, places=12)

    def test_bad_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            PlateauLR(two_group_optimizer(), **{**KW, "eta_min": 0.0})      # no floor -> no stop
        with self.assertRaises(ValueError):
            PlateauLR(two_group_optimizer(1e-5, 1e-5), **KW)                # nothing above the floor
        with self.assertRaises(ValueError):
            PlateauLR(two_group_optimizer(), **{**KW, "factor": 1.0})
        with self.assertRaises(ValueError):
            PlateauLR(two_group_optimizer(), **{**KW, "smooth": 0})


class TestBuildScheduler(unittest.TestCase):
    def test_defaults_are_the_legacy_cosine_path(self):
        self.assertEqual(RunConfig().scheduler, "cosine")
        self.assertIsInstance(build_scheduler(RunConfig(epochs=100, eta_min=1e-5), two_group_optimizer()), CosineLR)

    def test_plateau_builds_from_config(self):
        cfg = RunConfig(scheduler="plateau", eta_min=1e-5, epochs=150)
        sched = build_scheduler(cfg, two_group_optimizer())
        self.assertIsInstance(sched, PlateauLR)
        self.assertTrue(sched.owns_stopping)

    def test_unknown_scheduler_is_rejected(self):
        with self.assertRaises(ValueError):
            build_scheduler(RunConfig(scheduler="onecycle"), two_group_optimizer())

    def test_params_yaml_plateau_knobs_match_the_dataclass_defaults(self):
        # The two carry the same numbers on purpose; drift would make a bare RunConfig() and a
        # `dvc repro` disagree about what "plateau" means.
        from_yaml, default = RunConfig.from_params_yaml(), RunConfig()
        for k in ("plateau_smooth", "plateau_threshold", "plateau_patience", "plateau_cooldown",
                  "plateau_factor", "floor_epochs"):
            self.assertEqual(getattr(from_yaml, k), getattr(default, k), k)


if __name__ == "__main__":
    unittest.main()
