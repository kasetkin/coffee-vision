"""Tests for the ticket ML-2 D17 patch rule, geometry.sample_masked_bean_unit_boxes (plan §10). Plain unittest.

    python -m unittest tests.test_masked_sampler -v
"""
import math
import unittest

import numpy as np

from coffeecv.geometry import (Region, box_bean_share, mask_integral, rotated_box_side,
                               sample_bean_unit_patch_boxes, sample_masked_bean_unit_boxes)

REGION = Region(y0=10, y1=590, x0=12, x1=790)
ARGS = dict(pitch_px=40.0, beans_min=4.0, beans_max=7.0)


def legacy_bean_unit_boxes(rng, region, n, pitch_px, beans_min, beans_max, max_jitter_deg=0.0):
    """sample_bean_unit_patch_boxes as it was before P3, verbatim, to pin that the refactor drew the same."""
    room = min(region.width, region.height)
    beans = np.exp(rng.uniform(np.log(beans_min), np.log(beans_max), size=n))
    angles = rng.uniform(-max_jitter_deg, max_jitter_deg, size=n) if max_jitter_deg > 0 else np.zeros(n)
    out, clamped = [], 0
    for b, angle in zip(beans, angles):
        side = int(round(b * pitch_px))
        box_side = rotated_box_side(side, angle) if angle else side
        if box_side > room:
            box_side = room
            side = int(box_side / (rotated_box_side(1000, angle) / 1000)) if angle else box_side
            clamped += 1
        max_y0, max_x0 = region.y1 - box_side, region.x1 - box_side
        y = int(rng.integers(region.y0, max_y0 + 1))
        x = int(rng.integers(region.x0, max_x0 + 1))
        out.append((Region(y0=y, y1=y + box_side, x0=x, x1=x + box_side), float(angle), side))
    return out, clamped


def blob_mask(h=600, w=800, seed=0) -> np.ndarray:
    """A few random filled ellipses: a bean region with ragged gaps around it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    m = np.zeros((h, w), bool)
    for _ in range(4):
        cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
        ry, rx = rng.uniform(0.15, 0.35) * h, rng.uniform(0.15, 0.35) * w
        m |= ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
    return m


class TestLegacySamplerUnchanged(unittest.TestCase):
    def test_same_draw_as_before_p3(self):
        for jitter in (0.0, 5.0):
            for pitch in (40.0, 140.0):                       # 140 x 7 beans clamps to the room
                a = sample_bean_unit_patch_boxes(np.random.default_rng([3, 1]), REGION, 40, pitch, 4.0, 7.0, jitter)
                b = legacy_bean_unit_boxes(np.random.default_rng([3, 1]), REGION, 40, pitch, 4.0, 7.0, jitter)
                self.assertEqual(a, b)


class TestIntegralShare(unittest.TestCase):
    def test_equals_brute_force(self):
        m = blob_mask()
        ii = mask_integral(m)
        rng = np.random.default_rng(1)
        for _ in range(200):
            y0, x0 = int(rng.integers(0, 590)), int(rng.integers(0, 790))
            y1, x1 = int(rng.integers(y0 + 1, 601)), int(rng.integers(x0 + 1, 801))
            box = Region(y0=y0, y1=y1, x0=x0, x1=x1)
            self.assertAlmostEqual(box_bean_share(ii, box), m[y0:y1, x0:x1].mean(), places=12)


class TestMaskedSampler(unittest.TestCase):
    def draw(self, mask, seed=7, n=40, min_share=0.8, factor=20, jitter=0.0):
        return sample_masked_bean_unit_boxes(np.random.default_rng([seed, 2]), REGION, n, **ARGS, mask=mask,
                                             min_share=min_share, max_attempts_factor=factor,
                                             max_jitter_deg=jitter)

    def test_no_mask_is_the_plain_sampler(self):
        boxes, clamped, below = self.draw(None)
        self.assertEqual((boxes, clamped), sample_bean_unit_patch_boxes(
            np.random.default_rng([7, 2]), REGION, 40, **ARGS))
        self.assertEqual(below, 0)

    def test_full_mask_accepts_the_first_draws(self):
        boxes, _, below = self.draw(np.ones((600, 800), bool))
        self.assertEqual(below, 0)
        self.assertEqual(boxes, sample_bean_unit_patch_boxes(np.random.default_rng([7, 2]), REGION, 40, **ARGS)[0])

    def test_accepted_meet_the_bar_and_n_below_counts_the_rest(self):
        m = blob_mask()
        ii = mask_integral(m)
        for seed in range(5):
            boxes, _, below = self.draw(m, seed=seed)
            self.assertEqual(len(boxes), 40)
            shares = [box_bean_share(ii, b) for b, _, _ in boxes]
            self.assertEqual(below, sum(s < 0.8 for s in shares))
            # accepted first, then the top-up in descending share
            head, tail = shares[:40 - below], shares[40 - below:]
            self.assertTrue(all(s >= 0.8 for s in head))
            self.assertEqual(tail, sorted(tail, reverse=True))

    def test_empty_mask_tops_up_after_the_cap(self):
        boxes, _, below = self.draw(np.zeros((600, 800), bool), factor=3)
        self.assertEqual((len(boxes), below), (40, 40))
        # the cap: exactly 3 x 40 candidates were drawn, so the top-up is the first 40 (all share 0, draw order)
        cands = sample_bean_unit_patch_boxes(np.random.default_rng([7, 2]), REGION, 40, **ARGS)[0]
        self.assertEqual(boxes, cands)

    def test_top_up_takes_the_best_rejected(self):
        # Left half bean region: share depends on where the box lands, so the best rejected must win.
        m = np.zeros((600, 800), bool)
        m[:, :400] = True
        ii = mask_integral(m)
        boxes, _, below = self.draw(m, min_share=0.999, factor=2)
        self.assertGreater(below, 0)
        rng = np.random.default_rng([7, 2])
        cands = []
        while len(cands) < 80:
            cands += sample_bean_unit_patch_boxes(rng, REGION, 40, **ARGS)[0]
        acc = [c for c in cands if box_bean_share(ii, c[0]) >= 0.999][:40]
        rej = sorted((c for c in cands if box_bean_share(ii, c[0]) < 0.999),
                     key=lambda c: -box_bean_share(ii, c[0]))
        self.assertEqual(boxes, acc + rej[:40 - len(acc)])

    def test_deterministic_per_key(self):
        m = blob_mask(seed=3)
        self.assertEqual(self.draw(m, seed=11), self.draw(m, seed=11))
        self.assertNotEqual(self.draw(m, seed=11)[0], self.draw(m, seed=12)[0])

    def test_rotated_boxes_scored_on_their_bounding_box(self):
        m = blob_mask(seed=4)
        ii = mask_integral(m)
        boxes, _, below = self.draw(m, jitter=6.0)
        self.assertEqual(below, sum(box_bean_share(ii, b) < 0.8 for b, _, _ in boxes))
        self.assertTrue(any(a != 0 for _, a, _ in boxes))

    def test_refusals(self):
        with self.assertRaises(ValueError):
            self.draw(blob_mask(), factor=0)
        with self.assertRaises(ValueError):
            self.draw(np.ones((500, 800), bool))              # mask shorter than the region


if __name__ == "__main__":
    unittest.main()
