"""coffeecv.tone_map (ticket ML-29 P4, a prototype): the BT.2100 curves against the Recommendation's own numbers and
BT.2408's reference levels, and each candidate operator doing what its docstring says.

    python -m unittest tests.test_tone_map
"""
from __future__ import annotations

import unittest

import numpy as np

from coffeecv import tone_map as tm

GREY = np.linspace(0, 1, 1001)
RAMP = np.repeat(np.geomspace(1e-4, 60, 4000)[:, None], 3, axis=1)      # achromatic, far below and far above white
COLOURS = np.array([[0.9, 0.2, 0.05], [0.1, 0.6, 0.3], [0.05, 0.1, 0.8], [3.0, 1.0, 0.2], [0.3, 2.5, 0.4]])


class PQ(unittest.TestCase):
    def test_end_points(self):
        self.assertEqual(float(tm.pq_eotf(0.0)), 0.0)
        self.assertAlmostEqual(float(tm.pq_eotf(1.0)), 10000.0, places=6)
        self.assertAlmostEqual(float(tm.pq_inverse_eotf(10000.0)), 1.0, places=12)

    def test_round_trip(self):
        signal = GREY[GREY >= 0.01]      # signal 0 is 0 cd/m2, but 0 cd/m2 is signal c1 ** m2, 7e-7
        self.assertLess(np.abs(tm.pq_inverse_eotf(tm.pq_eotf(signal)) - signal).max(), 1e-9)
        nits = np.geomspace(0.001, 10000, 200)
        self.assertLess(np.abs(tm.pq_eotf(tm.pq_inverse_eotf(nits)) / nits - 1).max(), 1e-9)

    def test_reference_white_is_58_percent(self):
        """BT.2408-8 Table 1: 203 cd/m2 is 58% PQ; the files made here put it at 0.5807."""
        self.assertAlmostEqual(float(tm.pq_inverse_eotf(203.0)), 0.5807, places=4)
        self.assertAlmostEqual(float(tm.pq_to_relative(tm.pq_inverse_eotf(203.0))), 1.0, places=9)

    def test_grey_card_level(self):
        """Table 1: an 18% grey card is 26 cd/m2 at 38% PQ."""
        self.assertAlmostEqual(float(tm.pq_inverse_eotf(26.0)), 0.38, delta=0.005)

    def test_monotonic(self):
        self.assertTrue(np.all(np.diff(tm.pq_eotf(GREY)) >= 0))


class HLG(unittest.TestCase):
    def test_oetf_constants(self):
        """Note 5c, and the curve's joint: E = 1/12 is signal 0.5 from both pieces; E = 1 is signal 1."""
        self.assertAlmostEqual(tm.B, 0.28466892, places=8)
        self.assertAlmostEqual(float(tm.C), 0.55991073, places=8)
        self.assertAlmostEqual(float(tm.hlg_oetf(1 / 12)), 0.5, places=12)
        self.assertAlmostEqual(float(tm.A * np.log(12 / 12 - tm.B) + tm.C), 0.5, places=6)
        self.assertAlmostEqual(float(tm.hlg_oetf(1.0)), 1.0, places=6)

    def test_round_trip(self):
        self.assertLess(np.abs(tm.hlg_oetf(tm.hlg_inverse_oetf(GREY)) - GREY).max(), 1e-8)

    def test_system_gamma(self):
        """Note 5f: 1.2 at 1000 cd/m2, 1.2 + 0.42 log10(Lw/1000) inside 400-2000, the extended formula outside."""
        self.assertAlmostEqual(tm.hlg_system_gamma(1000), 1.2)
        self.assertAlmostEqual(tm.hlg_system_gamma(2000), 1.2 + 0.42 * np.log10(2), places=12)
        self.assertAlmostEqual(tm.hlg_system_gamma(4000), 1.2 * 1.111 ** 2, places=12)

    def test_reference_white_is_75_percent(self):
        """BT.2408-8 Table 1: 75% HLG is 203 cd/m2 on a 1000 cd/m2 display (within its rounding)."""
        fd = tm.hlg_eotf(np.full(3, 0.75))
        self.assertAlmostEqual(float(fd[0]), 203.0, delta=1.0)
        self.assertAlmostEqual(float(tm.hlg_to_relative(np.full(3, 0.75))[0]), 1.0, delta=0.005)
        self.assertAlmostEqual(float(tm.hlg_to_relative(np.full(3, 0.75), ootf=False)[0]), 1.0, places=12)

    def test_ootf_round_trip_and_peak(self):
        rgb = np.random.default_rng(29).random((500, 3))
        self.assertLess(np.abs(tm.hlg_inverse_ootf(tm.hlg_ootf(rgb)) - rgb).max(), 1e-9)
        self.assertAlmostEqual(float(tm.hlg_ootf(np.ones(3))[0]), 1000.0, places=9)
        self.assertAlmostEqual(tm.hlg_peak(), 1000 / 203, places=6)   # the OETF's 1.0 is 1 + 1.6e-7

    def test_black_lift(self):
        """BT.2100 Table 5: with a black level lb, signal 0 shows lb."""
        self.assertAlmostEqual(float(tm.hlg_eotf(np.zeros(3), lb=0.1)[0]), 0.1, places=9)

    def test_ootf_keeps_chromaticity(self):
        rgb = np.array([0.3, 0.2, 0.1])
        fd = tm.hlg_ootf(rgb)
        np.testing.assert_allclose(fd / fd.sum(), rgb / rgb.sum(), rtol=1e-12)


class Operators(unittest.TestCase):
    PEAKS = (1.0, 1.5, 4.0, 1000 / 203, 10000 / 203)

    def test_range_and_monotonic(self):
        for name, op in tm.OPERATORS.items():
            for peak in self.PEAKS:
                out = op(RAMP, peak)
                with self.subTest(name=name, peak=peak):
                    self.assertGreaterEqual(out.min(), 0.0)
                    self.assertLessEqual(out.max(), 1.0)
                    self.assertTrue(np.all(np.diff(out[:, 0]) >= -1e-12))

    def test_peak_one_is_identity_below_white(self):
        """With nothing above white to bring back, every candidate leaves standard-range light as it is."""
        x = RAMP[RAMP[:, 0] <= 1]
        for name, op in tm.OPERATORS.items():
            with self.subTest(name=name):
                np.testing.assert_allclose(op(x, 1.0), x, atol=1e-9)

    def test_clip(self):
        np.testing.assert_array_equal(tm.clip_at_white(np.array([[0.5, 1.0, 7.0]])), [[0.5, 1.0, 1.0]])

    def test_scale(self):
        np.testing.assert_allclose(tm.scale_by_peak(np.array([[1.0, 2.0, 4.0]]), 4.0), [[0.25, 0.5, 1.0]])

    def test_eetf_identity_below_knee_and_peak_to_white(self):
        for peak in self.PEAKS[1:]:
            knee = tm.eetf_knee(peak)
            below = RAMP[RAMP[:, 0] < knee * 0.999]
            with self.subTest(peak=peak):
                self.assertLess(knee, 1.0)
                np.testing.assert_allclose(tm.eetf_bt2408(below, peak), below, rtol=1e-9, atol=1e-12)
                self.assertAlmostEqual(float(tm.eetf_bt2408(np.full((1, 3), peak), peak)[0, 0]), 1.0, places=6)
                self.assertGreater(float(tm.eetf_bt2408(np.full((1, 3), 1.0), peak)[0, 0]), knee)

    def test_eetf_knee_documented_values(self):
        """The knee falls as the assumed peak rises: from 0.48 of white for 812 cd/m2, 0.43 for 1000, 0.12 for 10,000."""
        self.assertAlmostEqual(tm.eetf_knee(812 / 203), 0.48, delta=0.01)
        self.assertAlmostEqual(tm.eetf_knee(1000 / 203), 0.43, delta=0.01)
        self.assertAlmostEqual(tm.eetf_knee(10000 / 203), 0.12, delta=0.01)

    def test_reinhard(self):
        for peak in self.PEAKS[1:]:
            with self.subTest(peak=peak):
                self.assertAlmostEqual(float(tm.reinhard_max_rgb(np.full((1, 3), peak), peak)[0, 0]), 1.0, places=9)
                self.assertAlmostEqual(float(tm.reinhard_max_rgb(np.ones((1, 3)), peak)[0, 0]),
                                       (1 + 1 / peak**2) / 2, places=9)

    def test_max_rgb_operators_keep_ratios(self):
        """The EETF and Reinhard scale R, G and B alike: chromaticity kept, nothing leaves the gamut."""
        for op in (tm.eetf_bt2408, tm.reinhard_max_rgb):
            out = op(COLOURS, 4.0)
            with self.subTest(op=op.__name__):
                np.testing.assert_allclose(out / out.max(axis=1, keepdims=True),
                                           COLOURS / COLOURS.max(axis=1, keepdims=True), rtol=1e-9)

    def test_measured_white(self):
        self.assertEqual(tm.measured_pq_white(np.full(100, 150.0)), 203.0)
        self.assertEqual(tm.measured_pq_white(np.full(100, 2300.0)), 10000.0)


if __name__ == "__main__":
    unittest.main()


class InPlace(unittest.TestCase):
    """ML-29 P5 (D33): the chosen operator as canonicalization runs it, in place on float32 linear Rec.2020 strips."""

    def test_the_same_as_the_prototype_on_light_at_or_above_zero(self):
        for peak in (1.5, 4.93, 20.0, 49.0):
            with self.subTest(peak=peak):
                x = np.concatenate([RAMP, COLOURS]).astype(np.float32)
                want = tm.eetf_bt2408(x, peak)
                tm.eetf_in_place(x, peak)
                self.assertLess(float(np.abs(x - want).max()), 1e-5)

    def test_below_the_knee_untouched_to_the_bit(self):
        x = (np.random.default_rng(0).random((1000, 3)) * 0.48).astype(np.float32)   # below a peak of 4's knee, 0.485
        before = x.copy()
        tm.eetf_in_place(x, 4.0)
        self.assertEqual(x.tobytes(), before.tobytes())

    def test_a_peak_at_or_below_white_changes_nothing(self):
        x = np.concatenate([RAMP, COLOURS]).astype(np.float32)
        before = x.copy()
        tm.eetf_in_place(x, 1.0)
        self.assertEqual(x.tobytes(), before.tobytes())

    def test_nan_and_negative_light_kept_for_the_clip_after(self):
        x = np.array([[np.nan, 0.5, 0.5], [-0.1, 3.0, 0.2]], np.float32)
        tm.eetf_in_place(x, 4.0)
        self.assertTrue(np.isnan(x[0, 0]))
        self.assertLess(float(x[1, 0]), 0)                   # scaled with its pixel, clipped after the conversion
        self.assertLessEqual(float(x[1, 1]), 1.0)


class PicturePeak(unittest.TestCase):
    def test_the_9999th_percentile_of_max_rgb(self):
        m = np.ones(100_000, np.float32)
        m[:5] = 30.0                                         # 0.005% of the picture: above the percentile
        self.assertAlmostEqual(tm.picture_peak(m.copy()), 1.0)
        m[:50] = 30.0                                        # 0.05%: the percentile is theirs
        self.assertAlmostEqual(tm.picture_peak(m.copy()), 30.0)

    def test_at_least_white_and_at_most_pqs_top(self):
        self.assertEqual(tm.picture_peak(np.full(10, 0.3, np.float32)), 1.0)
        self.assertEqual(tm.picture_peak(np.full(10, 1e6, np.float32)), tm.PQ_PEAK / tm.REFERENCE_WHITE)

    def test_nan_and_infinity_do_not_make_the_peak(self):
        m = np.full(100_000, 2.0, np.float32)
        m[0], m[1] = np.nan, np.inf
        self.assertAlmostEqual(tm.picture_peak(m), 2.0, places=5)


class InPlaceEdges(unittest.TestCase):
    def test_positive_infinity_is_white_not_black(self):   # a float JPEG XL may hold it; the clip made it white
        x = np.array([[np.inf, 0.5, 0.2], [0.2, 0.2, 0.2]], np.float32)
        tm.eetf_in_place(x, 4.0)
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(float(x[0, 0]), 1.0, places=5)
        self.assertEqual(x[1].tolist(), np.float32([0.2, 0.2, 0.2]).tolist())
