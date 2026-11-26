"""coffeecv.readback (ticket ML-37 D2, D3, D9; plan P1): a linear Rec.2020 canonical photo's uint16 samples to the
8-bit sRGB both models read, against colour maths written here from the primaries' chromaticities, against the
published BT.709/BT.2020 matrix, and against LittleCMS.

    python -m unittest discover -s tests -p 'test_readback.py'
"""
from __future__ import annotations

import unittest

import numpy as np

from coffeecv import readback as rb
from tests._tiers import DATA_ROOT, real_data

# ----- independent colour maths, float64 -----
D65 = (0.3127, 0.3290)
SRGB = ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))
REC2020 = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))


def rgb_to_xyz(primaries) -> np.ndarray:
    """The RGB-to-XYZ matrix of a set of primaries with a D65 white (SMPTE RP 177)."""
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = D65
    return xyz * np.linalg.solve(xyz, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))


def srgb_decode(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def reference_levels(rec: np.ndarray) -> np.ndarray:
    """The read-back in float64, before rounding: 0-255."""
    lin = rec.astype(np.float64) / 65535 @ (np.linalg.inv(rgb_to_xyz(SRGB)) @ rgb_to_xyz(REC2020)).T
    lin = np.clip(lin, 0, 1)
    return 255 * np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def grey(values) -> np.ndarray:
    v = np.asarray(values, np.uint16)
    return np.repeat(v[None, :, None], 3, axis=2)


class Greys(unittest.TestCase):
    def test_every_8_bit_grey_reads_back_as_itself(self):
        """Each 8-bit sRGB grey, made linear and stored in 16 bits, reads back as the level it was."""
        levels = np.arange(256)
        stored = np.rint(srgb_decode(levels / 255) * 65535)
        out = rb.to_srgb(grey(stored))
        self.assertEqual(out.dtype, np.uint8)
        np.testing.assert_array_equal(out[0], np.repeat(levels[:, None], 3, axis=1))

    def test_a_16_bit_grey_ramp_matches_the_float64_reference(self):
        ramp = grey(np.arange(65536))
        self.assertLessEqual(np.abs(rb.srgb_levels(ramp) - reference_levels(ramp)).max(), 0.01)


class Matrix(unittest.TestCase):
    # ITU-R BT.2087-0, Annex 1: linear BT.709 RGB to linear BT.2020 RGB, as published (four decimals)
    BT2087 = np.array([[0.6274, 0.3293, 0.0433],
                       [0.0691, 0.9195, 0.0114],
                       [0.0164, 0.0880, 0.8956]])

    def test_the_matrix_is_the_inverse_of_bt2087s(self):
        np.testing.assert_allclose(np.linalg.inv(rb.REC2020_TO_SRGB), self.BT2087, rtol=0, atol=5e-5)
        np.testing.assert_allclose(rb.REC2020_TO_SRGB, np.linalg.inv(self.BT2087), rtol=0, atol=1e-3)

    def test_white_stays_white(self):
        np.testing.assert_allclose(rb.REC2020_TO_SRGB.sum(axis=1), 1.0, rtol=0, atol=1e-12)


class Colours(unittest.TestCase):
    def test_the_primaries_white_and_black(self):
        """Rec.2020's primaries lie outside sRGB: each clips to the sRGB values the reference gives."""
        rec = np.array([[[65535, 0, 0], [0, 65535, 0], [0, 0, 65535], [65535] * 3, [0] * 3]], np.uint16)
        out = rb.to_srgb(rec)
        np.testing.assert_array_equal(out, np.rint(reference_levels(rec)))
        np.testing.assert_array_equal(out[0, 3:], [[255] * 3, [0] * 3])
        np.testing.assert_array_equal(out[0, :3].diagonal(), [255] * 3)   # each primary saturates its own channel

    def test_random_pictures_match_the_float64_reference_before_rounding(self):
        rng = np.random.default_rng(37)
        for name, rec in {"uniform": rng.integers(0, 65536, (64, 96, 3)),
                          "shadows": rng.integers(0, 600, (64, 96, 3)),   # below about level 40 of 255
                          "near grey": rng.integers(20000, 21000, (64, 96, 3))}.items():
            with self.subTest(name):
                rec = rec.astype(np.uint16)
                levels = rb.srgb_levels(rec)
                self.assertEqual(levels.dtype, np.float32)
                self.assertLessEqual(np.abs(levels - reference_levels(rec)).max(), 0.01)
                np.testing.assert_array_equal(rb.to_srgb(rec), np.rint(levels).astype(np.uint8))


def littlecms_levels(rec: np.ndarray) -> np.ndarray:
    """LittleCMS 2's float32 transform from a linear Rec.2020 profile (D65, gamma 1) to its own sRGB, x 255: the
    library ML-19 D31 used for today's read-back."""
    import imagecodecs as ic
    linear_rec2020 = ic.cms_profile("rgb", whitepoint=D65, primaries=sum(REC2020, ()), gamma=1.0)
    srgb = ic.cms_transform(rec.astype(np.float32) / 65535, linear_rec2020, ic.cms_profile("srgb"),
                            intent=ic.CMS.INTENT.RELATIVE_COLORIMETRIC, flags=ic.CMS.FLAGS.NOOPTIMIZE,
                            outdtype=np.float32)
    return np.clip(srgb, 0, 1) * 255


class LittleCMS(unittest.TestCase):
    def test_within_half_a_level_of_littlecms(self):
        rng = np.random.default_rng(9)
        for name, rec in {"uniform": rng.integers(0, 65536, (64, 96, 3)), "shadows": rng.integers(0, 600, (64, 96, 3)),
                          "ramp": grey(np.arange(0, 65536, 7))}.items():
            with self.subTest(name):
                rec = rec.astype(np.uint16)
                self.assertLessEqual(np.abs(rb.srgb_levels(rec) - littlecms_levels(rec)).max(), 0.5)


def bt709_decode(v):
    """ITU-R BT.709's curve, inverted: today's canonical samples (ML-19 D7) to linear light."""
    v = np.asarray(v, np.float64)
    return np.where(v < 4.5 * 0.018, v / 4.5, ((v + 0.099) / 1.099) ** (1 / 0.45))


@real_data
class DatasetPhotos(unittest.TestCase):
    """20 of the store's photos, spread over it by path, made linear: a 256-row band through each photo's middle, from
    today's canonical samples through the BT.709 curve inverted, as uint16."""
    N, BAND = 20, 256

    @classmethod
    def setUpClass(cls):
        import imagecodecs as ic
        store = sorted((DATA_ROOT / "data" / "canonical").rglob("*.jxl"))
        if len(store) < cls.N:
            raise unittest.SkipTest(f"no canonical store under {DATA_ROOT} (COFFEECV_DATA_ROOT)")
        cls.bands = {}
        for path in (store[i * len(store) // cls.N] for i in range(cls.N)):
            today = ic.jpegxl_decode(path.read_bytes())
            y = max(0, today.shape[0] // 2 - cls.BAND // 2)
            band = today[y:y + cls.BAND]
            cls.bands[path.name] = np.rint(bt709_decode(band / 65535) * 65535).astype(np.uint16)

    def test_within_001_of_the_float64_reference_before_rounding(self):
        for name, rec in self.bands.items():
            with self.subTest(name):
                self.assertLessEqual(np.abs(rb.srgb_levels(rec) - reference_levels(rec)).max(), 0.01)

    def test_within_half_a_level_of_littlecms(self):
        for name, rec in self.bands.items():
            with self.subTest(name):
                self.assertLessEqual(np.abs(rb.srgb_levels(rec) - littlecms_levels(rec)).max(), 0.5)

    def test_the_strips_and_threads_change_no_pixel(self):
        for name, rec in self.bands.items():
            with self.subTest(name):
                np.testing.assert_array_equal(rb.to_srgb(rec, strip=37, numthreads=3), rb.to_srgb(rec, numthreads=1))


class Pin(unittest.TestCase):
    def test_the_numpy_pin_is_the_locks(self):   # D9, as coffeecv.canonical's imagecodecs (ML-19 D44)
        import tomllib
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        for lock in ("uv.lock", "webapp/uv.lock", "webapp/front/uv.lock"):
            with self.subTest(lock):
                held = tomllib.loads((repo / lock).read_text())["package"]
                self.assertEqual([p["version"] for p in held if p["name"] == "numpy"], [rb.NUMPY])


class Strips(unittest.TestCase):
    """R10: in strips of rows on a thread pool, never a whole-photo float copy, with the same pixels."""

    def test_the_strip_size_and_thread_count_do_not_change_a_pixel(self):
        rec = np.random.default_rng(7).integers(0, 65536, (601, 83, 3)).astype(np.uint16)
        whole = np.rint(rb.srgb_levels(rec)).astype(np.uint8)
        for strip, threads in ((256, 1), (256, 4), (1, 3), (100, 2), (1000, 8)):
            with self.subTest(strip=strip, threads=threads):
                np.testing.assert_array_equal(rb.to_srgb(rec, strip=strip, numthreads=threads), whole)

    def test_the_peak_memory_is_the_output_and_a_few_strips(self):
        import tracemalloc
        h, w = 2048, 1024
        rec = np.zeros((h, w, 3), np.uint16)
        strip_floats = rb.STRIP_ROWS * w * 3 * 4
        tracemalloc.start()
        try:
            rb.to_srgb(rec, numthreads=1)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        self.assertLess(peak, h * w * 3 + 4 * strip_floats)   # a whole float32 copy alone is h * w * 3 * 4


if __name__ == "__main__":
    unittest.main()
