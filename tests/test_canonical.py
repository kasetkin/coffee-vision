"""coffeecv.canonical (ADR 0033; ticket ML-19 D8): the colour conversion held fixed on every commit.

(i) Synthetic patches of known colour (sRGB and Display P3 primaries, a P3 green outside sRGB, greys, white) through
original -> canonical -> 8-bit sRGB, against colour maths written here from the primaries' chromaticities, not
LittleCMS; (ii) a canonical file's header (D7's encoding, nothing else); (iii) a committed golden file, exact on the
store's machine (D30, D51). Also: the eight EXIF orientations against Pillow's exif_transpose, the cap's two
thresholds, transparency, an untagged photo, is_canonical on near misses.

    python -m unittest discover -s tests -p 'test_canonical.py'

Regenerate the golden file only for a deliberate change to canonicalization, which is a re-canonicalization and a
retrain (ADR 0033): COFFEECV_WRITE_GOLDEN=1 python -m unittest tests.test_canonical
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image, ImageCms, ImageOps

from coffeecv import canonical as cn
from coffeecv.jxl_header import read_header
from tests._tiers import real_data

FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = FIXTURES / "canonical_golden.json"
DISPLAY_P3 = (FIXTURES / "display_p3_libjxl.icc").read_bytes()   # libjxl 0.12.0's DisplayP3 (D65, sRGB curve)

# ----- independent colour maths -----
D65 = (0.3127, 0.3290)
PRIMARIES = {"srgb": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)),
             "p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
             "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))}


def rgb_to_xyz(name: str) -> np.ndarray:
    """The RGB-to-XYZ matrix of a set of primaries with a D65 white (SMPTE RP 177)."""
    xyz = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in PRIMARIES[name]]).T
    wx, wy = D65
    s = np.linalg.solve(xyz, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))
    return xyz * s


def srgb_decode(v):
    v = np.asarray(v, np.float64)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def srgb_encode(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)


def bt709_encode(lin):
    lin = np.clip(lin, 0, 1)
    return np.where(lin < 0.018, 4.5 * lin, 1.099 * lin ** 0.45 - 0.099)


def convert(rgb8, source: str):
    """8-bit source values (sRGB curve, `source` primaries) to the expected canonical Rec.2020 encoding (0-1) and the
    expected 8-bit sRGB read-back."""
    lin = srgb_decode(np.asarray(rgb8, np.float64) / 255)
    rec_lin = np.clip(lin @ (np.linalg.inv(rgb_to_xyz("rec2020")) @ rgb_to_xyz(source)).T, 0, 1)
    srgb_lin = rec_lin @ (np.linalg.inv(rgb_to_xyz("srgb")) @ rgb_to_xyz("rec2020")).T
    return bt709_encode(rec_lin), np.round(srgb_encode(srgb_lin) * 255)


# ----- test photos -----
PATCH = 32
SRGB_PATCHES = [(255, 255, 255), (0, 0, 0), (64, 64, 64), (128, 128, 128), (192, 192, 192), (255, 0, 0), (0, 255, 0),
                (0, 0, 255), (200, 120, 60)]
P3_PATCHES = [(255, 255, 255), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255), (40, 220, 60)]  # P3 greens
# lie outside sRGB


def patches(colours) -> np.ndarray:
    return np.concatenate([np.full((PATCH, PATCH, 3), c, np.uint8) for c in colours], axis=1)


def png(a: np.ndarray, icc: bytes | None = None, exif: bytes | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {"icc_profile": icc} if icc else {}
    Image.fromarray(a).save(buf, "PNG", **kw, **({"exif": exif} if exif else {}))
    return buf.getvalue()


def jpeg(a: np.ndarray, orientation: int | None = None, icc: bytes | None = None) -> bytes:
    im = Image.fromarray(a)
    exif = im.getexif()
    if orientation is not None:
        exif[0x0112] = orientation
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95, exif=exif.tobytes(), **({"icc_profile": icc} if icc else {}))
    return buf.getvalue()


def centres(a: np.ndarray, n: int) -> np.ndarray:
    """Each patch's mean over its middle half, (n, 3)."""
    q = PATCH // 4
    return np.array([a[q:-q, i * PATCH + q:(i + 1) * PATCH - q].reshape(-1, 3).mean(0) for i in range(n)])


class Colour(unittest.TestCase):
    """D8 (i): original -> canonical -> sRGB against the maths above. The colour conversion alone is held to the
    maths within LittleCMS's own rounding; the whole path, through q95's lossy encode, in linear light: q95 moves a
    saturated colour's near-zero channel by up to 9 levels in 8-bit sRGB (where its curve is steepest), 0.003 in
    linear light, and leaves greys exact (2026-11-03)."""

    def check(self, colours, source: str, data: bytes):
        n = len(colours)
        want_stored, want_back = convert(colours, source)
        src = cn.decode(data)
        rec = cn.to_rec2020(src.pixels, src.full, src.profile)       # the conversion alone, no JPEG XL
        direct = ic.cms_transform(rec, cn.REC2020, cn.SRGB, intent=cn.RELATIVE, flags=cn.NOOPTIMIZE,
                                  outdtype=np.uint8)
        canon = cn.canonicalize(data)
        stored = centres(ic.jpegxl_decode(canon).astype(np.float64) / 65535, n)
        back = centres(cn.load_canonical(canon).astype(np.float64), n)
        for i, c in enumerate(colours):
            with self.subTest(source=source, colour=c):
                # s15.16 profile rounding: within 0.0005 of full scale (33 16-bit codes) and one 8-bit level
                np.testing.assert_allclose(centres(rec / 65535, n)[i], want_stored[i], atol=0.0005)
                np.testing.assert_allclose(centres(direct.astype(np.float64), n)[i], want_back[i], atol=1.0)
                np.testing.assert_allclose(stored[i], want_stored[i], atol=0.004)
                # each channel within one level, or within 0.004 in linear light (near black, where a level is tiny)
                near = (np.abs(back[i] - want_back[i]) <= 1) | \
                    (np.abs(srgb_decode(back[i] / 255) - srgb_decode(want_back[i] / 255)) <= 0.004)
                self.assertTrue(near.all(), f"read back {back[i]}, want {want_back[i]}")
                if len(set(c)) == 1:   # a grey stays a grey, to the level
                    np.testing.assert_allclose(back[i], want_back[i], atol=1.0)

    def test_srgb_patches_untagged(self):   # D16: an untagged photo is sRGB
        self.check(SRGB_PATCHES, "srgb", png(patches(SRGB_PATCHES)))

    def test_display_p3_patches_and_the_green_outside_srgb(self):
        self.check(P3_PATCHES, "p3", png(patches(P3_PATCHES), icc=DISPLAY_P3))
        # P3's green is outside sRGB: stored as more saturated than sRGB green can be, read back clipped to it
        stored, back = convert([(0, 255, 0)], "p3")
        srgb_green_stored, _ = convert([(0, 255, 0)], "srgb")
        self.assertGreater(srgb_green_stored[0][0] - stored[0][0], 0.05)
        self.assertEqual(back[0][1], 255)

    def test_a_tagged_srgb_photo_reads_as_the_untagged_one(self):
        a = patches(SRGB_PATCHES)
        tagged = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        one = cn.load_canonical(cn.canonicalize(png(a)))
        two = cn.load_canonical(cn.canonicalize(png(a, icc=tagged)))
        self.assertLessEqual(int(np.abs(one.astype(int) - two).max()), 1)

    def test_transparent_pixels_become_the_fill_grey(self):   # D13
        a = np.zeros((PATCH, 2 * PATCH, 4), np.uint8)
        a[:, :PATCH] = (10, 200, 30, 0)        # transparent: the fill, whatever the colour
        a[:, PATCH:] = (10, 200, 30, 255)      # opaque: its own colour
        src = cn.decode(png(a))
        rec = cn.to_rec2020(src.pixels, src.full, src.profile)
        back = centres(ic.cms_transform(rec, cn.REC2020, cn.SRGB, intent=cn.RELATIVE, flags=cn.NOOPTIMIZE,
                                        outdtype=np.uint8).astype(np.float64), 2)
        np.testing.assert_allclose(back[0], cn.FILL_SRGB, atol=1.0)
        np.testing.assert_allclose(back[1], (10, 200, 30), atol=1.0)
        self.assertTrue(cn.is_canonical(cn.canonicalize(png(a))))   # no alpha channel left

    def test_the_fill_is_the_segmenters(self):
        from dataclasses import fields

        from coffeecv.segment_beans import SegParams
        self.assertEqual(tuple(next(f.default for f in fields(SegParams) if f.name == "fill_rgb")), cn.FILL_SRGB)


class Header(unittest.TestCase):
    """D8 (ii): the canonical file declares D7's encoding and holds nothing else."""

    def test_the_file_declares_d7_and_nothing_else(self):
        canon = cn.canonicalize(jpeg(patches(SRGB_PATCHES), orientation=6))
        h = read_header(canon)
        c = h.colour
        self.assertEqual((c.want_icc, c.colour_space, c.white_point, c.primaries, c.transfer, c.rendering_intent),
                         (False, "RGB", "D65", "BT2100", "BT709", "Relative"))
        self.assertEqual((h.orientation, h.num_extra_channels, h.boxes, h.bits_per_sample, h.xyb_encoded),
                         (1, 0, [], 16, True))
        self.assertNotIn(b"Exif", canon)
        self.assertTrue(cn.is_canonical(canon))

    def test_the_imagecodecs_pin_is_the_locks(self):   # D44
        import tomllib
        lock = tomllib.loads((Path(__file__).resolve().parents[1] / "uv.lock").read_text())
        self.assertEqual([p["version"] for p in lock["package"] if p["name"] == "imagecodecs"], [cn.IMAGECODECS])

    def test_the_profile_is_the_committed_one(self):   # D46
        self.assertEqual(hashlib.sha256(cn.PROFILE_PATH.read_bytes()).hexdigest(), cn.PROFILE_SHA256)
        self.assertEqual(len(cn.REC2020), 536)


class Golden(unittest.TestCase):
    """D8 (iii): the canonical bytes and their read-back, exact. On another CPU libjxl may differ (D51): the golden
    file holds on the machine that builds the store and runs the tests."""

    def test_golden(self):
        inputs = {"srgb_patches_untagged.png": png(patches(SRGB_PATCHES)),
                  "p3_patches.png": png(patches(P3_PATCHES), icc=DISPLAY_P3),
                  "srgb_patches_rot6.jpg": jpeg(patches(SRGB_PATCHES), orientation=6)}
        got = {}
        for name, data in inputs.items():
            canon = cn.canonicalize(data, name)
            back = cn.load_canonical(canon)
            got[name] = {"input_sha256": hashlib.sha256(data).hexdigest(),
                         "canonical_sha256": hashlib.sha256(canon).hexdigest(),
                         "srgb_sha256": hashlib.sha256(back.data).hexdigest()}
        if os.environ.get("COFFEECV_WRITE_GOLDEN") == "1":
            GOLDEN.write_text(json.dumps(got, indent=1) + "\n")
        want = json.loads(GOLDEN.read_text())
        for name in inputs:
            with self.subTest(name):
                if got[name]["input_sha256"] != want[name]["input_sha256"]:
                    self.skipTest("Pillow made a different input file; the golden values are for the committed one")
                self.assertEqual(got[name], want[name])


class Orientation(unittest.TestCase):
    """D28: a JPEG turned by each of the eight EXIF orientations, as Pillow's exif_transpose turns it; 0 and 9 as
    stored."""

    def test_eight_orientations_and_invalid_values(self):
        a = np.zeros((48, 64, 3), np.uint8)
        a[:24, :32], a[:24, 32:], a[24:, :32], a[24:, 32:] = (250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 250)
        for orientation in range(10):
            with self.subTest(orientation=orientation):
                data = jpeg(a, orientation=orientation)
                want = np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")) \
                    if 1 <= orientation <= 8 else np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                got = cn.load_canonical(cn.canonicalize(data))
                self.assertEqual(got.shape, want.shape)
                self.assertLess(np.abs(got.astype(int) - want).mean(), 3)


class Cap(unittest.TestCase):
    """D27: at or under 60,000,000 pixels kept; over, at most 50,000,000 at the same shape."""

    def test_the_two_thresholds(self):
        self.assertEqual(cn.capped_size(10_000, 6_000), (10_000, 6_000))   # exactly 60 MP
        self.assertEqual(cn.capped_size(8160, 6144), (8160, 6144))         # the Pixel's 50.14 MP
        for w, h in ((10_001, 6_000), (16_320, 12_240), (7746, 7746), (12_240, 16_320), (60_000_001, 2)):
            with self.subTest(size=(w, h)):
                nw, nh = cn.capped_size(w, h)
                self.assertLessEqual(nw * nh, cn.CAP_PIXELS)
                self.assertGreater((nw + 1) * (nh + 1), cn.CAP_PIXELS)     # the largest such size
                self.assertLess(abs(nw / w - nh / h), 1 / min(nw, nh) + 1e-9)   # one factor for both sides

    @real_data
    def test_a_photo_just_over_the_threshold_is_downscaled_and_one_at_it_is_not(self):
        for (w, h), want in (((10_000, 6_000), (10_000, 6_000)), ((10_001, 6_000), cn.capped_size(10_001, 6_000))):
            with self.subTest(size=(w, h)):
                src = cn.decode(ic.png_encode(np.full((h, w, 3), 90, np.uint8)), "big.png")
                self.assertEqual(src.pixels.shape[1::-1], want)


class IsCanonical(unittest.TestCase):
    """D35: the one predicate behind the front's pass-through and the host's refusal."""

    def setUp(self):
        self.canon = cn.canonicalize(png(patches(SRGB_PATCHES)))

    def test_a_canonical_photo_passes_through_unchanged(self):   # D17
        self.assertTrue(cn.is_canonical(self.canon))
        self.assertEqual(cn.canonicalize(self.canon), self.canon)

    def test_near_misses(self):
        a16 = (np.random.default_rng(2).random((40, 60, 3)) * 65535).astype(np.uint16)
        a16a = (np.random.default_rng(3).random((40, 60, 4)) * 65535).astype(np.uint16)
        misses = {
            "an 8-bit sRGB JPEG XL": ic.jpegxl_encode(a16.astype(np.uint8), level=95),
            "a 16-bit Display P3 JPEG XL": ic.jpegxl_encode(a16, level=95, primaries="P3", transfer="SRGB"),
            "D7's encoding in a container": ic.jpegxl_encode(a16, level=95, primaries="BT2100", transfer="BT709",
                                                             usecontainer=True),
            "D7's encoding with alpha": ic.jpegxl_encode(a16a, level=95, primaries="BT2100", transfer="BT709"),
            "D7's encoding, lossless": ic.jpegxl_encode(a16, lossless=True, primaries="BT2100", transfer="BT709"),
            "a JPEG": jpeg(patches(SRGB_PATCHES)),
            "nothing": b"",
        }
        for what, data in misses.items():
            with self.subTest(what):
                self.assertFalse(cn.is_canonical(data))
                with self.assertRaises(cn.NotCanonical):
                    cn.load_canonical(data)

    def test_orientation_size_preview_and_animation(self):
        h = read_header(self.canon)
        self.assertEqual(cn.header_problems(h), [])
        for change in ({"orientation": 6}, {"width": 10_001, "height": 6_000}, {"have_preview": True},
                       {"have_animation": True}):
            with self.subTest(change=change):
                self.assertTrue(cn.header_problems(replace(h, **change)))


class Readers(unittest.TestCase):
    def test_a_srgb_jpeg_xl_upload_is_canonicalized(self):
        a = np.asarray(patches(SRGB_PATCHES))
        canon = cn.canonicalize(ic.jpegxl_encode(a, level=99))
        self.assertTrue(cn.is_canonical(canon))
        self.assertLess(np.abs(cn.load_canonical(canon).astype(int) - a).mean(), 2)

    def test_grey_photos(self):
        for data in (png(np.full((32, 32), 128, np.uint8)), ic.png_encode(np.dstack([np.full((32, 32), 128, np.uint8), np.full((32, 32), 255, np.uint8)]))):
            with self.subTest(n=len(data)):
                back = cn.load_canonical(cn.canonicalize(data))
                self.assertLessEqual(int(np.abs(back.astype(int) - 128).max()), 1)

    def test_unreadable(self):
        for data, name in ((b"not a photo", "x.jpg"), (b"\xff\xd8\xff\xe0 broken", "x.jpg"), (b"junk", "x.dng")):
            with self.subTest(name=name, data=data[:6]), self.assertRaises(cn.Unreadable):
                cn.canonicalize(data, name)

    def test_a_photo_over_the_ceiling_is_refused_before_its_pixels_are_decoded(self):   # D38
        data = png(patches(SRGB_PATCHES))                       # 288 x 32 = 9,216 pixels
        self.assertTrue(cn.is_canonical(cn.canonicalize(data, max_pixels=9_216)))
        for d in (data, jpeg(patches(SRGB_PATCHES)), ic.jpegxl_encode(patches(SRGB_PATCHES), level=95)):
            with self.subTest(n=len(d)), self.assertRaises(cn.TooLarge):
                cn.canonicalize(d, max_pixels=9_215)

    def test_cmyk_is_refused(self):
        buf = io.BytesIO()
        Image.new("CMYK", (16, 16), (0, 50, 100, 0)).save(buf, "JPEG")
        with self.assertRaises(cn.Unreadable):
            cn.canonicalize(buf.getvalue())


if __name__ == "__main__":
    unittest.main()
