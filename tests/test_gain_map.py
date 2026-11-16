"""coffeecv.gain_map (ML-29 P7): a gain map read from a file's bytes and applied to linear light.

Files are made here: by libultrahdr (imagecodecs.ultrahdr_encode, which writes ISO 21496-1), and by this module's own
Ultra HDR writer (a main JPEG, an MPF index and a gain map JPEG with hdrgm XMP), so the reader is not checked against
itself. ISO 21496-1 payloads are written here with struct, from the standard's field order.

    python -m unittest tests.test_gain_map
"""
from __future__ import annotations

import io
import struct
import unittest
from pathlib import Path

import imagecodecs as ic
import numpy as np
from PIL import Image

from coffeecv import canonical as cn
from coffeecv import gain_map as gm
from tests._tiers import real_data


def iso_payload(gain_min, gain_max, gamma, base_offset, alt_offset, base_headroom, alt_headroom, *,
                use_base=True, common=None, minimum_version=0) -> bytes:
    """ISO 21496-1's GainMapMetadata, minimum_version on. Per-channel sequences of 1 or 3; `common` writes every
    value over that one denominator (use_common_denominator), else each over 1,000,000."""
    channels = len(gain_min)
    flags = (0x80 if channels == 3 else 0) | (0x40 if use_base else 0) | (0x08 if common else 0)
    out = struct.pack(">HHB", minimum_version, 0, flags)
    if common:
        out += struct.pack(">III", common, round(base_headroom * common), round(alt_headroom * common))
        for c in range(channels):
            out += struct.pack(">iiIii", *(round(v[c] * common) for v in (gain_min, gain_max, gamma, base_offset,
                                                                           alt_offset)))
        return out
    d = 1_000_000
    out += struct.pack(">IIII", round(base_headroom * d), d, round(alt_headroom * d), d)
    for c in range(channels):
        for v in (gain_min, gain_max, gamma, base_offset, alt_offset):
            out += struct.pack(">iI", round(v[c] * d), d)
    return out


def jpeg(pixels: np.ndarray, *segments: bytes) -> bytes:
    """A baseline JPEG with `segments` (whole marker segments) right after its SOI."""
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, "JPEG", quality=95)
    data = buf.getvalue()
    return data[:2] + b"".join(segments) + data[2:]


def segment(marker: int, body: bytes) -> bytes:
    return struct.pack(">BBH", 0xFF, marker, len(body) + 2) + body


def ultra_hdr(main: np.ndarray, gain: np.ndarray, xmp: bytes, *, iso: bytes | None = None,
              orientation: int | None = None) -> bytes:
    """An Ultra HDR file: the main JPEG with an MPF index (CIPA DC-007, big-endian) whose second entry is the gain
    map JPEG appended after it; the map carries `xmp` (and an ISO 21496-1 APP2 if given)."""
    extra = [segment(0xE1, gm.XMP_URN + xmp)]
    if iso is not None:
        extra.append(segment(0xE2, gm.ISO_URN + iso))
    second = jpeg(gain, *extra)
    exif = []
    if orientation is not None:
        exif = [segment(0xE1, b"Exif\0\0MM\0*" + struct.pack(">IHHHIHH", 8, 1, 0x0112, 3, 1, orientation, 0)
                        + b"\0\0\0\0")]

    def mpf(first_size: int, second_offset: int) -> bytes:
        entries = struct.pack(">IIIHH", 0x030000, first_size, 0, 0, 0) + struct.pack(">IIIHH", 0, len(second),
                                                                                    second_offset, 0, 0)
        ifd = struct.pack(">H", 1) + struct.pack(">HHII", 0xB002, 7, 32, 8 + 2 + 12 + 4) + b"\0\0\0\0"
        return segment(0xE2, b"MPF\0" + b"MM\0*" + struct.pack(">I", 8) + ifd + entries)

    probe = jpeg(main, *exif, mpf(0, 0))
    tiff_at = probe.index(b"MPF\0") + 4
    first = jpeg(main, *exif, mpf(len(probe), len(probe) - tiff_at))
    return first + second


def srgb_decode(u: np.ndarray) -> np.ndarray:
    return np.where(u > 0.04045, ((u + 0.055) / 1.055) ** 2.4, u / 12.92).astype(np.float32)


def flat(value: float, h: int = 4, w: int = 6, channels: int = 1) -> np.ndarray:
    return np.full((h, w, channels), value, np.float32)


def meta(gain_min=0.0, gain_max=1.0, gamma=1.0, base_offset=0.0, alt_offset=0.0, base_headroom=0.0,
         alt_headroom=1.0, channels=1) -> gm.Metadata:
    return gm.Metadata((gain_min,) * channels, (gain_max,) * channels, (gamma,) * channels, (base_offset,) * channels,
                       (alt_offset,) * channels, base_headroom, alt_headroom)


class Iso(unittest.TestCase):
    def test_both_denominator_forms_read_back(self):
        values = ((-0.25, 0.0, 0.5), (1.25, 1.5, 2.0), (1.0, 0.5, 2.0), (1 / 64, 0.0, 0.01), (1 / 64, 0.02, 0.0))
        for common in (None, 1 << 16):
            with self.subTest(common=common):
                m = gm.parse_iso(iso_payload(*values, 0.5, 2.25, use_base=False, common=common))
                for got, want in zip((m.gain_min, m.gain_max, m.gamma, m.base_offset, m.alternate_offset), values):
                    np.testing.assert_allclose(got, want, atol=1e-4)
                self.assertEqual((m.base_headroom, m.alternate_headroom, m.use_base_colour_space), (0.5, 2.25, False))

    def test_one_channel(self):
        m = gm.parse_iso(iso_payload((0.0,), (2.3,), (1.0,), (0.0,), (0.0,), 0.0, 2.3))
        self.assertEqual((m.gain_max, m.use_base_colour_space, m.leads_down), ((2.3,), True, False))

    def test_an_unknown_version_and_the_version_alone_are_no_map(self):
        self.assertIsNone(gm.parse_iso(iso_payload((0.0,), (1.0,), (1.0,), (0.0,), (0.0,), 0, 1, minimum_version=1)))
        self.assertIsNone(gm.parse_iso(struct.pack(">HH", 0, 0)))

    def test_a_zero_gamma_or_denominator_is_no_map(self):
        self.assertIsNone(gm.parse_iso(iso_payload((0.0,), (1.0,), (0.0,), (0.0,), (0.0,), 0, 1)))
        zero_denominator = struct.pack(">HHB", 0, 0, 0x08) + struct.pack(">III", 0, 0, 1) + b"\0" * 20
        self.assertIsNone(gm.parse_iso(zero_denominator))


class Hdrgm(unittest.TestCase):
    def test_attributes_with_the_formats_defaults(self):
        m = gm.parse_hdrgm(b'<rdf:Description hdrgm:Version="1.0" hdrgm:GainMapMax="2.3" hdrgm:HDRCapacityMax="2.3"/>')
        self.assertEqual((m.gain_min, m.gain_max, m.gamma), ((0.0,), (2.3,), (1.0,)))
        self.assertEqual((m.base_offset, m.alternate_offset, m.base_headroom, m.alternate_headroom),
                         ((1 / 64,), (1 / 64,), 0.0, 2.3))

    def test_per_channel_sequences(self):   # Adobe's form, which libultrahdr refuses
        xmp = (b'<rdf:Description hdrgm:Version="1.0" hdrgm:HDRCapacityMax="1.3" hdrgm:OffsetSDR="0">'
               b"<hdrgm:GainMapMax><rdf:Seq><rdf:li>1.2</rdf:li><rdf:li>1.25</rdf:li><rdf:li>1.3</rdf:li>"
               b"</rdf:Seq></hdrgm:GainMapMax></rdf:Description>")
        m = gm.parse_hdrgm(xmp)
        self.assertEqual((m.gain_max, m.gain_min, m.base_offset), ((1.2, 1.25, 1.3), (0.0,) * 3, (0.0,) * 3))

    def test_an_hdr_base_leads_down(self):
        m = gm.parse_hdrgm(b'hdrgm:Version="1.0" hdrgm:GainMapMax="2" hdrgm:HDRCapacityMax="2" '
                           b'hdrgm:BaseRenditionIsHDR="True" hdrgm:OffsetSDR="0.01" hdrgm:OffsetHDR="0.02"')
        self.assertTrue(m.leads_down)
        self.assertEqual((m.base_headroom, m.alternate_headroom, m.base_offset, m.alternate_offset),
                         (2.0, 0.0, (0.02,), (0.01,)))

    def test_without_its_required_fields_or_another_version_no_map(self):
        self.assertIsNone(gm.parse_hdrgm(b'hdrgm:Version="1.0" hdrgm:GainMapMax="2"'))
        self.assertIsNone(gm.parse_hdrgm(b'hdrgm:Version="2.0" hdrgm:GainMapMax="2" hdrgm:HDRCapacityMax="2"'))


class Apply(unittest.TestCase):
    def setUp(self):
        self.base = np.random.default_rng(0).random((4, 6, 3), dtype=np.float32)

    def test_a_zero_map_changes_nothing(self):
        out = gm.apply(self.base, gm.GainMap("test", meta(gain_min=0.0, gain_max=0.0), flat(0.7)))
        np.testing.assert_array_equal(out, self.base)

    def test_a_flat_map_is_a_pure_brightness_change(self):
        for value, stops in ((1.0, 1.0), (0.5, 0.5), (0.25, 0.25)):
            with self.subTest(value=value):
                out = gm.apply(self.base, gm.GainMap("test", meta(gain_max=1.0), flat(value)))
                np.testing.assert_allclose(out, self.base * 2**stops, rtol=1e-6)

    def test_offsets_and_gamma_as_iso_writes_them(self):
        m = meta(gain_min=-0.5, gain_max=1.5, gamma=0.5, base_offset=1 / 64, alt_offset=1 / 32)
        out = gm.apply(self.base, gm.GainMap("test", m, flat(0.36)))
        stops = -0.5 + 2.0 * 0.36 ** 2
        np.testing.assert_allclose(out, (self.base + 1 / 64) * 2**stops - 1 / 32, rtol=1e-5)

    def test_a_map_from_hdr_leads_down(self):
        m = meta(gain_max=2.0, base_headroom=2.0, alt_headroom=0.0)
        np.testing.assert_allclose(gm.apply(self.base * 4, gm.GainMap("test", m, flat(1.0))), self.base, rtol=1e-6)

    def test_equal_headrooms_leave_the_map_unused(self):
        m = meta(gain_max=2.0, base_headroom=1.0, alt_headroom=1.0)
        np.testing.assert_array_equal(gm.apply(self.base, gm.GainMap("test", m, flat(1.0))), self.base)

    def test_three_channels_each_their_own(self):
        m = gm.Metadata((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (1.0,) * 3, (0.0,) * 3, (0.0,) * 3, 0.0, 3.0)
        out = gm.apply(self.base, gm.GainMap("test", m, flat(1.0, channels=3)))
        np.testing.assert_allclose(out, self.base * np.array([2, 4, 8], np.float32), rtol=1e-6)

    def test_apples_formula(self):
        m = gm.Metadata((0.0,), (0.0,), (1.0,), (0.0,), (0.0,), 0.0, 2.0, apple_headroom=4.0)
        for value in (0.0, 0.5, 1.0):
            with self.subTest(value=value):
                out = gm.apply(self.base, gm.GainMap("Apple", m, flat(value)))
                np.testing.assert_allclose(out, self.base * (1 + 3 * srgb_decode(np.float32(value))), rtol=1e-6)

    def test_a_smaller_map_is_stretched_to_the_photo(self):
        small = np.zeros((2, 3, 1), np.float32)
        small[:, 1:] = 1.0
        out = gm.apply(np.ones((4, 6, 3), np.float32), gm.GainMap("test", meta(gain_max=1.0), small))
        self.assertEqual(out.shape, (4, 6, 3))
        self.assertEqual((out[0, 0, 0], out[0, -1, 0]), (1.0, 2.0))


class Jpeg(unittest.TestCase):
    def setUp(self):
        self.main = np.full((48, 64, 3), 128, np.uint8)
        self.gain = np.zeros((24, 32), np.uint8)
        self.gain[:, 16:] = 255                                   # the right half boosted
        self.xmp = b'<x hdrgm:Version="1.0" hdrgm:GainMapMax="1" hdrgm:HDRCapacityMax="1" hdrgm:OffsetSDR="0" ' \
                   b'hdrgm:OffsetHDR="0"/>'

    def test_ultra_hdr_through_its_mpf_index(self):
        g = gm.read(ultra_hdr(self.main, self.gain, self.xmp))
        self.assertEqual((g.form, g.pixels.shape, g.metadata.gain_max), ("Ultra HDR", (24, 32, 1), (1.0,)))
        self.assertLess(g.pixels[:, :12].max(), 0.05)
        self.assertGreater(g.pixels[:, 20:].min(), 0.95)

    def test_iso_comes_before_the_vendors(self):   # D14
        iso = iso_payload((0.0,), (2.0,), (1.0,), (0.0,), (0.0,), 0.0, 2.0)
        g = gm.read(ultra_hdr(self.main, self.gain, self.xmp, iso=iso))
        self.assertEqual((g.form, g.metadata.gain_max), ("ISO 21496-1", (2.0,)))
        both = gm.read_all(ultra_hdr(self.main, self.gain, self.xmp, iso=iso))
        self.assertEqual([x.form for x in both], ["ISO 21496-1", "Ultra HDR"])

    def test_a_jpeg_without_a_second_image_has_no_map(self):
        self.assertIsNone(gm.read(jpeg(self.main)))
        self.assertIsNone(gm.read(b"\x89PNG\r\n\x1a\n"))

    def test_orientation_follows_the_main_image(self):
        """The map comes as stored, like a JPEG's main image before canonicalization turns it; turned by the same
        EXIF orientation, the boosted half lands where the main image's right half lands (the top, for 6)."""
        main = self.main.copy()
        main[:, 32:] = 200                                         # the main image's right half brighter
        data = ultra_hdr(main, self.gain, self.xmp, orientation=6)
        turn = cn.applied_orientation(data, "x.jpg")
        self.assertEqual(turn, 6)
        upright_main = cn.decode(data, "x.jpg").pixels
        upright_map = cn.orient(gm.read(data).pixels, turn)
        self.assertEqual(upright_map.shape[0] / upright_map.shape[1], upright_main.shape[0] / upright_main.shape[1])
        bright = upright_main[..., 0] > 160
        rows = np.linspace(0, upright_map.shape[0] - 1, upright_main.shape[0]).round().astype(int)
        boosted = upright_map[rows][:, :, 0].mean(axis=1) > 0.5
        np.testing.assert_array_equal(boosted, bright.mean(axis=1) > 0.5)

    def test_libultrahdrs_files_read_and_agree_with_libultrahdr(self):
        """A file libultrahdr writes (ISO 21496-1) gives, through this module, the light libultrahdr itself decodes:
        an HDR picture twice the standard-range one, and a brighter patch with its own boost."""
        h, w = 64, 96
        sdr = np.full((h, w, 4), 255, np.uint8)
        sdr[..., :3] = 128
        sdr[16:48, 24:72, :3] = 90
        lin = srgb_decode(sdr[..., :3].astype(np.float32) / 255)
        hdr = np.ones((h, w, 4), np.float16)
        hdr[..., :3] = lin * 2
        hdr[16:48, 24:72, :3] = lin[16:48, 24:72] * 4
        data = ic.ultrahdr_encode(hdr, sdr=sdr, level=95)
        g = gm.read(data)
        self.assertEqual(g.form, "ISO 21496-1")
        base = srgb_decode(ic.jpeg8_decode(data).astype(np.float32) / 255)
        ours = gm.apply(base, g)
        theirs = ic.ultrahdr_decode(data).astype(np.float32)[..., :3]
        inner = (slice(20, 44), slice(28, 68))
        self.assertLess(np.abs(ours[inner] / theirs[inner] - 1).max(), 0.01)
        self.assertLess(np.abs(ours[:8, :8] / theirs[:8, :8] - 1).max(), 0.01)
        np.testing.assert_allclose(np.median(ours[inner] / base[inner]), 4, rtol=0.02)
        np.testing.assert_allclose(np.median(ours[:8, :8] / base[:8, :8]), 2, rtol=0.02)


class Containers(unittest.TestCase):
    def test_an_avif_without_a_tmap_has_none(self):
        self.assertIsNone(gm.read(ic.avif_encode(np.full((16, 16, 3), 128, np.uint8))))

    @real_data
    def test_an_iphone_heic_reads_its_tmap_turned_like_its_main_image(self):
        """The dataset's iPhone HEICs carry Apple's map and an ISO 21496-1 tmap item (D14: the ISO one is read)."""
        import pillow_heif
        root = Path(__file__).resolve().parents[1] / "dataset"
        path = next(root.glob("2026-08-25__iphone/*/IMG_6301D.HEIC"))
        data = path.read_bytes()
        g = gm.read(data)
        self.assertEqual((g.form, g.metadata.use_base_colour_space, g.metadata.leads_down), ("ISO 21496-1", True, False))
        w, h = pillow_heif.open_heif(io.BytesIO(data)).size
        self.assertAlmostEqual(g.pixels.shape[0] / g.pixels.shape[1], h / w, places=3)


if __name__ == "__main__":
    unittest.main()
