"""The front's format check refuses what is in no accepted format, and lets through what canonicalization decodes
(tickets OPS-15 P1, OPS-18; ADR 0034). Plain unittest.

    python -m unittest tests.test_front_upload_check

webapp/front/upload_check.py words the refusal; the recogniser it calls is coffeecv.canonical's, the one decode
dispatches on (OPS-18 D17). So this checks that every accepted format, written here, is both let through by the
front and decoded by canonical.decode (D7), and that a plain TIFF and the near misses of each signature are refused.
"""
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import imagecodecs as ic
import numpy as np
import pillow_heif
from PIL import Image

from coffeecv import canonical as cn
from tests._tiers import real_data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "front"))
import upload_check  # noqa: E402

# A RAW file's first bytes, by extension: the signature of the family LibRaw identifies it by.
TIFF_LE, TIFF_BE = b"II*\x00\x08\x00\x00\x00", b"MM\x00*\x00\x00\x00\x08"
RAW_HEADS = {
    ".dng": TIFF_LE, ".cr2": TIFF_LE + b"CR\x02\x00", ".nef": TIFF_BE, ".arw": TIFF_LE, ".pef": TIFF_LE,
    ".srw": TIFF_LE, ".orf": b"IIRO\x08\x00\x00\x00", ".rw2": b"IIU\x00\x18\x00\x00\x00",
    ".cr3": b"\x00\x00\x00\x18ftypcrx \x00\x00\x00\x01crx isom", ".raf": b"FUJIFILMCCD-RAW 0201FF383501",
}
PIXELS = np.full((24, 32, 3), (120, 80, 40), np.uint8)


def pillow_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(PIXELS).save(buf, format=fmt)
    return buf.getvalue()


def heic(major: bytes = b"heic", compatible: bytes = b"mif1heicmiaf") -> bytes:
    """A HEIC from pillow-heif, its ftyp box's brands replaced (same length: the box keeps its size)."""
    buf = io.BytesIO()
    pillow_heif.from_bytes("RGB", (32, 24), PIXELS.tobytes()).save(buf, format="HEIF", quality=90)
    data = buf.getvalue()
    assert data[4:8] == b"ftyp" and data[16:28] == b"mif1heicmiaf", data[:32]
    return data[:8] + major + data[12:16] + compatible + data[28:]


# One photo per accepted format but RAW, with the extension a phone, camera or editor gives it.
SAMPLES = {
    "jpeg": (".jpg", lambda: pillow_bytes("JPEG")), "png": (".png", lambda: pillow_bytes("PNG")),
    "webp": (".webp", lambda: pillow_bytes("WEBP")), "heif": (".heic", heic),
    "avif": (".avif", lambda: pillow_bytes("AVIF")), "jxl": (".jxl", lambda: ic.jpegxl_encode(PIXELS, level=95)),
    "gif": (".gif", lambda: pillow_bytes("GIF")), "bmp": (".bmp", lambda: pillow_bytes("BMP")),
}


def head(data: bytes) -> bytes:
    return data[:upload_check.HEAD_BYTES]


class AcceptsWhatCanonicalizationDecodes(unittest.TestCase):
    def test_every_accepted_format_has_a_sample_here(self):
        self.assertEqual(set(SAMPLES) | {f.id for f in cn.ACCEPTED_FORMATS.values() if f.is_raw},
                         set(cn.ACCEPTED_FORMATS))
        self.assertEqual(set(RAW_HEADS), cn.RAW_EXTENSIONS)

    def test_each_format_is_let_through_and_decoded(self):   # D7
        for fid, (ext, make) in SAMPLES.items():
            with self.subTest(format=fid):
                data = make()
                self.assertIsNone(upload_check.refusal(f"photo{ext}", head(data)))
                self.assertEqual(cn.decode(data, f"photo{ext}").pixels.shape[:2], (24, 32))

    def test_gif_and_bmp(self):   # OPS-18 D2
        for fmt in ("GIF", "BMP"):
            with self.subTest(format=fmt):
                self.assertIsNone(upload_check.refusal(f"photo.{fmt.lower()}", head(pillow_bytes(fmt))))

    def test_a_heif_by_a_hevm_major_or_a_compatible_heic_brand(self):   # D18
        # The second: major brand isom, heic and mif1 compatible only (libheif needs mif1 to read it as an image).
        for data in (heic(major=b"hevm"), heic(major=b"isom")):
            with self.subTest(brands=data[8:28]):
                self.assertIsNone(upload_check.refusal("IMG_0001.HEIC", head(data)))
                self.assertEqual(cn.decode(data, "IMG_0001.HEIC").pixels.shape[:2], (24, 32))

    def test_content_decides_not_the_name(self):
        # Canonicalization opens by content, so a PNG named .jpg is read; the front lets it through too.
        self.assertIsNone(upload_check.refusal("photo.jpg", head(pillow_bytes("PNG"))))
        self.assertIsNone(upload_check.refusal("photo", head(pillow_bytes("JPEG"))))
        self.assertIsNone(upload_check.refusal("photo.bmp", head(pillow_bytes("GIF"))))

    def test_each_raw_extension_with_its_header_is_accepted(self):
        for ext, raw_head in RAW_HEADS.items():
            for name in (f"IMG_0001{ext}", f"IMG_0001{ext.upper()}"):
                with self.subTest(name=name):
                    self.assertIsNone(upload_check.refusal(name, raw_head + bytes(64)))

    def test_a_tiff_based_raw_may_be_big_endian(self):
        self.assertIsNone(upload_check.refusal("DSC_0001.nef", TIFF_BE + bytes(64)))
        self.assertIsNone(upload_check.refusal("DSC_0001.dng", TIFF_BE + bytes(64)))


class RefusesWhatItCannotBe(unittest.TestCase):
    def assert_refused(self, filename: str, data: bytes) -> str:
        reason = upload_check.refusal(filename, head(data))
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)
        return reason

    def test_the_message_names_the_accepted_formats(self):   # D20
        self.assertEqual(self.assert_refused("photo.jpg", b"not a photo" * 8),
                         "not a JPEG, PNG, WebP, HEIC/HEIF, AVIF, JPEG XL, GIF, BMP or camera RAW file")

    def test_a_renamed_text_file(self):
        self.assert_refused("photo.jpg", b"Dear visitor, this is not a photo.\n" * 4)

    def test_an_empty_file(self):
        self.assert_refused("photo.jpg", b"")

    def test_a_truncated_header(self):
        self.assert_refused("photo.jpg", b"\xff\xd8")
        self.assert_refused("photo.png", pillow_bytes("PNG")[:5])
        self.assert_refused("photo.webp", pillow_bytes("WEBP")[:10])
        self.assert_refused("photo.gif", b"GIF8")

    def test_a_plain_tiff(self):   # D2: refused for now
        self.assert_refused("scan.tif", TIFF_LE + bytes(64))
        self.assert_refused("scan.jpg", TIFF_BE + bytes(64))
        self.assert_refused("scan.tiff", pillow_bytes("TIFF"))

    def test_a_file_that_starts_bm_and_is_not_a_bmp(self):   # R4
        self.assert_refused("photo.bmp", b"BM" + b"Dear visitor, this is not a photo.\n" * 2)
        self.assert_refused("photo.bmp", b"BM" + bytes(12) + (41).to_bytes(4, "little") + bytes(46))

    def test_other_formats_pillow_reads(self):   # ADR 0034: refused though a decoder exists
        for fmt in ("PCX", "TGA", "PPM", "ICO"):
            with self.subTest(format=fmt):
                self.assert_refused(f"photo.{fmt.lower()}", pillow_bytes(fmt))

    def test_a_raw_extension_on_another_format(self):
        # rawpy is handed a .nef whatever it holds, and cannot read a JPEG.
        self.assertEqual(self.assert_refused("DSC_0001.nef", pillow_bytes("JPEG")),
                         "the file is named .nef but is not a NEF camera RAW file")
        self.assert_refused("IMG_0001.cr3", heic())
        self.assert_refused("DSCF0001.raf", TIFF_LE + bytes(64))

    def test_an_iso_media_file_that_is_not_an_image(self):
        self.assert_refused("photo.heic", b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + bytes(64))

    def test_head_bytes_is_small(self):
        self.assertLessEqual(upload_check.HEAD_BYTES, 4096)


@real_data
class AcceptsRealPhotos(unittest.TestCase):
    def test_one_real_photo_per_extension(self):
        for ext in ("jpg", "JPG", "HEIC", "heic", "heif", "dng", "png"):
            photo = next((p for p in sorted((REPO / "dataset").rglob(f"*.{ext}")) if p.is_file()), None)
            if photo is None:
                continue
            with self.subTest(photo=str(photo.relative_to(REPO))), photo.open("rb") as f:
                self.assertIsNone(upload_check.refusal(photo.name, f.read(upload_check.HEAD_BYTES)))

    def test_a_real_raw_photo_is_let_through_and_decoded(self):   # D7: the RAW rows' decoder, on the one RAW kind
        # the dataset holds (DNG); the other RAW families are checked by their signatures only.
        photo = next((p for p in sorted((REPO / "dataset").rglob("*")) if p.suffix.lower() == ".dng"), None)
        if photo is None:
            self.skipTest("no DNG under dataset/ (dvc pull)")
        data = photo.read_bytes()
        self.assertIsNone(upload_check.refusal(photo.name, head(data)))
        src = cn.decode(data, photo.name)
        self.assertEqual((src.decoder, src.pixels.ndim, src.pixels.shape[2]), ("rawpy16", 3, 3))


if __name__ == "__main__":
    unittest.main()
