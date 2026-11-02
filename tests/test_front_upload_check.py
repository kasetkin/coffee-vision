"""The front's format check agrees with what the inference host decodes (ticket OPS-15 P1). Plain unittest.

    python -m unittest tests.test_front_upload_check

webapp/front/upload_check.py runs on the front, which has the standard library only (D7), so it cannot import
coffeecv.photo_io: this holds its RAW extensions to photo_io's, and checks that every format the public contract
names (webapp/static/llms.txt), written by Pillow here, is both accepted by the front and decoded by load_rgb_image.
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from coffeecv import photo_io
from tests._tiers import real_data

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "front"))
import upload_check  # noqa: E402

# Pillow's name for each format the contract names, with the extension a phone or camera gives it.
PILLOW_FORMATS = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "HEIF": ".heic", "AVIF": ".avif", "JXL": ".jxl"}
# A RAW file's first bytes, by extension: the signature of the family LibRaw identifies it by.
TIFF_LE, TIFF_BE = b"II*\x00\x08\x00\x00\x00", b"MM\x00*\x00\x00\x00\x08"
RAW_HEADS = {
    ".dng": TIFF_LE, ".cr2": TIFF_LE + b"CR\x02\x00", ".nef": TIFF_BE, ".arw": TIFF_LE, ".pef": TIFF_LE,
    ".srw": TIFF_LE, ".orf": b"IIRO\x08\x00\x00\x00", ".rw2": b"IIU\x00\x18\x00\x00\x00",
    ".cr3": b"\x00\x00\x00\x18ftypcrx \x00\x00\x00\x01crx isom", ".raf": b"FUJIFILMCCD-RAW 0201FF383501",
}


def pillow_bytes(fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 24), (120, 80, 40)).save(buf, format=fmt)
    return buf.getvalue()


def head(data: bytes) -> bytes:
    return data[:upload_check.HEAD_BYTES]


class SameRawExtensionsAsTheDecoder(unittest.TestCase):
    def test_raw_extensions_are_photo_ios(self):
        self.assertEqual(upload_check.RAW_EXTENSIONS, frozenset(photo_io.RAW_EXTENSIONS))

    def test_every_raw_extension_has_a_test_header(self):
        self.assertEqual(set(RAW_HEADS), set(photo_io.RAW_EXTENSIONS))


class AcceptsWhatTheDecoderReads(unittest.TestCase):
    def test_each_pillow_format_is_accepted_and_decoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fmt, ext in PILLOW_FORMATS.items():
                with self.subTest(format=fmt):
                    data = pillow_bytes(fmt)
                    self.assertIsNone(upload_check.refusal(f"photo{ext}", head(data)))
                    path = Path(tmp) / f"photo{ext}"
                    path.write_bytes(data)
                    self.assertEqual(photo_io.load_rgb_image(path).shape, (24, 32, 3))

    def test_content_decides_not_the_name(self):
        # Pillow opens by content, so a PNG named .jpg is read; the front lets it through too.
        self.assertIsNone(upload_check.refusal("photo.jpg", head(pillow_bytes("PNG"))))
        self.assertIsNone(upload_check.refusal("photo", head(pillow_bytes("JPEG"))))

    def test_each_raw_extension_with_its_header_is_accepted(self):
        for ext, raw_head in RAW_HEADS.items():
            for name in (f"IMG_0001{ext}", f"IMG_0001{ext.upper()}"):
                with self.subTest(name=name):
                    self.assertIsNone(upload_check.refusal(name, raw_head + bytes(64)))

    def test_a_tiff_based_raw_may_be_big_endian(self):
        self.assertIsNone(upload_check.refusal("DSC_0001.nef", TIFF_BE + bytes(64)))
        self.assertIsNone(upload_check.refusal("DSC_0001.dng", TIFF_BE + bytes(64)))


class RefusesWhatItCannotBe(unittest.TestCase):
    def assert_refused(self, filename: str, data: bytes):
        reason = upload_check.refusal(filename, head(data))
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)

    def test_a_renamed_text_file(self):
        self.assert_refused("photo.jpg", b"Dear visitor, this is not a photo.\n" * 4)

    def test_an_empty_file(self):
        self.assert_refused("photo.jpg", b"")

    def test_a_truncated_header(self):
        self.assert_refused("photo.jpg", b"\xff\xd8")
        self.assert_refused("photo.png", pillow_bytes("PNG")[:5])
        self.assert_refused("photo.webp", pillow_bytes("WEBP")[:10])

    def test_formats_the_contract_does_not_name(self):
        for fmt in ("GIF", "BMP"):
            with self.subTest(format=fmt):
                self.assert_refused(f"photo.{fmt.lower()}", pillow_bytes(fmt))

    def test_a_plain_tiff_which_the_contract_does_not_name(self):
        self.assert_refused("scan.tif", TIFF_LE + bytes(64))
        self.assert_refused("scan.jpg", TIFF_BE + bytes(64))

    def test_a_raw_extension_on_another_format(self):
        # The app hands a .nef to rawpy whatever it holds, which cannot read a JPEG.
        self.assert_refused("DSC_0001.nef", pillow_bytes("JPEG"))
        self.assert_refused("IMG_0001.cr3", pillow_bytes("HEIF"))
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


if __name__ == "__main__":
    unittest.main()
