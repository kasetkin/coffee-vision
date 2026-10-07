"""coffeecv.strip_metadata (ticket ML-3, D13): private tags go, pixels and every other tag stay, and a photo
that fails any check keeps its original bytes. Plain unittest, on JPEG and HEIC fixtures written to a
temp dir (Orientation 6, an sRGB ICC profile, GPS, an artist and a serial number), plus one real Pixel DNG
when the DVC data is present.

    python -m unittest discover -s tests -p 'test_strip_metadata.py'

Needs exiftool (libimage-exiftool-perl, in the devcontainer).
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pillow_heif
from PIL import Image, ImageCms

from coffeecv import strip_metadata as sm
from coffeecv.config import REPO_ROOT

from tests._tiers import real_data

HAVE_EXIFTOOL = shutil.which("exiftool") is not None
DNG = REPO_ROOT / "dataset/2026-07-24__first_pictures/dng/PXL_20260724_114105353.RAW-02.ORIGINAL.dng"
PRIVATE = ["-GPSLatitude=48.1", "-GPSLatitudeRef=N", "-GPSLongitude=11.5", "-GPSLongitudeRef=E",
           "-Artist=Someone", "-SerialNumber=SN123", "-ExifIFD:DateTimeOriginal=2026:10:05 10:00:00"]


def write_fixture(path: Path) -> Path:
    """A small photo with kept fields (Orientation, ICC, Make, Model, DateTimeOriginal) and private ones."""
    pillow_heif.register_heif_opener()
    im = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 96, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112], exif[0x010F], exif[0x0110] = 6, "TestMake", "TestModel"
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    im.save(path, exif=exif.tobytes(), icc_profile=icc, quality=90)
    subprocess.run(["exiftool", "-q", "-overwrite_original", *PRIVATE, str(path)], check=True)
    return path


@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool is not installed (libimage-exiftool-perl)")
class TestStrip(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def assert_clean_and_lossless(self, path: Path):
        before = sm.read_tags(path)
        pixels = sm.pixel_digest(path)
        manifest = self.tmp / "manifest.csv"
        self.assertEqual(sm.main([str(path), "--manifest", str(manifest)]), 0)
        after = sm.read_tags(path)
        self.assertEqual([k for k in after if sm.is_denied(k)], [])
        self.assertEqual(sm.pixel_digest(path), pixels)
        for tag in ("Orientation", "ICC_Profile", "Make", "Model", "DateTimeOriginal"):
            keys = [k for k in before if k.endswith(f":{tag}") and sm.is_compared(k)]
            self.assertTrue(keys, tag)
            for k in keys:
                self.assertEqual(after.get(k), before[k], k)
        [row] = list(csv.DictReader(line for line in manifest.read_text().splitlines() if not line.startswith("#")))
        self.assertEqual(row["sha256_after"], sm.sha256_file(path))
        self.assertNotEqual(row["sha256_before"], row["sha256_after"])
        self.assertIn("EXIF:GPS:GPSLatitude", row["removed"].split())
        self.assertIn("EXIF:IFD0:Artist", row["removed"].split())
        # A second run finds the photo in the manifest with the recorded bytes and leaves it alone.
        self.assertEqual(sm.main([str(path), "--manifest", str(manifest)]), 0)
        self.assertEqual(len(manifest.read_text().splitlines()), 5)   # 3 header lines, the columns, one row

    def test_jpeg(self):
        self.assert_clean_and_lossless(write_fixture(self.tmp / "a.jpg"))

    def test_heic(self):
        self.assert_clean_and_lossless(write_fixture(self.tmp / "a.heic"))

    def check_fails_untouched(self, path: Path, expect: str, **patches):
        sha = sm.sha256_file(path)
        with mock.patch.multiple(sm, **patches):
            row = sm.strip_one(path, dry_run=False)
        self.assertTrue(any(expect in p for p in row["problems"]), row["problems"])
        self.assertEqual(sm.sha256_file(path), sha)
        self.assertEqual([p.name for p in self.tmp.iterdir() if p.name.startswith(".")], [])  # no temp left

    def test_pixel_difference_leaves_the_original(self):
        path = write_fixture(self.tmp / "a.jpg")
        real = sm.pixel_digest
        self.check_fails_untouched(path, "decoded pixels differ",
                                   pixel_digest=lambda p: real(p) if p == path else "different")

    def test_surviving_deny_listed_tag_fails(self):
        path = write_fixture(self.tmp / "a.jpg")
        self.check_fails_untouched(path, "deny-listed tag survived: EXIF:GPS",
                                   DELETE_ARGS=[a for a in sm.DELETE_ARGS if not a.startswith("-gps")])

    def test_dropped_orientation_fails(self):
        path = write_fixture(self.tmp / "a.jpg")
        self.check_fails_untouched(path, "kept field lost: EXIF:IFD0:Orientation",
                                   DELETE_ARGS=sm.DELETE_ARGS + ["-ifd0:orientation="])

    def test_dropped_icc_profile_fails(self):
        path = write_fixture(self.tmp / "a.jpg")
        self.check_fails_untouched(path, "kept field lost: ICC_Profile:ICC_Profile",
                                   DELETE_ARGS=sm.DELETE_ARGS + ["-icc_profile:all="])

    def test_dry_run_changes_nothing(self):
        path = write_fixture(self.tmp / "a.jpg")
        sha = sm.sha256_file(path)
        row = sm.strip_one(path, dry_run=True)
        self.assertEqual(row["problems"], [])
        self.assertTrue(row["removed"])
        self.assertEqual(sm.sha256_file(path), sha)

    @real_data
    @unittest.skipUnless(DNG.exists(), "the 2026-07-24 DNGs are not present (dvc pull)")
    def test_pixel_dng(self):
        """A Pixel DNG loses GPS and its artist with its sensor data and rendering unchanged. Its IFD0
        preview is image data exiftool cannot remove; the owner keeps it (2026-10-05), so it is noted, not
        failed -- and nothing else deny-listed may stay."""
        path = self.tmp / DNG.name
        shutil.copyfile(DNG, path)
        # The dataset's DNGs are stripped since 2026-10-05: give the copy GPS and an artist back first.
        subprocess.run(["exiftool", "-q", "-overwrite_original", *PRIVATE[:5], str(path)], check=True)
        row = sm.strip_one(path, dry_run=True)
        self.assertIn("EXIF:GPS:GPSLatitude", row["removed"].split())
        self.assertEqual(row["problems"], [])
        self.assertEqual(row["notes"], ["RAW preview image stays (owner, 2026-10-05)"])

if __name__ == "__main__":
    unittest.main()
