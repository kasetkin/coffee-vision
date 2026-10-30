"""coffeecv.strip_metadata (ticket ML-3, D13): private tags go, pixels and every other tag stay, and a photo
that fails any check keeps its original bytes. Plain unittest, on JPEG and HEIC fixtures written to a
temp dir (Orientation 6, an sRGB ICC profile, GPS, an artist and a serial number), plus one real Pixel DNG
and one real iPhone 16 HEIC when the DVC data is present.

    python -m unittest discover -s tests -p 'test_strip_metadata.py'

Needs exiftool (libimage-exiftool-perl, in the devcontainer).
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pillow_heif
from PIL import Image, ImageCms

from coffeecv import strip_metadata as sm
from coffeecv.config import REPO_ROOT

from tests._tiers import real_data

HAVE_EXIFTOOL = shutil.which("exiftool") is not None
DNG = REPO_ROOT / "dataset/2026-07-24__first_pictures/dng/PXL_20260724_114105353.RAW-02.ORIGINAL.dng"
# ML-6 D10: two of its auxiliary images are 10-bit, which pillow-heif cannot decode.
IPHONE16 = REPO_ROOT / "dataset/segmenter_positives/IMG_7252.HEIC"
# 16320x12240, above Pillow's own limit (tests/test_large_photos.py).
GALAXY_200MP = REPO_ROOT / "dataset/segmenter_positives/20261028_112339_Galaxy_Z_Fold8_Ultra_Brazil.jpg"
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


def manifest_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith("#")))


def run_main(*argv: str) -> tuple[int, str]:
    """sm.main's exit code and what it printed on stdout."""
    with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()):
        code = sm.main(list(argv))
    return code, out.getvalue()


def samsung_trailer(blocks: dict[int, tuple[bytes, bytes]]) -> bytes:
    """A Samsung SEF trailer as a Galaxy appends it after a JPEG: {type: (name, value)} blocks, the SEFH
    directory pointing back at them, its length and SEFT (the layout of the 2026-10-23 photos, made-up values)."""
    body, entries = b"", []
    for typ, (name, value) in blocks.items():
        block = struct.pack("<HHI", 0, typ, len(name)) + name + value
        entries.append((typ, len(body), len(block)))
        body += block
    directory = b"SEFH" + struct.pack("<II", 107, len(entries)) + b"".join(
        struct.pack("<HHII", 0, typ, len(body) - start, size) for typ, start, size in entries)
    return body + directory + struct.pack("<I", len(directory)) + b"SEFT"


SAMSUNG = {0x0A01: (b"Image_UTC_Data", b"1700000000000"), 0x0AA1: (b"MCC_Data", b"262")}


def flip_coded_byte(path: Path, item_id: int) -> None:
    """Change one bit in the middle of a HEIF item's coded data, stored in the file (iloc method 0)."""
    method, _, [(off, length), *_] = sm._heif_meta(path.read_bytes())["extents"][item_id]
    assert method == 0, method
    b = bytearray(path.read_bytes())
    b[off + length // 2] ^= 1
    path.write_bytes(bytes(b))


class TestImports(unittest.TestCase):
    """ML-14 D8: a worker spawned for the strip imports the module afresh; it must not pay for torch, and it
    must get the dataset's pixel limit, not Pillow's."""

    def test_no_torch_and_the_dataset_pixel_limit(self):
        code = ("import json, sys\nfrom PIL import Image\nimport coffeecv.strip_metadata\n"
                "from coffeecv.photo_io import DATASET_MAX_PIXELS\n"
                "print(json.dumps(['torch' in sys.modules, Image.MAX_IMAGE_PIXELS, DATASET_MAX_PIXELS]))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(json.loads(out.stdout.splitlines()[-1]), [False, 2**28, 2**28])

    @real_data
    @unittest.skipUnless(GALAXY_200MP.exists(), "the 2026-10-28 200 MP photo is not present (dvc pull)")
    def test_a_200_mp_photo_digests(self):
        self.assertRegex(sm.pixel_digest(GALAXY_200MP), r"^[0-9a-f]{64}$")


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

    def test_samsung_trailer_is_cut(self):
        """2026-10-23: a Galaxy JPEG ends with an SEF trailer exiftool reads but cannot delete; it is cut, and
        the photo before it is what exiftool wrote."""
        path = write_fixture(self.tmp / "a.jpg")
        path.write_bytes(path.read_bytes() + samsung_trailer(SAMSUNG))
        self.assertIn("MakerNotes:Samsung:MCCData", sm.read_tags(path))
        self.assert_clean_and_lossless(path)
        self.assertEqual(sm.samsung_trailer_start(path.read_bytes()), None)
        self.assertTrue(path.read_bytes().endswith(b"\xff\xd9"))

    def test_trailer_start_is_its_earliest_block(self):
        trailer = samsung_trailer(SAMSUNG)
        self.assertEqual(sm.samsung_trailer_start(b"jpeg" + trailer), 4)
        self.assertEqual(sm.samsung_trailer_start(b"jpeg\xff\xd9"), None)

    def test_malformed_samsung_trailer_fails(self):
        path = write_fixture(self.tmp / "a.jpg")
        trailer = bytearray(samsung_trailer(SAMSUNG))
        trailer[-8] += 1                                # the directory's length no longer matches
        path.write_bytes(path.read_bytes() + bytes(trailer))
        sha = sm.sha256_file(path)
        row = sm.strip_one(path, dry_run=False)
        self.assertTrue(any("Samsung trailer" in p for p in row["problems"]), row["problems"])
        self.assertEqual(sm.sha256_file(path), sha)

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

    def test_undecodable_aux_image_compared_by_stored_bytes(self):
        """An auxiliary image pillow-heif cannot decode (ML-6 D10) counts by its stored bytes: the strip
        passes, and a changed byte in its coded data changes the digest. The fixture has no auxiliary image,
        so its primary item stands in for one."""
        path = write_fixture(self.tmp / "a.heic")
        primary = sm._heif_meta(path.read_bytes())["primary"]

        class Undecodable:
            def __init__(self, img):
                self._img, self.info = img, {**img.info, "aux": {"urn:test:aux": [primary]}}

            def __array__(self, dtype=None, copy=None):
                return np.asarray(self._img)

            def get_aux_image(self, aux_id):
                raise NotImplementedError("Only 8-bit AUX images are currently supported. Got 10-bit image.")

        heif = SimpleNamespace(open_heif=lambda *a, **k: [Undecodable(i) for i in pillow_heif.open_heif(*a, **k)])
        with mock.patch.object(sm, "pillow_heif", heif), \
                mock.patch.object(sm, "heif_stored_digest", wraps=sm.heif_stored_digest) as stored_digest:
            row = sm.strip_one(path, dry_run=True)
        self.assertEqual(row["problems"], [])
        self.assertIn("EXIF:GPS:GPSLatitude", row["removed"].split())
        self.assertTrue(stored_digest.called)
        stored = sm.heif_stored_digest(path, primary)
        flip_coded_byte(path, primary)
        self.assertNotEqual(sm.heif_stored_digest(path, primary), stored)

    @real_data
    @unittest.skipUnless(IPHONE16.exists(), "the 2026-10-11 iPhone HEIC is not present (dvc pull)")
    def test_iphone16_heic(self):
        """An iPhone 16 Pro Max HEIC with two 10-bit auxiliary images (Apple's style delta map, a grid of 30
        tiles, and its linear thumbnail) loses GPS and its maker notes with every image unchanged; a changed
        byte in the thumbnail, or in one tile of the style map, fails the check."""
        path = self.tmp / IPHONE16.name
        shutil.copyfile(IPHONE16, path)
        subprocess.run(["exiftool", "-q", "-overwrite_original", *PRIVATE[:5], str(path)], check=True)
        row = sm.strip_one(path, dry_run=True)
        self.assertIn("EXIF:GPS:GPSLatitude", row["removed"].split())
        self.assertEqual(row["problems"], [])
        meta = sm._heif_meta(path.read_bytes())
        for item in (97, meta["dimg"][83][5]):
            planted = self.tmp / f"planted_{item}.heic"
            shutil.copyfile(path, planted)
            flip_coded_byte(planted, item)
            self.assertNotEqual(sm.pixel_digest(planted), row["pixel_sha256"], item)

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


@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool is not installed (libimage-exiftool-perl)")
class TestRun(unittest.TestCase):
    """ML-14 P2: a run over several photos, one at a time (--jobs 1: a mock does not reach a spawned worker). An exception is a failing row (D5), the resume check
    comes before any work (D6), and each good row is in the manifest as soon as its photo is done (D10)."""

    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, cls.fixture_dir)
        cls.fixture = write_fixture(cls.fixture_dir / "fixture.jpg")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "photos").mkdir()
        self.photos = [Path(shutil.copyfile(self.fixture, self.tmp / "photos" / f"{c}.jpg")) for c in "abc"]
        self.manifest = self.tmp / "manifest.csv"

    def test_an_exception_is_a_failing_row_and_the_run_goes_on(self):
        real, b = sm.strip_one, self.photos[1]
        sha = sm.sha256_file(b)

        def strip_one(path, *args):
            if path == b:
                raise sm.StripError("exiftool could not read b.jpg")
            return real(path, *args)

        with mock.patch.object(sm, "strip_one", strip_one):
            code, out = run_main(str(self.tmp / "photos"), "--manifest", str(self.manifest), "--jobs", "1")
        self.assertEqual(code, 1)
        self.assertIn("b.jpg: FAIL error: StripError: exiftool could not read b.jpg", out)
        self.assertEqual([Path(r["path"]).name for r in manifest_rows(self.manifest)], ["a.jpg", "c.jpg"])
        self.assertEqual(sm.sha256_file(b), sha)

    def test_a_changed_manifest_photo_stops_the_run_before_any_work(self):
        a, b, c = self.photos
        self.assertEqual(run_main(str(c), "--manifest", str(self.manifest))[0], 0)
        c.write_bytes(c.read_bytes() + b"changed")             # no longer the recorded sha256_after
        manifest, shas = self.manifest.read_bytes(), [sm.sha256_file(p) for p in (a, b)]
        self.assertEqual(run_main(str(self.tmp / "photos"), "--manifest", str(self.manifest))[0], 1)
        self.assertEqual([sm.sha256_file(p) for p in (a, b)], shas)
        self.assertEqual(self.manifest.read_bytes(), manifest)

    def test_an_interrupted_run_keeps_the_rows_of_the_photos_done(self):
        real, c = sm.strip_one, self.photos[2]

        def strip_one(path, *args):
            if path == c:
                raise KeyboardInterrupt
            return real(path, *args)

        sha = sm.sha256_file(c)
        with mock.patch.object(sm, "strip_one", strip_one):
            code, out = run_main(str(self.tmp / "photos"), "--manifest", str(self.manifest), "--jobs", "1")
        self.assertEqual(code, 130)
        self.assertIn("STOPPED by Ctrl-C: 1 of 3 photos not done", out)
        rows = manifest_rows(self.manifest)
        self.assertEqual([Path(r["path"]).name for r in rows], ["a.jpg", "b.jpg"])
        self.assertEqual([r["sha256_after"] for r in rows], [sm.sha256_file(p) for p in self.photos[:2]])
        self.assertEqual(sm.sha256_file(c), sha)

    def test_a_new_manifests_start_time_is_the_runs_start(self):
        start = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
        clock = [start]
        real = sm.strip_one

        def strip_one(*args):                                   # each photo takes an hour on this clock
            clock[0] += timedelta(hours=1)
            return real(*args)

        fake = SimpleNamespace(now=lambda tz=None: clock[0])
        with mock.patch.object(sm, "datetime", fake), mock.patch.object(sm, "strip_one", strip_one):
            self.assertEqual(run_main(str(self.photos[0]), "--manifest", str(self.manifest))[0], 0)
        self.assertIn("# started 2026-10-30T10:00:00+00:00", self.manifest.read_text().splitlines())

    def test_a_run_with_no_good_row_writes_no_manifest(self):
        with mock.patch.object(sm, "DELETE_ARGS", sm.DELETE_ARGS + ["-ifd0:orientation="]):
            code, out = run_main(str(self.photos[0]), "--manifest", str(self.manifest))
        self.assertEqual(code, 1)
        self.assertIn("kept field lost", out)
        self.assertFalse(self.manifest.exists())



@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool is not installed (libimage-exiftool-perl)")
class TestMemoryCeiling(unittest.TestCase):
    """ML-14 P3, D13 (a): a photo whose decode would need more memory than the budget fails before it is
    decoded, its original untouched, rather than run the machine out of memory."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_a_photo_above_the_ceiling_fails_untouched(self):
        path, manifest = write_fixture(self.tmp / "a.jpg"), self.tmp / "manifest.csv"
        sha = sm.sha256_file(path)
        with mock.patch.object(sm, "pixel_digest", side_effect=AssertionError("decoded")):
            code, out = run_main(str(path), "--manifest", str(manifest), "--max-memory", "0.00001")
        self.assertEqual(code, 1)
        self.assertRegex(out, r"a\.jpg: FAIL too big: about [0-9.e-]+ GB to decode, budget 1e-05 GB \(--max-memory\)")
        self.assertEqual(sm.sha256_file(path), sha)
        self.assertFalse(manifest.exists())

    def test_estimate_per_format(self):
        """Measured on 2026-10-30 (ML-14 P3), peak memory above the imports: 10 bytes per pixel for a JPEG,
        12 for an RGBA PNG, 11 for a HEIC, 13 for a DNG. The estimate must not be below any of them."""
        pixels = 64 * 96
        im = Image.fromarray(np.zeros((64, 96, 4), np.uint8))
        im.save(self.tmp / "a.png")
        for path, at_least in [(write_fixture(self.tmp / "a.jpg"), 10), (self.tmp / "a.png", 12),
                               (write_fixture(self.tmp / "a.heic"), 11)]:
            self.assertGreaterEqual(sm.estimate_decode_bytes(path), at_least * pixels, path.name)
            self.assertLess(sm.estimate_decode_bytes(path), 16 * pixels, path.name)

    @real_data
    @unittest.skipUnless(DNG.exists(), "the 2026-07-24 DNGs are not present (dvc pull)")
    def test_estimate_of_a_dng_is_from_its_output_size(self):
        import rawpy
        with rawpy.imread(str(DNG)) as raw:
            pixels = raw.sizes.width * raw.sizes.height
        self.assertGreaterEqual(sm.estimate_decode_bytes(DNG), 13 * pixels)
        self.assertLess(sm.estimate_decode_bytes(DNG), 16 * pixels)

    @real_data
    @unittest.skipUnless(GALAXY_200MP.exists(), "the 2026-10-28 200 MP photo is not present (dvc pull)")
    def test_estimate_of_the_200_mp_photo_covers_its_measured_peak(self):
        self.assertGreaterEqual(sm.estimate_decode_bytes(GALAXY_200MP), 1.95e9)   # measured 2026-10-30



@unittest.skipUnless(HAVE_EXIFTOOL, "exiftool is not installed (libimage-exiftool-perl)")
class TestPool(unittest.TestCase):
    """ML-14 P4: several photos at once on spawned workers. A mock does not reach a spawned worker, so every
    behaviour checked in a worker is a real file's."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def mixed_set(self, folder: Path, real: tuple[Path, ...] = ()) -> Path:
        """JPEG and HEIC fixtures with private tags, an already clean JPEG, a truncated JPEG that fails, and the
        `real` photos with private tags added back."""
        folder.mkdir()
        whole = write_fixture(folder / "a.jpg").read_bytes()
        (folder / "a2.jpg").write_bytes(whole[:len(whole) // 2])         # decodes no further than its middle
        write_fixture(folder / "b.heic")
        Image.fromarray(np.random.default_rng(1).integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(folder / "c.jpg")
        for photo in real:
            shutil.copyfile(photo, folder / photo.name)
            subprocess.run(["exiftool", "-q", "-overwrite_original", *PRIVATE[:5], str(folder / photo.name)], check=True)
        return folder

    def run_on(self, folder: Path, *argv: str) -> tuple[int, str, list[dict], dict[str, str]]:
        """main's exit code, output and manifest rows with `folder` taken out, and each photo's sha256 by name."""
        manifest = folder.with_suffix(".csv")
        code, out = run_main(str(folder), *(["--dry-run"] if "--dry-run" in argv else ["--manifest", str(manifest)]),
                             *[a for a in argv if a != "--dry-run"])
        rows = manifest_rows(manifest) if manifest.exists() else []
        for r in rows:
            r["path"] = Path(r["path"]).name
        return (code, out.replace(str(folder), "<dir>"), rows,
                {p.name: sm.sha256_file(p) for p in sorted(folder.iterdir())})

    def assert_parallel_matches_serial(self, real: tuple[Path, ...] = ()):
        photos = self.mixed_set(self.tmp / "photos", real)
        for mode in ([], ["--dry-run"]):
            with self.subTest(mode=mode):
                runs = []
                for jobs in ("1", "3"):
                    folder = self.tmp / f"jobs{jobs}{''.join(mode)}"
                    shutil.copytree(photos, folder)
                    runs.append(self.run_on(folder, "--jobs", jobs, *mode))
                serial, parallel = runs
                self.assertEqual(parallel, serial)
                self.assertEqual(serial[0], 1)
                self.assertIn("a2.jpg: FAIL error: OSError: image file is truncated", serial[1])
                self.assertEqual(serial[1].count("\n[") + serial[1].startswith("["), 4 + len(real))
                self.assertEqual([r["path"] for r in serial[2]], [] if mode else sorted(set(serial[3]) - {"a2.jpg"}))

    @real_data
    def test_a_parallel_run_matches_a_serial_one(self):
        self.assert_parallel_matches_serial()

    @real_data
    @unittest.skipUnless(DNG.exists() and IPHONE16.exists(), "the DNG and iPhone HEIC are not present (dvc pull)")
    def test_a_parallel_run_matches_a_serial_one_on_real_photos(self):
        self.assert_parallel_matches_serial((DNG, IPHONE16))

    @real_data
    def test_an_exception_in_a_worker_is_a_failing_row(self):
        photos = self.mixed_set(self.tmp / "photos")
        sha = sm.sha256_file(photos / "a2.jpg")
        code, out, rows, _ = self.run_on(photos, "--jobs", "2")
        self.assertEqual(code, 1)
        self.assertRegex(out, r"<dir>/a2\.jpg: FAIL error: OSError: image file is truncated")
        self.assertEqual([r["path"] for r in rows], ["a.jpg", "b.heic", "c.jpg"])
        self.assertEqual(sm.sha256_file(photos / "a2.jpg"), sha)

    @real_data
    def test_a_ctrl_c_leaves_every_stripped_photo_with_its_row(self):
        """D4: a Ctrl-C at a terminal reaches the whole process group; the workers ignore it, the run cancels the
        photos not started, waits for those in progress and writes every row done."""
        photos, manifest = self.tmp / "photos", self.tmp / "manifest.csv"
        photos.mkdir()
        fixture = write_fixture(self.tmp / "fixture.jpg")
        for i in range(12):
            shutil.copyfile(fixture, photos / f"{i:02}.jpg")
        original = sm.sha256_file(fixture)
        cmd = [sys.executable, "-m", "coffeecv.strip_metadata", str(photos), "--manifest", str(manifest), "--jobs", "2"]
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                start_new_session=True)
        self.addCleanup(lambda: proc.poll() is None and os.killpg(proc.pid, signal.SIGKILL))
        self.assertTrue(proc.stdout.readline().startswith("[1/12] "))
        os.killpg(proc.pid, signal.SIGINT)
        out, err = proc.communicate(timeout=60)
        self.assertEqual(proc.returncode, 130, err)
        self.assertIn("stopping: waiting for the", err)
        self.assertIn("STOPPED by Ctrl-C", out)
        self.assertNotIn("FAIL", out)                          # the photos in progress finished, exiftool too
        rows = {Path(r["path"]).name: r for r in manifest_rows(manifest)}
        changed = {p.name for p in photos.iterdir() if sm.sha256_file(p) != original}
        self.assertEqual(set(rows), changed)
        self.assertLess(len(changed), 12)                       # the Ctrl-C did stop the run
        for name, r in rows.items():
            self.assertEqual(r["sha256_after"], sm.sha256_file(photos / name), name)
        self.assertEqual([p.name for p in photos.iterdir() if p.name.startswith(".")], [])
        # A rerun leaves the photos done alone and strips the rest.
        done = len(rows)
        self.assertEqual(run_main(str(photos), "--manifest", str(manifest), "--jobs", "2")[0], 0)
        rows = manifest_rows(manifest)
        self.assertEqual(sorted(Path(r["path"]).name for r in rows), [f"{i:02}.jpg" for i in range(12)])
        self.assertEqual(sum(1 for line in manifest.read_text().splitlines() if line.startswith("# started")), 1)
        self.assertEqual(len(rows) - done, 12 - done)
        self.assertTrue(all(sm.sha256_file(photos / Path(r["path"]).name) == r["sha256_after"] for r in rows))

if __name__ == "__main__":
    unittest.main()
