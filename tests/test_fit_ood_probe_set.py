"""What the shipped probe is fitted on (ADR 0016, ticket ML-5 D16): `fit_ood_probe` takes its negatives from
the segmenter dataset's training and validation splits, clean tags only, and drops from its bean photos any
photo that would share a D13 group with a test photo, which `ood_eval` measures it on (D4): the same bytes,
or a burst shot within seg_dataset.GROUP_SECONDS of one on the same clock. Plain unittest, on
tests/fixtures/ml5_seg_dataset.yaml with stand-in shot times; no model and no photo.

    python -m unittest tests.test_fit_ood_probe_set -v
"""
import hashlib
import unittest
from datetime import datetime
from pathlib import Path

from coffeecv import fit_ood_probe, seg_dataset
from coffeecv.config import REPO_ROOT

from tests._tiers import real_data

FIXTURE = Path(__file__).parent / "fixtures" / "ml5_seg_dataset.yaml"
POOL = "dataset/2026-09-11__pixel/class_001__Ethiopia_Sidamo/PXL_20260911_10"
SESSION = "dataset/2026-10-01__pixel/class_002__Kenya_Nyeri/"
TEST_POSITIVE = "dataset/segmenter_positives/PXL_20261001_120000000.jpg"
TEST_NEGATIVE = "dataset/ood_negatives/2026-09__internet_proxy/empty_tray/empty_tray_001.jpg"


def at(clock: str, hhmmss: str, day: str = "20261001") -> tuple[str, datetime]:
    return clock, datetime.strptime(day + hhmmss, "%Y%m%d%H%M%S")


# Shot times as seg_dataset.shot_time gives them; a path not here has none.
TIMES = {
    TEST_POSITIVE: at("utc", "120000"),
    TEST_NEGATIVE: at("utc", "080000"),
    SESSION + "PXL_burst.jpg": at("utc", "120100"),            # 60 s after a test photo: one burst
    SESSION + "DSC_other_clock.jpg": at("local sony", "120100"),  # the same reading on another camera's clock
    SESSION + "PXL_later.jpg": at("utc", "120500"),            # 300 s after
    SESSION + "PXL_after_negative.jpg": at("utc", "080030"),   # 30 s after a test negative: another subject
}
COPY = SESSION + "PXL_copy.jpg"                                # the bytes of a test pool photo


class FakeScan:
    """seg_dataset.scan(only=...) on the fixture: each asked-for photo's record, with TIMES' shot times."""

    def __init__(self, photos):
        self.listed = {e["path"]: e for e in photos}
        self.asked = None

    def __call__(self, only):
        self.asked = set(only)
        out = []
        for path in sorted(only):
            e = self.listed.get(path, {"source": "pool", "sha256": hashlib.sha256(path.encode()).hexdigest()})
            sha = self.listed[POOL + "0000000.jpg"]["sha256"] if path == COPY else e["sha256"]
            out.append({"path": path, "source": e["source"], "batch": e.get("batch"), "tag": e.get("tag"),
                        "url": "", "sha256": sha, "time": TIMES.get(path)})
        return out


class TestProbeFitSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.photos = seg_dataset.load_seg_dataset(FIXTURE)

    def test_negatives_are_training_and_validation_ones_with_a_clean_tag(self):
        self.assertEqual(
            [e["path"] for e in fit_ood_probe.probe_negatives(self.photos)],
            ["dataset/ood_negatives/2026-09__internet_proxy/empty_tray/empty_tray_003.jpg",
             "dataset/ood_negatives/2026-09-11__user_samerig/confusable_grain/PIC_20260911_132408.JPG"])

    def test_bean_photos_in_the_test_split_are_dropped(self):
        beans = [Path(POOL + s) for s in ("0000000.jpg", "0200000.jpg", "0300000.jpg", "9900000.jpg")]
        kept, dropped = fit_ood_probe.probe_beans(beans, self.photos, describe=FakeScan(self.photos))
        self.assertEqual(kept, [beans[2], beans[3]])     # a validation photo, and one not in the dataset
        self.assertEqual(dropped, 2)

    def test_copies_and_bursts_of_a_test_photo_are_dropped(self):
        names = ("PXL_burst.jpg", "DSC_other_clock.jpg", "PXL_later.jpg", "PXL_copy.jpg", "PXL_after_negative.jpg")
        beans = [REPO_ROOT / (SESSION + n) for n in names]              # absolute, as id_photos returns them
        scan = FakeScan(self.photos)
        kept, dropped = fit_ood_probe.probe_beans(beans, self.photos, describe=scan)
        self.assertEqual([q.name for q in kept], ["DSC_other_clock.jpg", "PXL_later.jpg", "PXL_after_negative.jpg"])
        self.assertEqual(dropped, 2)
        self.assertTrue({TEST_POSITIVE, TEST_NEGATIVE, SESSION + "PXL_burst.jpg"} <= scan.asked)



@real_data
class TestProbeFitSetOnThePhotos(unittest.TestCase):
    """probe_beans through seg_dataset.scan on the committed dataset file and the checked-out photos: a pool photo
    the dataset did not draw, shot 103 s before a test photo on the same Pixel, goes; one shot an hour earlier
    stays."""

    def test_a_burst_neighbour_of_a_test_photo_is_dropped(self):
        box = REPO_ROOT / "dataset" / "2026-08-07__box_pictures_all_classes"
        photos = seg_dataset.load_seg_dataset()
        test = {e["path"]: e["split"] for e in photos}
        self.assertEqual(test.get(f"{box.relative_to(REPO_ROOT)}/class_002__Kenya_AA/PXL_20260807_081158015.jpg"),
                         "test")
        burst = box / "class_002__Kenya_AA" / "PXL_20260807_081015382.jpg"
        earlier = box / "class_006__Brazil_Cerrado" / "PXL_20260807_070742209.jpg"
        kept, dropped = fit_ood_probe.probe_beans([burst, earlier], photos)
        self.assertEqual((kept, dropped), ([earlier], 1))

if __name__ == "__main__":
    unittest.main()
