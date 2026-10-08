"""What the shipped probe is fitted on (ADR 0016, ticket ML-5 D16): `fit_ood_probe` takes its negatives from
the segmenter dataset's training and validation splits, clean tags only, and drops from its bean photos any
photo whose bytes are in the test split, which `ood_eval` measures it on. Plain unittest, on
tests/fixtures/ml5_seg_dataset.yaml; no model.

    python -m unittest tests.test_fit_ood_probe_set -v
"""
import hashlib
import unittest
from pathlib import Path

from coffeecv import fit_ood_probe, ood_eval

FIXTURE = Path(__file__).parent / "fixtures" / "ml5_seg_dataset.yaml"
POOL = "dataset/2026-09-11__pixel/class_001__Ethiopia_Sidamo/PXL_20260911_10"


def fixture_sha(path: Path) -> str:
    """The fixture's sha256 of a photo: of its path string."""
    return hashlib.sha256(str(path).encode()).hexdigest()


class TestProbeFitSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.photos = ood_eval.load_seg_dataset(FIXTURE)

    def test_negatives_are_training_and_validation_ones_with_a_clean_tag(self):
        self.assertEqual(
            [e["path"] for e in fit_ood_probe.probe_negatives(self.photos)],
            ["dataset/ood_negatives/2026-09__internet_proxy/empty_tray/empty_tray_003.jpg",
             "dataset/ood_negatives/2026-09-11__user_samerig/confusable_grain/PIC_20260911_132408.JPG"])

    def test_bean_photos_in_the_test_split_are_dropped(self):
        beans = [Path(POOL + s) for s in ("0000000.jpg", "0200000.jpg", "0300000.jpg", "9900000.jpg")]
        kept, dropped = fit_ood_probe.probe_beans(beans, self.photos, sha256=fixture_sha)
        self.assertEqual(kept, [beans[2], beans[3]])     # a validation photo, and one not in the dataset
        self.assertEqual(dropped, 2)


if __name__ == "__main__":
    unittest.main()
