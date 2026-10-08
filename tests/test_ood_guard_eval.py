"""The guard's measurement of ADR 0016 (ticket ML-5 D16): `ood_eval` reads the segmenter dataset file
(labels/ml5/seg_dataset.yaml, schema in `ood_eval.load_seg_dataset`) and reports, on its test split only,
negatives caught and positives refused at the fixed 0.5, the positives by source and by whether the
classifier's train split holds the same bytes. Plain unittest, on tests/fixtures/ml5_seg_dataset.yaml with
hand-set scores; no model.

    python -m unittest tests.test_ood_guard_eval -v
"""
import hashlib
import tempfile
import unittest
from pathlib import Path

from coffeecv import ood_eval

FIXTURE = Path(__file__).parent / "fixtures" / "ml5_seg_dataset.yaml"
NEG = "dataset/ood_negatives/2026-09__internet_proxy/empty_tray/empty_tray_00"
POOL = "dataset/2026-09-11__pixel/class_001__Ethiopia_Sidamo/PXL_20260911_10"
# Probe scores of the fixture's test photos; None = unmeasurable (classify_one refuses it). Train and
# validation photos are absent, so scoring one raises KeyError.
SCORES = {
    NEG + "1.jpg": 0.99,
    NEG + "2.jpg": 0.5,                    # on the boundary: not caught, the probe refuses above 0.5
    "dataset/ood_negatives/2026-09__user_realworld/real_world_negatives/real_world_negatives_001.jpg": None,
    POOL + "0000000.jpg": 0.01,            # trained on
    POOL + "0100000.jpg": 0.7,             # trained on
    POOL + "0200000.jpg": 0.6,
    "dataset/segmenter_positives/PXL_20260911_100100000.jpg": 0.2,   # byte copy of a trained pool photo
    "dataset/segmenter_positives/PXL_20261001_120000000.jpg": 0.55,
    "dataset/ood_positives_internet/internet_beans/internet_beans_001.jpg": None,
}
TRAINED = {hashlib.sha256((POOL + s).encode()).hexdigest() for s in ("0000000.jpg", "0100000.jpg")}


class TestTestSplitReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        photos = ood_eval.load_seg_dataset(FIXTURE)
        cls.report = ood_eval.guard_report(photos, lambda path: SCORES[path], TRAINED)

    def test_negatives_caught(self):
        self.assertEqual(self.report["threshold"], 0.5)
        self.assertEqual(self.report["negatives"]["all"], {"n": 3, "caught": 2, "unmeasurable": 1})
        self.assertEqual(self.report["negatives"]["by_tag"],
                         {"empty_tray": {"n": 2, "caught": 1, "unmeasurable": 0},
                          "real_world_negatives": {"n": 1, "caught": 1, "unmeasurable": 1}})

    def test_positives_refused_by_source_and_by_whether_the_classifier_trained_on_them(self):
        pos = self.report["positives"]
        self.assertEqual(pos["pool"], {"trained": {"n": 2, "refused": 1, "unmeasurable": 0},
                                       "not_trained": {"n": 1, "refused": 1, "unmeasurable": 0}})
        self.assertEqual(pos["segmenter_positive"], {"trained": {"n": 1, "refused": 0, "unmeasurable": 0},
                                                     "not_trained": {"n": 1, "refused": 1, "unmeasurable": 0}})
        self.assertEqual(pos["internet_positive"], {"trained": {"n": 0, "refused": 0, "unmeasurable": 0},
                                                    "not_trained": {"n": 1, "refused": 1, "unmeasurable": 1}})
        self.assertEqual(pos["all"], {"trained": {"n": 3, "refused": 1, "unmeasurable": 0},
                                      "not_trained": {"n": 3, "refused": 3, "unmeasurable": 1}})

    def test_only_the_test_split_is_scored(self):
        self.assertEqual({r["path"] for r in self.report["per_photo"]}, set(SCORES))

    def test_the_printout_names_every_count(self):
        text = ood_eval.format_report(self.report)
        self.assertIn("negatives caught at 0.5: 2/3", text)
        self.assertIn("not trained on", text)


class TestLoadSegDataset(unittest.TestCase):
    ENTRY = "- path: {path}\n  sha256: {sha}\n  source: {source}\n  split: {split}\n  group: g\n{extra}"

    def load(self, *entries: dict) -> list[dict]:
        body = "photos:\n" + "".join(self.ENTRY.format(**{"path": "dataset/x.jpg", "sha": "a" * 64,
                                                            "source": "pool", "split": "test", "extra": "",
                                                            **e}) for e in entries)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "seg_dataset.yaml"
            f.write_text(body)
            return ood_eval.load_seg_dataset(f)

    def test_reads_the_fixture(self):
        photos = ood_eval.load_seg_dataset(FIXTURE)
        self.assertEqual(len(photos), 15)
        self.assertEqual({p["source"] for p in photos},
                         {"negative", "pool", "segmenter_positive", "internet_positive"})

    def test_refuses_what_the_schema_forbids(self):
        cases = {"an unknown source": [{"source": "ood_positive"}],
                 "a split not written out": [{"split": "val"}],
                 "a negative without its tag": [{"source": "negative", "extra": "  batch: b\n"}],
                 "a short sha256": [{"sha": "abc"}],
                 "a path twice": [{}, {"sha": "b" * 64}]}
        for name, entries in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.load(*entries)


if __name__ == "__main__":
    unittest.main()
