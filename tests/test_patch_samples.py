"""coffeecv.patch_samples: the README's sample sheet. The committed list names every country once or twice and only
crops of that country; a malformed line is refused. Plain unittest; the crops themselves only in the real-data tier.

    python -m unittest discover -s tests -p 'test_patch_samples.py'
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coffeecv.class_list import load_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.patch_samples import LIST_PATH, display_name, read_list
from tests._tiers import real_data

CLASSES = load_classes(REPO_ROOT / "dataset" / "classes.txt")
GOOD = "Brazil green data/segcropped/s/class_006__Brazil_Cerrado/a__cropped.jpg\n"


def _read(text: str):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "list.txt"
        path.write_text(text)
        return read_list(path, CLASSES)


class CommittedList(unittest.TestCase):
    def test_every_country_once_or_twice(self):
        samples = read_list(LIST_PATH, CLASSES)
        by_country = {}
        for s in samples:
            by_country.setdefault(s.country, []).append(s.state)
        self.assertEqual(set(by_country), set(CLASSES.keys))
        self.assertTrue(all(1 <= len(states) <= 2 for states in by_country.values()))

    @real_data
    def test_every_crop_is_in_the_pool(self):
        pool = REPO_ROOT / "data" / "segcropped" / "pool"
        for s in read_list(LIST_PATH, CLASSES):
            with self.subTest(crop=str(s.crop)):
                self.assertTrue((REPO_ROOT / s.crop).is_file())
                # The pool names a class folder by its first session's name (class_009__Vietnam), so match the id.
                class_prefix = s.crop.parent.name.split("__")[0] + "__"
                self.assertTrue(any((d / s.crop.name).is_file() for d in pool.glob(class_prefix + "*")))


class ReadList(unittest.TestCase):
    def test_comments_and_blank_lines(self):
        samples = _read("# a comment\n\n" + GOOD)
        self.assertEqual([(s.country, s.state) for s in samples], [("Brazil", "green")])

    def test_refuses(self):
        cases = {
            "unknown country": "Narnia green data/segcropped/s/class_006__Brazil_Cerrado/a__cropped.jpg\n",
            "unknown state": "Brazil medium data/segcropped/s/class_006__Brazil_Cerrado/a__cropped.jpg\n",
            "another country's folder": "Brazil green data/segcropped/s/class_002__Kenya_AA/a__cropped.jpg\n",
            "a repeated line": GOOD + GOOD,
            "a missing field": "Brazil green\n",
        }
        for what, text in cases.items():
            with self.subTest(what), self.assertRaises(ValueError):
                _read(text)

    def test_two_folders_of_one_country(self):
        samples = _read(GOOD + "Brazil roasted data/segcropped/s/class_013__Brazil_SulDeMinas/b__cropped.jpg\n")
        self.assertEqual([s.state for s in samples], ["green", "roasted"])


class DisplayName(unittest.TestCase):
    def test_splits_camel_case(self):
        self.assertEqual(display_name("CostaRica"), "Costa Rica")
        self.assertEqual(display_name("Brazil"), "Brazil")


class ListMatchesConfig(unittest.TestCase):
    def test_params_yaml_names_the_same_class_list(self):
        self.assertEqual(load_classes(REPO_ROOT / RunConfig.from_params_yaml().classes_file).keys, CLASSES.keys)


if __name__ == "__main__":
    unittest.main()
