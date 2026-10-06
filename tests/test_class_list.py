"""coffeecv.class_list (ticket ML-3): one parser for classes.txt, in both formats. The D15 file gives one class
per country; an old "id;label" file -- every shipped model's frozen list -- reads exactly as before. Plain
unittest, no data needed.

    python -m unittest discover -s tests -p 'test_class_list.py'
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from coffeecv.class_list import ClassList, folder_classes, load_classes, read_coffees
from coffeecv.config import REPO_ROOT

# The D15 content, word for word from the ticket (docs/ticket_country_classes.html, D15).
D15 = """\
001;Ethiopia,Sidamo;
002;Kenya,Nyeri;AA
003;Colombia,Huila;PinkBourbon
004;CostaRica,Tarrazu;LaPastora
005;Guatemala,Acatenango;TataNahual
006;Brazil,Cerrado;
007;Brazil,Cerrado;MonteCristo
008;Ethiopia,Yirgacheffe,Kochere;
009;Vietnam,DakLak;Robusta
010;Indonesia,Java;
011;Peru,Junin,Satipo;Minca
012;Rwanda,Western,Rusizi;Gisuma
013;Brazil,SulDeMinas;
014;Colombia,Antioquia;Excelso
"""
# The ten countries in dataset_new_ignored/classes_short_map.txt's order, pinned here (Q5): the file goes
# with dataset_new_ignored/, and this list stays.
COUNTRIES = ("Ethiopia", "Kenya", "Colombia", "CostaRica", "Guatemala", "Brazil", "Vietnam", "Indonesia",
             "Peru", "Rwanda")
SHORT_MAP = REPO_ROOT / "dataset_new_ignored" / "classes_short_map.txt"


def legacy_load_class_labels(path: Path) -> dict[str, str]:
    """dataset.load_class_labels as it was before ML-3, the reference for the old format."""
    labels = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or ";" not in line:
            continue
        cid, label = line.split(";", 1)
        labels[cid.strip()] = label.strip()
    return labels


class TestClassList(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def write(self, text: str) -> Path:
        path = self.tmp / "classes.txt"
        path.write_text(text)
        return path

    def test_d15_gives_one_class_per_country_in_first_appearance_order(self):
        classes = load_classes(self.write(D15))
        self.assertEqual(classes.keys, COUNTRIES)
        self.assertEqual(classes.labels, {k: k for k in COUNTRIES})      # D2: the key is shown as written
        self.assertEqual(classes.folders["Brazil"], ("006", "007", "013"))
        self.assertEqual(classes.folders["Ethiopia"], ("001", "008"))
        self.assertEqual(classes.folders["Colombia"], ("003", "014"))
        self.assertEqual(classes.folders["Peru"], ("011",))
        self.assertEqual(classes.key_of("013"), "Brazil")
        self.assertEqual(classes.index("CostaRica"), 3)
        with self.assertRaises(KeyError):
            classes.key_of("015")

    def test_order_follows_ids_not_line_order(self):
        lines = D15.splitlines()
        shuffled = "\n".join(lines[7:] + lines[:7]) + "\n"
        self.assertEqual(load_classes(self.write(shuffled)), load_classes(self.write(D15)))

    @unittest.skipUnless(SHORT_MAP.exists(), "dataset_new_ignored/ is gone (ticket ML-3 P10); the list is pinned")
    def test_countries_match_classes_short_map(self):
        self.assertEqual(tuple(SHORT_MAP.read_text().split()), COUNTRIES)

    def test_dataset_classes_txt_is_d15(self):
        """Ticket ML-3 P4 wrote D15 to dataset/classes.txt, word for word; before it, the old ten-line list."""
        self.assertEqual((REPO_ROOT / "dataset" / "classes.txt").read_text(), D15)

    def test_coffees_keep_farm_region_and_misc_apart(self):
        coffees = {c.folder_id: c for c in read_coffees(self.write(D15))}
        self.assertEqual(len(coffees), 14)
        kochere = coffees["008"]
        self.assertEqual((kochere.country, kochere.farm_region, kochere.misc),
                         ("Ethiopia", "Ethiopia,Yirgacheffe,Kochere", None))
        self.assertEqual((coffees["002"].farm_region, coffees["002"].misc), ("Kenya,Nyeri", "AA"))
        self.assertEqual(coffees["012"].label, "Rwanda,Western,Rusizi;Gisuma")

    def test_folder_classes_of_d15_are_sorted_folder_ids(self):
        classes = folder_classes(self.write(D15))
        self.assertEqual(classes.keys, tuple(f"{i:03d}" for i in range(1, 15)))
        self.assertEqual(classes.folders["013"], ("013",))

    def test_every_shipped_frozen_list_reads_as_before(self):
        frozen = sorted((REPO_ROOT / "models").glob("*.classes.txt"))
        old = [p for p in frozen if read_coffees(p)[0].country is None]
        self.assertTrue(old)
        for path in sorted(set(frozen) - set(old)):     # shipped since ML-3: exactly the D15 file
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(), D15)
        for path in old:
            with self.subTest(path=path.name):
                before = legacy_load_class_labels(path)
                for classes in (load_classes(path), folder_classes(path)):
                    self.assertEqual(list(classes.keys), sorted(before))
                    self.assertEqual(classes.labels, before)
                    self.assertEqual(classes.folders, {k: (k,) for k in before})
                self.assertTrue(all(c.country is None for c in read_coffees(path)))

    def test_bad_files_are_refused(self):
        bad = {
            "mixed": "001;Ethiopia,Sidamo;\n002;Kenya,AA\n",
            "non-ASCII": "001;Ethiopia,Sidamo;\n002;Colombia,Tarrazú;\n",
            "duplicate id": "001;Ethiopia,Sidamo;\n001;Kenya,Nyeri;AA\n",
            "duplicate id, old format": "001;Ethiopia,Sidamo\n001;Kenya,AA\n",
            "no country": "001;;\n",
            "three ;": "001;Kenya,Nyeri;AA;x\n",
            "four places": "001;A,B,C,D;\n",
            "no ;": "001 Ethiopia\n",
            "id not a number": "x01;Ethiopia,Sidamo;\n",
            "empty": "\n",
        }
        for name, text in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                load_classes(self.write(text))

    def test_select_keeps_the_given_order(self):
        classes = folder_classes(self.write(D15)).select(["003", "001"])
        self.assertEqual(classes.keys, ("003", "001"))
        self.assertEqual(classes.labels["001"], "Ethiopia,Sidamo;")

    def test_a_folder_cannot_belong_to_two_classes(self):
        with self.assertRaises(ValueError):
            ClassList(("A", "B"), {"A": "A", "B": "B"}, {"A": ("001",), "B": ("001",)})


if __name__ == "__main__":
    unittest.main()
