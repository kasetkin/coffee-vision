"""The page's example photos (ticket OPS-16, ADR 0032): their one list and their files. Plain unittest.

    python -m unittest tests.test_example_photos

webapp/static/examples/examples.json is the one list of examples (D13, D18): the page builds its row from it, the
release manifest ships what it names, and --verify expects each one's country. scripts/make_example_photos.py keeps
only each name's original, which never goes public.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO / "webapp" / "static" / "examples"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
from coffeecv.class_list import load_classes  # noqa: E402
from release_manifest import STATIC  # noqa: E402


def examples() -> list[dict]:
    return json.loads((EXAMPLES_DIR / "examples.json").read_text())["examples"]


def served_countries() -> set[str]:
    """The class ids /analyze answers with: the countries of the model the app serves by default."""
    ckpt = re.search(r'"COFFEE_CV_CHECKPOINT", "([^"]+)\.pt"', (REPO / "webapp" / "app.py").read_text())[1]
    return set(load_classes(REPO / f"{ckpt}.classes.txt").keys)


class TheList(unittest.TestCase):
    def test_d10s_five_in_its_order_with_their_captions(self):
        self.assertEqual([(e["name"], e["caption"], e["country"]) for e in examples()], [
            ("brazil-grinder", "Roasted, in a grinder", "Brazil"),
            ("ethiopia-green", "Green, in a tray", "Ethiopia"),
            ("vietnam-robusta", "Robusta, on a table", "Vietnam"),
            ("indonesia-roasted", "Roasted, close up", "Indonesia"),
            ("cappuccino", "A cappuccino, no beans", None),
        ])

    def test_each_country_is_one_the_served_model_answers_with(self):
        countries = served_countries()
        for e in examples():
            with self.subTest(example=e["name"]):
                self.assertTrue(e["country"] is None or e["country"] in countries, e["country"])

    def test_each_example_has_its_two_files_and_the_folder_nothing_else(self):
        want = {"examples.json"} | {f"{e['name']}.{ext}" for e in examples() for ext in ("jxl", "webp")}
        self.assertEqual({p.name for p in EXAMPLES_DIR.iterdir()}, want)

    def test_the_front_ships_the_list_and_every_file_it_names(self):
        want = {"webapp/static/examples/examples.json"} | {
            f"webapp/static/examples/{e['name']}.{ext}" for e in examples() for ext in ("jxl", "webp")}
        self.assertEqual({p for p in STATIC if p.startswith("webapp/static/examples/")}, want)

    def test_the_script_makes_exactly_the_listed_examples(self):
        spec = importlib.util.spec_from_file_location("make_example_photos", REPO / "scripts/make_example_photos.py")
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script   # its @dataclass looks its module up while the class is made
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        self.assertEqual([ex.name for ex in script.EXAMPLES], [e["name"] for e in examples()])


if __name__ == "__main__":
    unittest.main()
