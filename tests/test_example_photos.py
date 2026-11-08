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
import subprocess
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO / "webapp" / "static" / "examples"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
from coffeecv.class_list import load_classes  # noqa: E402
from release_manifest import EXAMPLES_LIST, _readers, front_manifest  # noqa: E402


def examples() -> list[dict]:
    return json.loads((EXAMPLES_DIR / "examples.json").read_text())["examples"]


def served_countries() -> set[str]:
    """The class ids /analyze answers with: the countries of the model the app serves by default."""
    ckpt = re.search(r'"COFFEE_CV_CHECKPOINT", "([^"]+)\.pt"', (REPO / "webapp" / "app.py").read_text())[1]
    return set(load_classes(REPO / f"{ckpt}.classes.txt").keys)


class TheList(unittest.TestCase):
    def test_each_example_is_a_name_a_caption_and_a_country(self):
        names = [e["name"] for e in examples()]
        self.assertEqual(len(names), len(set(names)), "a name twice")
        for e in examples():
            with self.subTest(example=e.get("name")):
                self.assertEqual(set(e), {"name", "caption", "country"})
                self.assertRegex(e["name"], r"^[a-z0-9]+(-[a-z0-9]+)*$")   # a file name and a URL path as it is
                self.assertTrue(e["caption"].strip())

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
        self.assertEqual({p for p in front_manifest()["git"] if p.startswith("webapp/static/examples/")}, want)

    def test_a_release_ships_the_examples_its_own_commit_lists(self):
        """Read at the release's ref, not the working tree's: a commit from before the list ships none."""
        added = subprocess.run(["git", "log", "-1", "--format=%H", "--diff-filter=A", "--", EXAMPLES_LIST],
                               cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        if not added:
            self.skipTest(f"{EXAMPLES_LIST} is not committed yet")
        at = {p for p in front_manifest(*_readers(added))["git"] if p.startswith("webapp/static/examples/")}
        before = {p for p in front_manifest(*_readers(f"{added}^"))["git"] if p.startswith("webapp/static/examples/")}
        self.assertIn(EXAMPLES_LIST, at)
        self.assertEqual(before, set())

    def test_the_script_makes_exactly_the_listed_examples(self):
        spec = importlib.util.spec_from_file_location("make_example_photos", REPO / "scripts/make_example_photos.py")
        script = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = script   # its @dataclass looks its module up while the class is made
        self.addCleanup(sys.modules.pop, spec.name)
        spec.loader.exec_module(script)
        self.assertEqual([ex.name for ex in script.EXAMPLES], [e["name"] for e in examples()])


class ThePage(unittest.TestCase):
    """The page's example row (P4), as text: what runs is checked in a browser (the plan's P4 checks)."""

    INDEX = (REPO / "webapp" / "static" / "index.html").read_text()

    def test_it_builds_the_row_from_the_list(self):
        self.assertIn("fetch('/examples/examples.json')", self.INDEX)
        self.assertIn("No photo? Try one of ours:", self.INDEX)

    def test_its_truth_lines_are_d10s(self):
        self.assertIn("const NO_BEANS_TRUTH = 'No coffee beans here: the classifier should refuse it.';", self.INDEX)
        self.assertIn("`This coffee is from ${example.country}.`", self.INDEX)


class _Card(HTMLParser):
    """The card's elements in document order, as (tag, attrs, own text), and each button's text by id."""

    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict]] = []
        self.text: dict[str, str] = {}
        self._button: str | None = None
        self._hidden_depth = 0   # inside an aria-hidden element: its text is not the button's label
        self._stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = {k: (v if v is not None else "") for k, v in attrs}
        self.elements.append((tag, a))
        if tag in ("br", "img", "input", "meta", "link"):
            return
        self._stack.append(tag)
        if tag == "button" and a.get("id"):
            self._button = a["id"]
            self.text[self._button] = ""
        if a.get("aria-hidden") == "true" or self._hidden_depth:
            self._hidden_depth += 1

    def handle_endtag(self, tag):
        if self._stack and self._stack[-1] == tag:
            self._stack.pop()
        if self._hidden_depth:
            self._hidden_depth -= 1
        if tag == "button":
            self._button = None

    def handle_data(self, data):
        if self._button and not self._hidden_depth:
            self.text[self._button] += data


class TheFolds(unittest.TestCase):
    """The two buttons above the photo (ticket OPS-20), as text: what they do is checked in a browser (the plan's P1
    checks)."""

    @classmethod
    def setUpClass(cls):
        card = _Card()
        card.feed(ThePage.INDEX)
        cls.card = card
        cls.ids = [a.get("id") for _, a in card.elements]
        cls.by_id = {a["id"]: (tag, a) for tag, a in card.elements if a.get("id")}

    def index(self, pred) -> int:
        return next(i for i, (tag, a) in enumerate(self.card.elements) if pred(tag, a))

    def test_the_buttons_are_a_line_of_their_own_under_the_controls(self):   # D3
        controls = self.index(lambda t, a: "controls" in a.get("class", "").split())
        folds = self.index(lambda t, a: "folds" in a.get("class", "").split())
        # Nothing but the controls' two labels between them.
        between = [t for t, _ in self.card.elements[controls + 1:folds]]
        self.assertEqual(between, ["label", "label"])

    def test_accepted_formats_first_then_examples(self):   # D7, D12
        folds = self.index(lambda t, a: "folds" in a.get("class", "").split())
        buttons = [a["id"] for t, a in self.card.elements[folds + 1:folds + 6] if t == "button"]
        self.assertEqual(buttons, ["formats-toggle", "examples-toggle"])
        self.assertEqual(" ".join(self.card.text["formats-toggle"].split()), "Accepted formats")
        self.assertEqual(" ".join(self.card.text["examples-toggle"].split()), "Examples")

    def test_each_button_controls_its_own_list_and_starts_folded(self):   # D4, D5
        for button, panel in (("formats-toggle", "formats"), ("examples-toggle", "examples")):
            with self.subTest(button=button):
                tag, a = self.by_id[button]
                self.assertEqual(a.get("type"), "button")
                self.assertEqual(a.get("aria-controls"), panel)
                self.assertEqual(a.get("aria-expanded"), "false")
                self.assertIn("hidden", self.by_id[panel][1])

    def test_each_arrow_is_hidden_from_screen_readers_and_points_folded(self):   # D4
        for button in ("formats-toggle", "examples-toggle"):
            with self.subTest(button=button):
                i = self.ids.index(button)
                tag, a = self.card.elements[i + 1]
                self.assertEqual((tag, a.get("class"), a.get("aria-hidden")), ("span", "arrow", "true"))
        self.assertEqual(ThePage.INDEX.count('<span class="arrow" aria-hidden="true">&#9656;</span>'), 2)

    def test_examples_appears_only_after_a_pick(self):   # D1
        self.assertIn("hidden", self.by_id["examples-toggle"][1])
        self.assertNotIn("hidden", self.by_id["formats-toggle"][1])

    def test_the_chips_are_the_formats_list_and_outside_the_buttons(self):   # D5, R5
        tag, a = self.by_id["formats"]
        self.assertEqual(tag, "div")
        self.assertIn("formats-hint", a.get("class", "").split())
        # A folded flex box needs its own rule: display: flex wins over the hidden attribute.
        self.assertIn(".formats-hint[hidden] { display: none; }", ThePage.INDEX)

    def test_the_lists_stack_in_todays_order_under_the_buttons(self):   # D11
        folds = self.index(lambda t, a: "folds" in a.get("class", "").split())
        self.assertLess(folds, self.ids.index("formats"))
        self.assertLess(self.ids.index("formats"), self.ids.index("examples"))
        self.assertLess(self.ids.index("examples"), self.ids.index("preview-wrap"))


if __name__ == "__main__":
    unittest.main()
