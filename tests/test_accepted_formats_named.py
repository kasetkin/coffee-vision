"""The public texts name the accepted formats, every one and nothing else (ticket OPS-18 D5, D12; ADR 0034). Plain
unittest.

    python -m unittest tests.test_accepted_formats_named

coffeecv/data/accepted_formats.csv is the one list of the formats the service reads. llms.txt, openapi.json, the page's
format chips and its file inputs' accept lists are written by hand; this holds each to the file, in its order, a
group's rows (camera RAW) gathered under the group's name as upload_check.named_formats gathers them.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path

from coffeecv import canonical as cn

REPO = Path(__file__).resolve().parents[1]
STATIC = REPO / "webapp" / "static"
sys.path.insert(0, str(REPO / "webapp" / "front"))
import upload_check  # noqa: E402

LLMS = " ".join((STATIC / "llms.txt").read_text().split())     # its lines rewrapped as one
OPENAPI = json.loads((STATIC / "openapi.json").read_text())
PHOTO_PART = OPENAPI["paths"]["/analyze"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"][
    "properties"]["photo"]["description"]
INDEX = (STATIC / "index.html").read_text()
# Formats Pillow (or another decoder here) reads that are not accepted: none may be named as if it were.
NOT_ACCEPTED = ("TIFF", "TGA", "PSD", "PCX", "EPS", "ICO", "JPEG 2000", "PPM", "SVG")


class _Page(HTMLParser):
    """The page's format chips (the spans of div.formats-hint) and its file inputs' accept lists, by id."""

    def __init__(self) -> None:
        super().__init__()
        self.chips: list[str] = []
        self.accept: dict[str, str] = {}
        self._in_hint = self._in_chip = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and "formats-hint" in (a.get("class") or "").split():
            self._in_hint = True
        elif tag == "span" and self._in_hint:
            self._in_chip = True
            self.chips.append("")
        elif tag == "input" and a.get("type") == "file":
            self.accept[a.get("id") or ""] = a.get("accept") or ""

    def handle_endtag(self, tag):
        if tag == "span":
            self._in_chip = False
        elif tag == "div":
            self._in_hint = False

    def handle_data(self, data):
        if self._in_chip:
            self.chips[-1] += data


def page() -> _Page:
    p = _Page()
    p.feed(INDEX)
    return p


class TheTextsNameTheAcceptedFormats(unittest.TestCase):
    def test_llms_txt(self):
        self.assertIn(upload_check.named_formats("and", list_groups=True), LLMS)
        self.assertIn("An animated GIF is read as its first frame.", LLMS)   # D1

    def test_openapi_json(self):
        self.assertIn(upload_check.named_formats("or", list_groups=True), PHOTO_PART)
        self.assertIn("An animated GIF is read as its first frame.", PHOTO_PART)

    def test_the_pages_chips(self):   # D5: a chip per format; a group's chip lists some of its rows
        ungrouped = [f.label for f in cn.ACCEPTED_FORMATS.values() if not f.group]
        groups: dict[str, list[str]] = {}
        for f in cn.ACCEPTED_FORMATS.values():
            if f.group:
                groups.setdefault(f.group, []).append(f.label)
        chips = page().chips
        self.assertEqual(chips[:len(ungrouped)], ungrouped)
        self.assertEqual(len(chips), len(ungrouped) + len(groups))
        for chip, (group, labels) in zip(chips[len(ungrouped):], groups.items()):
            with self.subTest(group=group):
                m = re.fullmatch(rf"{re.escape(group)} \((.+?)(, …)?\)", chip)
                self.assertIsNotNone(m, chip)
                named = m.group(1).split(", ")
                self.assertLessEqual(set(named), set(labels))
                self.assertEqual(m.group(2) is None, named == labels)   # an ellipsis when some are left out

    def test_every_accept_list_is_the_files(self):   # every MIME type, then every extension, in the file's order
        want = [m for f in cn.ACCEPTED_FORMATS.values() for m in f.mime_types] + \
            [e for f in cn.ACCEPTED_FORMATS.values() for e in f.extensions]
        accept = page().accept
        self.assertEqual(set(accept), {"photo-input", "camera-input"})
        for input_id, value in accept.items():
            with self.subTest(input=input_id):
                self.assertEqual(value.split(","), want)

    def test_nothing_else_is_named(self):
        texts = {"llms.txt": LLMS, "openapi.json": PHOTO_PART, "the page's chips": " ".join(page().chips),
                 "the refusal": upload_check.NOT_A_PHOTO}
        for where, text in texts.items():
            for other in NOT_ACCEPTED:
                with self.subTest(where=where, format=other):
                    self.assertNotRegex(text, rf"\b{re.escape(other)}\b")


if __name__ == "__main__":
    unittest.main()
