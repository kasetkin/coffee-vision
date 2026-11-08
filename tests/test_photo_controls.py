"""The page's two photo controls (ticket OPS-17): Choose Photo and Take Photo. Plain unittest.

    python -m unittest tests.test_photo_controls

Choose Photo carried capture="environment" from the page's first commit, so iOS opened the camera with no way to
pick a photo already on the phone. D1: Choose Photo has no capture attribute, and a Take Photo control with it opens
the camera. D2: Take Photo shows on every phone and tablet, iOS and Android, and on no desktop. No test here can open a
phone's picker; the owner checks that on the devices.
"""
from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path

INDEX = (Path(__file__).resolve().parents[1] / "webapp" / "static" / "index.html").read_text()


class _Elements(HTMLParser):
    """Every element with an id, by id: its tag and attributes."""

    def __init__(self) -> None:
        super().__init__()
        self.by_id: dict[str, tuple[str, dict[str, str | None]]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if a.get("id"):
            self.by_id[a["id"]] = (tag, a)


def element(id_: str) -> tuple[str, dict[str, str | None]]:
    parser = _Elements()
    parser.feed(INDEX)
    return parser.by_id[id_]


class PhotoControls(unittest.TestCase):
    def test_choose_photo_offers_the_photos_already_there(self):   # D1
        tag, attrs = element("photo-input")
        self.assertEqual((tag, attrs.get("type")), ("input", "file"))
        self.assertNotIn("capture", attrs)

    def test_take_photo_opens_the_rear_camera(self):   # D1
        tag, attrs = element("camera-input")
        self.assertEqual((tag, attrs.get("type"), attrs.get("capture")), ("input", "file", "environment"))

    def test_both_take_the_same_formats(self):
        self.assertTrue(element("photo-input")[1]["accept"])
        self.assertEqual(element("camera-input")[1]["accept"], element("photo-input")[1]["accept"])

    def test_each_label_names_its_input(self):
        self.assertEqual(element("photo-label")[1].get("for"), "photo-input")
        self.assertEqual(element("camera-label")[1].get("for"), "camera-input")

    def test_take_photo_is_hidden_until_the_page_finds_a_phone_or_tablet(self):   # D2
        self.assertIn("hidden", element("camera-label")[1])
        self.assertIn("cameraLabel.hidden = !isMobile();", INDEX)
        for platform in ("Android", "iPhone", "iPad", "iPod", "Mobi"):
            self.assertIn(platform, INDEX.split("function isMobile()")[1].split("}")[0])
        self.assertIn("navigator.maxTouchPoints > 1", INDEX)   # iPadOS Safari, which reports a Mac

    def test_both_answer_a_pick_and_both_wait_while_a_photo_is_in_flight(self):   # as OPS-7 D4
        self.assertIn("for (const input of [fileInput, cameraInput])", INDEX)
        busy = INDEX.split("function setBusy(busy) {")[1].split("\n}")[0]
        for line in ("fileInput.disabled = busy;", "cameraInput.disabled = busy;",
                     "photoLabel.classList.toggle('disabled', busy);", "cameraLabel.classList.toggle('disabled', busy);"):
            self.assertIn(line, busy)


if __name__ == "__main__":
    unittest.main()
