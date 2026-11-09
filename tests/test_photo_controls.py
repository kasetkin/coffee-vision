"""The page's photo controls (tickets OPS-17, OPS-21): Choose Photo, Select File and the camera button. Plain unittest.

    python -m unittest tests.test_photo_controls

OPS-17: Choose Photo has no capture attribute, so a phone offers the photos it already has, and Take Photo, with
capture="environment", opens the camera on every phone and tablet. OPS-21: on Android each control does one thing.
Choose Photo opens the photo picker, Select File a file browser, and the camera button the camera (D1, D9). A tablet
that asks for the desktop site reports Linux, not Android, and gets the desktop controls (D6). Every input is written in
the HTML (D14): two for iOS, desktop and any other phone (D15), three for Android, which the script points the labels
at.
The camera button is a square with an icon and no visible words, on every phone (D8, D13). The accept lists are held to
the accepted formats by tests/test_accepted_formats_named.py (ADR 0034). No test here can open a phone's picker; the
owner checks that on the devices (D7).
"""
from __future__ import annotations

import re
import unittest
from html.parser import HTMLParser
from pathlib import Path

INDEX = (Path(__file__).resolve().parents[1] / "webapp" / "static" / "index.html").read_text()
SCRIPT = INDEX.split("<script>")[1].split("</script>")[0]
STYLE = INDEX.split("<style>")[1].split("</style>")[0]


class _Elements(HTMLParser):
    """Every element with an id, by id: its tag and attributes; and the ids in the order they open."""

    def __init__(self) -> None:
        super().__init__()
        self.by_id: dict[str, tuple[str, dict[str, str | None]]] = {}
        self.order: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        by_name = dict(attrs)
        if by_name.get("id"):
            self.by_id[by_name["id"]] = (tag, by_name)
            self.order.append(by_name["id"])


class _Inside(HTMLParser):
    """What one element holds: the tags inside it with their attributes, and its text, each piece with the classes
    of the elements around it."""

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.text: list[tuple[str, set[str]]] = []
        self._classes: list[set[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        self.tags.append((tag, a))
        self._classes.append(set((a.get("class") or "").split()))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag: str) -> None:
        self._classes.pop()

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.text.append((data.strip(), set().union(*self._classes)))


def page() -> _Elements:
    parser = _Elements()
    parser.feed(INDEX)
    return parser


def element(id_: str) -> tuple[str, dict[str, str | None]]:
    return page().by_id[id_]


def inside(label_id: str) -> _Inside:
    m = re.search(rf'<label id="{label_id}"[^>]*>(.*?)</label>', INDEX, re.S)
    assert m, label_id
    parser = _Inside()
    parser.feed(m.group(1))
    return parser


def js_block(head: str) -> str:
    """The page script's function or block that opens with `head`, up to its closing brace at the start of a line."""
    return SCRIPT.split(head)[1].split("\n}")[0]


INPUTS = ("photo-input", "camera-input", "android-photo-input", "select-file-input", "android-camera-input")
LABELS = ("photo-label", "select-file-label", "camera-label")


class PhotoControls(unittest.TestCase):   # OPS-17, kept for iOS, desktop and other phones (OPS-21 D2, D15)
    def test_choose_photo_offers_the_photos_already_there(self):   # OPS-17 D1
        tag, attrs = element("photo-input")
        self.assertEqual((tag, attrs.get("type")), ("input", "file"))
        self.assertNotIn("capture", attrs)

    def test_take_photo_opens_the_rear_camera(self):   # OPS-17 D1
        tag, attrs = element("camera-input")
        self.assertEqual((tag, attrs.get("type"), attrs.get("capture")), ("input", "file", "environment"))

    def test_both_take_the_same_formats(self):
        self.assertTrue(element("photo-input")[1]["accept"])
        self.assertEqual(element("camera-input")[1]["accept"], element("photo-input")[1]["accept"])

    def test_each_label_names_its_input_until_the_page_finds_android(self):
        self.assertEqual(element("photo-label")[1].get("for"), "photo-input")
        self.assertEqual(element("camera-label")[1].get("for"), "camera-input")
        self.assertEqual(element("select-file-label")[1].get("for"), "select-file-input")

    def test_the_camera_is_hidden_until_the_page_finds_a_phone_or_tablet(self):   # OPS-17 D2
        self.assertIn("hidden", element("camera-label")[1])
        self.assertIn("cameraLabel.hidden = !isMobile();", SCRIPT)
        for platform in ("Android", "iPhone", "iPad", "iPod", "Mobi"):
            self.assertIn(platform, js_block("function isMobile() {"))
        self.assertIn("navigator.maxTouchPoints > 1", SCRIPT)   # iPadOS Safari, which reports a Mac


class AndroidControls(unittest.TestCase):   # OPS-21
    def test_every_input_is_in_the_html(self):   # D14
        elements = page().by_id
        for id_ in INPUTS:
            with self.subTest(input=id_):
                tag, attrs = elements[id_]
                self.assertEqual((tag, attrs.get("type")), ("input", "file"))

    def test_androids_choose_photo_and_select_file_have_no_capture(self):   # D1: the picker and Files, no camera
        for id_ in ("android-photo-input", "select-file-input"):
            with self.subTest(input=id_):
                self.assertNotIn("capture", element(id_)[1])

    def test_androids_camera_opens_the_rear_camera(self):   # D1: the test's C1; its list is held with the others
        self.assertEqual(element("android-camera-input")[1].get("capture"), "environment")

    def test_select_file_shows_on_android_only(self):   # D2, D6
        self.assertIn("hidden", element("select-file-label")[1])
        self.assertEqual(inside("select-file-label").text, [("Select File", set())])
        self.assertIn("/Android/i.test(navigator.userAgent)", js_block("function isAndroid() {"))
        android = js_block("if (isAndroid()) {")
        self.assertIn("selectFileLabel.hidden = false;", android)

    def test_on_android_the_labels_point_at_androids_inputs(self):   # D14
        android = js_block("if (isAndroid()) {")
        self.assertIn("photoLabel.htmlFor = 'android-photo-input';", android)
        self.assertIn("cameraLabel.htmlFor = 'android-camera-input';", android)

    def test_the_controls_in_their_order(self):   # D9: the camera button last
        order = [i for i in page().order if i in LABELS]
        self.assertEqual(order, ["photo-label", "select-file-label", "camera-label"])

    def test_the_android_row_never_wraps(self):   # D12, R7
        self.assertIn("controls.classList.add('android');", js_block("if (isAndroid()) {"))
        rule = re.search(r"\.controls\.android \{([^}]*)\}", STYLE)
        self.assertIsNotNone(rule)
        self.assertIn("flex-wrap: nowrap;", rule.group(1))


class CameraButton(unittest.TestCase):   # OPS-21 D1, D8, D13
    def test_an_icon_hidden_from_a_screen_reader(self):
        svgs = [a for tag, a in inside("camera-label").tags if tag == "svg"]
        self.assertEqual(len(svgs), 1)
        self.assertEqual(svgs[0].get("aria-hidden"), "true")
        self.assertIn("camera-btn", element("camera-label")[1]["class"].split())

    def test_take_photo_read_aloud_but_not_shown(self):
        self.assertEqual(inside("camera-label").text, [("Take Photo", {"visually-hidden"})])
        self.assertEqual(element("camera-label")[1].get("title"), "Take Photo")
        self.assertRegex(STYLE, r"\.visually-hidden \{[^}]*clip")

    def test_a_square_at_the_text_buttons_height(self):   # D1, D12
        text = re.search(r"\.controls\.with-camera \.file-btn \{([^}]*)\}", STYLE).group(1)
        square = re.search(r"\.file-btn\.camera-btn \{([^}]*)\}", STYLE).group(1)
        self.assertIn("line-height: 1.25rem;", text)
        for line in ("width: 2.75rem;", "height: 2.75rem;", "flex: none;"):   # 1.25rem of line, 0.75rem padding a side
            self.assertIn(line, square)
        self.assertIn("controls.classList.toggle('with-camera', isMobile());", SCRIPT)   # desktop's line as before (D2)

    def test_hidden_until_the_page_finds_a_phone(self):   # its display: flex would otherwise win over hidden
        self.assertIn(".file-btn.camera-btn[hidden] { display: none; }", STYLE)


class EveryControlTakesThePath(unittest.TestCase):   # OPS-21 scope, as OPS-7 D4
    def test_each_input_answers_a_pick(self):
        self.assertIn("const pickInputs = [fileInput, cameraInput, androidPhotoInput, selectFileInput, "
                      "androidCameraInput];", SCRIPT)
        self.assertIn("for (const input of pickInputs)", SCRIPT)
        for name, id_ in (("fileInput", "photo-input"), ("cameraInput", "camera-input"),
                          ("androidPhotoInput", "android-photo-input"), ("selectFileInput", "select-file-input"),
                          ("androidCameraInput", "android-camera-input")):
            self.assertIn(f"const {name} = document.getElementById('{id_}');", SCRIPT)

    def test_each_waits_while_a_photo_is_in_flight(self):
        self.assertIn("const pickLabels = [photoLabel, selectFileLabel, cameraLabel];", SCRIPT)
        busy = js_block("function setBusy(busy) {")
        self.assertIn("for (const input of pickInputs) input.disabled = busy;", busy)
        self.assertIn("for (const label of pickLabels) label.classList.toggle('disabled', busy);", busy)


if __name__ == "__main__":
    unittest.main()
