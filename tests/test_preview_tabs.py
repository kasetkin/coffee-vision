"""The page's preview format tabs, ticket OPS-25's trial (D2, D5). Plain unittest.

    python -m unittest tests.test_preview_tabs

Static: the tabs are today's preview then the previews part's formats, in the server's order, under the preview;
the script reads the previews part. What a browser shows on each tab is the owner's check on the phones (P5).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from coffeecv import preview_formats

INDEX = (Path(__file__).resolve().parents[1] / "webapp" / "static" / "index.html").read_text()
SCRIPT = INDEX.split("<script>")[1].split("</script>")[0]


class PreviewTabs(unittest.TestCase):
    def test_the_tabs_are_todays_preview_then_the_servers_formats_in_its_order(self):
        tabs = re.search(r'<div id="preview-tabs"[^>]*>(.*?)</div>', INDEX, re.S).group(1)
        self.assertEqual(re.findall(r'data-format="([^"]+)"', tabs), ["srgb_jpeg", *preview_formats.MIME])
        self.assertEqual(re.findall(r'>([^<]+)<span class="size">', tabs),
                         ["sRGB JPEG", "P3 JPEG", "AVIF 10-bit", "JPEG XL"])

    def test_they_sit_under_the_preview(self):
        self.assertLess(INDEX.index('<div id="preview-wrap">'), INDEX.index('<div id="preview-tabs"'))
        self.assertLess(INDEX.index('<div id="preview-tabs"'), INDEX.index('<label id="outline-label">'))

    def test_the_script_reads_the_previews_part_and_settles_the_tabs(self):
        self.assertIn("part.part === 'previews'", SCRIPT)
        self.assertIn("selectPreview('srgb_jpeg')", SCRIPT)
        # a photo the page clears takes its tabs with it
        clear = SCRIPT[SCRIPT.index("function clearPhoto()"):]
        self.assertIn("previewTabs", clear[:clear.index("\n}\n")])


if __name__ == "__main__":
    unittest.main()
