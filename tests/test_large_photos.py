"""The dataset tools read photos above Pillow's decompression-bomb limit; the web service keeps it (owner,
2026-10-28: a 200 MP Galaxy photo of 2026-10-28, 16320x12240, which Pillow refuses above 179 MP). Plain unittest.

    python -m unittest tests.test_large_photos -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from coffeecv.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "webapp" / "deploy"))
from release_manifest import LIBRARY  # noqa: E402

# The 200 MP photo's size, and Pillow's own limit (Image.MAX_IMAGE_PIXELS when nothing sets it).
GALAXY_200MP = (16320, 12240)
PILLOW_DEFAULT = 89_478_485


class LargePhotos(unittest.TestCase):
    def test_a_200_mp_photo_decodes_through_the_dataset_tools_loader(self):
        from coffeecv.dataset import load_rgb_image
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "big.png"
            Image.new("1", GALAXY_200MP).save(path)          # all black: a small file, a 200 MP decode
            rgb = load_rgb_image(path)
        self.assertEqual(rgb.shape, (GALAXY_200MP[1], GALAXY_200MP[0], 3))

    def test_the_web_services_modules_keep_pillows_limit(self):
        modules = [m.removesuffix(".py").replace("/", ".") for m in LIBRARY]
        code = ("import json, sys\nfrom PIL import Image\n"
                + "".join(f"import {m}\n" for m in modules)
                + "print(json.dumps([Image.MAX_IMAGE_PIXELS, 'coffeecv.dataset' in sys.modules]))")
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(json.loads(out.stdout.splitlines()[-1]), [PILLOW_DEFAULT, False])


if __name__ == "__main__":
    unittest.main()
