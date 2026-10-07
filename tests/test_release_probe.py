"""The deploy's smoke comparison, release_probe.py compare-smoke (ticket OPS-6 D10, R2). Plain unittest.

    python -m unittest tests.test_release_probe -v

Builds a locally computed PROBE line and a served smoke file the way scripts/deploy_webapp.sh writes them,
with /crop masks made here as 1-bit palette PNGs (OPS-6 D7), and runs compare_smoke on the pair. No model,
no photo. The masks are an ellipse filling about 40% of a 300 x 400 frame, as the smoke photo's 768 x 1024
mask fills 41%, so a one-pixel shift costs about a percent of IoU, as it does on a real /crop mask.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
from release_probe import compare_smoke  # noqa: E402

BUILD = {"model_sha": "1ef12a19f56c9070", "code": {"commit": "0123456789abcdef0123456789abcdef01234567"}}
BOX = [0.11366666666666667, 0.25325, 0.881, 0.7655]


def _ellipse(h: int = 400, w: int = 300) -> np.ndarray:
    y, x = np.mgrid[:h, :w]
    return ((y - h / 2) / (0.42 * h)) ** 2 + ((x - w / 2) / (0.36 * w)) ** 2 <= 1


def _png_b64(mask: np.ndarray) -> str:
    """A 1-bit palette PNG, index 0 (not bean) transparent, as /crop sends it (OPS-6 D7)."""
    im = Image.fromarray(mask.astype(np.uint8), mode="P")
    im.putpalette([0, 0, 0, 255, 255, 255])
    buf = io.BytesIO()
    im.save(buf, "PNG", bits=1, transparency=0, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _crop_body(mask: np.ndarray | None, box=BOX) -> dict:
    return {"cropped": mask is not None, "box": box if mask is not None else None, "needs_review": False,
            "mask": None if mask is None else _png_b64(mask), "seg_fallback": mask is None}


class TestCompareSmoke(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.mask = _ellipse()

    def compare(self, expected_crop: dict, served_crop: dict) -> tuple[bool, str]:
        """compare_smoke on the two /crop bodies (the other endpoints agree): (passed, what it printed)."""
        classify = {"verdict": "ok", "ranked": [{"id": "Colombia", "label": "Colombia", "score": 0.9}]}
        exp = self.dir / "expected.json"
        exp.write_text("PROBE " + json.dumps({"build": BUILD, "classify": {"body": classify},
                                              "crop": {"body": expected_crop}}) + "\n")
        smoke = self.dir / "smoke.txt"
        smoke.write_text(f"SMOKE_CLASSIFY 200 {json.dumps(classify)}\n"
                         f"SMOKE_CROP 200 {json.dumps(served_crop)}\n"
                         "SMOKE_PREVIEW 200 image/jpeg 81234\n"
                         f"SMOKE_LOG {json.dumps({'build': BUILD})}\n")
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                compare_smoke(exp, smoke)
        except SystemExit as e:
            self.assertEqual(e.code, 1, out.getvalue())
            return False, out.getvalue()
        return True, out.getvalue()

    def test_a_few_flipped_boundary_pixels_pass_and_are_reported(self):
        served = self.mask.copy()
        edge = np.argwhere(self.mask & ~np.roll(self.mask, 1, axis=0))   # the ellipse's top boundary pixels
        for y, x in edge[::len(edge) // 5][:5]:
            served[y, x] = False
        passed, out = self.compare(_crop_body(self.mask), _crop_body(served))
        self.assertTrue(passed, out)
        self.assertIn("mask IoU 0.9999", out)
        self.assertIn("5 pixels differ", out)

    def test_the_mask_shifted_by_one_pixel_fails(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                passed, out = self.compare(_crop_body(self.mask), _crop_body(np.roll(self.mask, 1, axis=axis)))
                self.assertFalse(passed, out)
                self.assertIn("FAIL: /crop mask IoU", out)

    def test_a_mask_of_another_size_fails(self):
        for served in (self.mask[:-1], self.mask.T):   # one row short; width and height swapped
            with self.subTest(shape=served.shape):
                passed, out = self.compare(_crop_body(self.mask), _crop_body(served))
                self.assertFalse(passed, out)
                self.assertIn("FAIL: /crop mask is", out)

    def test_null_against_a_mask_fails(self):
        for expected, served in ((self.mask, None), (None, self.mask)):
            with self.subTest(expected_null=expected is None):
                passed, out = self.compare(_crop_body(expected), _crop_body(served))
                self.assertFalse(passed, out)
                self.assertIn("FAIL: /crop mask is", out)

    def test_null_on_both_sides_passes(self):
        """The fallback (ML-2 D18): /crop answers mask null and seg_fallback true, on both machines."""
        passed, out = self.compare(_crop_body(None), _crop_body(None))
        self.assertTrue(passed, out)
        self.assertIn("mask null", out)

    def test_a_mask_field_on_one_side_only_fails(self):
        """A server from before OPS-6 answers /crop with no mask field at all."""
        old = _crop_body(self.mask)
        del old["mask"], old["seg_fallback"]
        for expected, served in ((_crop_body(self.mask), old), (old, _crop_body(self.mask))):
            with self.subTest(served_has_mask="mask" in served):
                passed, out = self.compare(expected, served)
                self.assertFalse(passed, out)
                self.assertIn("/mask: present on one side only", out)

    def test_no_mask_field_on_either_side_passes(self):
        """A release from before OPS-6, as the deploy rehearsal's first deploy is, compared by today's probe."""
        old = _crop_body(self.mask)
        del old["mask"], old["seg_fallback"]
        passed, out = self.compare(old, dict(old))
        self.assertTrue(passed, out)
        self.assertNotIn("mask", out)

    def test_the_box_is_still_compared(self):
        moved = [BOX[0] + 0.002, *BOX[1:]]
        passed, out = self.compare(_crop_body(self.mask), _crop_body(self.mask, box=moved))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: /crop box differs by up to 2.00e-03", out)
        self.assertIn("mask IoU 1.000000, 0 pixels differ", out)


if __name__ == "__main__":
    unittest.main()
