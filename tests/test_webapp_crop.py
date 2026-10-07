"""/crop's answer carries the photo's mask (ticket OPS-6 D7, D10). Plain unittest.

    python -m unittest tests.test_webapp_crop -v

Sends real photos to webapp.app's /crop through Flask's test client. webapp.app makes its log directory and
loads the shipped model at import, so the import happens in setUpClass with COFFEE_CV_LOG_DIR pointed at a
temporary directory. Needs the DVC-tracked model, the pretrained weights and the deploy's smoke photo; skips
with a reason otherwise.
"""
from __future__ import annotations

import base64
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tests._tiers import real_data

REPO = Path(__file__).resolve().parents[1]
# The deploy's smoke photo, from the same list the deploy uses: a tray photo, 3000 x 4000, not square.
PHOTO = REPO / next(line.split()[2] for line in (REPO / "webapp/deploy/fixtures.txt").read_text().splitlines()
                    if line.startswith("smoke "))
MODEL = REPO / "models/allrigs_dino3b16_seg_country_s123.pt"


def _post(client, endpoint: str, data: bytes):
    return client.post(endpoint, data={"photo": (io.BytesIO(data), "photo.jpg")},
                       content_type="multipart/form-data")


def _decode_mask(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _sampled_at_centres(full: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`full` read at each centre of a `shape` grid laid over it: a nearest-neighbour downscale that shares
    no code with /crop's resize, so agreeing with it says /crop's mask is the full mask, in its frame."""
    fh, fw = full.shape
    h, w = shape
    ys = ((np.arange(h) + 0.5) * fh / h).astype(int)
    xs = ((np.arange(w) + 0.5) * fw / w).astype(int)
    return full[np.ix_(ys, xs)]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


@real_data
class TestCropMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in (MODEL, PHOTO):
            if not p.is_file():
                raise unittest.SkipTest(f"{p.relative_to(REPO)} is not present (dvc pull)")
        logs = tempfile.mkdtemp(prefix="test_webapp_crop.")
        cls.addClassCleanup(shutil.rmtree, logs, ignore_errors=True)
        old = os.environ.get("COFFEE_CV_LOG_DIR")
        os.environ["COFFEE_CV_LOG_DIR"] = logs
        try:
            import webapp.app as app
        finally:
            if old is None:
                os.environ.pop("COFFEE_CV_LOG_DIR")
            else:
                os.environ["COFFEE_CV_LOG_DIR"] = old
        cls.app = app
        cls.client = app.app.test_client()
        cls.photo = PHOTO.read_bytes()
        r = _post(cls.client, "/crop", cls.photo)
        assert r.status_code == 200, r.data
        cls.answer = r.get_json()

    def test_tray_photo_mask_is_the_photos_mask_scaled_down(self):
        from coffeecv.dataset import load_rgb_image
        a = self.answer
        self.assertIs(a["seg_fallback"], False)
        self.assertIs(a["cropped"], True)
        img = _decode_mask(a["mask"])
        rgb = load_rgb_image(PHOTO)
        h, w = rgb.shape[:2]
        self.assertNotEqual(h, w, "the test needs a non-square photo")
        mw, mh = img.size
        self.assertEqual(max(mw, mh), min(self.app.PREVIEW_MAX_DIM, max(w, h)))
        self.assertLessEqual(abs(mw - w * mh / h), 1, (img.size, (w, h)))
        self.assertLessEqual(abs(mh - h * mw / w), 1, (img.size, (w, h)))
        # A 1-bit palette PNG: index 0 (not bean) transparent, index 1 (bean) opaque (D7, D14).
        self.assertEqual(img.format, "PNG")
        self.assertEqual(img.mode, "P")
        self.assertEqual(img.info.get("transparency"), 0)
        idx = np.asarray(img)
        self.assertLessEqual(set(np.unique(idx).tolist()), {0, 1})
        mask = idx == 1
        self.assertTrue(mask.any(), "the mask is empty")
        full = self.app.segmenter.predict_mask(rgb)
        iou = _iou(mask, _sampled_at_centres(full, mask.shape))
        self.assertGreaterEqual(iou, 0.99)
        # The same check, transposed, must fail: it is what a swapped width and height would give.
        self.assertLess(_iou(mask.T, _sampled_at_centres(full, mask.T.shape)), 0.99)


if __name__ == "__main__":
    unittest.main()
