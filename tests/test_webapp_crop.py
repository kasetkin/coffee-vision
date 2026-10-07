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
from unittest import mock

import numpy as np
from PIL import Image

from tests._tiers import real_data

REPO = Path(__file__).resolve().parents[1]
# The deploy's smoke photo, from the same list the deploy uses: a tray photo, 3000 x 4000, not square.
PHOTO = REPO / next(line.split()[2] for line in (REPO / "webapp/deploy/fixtures.txt").read_text().splitlines()
                    if line.startswith("smoke "))
# The model webapp.app will load: COFFEE_CV_CHECKPOINT, else the same default as its CHECKPOINT. The literal is
# repeated because webapp.app cannot be imported to read it before this check: the import loads the model.
MODEL = REPO / os.environ.get("COFFEE_CV_CHECKPOINT", "models/allrigs_dino3b16_seg_country_s123.pt")


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
        alpha = np.asarray(img.convert("RGBA"))[..., 3]
        self.assertTrue((alpha[idx == 1] == 255).all(), "bean pixels (index 1) are not opaque")
        mask = idx == 1
        self.assertTrue(mask.any(), "the mask is empty")
        full = self.app.segmenter.predict_mask(rgb)
        iou = _iou(mask, _sampled_at_centres(full, mask.shape))
        self.assertGreaterEqual(iou, 0.99)
        # The same check, transposed, must fail: it is what a swapped width and height would give.
        self.assertLess(_iou(mask.T, _sampled_at_centres(full, mask.T.shape)), 0.99)

    def test_orientation_tag_is_ignored_by_mask_and_thumbnail_alike(self):
        # load_rgb_image ignores EXIF orientation on purpose (CODING_STANDARDS.md, "Invariance over
        # normalization"); the mask and /preview's thumbnail must both stay in the stored pixels' frame.
        src = Image.open(PHOTO)
        w, h = src.size
        exif = src.getexif()
        exif[0x0112] = 6                       # "rotate 90 CW to display": a sideways-stored portrait
        buf = io.BytesIO()
        src.save(buf, "JPEG", quality=95, exif=exif.tobytes())
        tagged = buf.getvalue()
        from PIL import ImageOps
        self.assertEqual(ImageOps.exif_transpose(Image.open(io.BytesIO(tagged))).size, (h, w),
                         "the copy's tag would rotate it, if anything honoured it")
        crop, preview = _post(self.client, "/crop", tagged), _post(self.client, "/preview", tagged)
        self.assertEqual((crop.status_code, preview.status_code), (200, 200))
        mask_size = _decode_mask(crop.get_json()["mask"]).size
        self.assertLessEqual(max(mask_size), 1024, "the mask's long side (D7, D10)")
        for name, (iw, ih) in (("mask", mask_size),
                               ("thumbnail", Image.open(io.BytesIO(preview.data)).size)):
            with self.subTest(name):
                self.assertLessEqual(abs(iw - w * ih / h), 1, ((iw, ih), (w, h)))
                self.assertLessEqual(abs(ih - h * iw / w), 1, ((iw, ih), (w, h)))

    def test_empty_mask_falls_back_to_no_mask(self):
        # ML-2 D18: an empty mask means the whole photo is classified, unfilled; /crop says so (OPS-6 D5).
        def empty(_self, rgb):
            return np.zeros(rgb.shape[:2], bool)

        with mock.patch("coffeecv.segment_beans.BeanSegmenter.predict_mask", empty):
            r = _post(self.client, "/crop", self.photo)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), {"cropped": False, "box": None, "needs_review": False,
                                        "mask": None, "seg_fallback": True})


if __name__ == "__main__":
    unittest.main()
