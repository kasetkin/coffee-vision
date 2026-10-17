"""/analyze answers a photo in one streamed request (ticket OPS-7, ADR 0023): a preview part with the photo's mask
(OPS-6 D7, D10), then a classification part. Plain unittest.

    python -m unittest tests.test_webapp_analyze -v

Sends real photos to webapp.app's /analyze through Flask's test client. webapp.app makes its log directory and
loads the shipped model at import, so the import happens in setUpClass with COFFEE_CV_LOG_DIR pointed at a
temporary directory. Needs the DVC-tracked model, the pretrained weights and the deploy's smoke photo; skips
with a reason otherwise.

The app serves LiteRT (ticket ML-7 P8). Until a model ships its converted directory (plan P9), the test serves
ML-7 P7's files on the CPU for their checkpoint (tests.test_litert.build_cpu_dir).
"""
from __future__ import annotations

import base64
import io
import json
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


def _post(client, data: bytes, skip_crop: bool = False, buffered: bool = True):
    form = {"photo": (io.BytesIO(data), "photo.jpg")}
    if skip_crop:
        form["skip_crop"] = "1"
    return client.post("/analyze", data=form, content_type="multipart/form-data", buffered=buffered)


def _parts(response) -> list[dict]:
    """The streamed answer's lines, one JSON object each (ticket OPS-7 D8)."""
    return [json.loads(line) for line in response.get_data(as_text=True).splitlines()]


def _decode_mask(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _decode_thumbnail(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _sampled_at_centres(full: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`full` read at each centre of a `shape` grid laid over it: a nearest-neighbour downscale that shares
    no code with the preview's resize, so agreeing with it says the preview's mask is the full mask, in its frame."""
    fh, fw = full.shape
    h, w = shape
    ys = ((np.arange(h) + 0.5) * fh / h).astype(int)
    xs = ((np.arange(w) + 0.5) * fw / w).astype(int)
    return full[np.ix_(ys, xs)]


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return np.count_nonzero(a & b) / np.count_nonzero(a | b)


@real_data
class TestAnalyze(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in (MODEL, PHOTO):
            if not p.is_file():
                raise unittest.SkipTest(f"{p.relative_to(REPO)} is not present (dvc pull)")
        logs = tempfile.mkdtemp(prefix="test_webapp_analyze.")
        cls.addClassCleanup(shutil.rmtree, logs, ignore_errors=True)
        env = {"COFFEE_CV_LOG_DIR": logs}
        if not MODEL.with_suffix(".litert").is_dir():
            from tests import test_litert
            missing = test_litert._have_served()
            if missing:
                raise unittest.SkipTest(missing)
            test_litert.build_cpu_dir(Path(logs))
            env.update(COFFEE_CV_CHECKPOINT=str(test_litert.CHECKPOINT.relative_to(REPO)), COFFEE_CV_LITERT_DIR=logs)
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            import webapp.app as app
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k)
                else:
                    os.environ[k] = v
        cls.app = app
        cls.client = app.app.test_client()
        cls.photo = PHOTO.read_bytes()
        cls.response = _post(cls.client, cls.photo)
        assert cls.response.status_code == 200, cls.response.data
        cls.parts = _parts(cls.response)
        cls.preview = cls.parts[0]

    def _log_lines(self, response, msg: str = "request") -> list[dict]:
        """The request log's `msg` lines for one answer, found by its X-Request-Id."""
        rid = response.headers["X-Request-Id"]
        lines = [json.loads(ln) for ln in (self.app.LOG_DIR / "app.jsonl").read_text().splitlines()]
        return [ln for ln in lines if ln.get("request_id") == rid and ln["msg"] == msg]

    def test_answer_is_a_preview_then_a_classification(self):
        r = self.response
        self.assertEqual(r.mimetype, "application/x-ndjson")
        # Both nginx servers stream it only if the app asks them to (ticket OPS-7 D11).
        self.assertEqual(r.headers.get("X-Accel-Buffering"), "no")
        self.assertEqual([p["part"] for p in self.parts], ["preview", "classification"])

    def test_preview_mask_is_the_photos_mask_scaled_down(self):
        from coffeecv.dataset import load_rgb_image
        a = self.preview
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

    def test_preview_box_is_a_fraction_of_the_photo(self):
        x0, y0, x1, y1 = self.preview["box"]
        self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, self.preview["box"])
        self.assertIs(self.preview["needs_review"], False)

    def test_preview_thumbnail_is_the_photo_scaled_down(self):
        img = _decode_thumbnail(self.preview["thumbnail"])
        self.assertEqual(img.format, "JPEG")
        w, h = Image.open(PHOTO).size
        tw, th = img.size
        self.assertEqual(max(tw, th), self.app.PREVIEW_MAX_DIM)
        self.assertLessEqual(abs(tw - w * th / h), 1, ((tw, th), (w, h)))

    def test_classification_is_classify_ones_answer(self):
        from coffeecv.infer import classify_one
        app = self.app
        entry = classify_one(PHOTO, app.cfg, app.class_ids, app.class_labels, app.classifier, app.ref,
                             n_patches=app.N_PATCHES, tta=app.TTA, probe=app.probe, segmenter=app.segmenter)
        body, status = app._entry_to_response(entry)
        self.assertEqual(status, 200)
        self.assertEqual(self.parts[1], {"part": "classification", **body})
        self.assertIn(body["verdict"], ("predicted", "warned"))

    def test_one_log_line_with_the_classify_fields_and_how_it_ended(self):
        lines = self._log_lines(self.response)
        self.assertEqual(len(lines), 1, lines)
        line = lines[0]
        self.assertEqual((line["endpoint"], line["status"]), ("/analyze", 200))
        for field in ("verdict", "top1_class", "top1_score", "inference_ms", "decode_ms", "crop_detect_ms",
                      "decoded_w", "decoded_h", "upload_bytes"):
            self.assertIn(field, line)
        self.assertLessEqual(line["preview_ms"], line["latency_ms"])
        self.assertIs(line["skip_crop"], False)
        # Ticket OPS-7 D19.
        self.assertEqual((line["outcome"], line["parts_sent"]), ("complete", 2))

    def test_orientation_tag_is_ignored_by_mask_and_thumbnail_alike(self):
        # load_rgb_image ignores EXIF orientation on purpose (CODING_STANDARDS.md, "Invariance over
        # normalization"); the mask and the thumbnail must both stay in the stored pixels' frame.
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
        r = _post(self.client, tagged)
        self.assertEqual(r.status_code, 200)
        preview = _parts(r)[0]
        mask_size = _decode_mask(preview["mask"]).size
        self.assertLessEqual(max(mask_size), 1024, "the mask's long side (D7, D10)")
        for name, (iw, ih) in (("mask", mask_size), ("thumbnail", _decode_thumbnail(preview["thumbnail"]).size)):
            with self.subTest(name):
                self.assertLessEqual(abs(iw - w * ih / h), 1, ((iw, ih), (w, h)))
                self.assertLessEqual(abs(ih - h * iw / w), 1, ((iw, ih), (w, h)))

    def test_empty_mask_falls_back_to_no_mask(self):
        # ML-2 D18: an empty mask means the whole photo is classified, unfilled; the preview says so (OPS-6 D5).
        def empty(_self, rgb):
            return np.zeros(rgb.shape[:2], bool)

        with mock.patch("coffeecv.segment_beans.BeanSegmenter.predict_mask", empty):
            r = _post(self.client, self.photo)
        self.assertEqual(r.status_code, 200)
        preview, classification = _parts(r)
        self.assertTrue(preview.pop("thumbnail"), "the whole photo's thumbnail")
        self.assertEqual(preview, {"part": "preview", "cropped": False, "box": None, "needs_review": False,
                                   "mask": None, "seg_fallback": True})
        self.assertEqual(classification["part"], "classification")

    def test_skip_crop_previews_the_whole_photo_then_classifies_it(self):
        # Ticket OPS-7 D9: the same two parts, with no mask and no box.
        r = _post(self.client, self.photo, skip_crop=True)
        self.assertEqual(r.status_code, 200)
        preview, classification = _parts(r)
        self.assertEqual({k: preview[k] for k in ("part", "cropped", "box", "needs_review", "mask", "seg_fallback")},
                         {"part": "preview", "cropped": False, "box": None, "needs_review": False, "mask": None,
                          "seg_fallback": False})
        self.assertEqual(_decode_thumbnail(preview["thumbnail"]).size,
                         _decode_thumbnail(self.preview["thumbnail"]).size)
        self.assertEqual(classification["part"], "classification")
        self.assertIs(self._log_lines(r)[0]["skip_crop"], True)

    def test_unreadable_file_is_a_400_with_no_stream(self):
        r = _post(self.client, b"not an image")
        self.assertEqual(r.status_code, 400)
        self.assertTrue(r.is_json)
        self.assertTrue(r.get_json()["error"].startswith("could not read this as an image"), r.get_json())
        self.assertEqual(len(self._log_lines(r, "decode failure")), 1)

    def test_no_file_is_a_400(self):
        r = self.client.post("/analyze", data={}, content_type="multipart/form-data")
        self.assertEqual((r.status_code, r.get_json()), (400, {"error": "no photo uploaded"}))

    def test_error_after_the_preview_is_an_error_part(self):
        # Ticket OPS-7 R4: the 200 has gone out with the preview, so a later failure is a part, not a status.
        with mock.patch.object(self.app, "classify_segmented", side_effect=RuntimeError("boom")):
            r = _post(self.client, self.photo)
        self.assertEqual(r.status_code, 200)
        parts = _parts(r)
        self.assertEqual([p["part"] for p in parts], ["preview", "error"])
        self.assertEqual(parts[1]["message"], "Something went wrong -- try a different photo.")
        line, = self._log_lines(r)
        self.assertEqual((line["outcome"], line["parts_sent"]), ("error", 2))
        unhandled, = self._log_lines(r, "unhandled exception")
        self.assertEqual(unhandled["exc_type"], "RuntimeError")

    def test_a_client_gone_after_the_preview_is_logged_once(self):
        r = _post(self.client, self.photo, buffered=False)
        first = json.loads(next(iter(r.response)))
        self.assertEqual(first["part"], "preview")
        r.close()
        line, = self._log_lines(r)
        self.assertEqual((line["outcome"], line["parts_sent"]), ("client_gone", 1))

    def test_the_three_old_endpoints_are_gone(self):
        # Ticket OPS-7 D5, D7.
        for endpoint in ("/preview", "/crop", "/classify"):
            with self.subTest(endpoint):
                r = self.client.post(endpoint, data={"photo": (io.BytesIO(self.photo), "photo.jpg")},
                                     content_type="multipart/form-data")
                self.assertEqual(r.status_code, 404)


if __name__ == "__main__":
    unittest.main()
