"""The deploy's comparisons in release_probe.py: compare-smoke (tickets OPS-6 D10, R2 and OPS-7 P3) and
compare-classify --d7, a release against PyTorch within ML-7 D7's bars (ADR 0020). Plain unittest.

    python -m unittest tests.test_release_probe -v

Builds a locally computed PROBE line and a served smoke file the way scripts/deploy_webapp.sh writes them: /analyze's
two parts (OPS-7 D8), with preview masks made here as 1-bit palette PNGs (OPS-6 D7) and thumbnails as JPEGs, and runs
compare_smoke on the pair. No model, no photo. The masks are an ellipse filling about 40% of a 300 x 400 frame, as
the smoke photo's 768 x 1024 mask fills 41%, so a one-pixel shift costs about a percent of IoU, as it does on a real
mask. The D7 comparison runs on classify entries made here, as classify-torch and classify print them.
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
from release_probe import compare_classify, compare_smoke  # noqa: E402

BUILD = {"model_sha": "1ef12a19f56c9070", "code": {"commit": "0123456789abcdef0123456789abcdef01234567"}}
BOX = [0.11366666666666667, 0.25325, 0.881, 0.7655]
CLASSIFICATION = {"part": "classification", "verdict": "predicted",
                  "ranked": [{"id": "Colombia", "label": "Colombia", "score": 0.9}]}


def _ellipse(h: int = 400, w: int = 300) -> np.ndarray:
    y, x = np.mgrid[:h, :w]
    return ((y - h / 2) / (0.42 * h)) ** 2 + ((x - w / 2) / (0.36 * w)) ** 2 <= 1


def _png_b64(mask: np.ndarray) -> str:
    """A 1-bit palette PNG, index 0 (not bean) transparent, as the preview part sends it (OPS-6 D7)."""
    im = Image.fromarray(mask.astype(np.uint8), mode="P")
    im.putpalette([0, 0, 0, 255, 255, 255])
    buf = io.BytesIO()
    im.save(buf, "PNG", bits=1, transparency=0, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_b64(size: tuple[int, int] = (768, 1024), fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 80, 40)).save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _preview(mask: np.ndarray | None, box=BOX, thumbnail: str | None = None) -> dict:
    return {"part": "preview", "thumbnail": thumbnail or _jpeg_b64(), "cropped": mask is not None,
            "box": box if mask is not None else None, "needs_review": False,
            "mask": None if mask is None else _png_b64(mask), "seg_fallback": mask is None}


class TestCompareSmoke(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        self.mask = _ellipse()
        self.boot = None   # the SMOKE_BOOT record, when a test sets one

    def compare(self, expected_preview: dict, served_preview: dict | None, served_rest=(CLASSIFICATION,),
                status: str = "200", content_type: str = "application/x-ndjson") -> tuple[bool, str]:
        """compare_smoke on the two previews, the classifications agreeing unless `served_rest` says otherwise:
        (passed, what it printed)."""
        exp = self.dir / "expected.json"
        exp.write_text("PROBE " + json.dumps({"build": BUILD, "analyze": {
            "status": 200, "content_type": "application/x-ndjson", "parts": [expected_preview, CLASSIFICATION]}}) + "\n")
        served = [p for p in (served_preview, *served_rest) if p is not None]
        smoke = self.dir / "smoke.txt"
        smoke.write_text(f"SMOKE_ANALYZE {status} 7.25 {content_type}\n"
                         + "".join(f"SMOKE_PART {json.dumps(p)}\n" for p in served)
                         + f"SMOKE_LOG {json.dumps({'build': BUILD})}\n"
                         + (f"SMOKE_BOOT {json.dumps(self.boot)}\n" if self.boot else ""))
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                compare_smoke(exp, smoke)
        except SystemExit as e:
            self.assertEqual(e.code, 1, out.getvalue())
            return False, out.getvalue()
        return True, out.getvalue()

    def test_the_same_answer_passes_and_is_summarised(self):
        passed, out = self.compare(_preview(self.mask), _preview(self.mask))
        self.assertTrue(passed, out)
        self.assertIn("classification  predicted, top-1 Colombia", out)
        self.assertIn("thumbnail 768 x 1024 JPEG", out)

    def test_a_few_flipped_boundary_pixels_pass_and_are_reported(self):
        served = self.mask.copy()
        edge = np.argwhere(self.mask & ~np.roll(self.mask, 1, axis=0))   # the ellipse's top boundary pixels
        for y, x in edge[::len(edge) // 5][:5]:
            served[y, x] = False
        passed, out = self.compare(_preview(self.mask), _preview(served))
        self.assertTrue(passed, out)
        self.assertIn("mask IoU 0.9999", out)
        self.assertIn("5 pixels differ", out)

    def test_the_mask_shifted_by_one_pixel_fails(self):
        for axis in (0, 1):
            with self.subTest(axis=axis):
                passed, out = self.compare(_preview(self.mask), _preview(np.roll(self.mask, 1, axis=axis)))
                self.assertFalse(passed, out)
                self.assertIn("FAIL: preview mask IoU", out)

    def test_a_mask_of_another_size_fails(self):
        for served in (self.mask[:-1], self.mask.T):   # one row short; width and height swapped
            with self.subTest(shape=served.shape):
                passed, out = self.compare(_preview(self.mask), _preview(served))
                self.assertFalse(passed, out)
                self.assertIn("FAIL: preview mask is", out)

    def test_null_against_a_mask_fails(self):
        for expected, served in ((self.mask, None), (None, self.mask)):
            with self.subTest(expected_null=expected is None):
                passed, out = self.compare(_preview(expected), _preview(served))
                self.assertFalse(passed, out)
                self.assertIn("FAIL: preview mask is", out)

    def test_null_on_both_sides_passes(self):
        """The fallback (ML-2 D18): the preview has mask null and seg_fallback true, on both machines."""
        passed, out = self.compare(_preview(None), _preview(None))
        self.assertTrue(passed, out)
        self.assertIn("mask null", out)

    def test_the_box_is_still_compared(self):
        moved = [BOX[0] + 0.002, *BOX[1:]]
        passed, out = self.compare(_preview(self.mask), _preview(self.mask, box=moved))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: preview box differs by up to 2.00e-03", out)
        self.assertIn("mask IoU 1.000000, 0 pixels differ", out)

    def test_a_thumbnail_of_another_size_fails(self):
        passed, out = self.compare(_preview(self.mask), _preview(self.mask, thumbnail=_jpeg_b64((767, 1024))))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: preview thumbnail is 767 x 1024, expected 768 x 1024", out)

    def test_a_thumbnail_that_is_not_a_jpeg_fails(self):
        passed, out = self.compare(_preview(self.mask), _preview(self.mask, thumbnail=_jpeg_b64(fmt="PNG")))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: preview thumbnail is a PNG, not a JPEG", out)

    def test_a_missing_part_fails(self):
        for preview, rest, parts in ((_preview(self.mask), (), "preview"), (None, (CLASSIFICATION,), "classification")):
            with self.subTest(parts=parts):
                passed, out = self.compare(_preview(self.mask), preview, served_rest=rest)
                self.assertFalse(passed, out)
                self.assertIn(f"FAIL: /analyze answered the parts [{parts}], expected [preview, classification]", out)

    def test_an_error_part_fails_with_its_message(self):
        error = {"part": "error", "message": "Something went wrong -- try a different photo."}
        passed, out = self.compare(_preview(self.mask), _preview(self.mask), served_rest=(error,))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: /analyze answered the parts [preview, error], expected [preview, classification]: "
                      "Something went wrong -- try a different photo.", out)

    def test_another_status_or_content_type_fails(self):
        passed, out = self.compare(_preview(self.mask), None, served_rest=(), status="503", content_type="text/html")
        self.assertFalse(passed, out)
        self.assertIn("FAIL: /analyze answered 503 text/html", out)

    def test_the_classification_is_still_compared(self):
        other = {**CLASSIFICATION, "ranked": [{"id": "Kenya", "label": "Kenya", "score": 0.9}]}
        passed, out = self.compare(_preview(self.mask), _preview(self.mask), served_rest=(other,))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: classification top-1", out)

    def test_the_boot_record_is_reported(self):
        gpu = {"ok": True, "why_not": [], "busy_frac": 0.71,
               "libraries": ["libLiteRtWebGpuAccelerator.so", "libvulkan_intel.so"], "render_nodes": ["/dev/dri/renderD128"]}
        self.boot = {"boot_s": 19.97, "warm_up_s": 10.33, "peak_rss_mb": 1830.5, "gpu": gpu}
        passed, out = self.compare(_preview(self.mask), _preview(self.mask))
        self.assertTrue(passed, out)
        self.assertIn("boot       19.97 s (warm-up 10.33 s), peak RSS 1830.5 MB, GPU proven (busy 0.71, "
                      "libLiteRtWebGpuAccelerator.so, libvulkan_intel.so)", out)

    def test_an_unproven_gpu_fails(self):
        gpu = {"ok": False, "why_not": ["no i915 counters"], "busy_frac": None, "libraries": [], "render_nodes": []}
        self.boot = {"boot_s": 19.97, "warm_up_s": 10.33, "peak_rss_mb": 1830.5, "gpu": gpu}
        passed, out = self.compare(_preview(self.mask), _preview(self.mask))
        self.assertFalse(passed, out)
        self.assertIn("FAIL: the GPU is not proven (ADR 0020): ['no i915 counters']", out)


def _entry(verdict: str = "predicted", probs: tuple = (0.7, 0.2, 0.1)) -> dict:
    """A classify_one entry as the probes print it: ranked (id, label, probability), highest first."""
    ids = ("Colombia", "Kenya", "Brazil")
    ranked = sorted(zip(ids, ids, probs), key=lambda r: -r[2])
    return {"verdict": verdict, "ranked": [list(r) for r in ranked]}


class TestCompareD7(unittest.TestCase):
    """compare-classify --d7: PyTorch's answers (A) against the release's (B)."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def compare(self, torch_entries: dict, release_entries: dict, release_sha: str = BUILD["model_sha"]
                ) -> tuple[bool, str]:
        a, b = self.dir / "torch.json", self.dir / "release.json"
        a.write_text("PROBE " + json.dumps({"build": BUILD, "entries": torch_entries, "python": "3.12.14",
                                            "numpy": "2.3", "torch": "2.14.0", "torch_threads": 12}) + "\n")
        b.write_text("PROBE " + json.dumps({"build": {**BUILD, "model_sha": release_sha}, "entries": release_entries,
                                            "python": "3.12.14", "numpy": "2.3", "ai_edge_litert": "2.3.0"}) + "\n")
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                compare_classify(a, b, 0.0, d7=True)
        except SystemExit as e:
            return False, f"{out.getvalue()}{e.code if e.code != 1 else ''}"
        return True, out.getvalue()

    @staticmethod
    def photos(n: int, **kw) -> dict:
        return {f"p{i:03}.jpg": _entry(**kw) for i in range(n)}

    def test_small_probability_changes_pass(self):
        """Not bit for bit: fp16 on the GPU moves every probability a little (ML-7 P8: p95 0.004)."""
        passed, out = self.compare(self.photos(12), self.photos(12, probs=(0.69, 0.205, 0.105)))
        self.assertTrue(passed, out)
        self.assertIn("12 photos: 0 verdict changes, 0 top-1 changes, p95 largest change 0.0100", out)

    def test_a_verdict_change_fails(self):
        release = self.photos(12)
        release["p003.jpg"] = _entry(verdict="refused (not coffee beans)")
        passed, out = self.compare(self.photos(12), release)
        self.assertFalse(passed, out)
        self.assertIn("FAIL: p003.jpg: verdict refused (not coffee beans) != PyTorch's predicted", out)

    def test_one_top1_change_in_the_twelve_fixtures_fails(self):
        """1% of 12 photos is no photo."""
        release = self.photos(12)
        release["p000.jpg"] = _entry(probs=(0.44, 0.46, 0.1))
        passed, out = self.compare(self.photos(12), release)
        self.assertFalse(passed, out)
        self.assertIn("FAIL: top-1 changed on 1 of 12 photos (bar: 1%)", out)

    def test_one_top1_change_in_a_hundred_passes(self):
        release = self.photos(100)
        release["p000.jpg"] = _entry(probs=(0.44, 0.46, 0.1))
        passed, out = self.compare(self.photos(100), release)
        self.assertTrue(passed, out)
        self.assertIn("1 top-1 changes", out)

    def test_the_p95_of_the_largest_change_is_the_bar(self):
        one_far = self.photos(20)
        one_far["p000.jpg"] = _entry(probs=(0.6, 0.3, 0.1))          # 0.1 on one photo of 20: p95 below 0.05
        self.assertTrue(self.compare(self.photos(20), one_far)[0])
        all_far = self.photos(20, probs=(0.62, 0.28, 0.1))           # 0.08 on every photo
        passed, out = self.compare(self.photos(20), all_far)
        self.assertFalse(passed, out)
        self.assertIn("FAIL: p95 of the largest probability change 0.0800 > 0.05", out)

    def test_a_photo_the_release_did_not_answer_fails(self):
        release = self.photos(12)
        del release["p005.jpg"]
        passed, out = self.compare(self.photos(12), release)
        self.assertFalse(passed, out)
        self.assertIn("FAIL: p005.jpg: no answer from the release", out)

    def test_a_non_finite_probability_fails(self):
        release = self.photos(12)
        release["p001.jpg"] = _entry(probs=(0.7, float("nan"), 0.1))
        passed, out = self.compare(self.photos(12), release)
        self.assertFalse(passed, out)
        self.assertIn("FAIL: p001.jpg: a non-finite class probability", out)

    def test_another_model_is_refused(self):
        passed, out = self.compare(self.photos(12), self.photos(12), release_sha="0" * 16)
        self.assertFalse(passed)
        self.assertIn("different models", out)


if __name__ == "__main__":
    unittest.main()
