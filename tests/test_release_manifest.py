"""The release manifest is complete: serving never opens a repo file outside it. Plain unittest.

    python -m unittest discover -s tests -p 'test_release_manifest.py'

For each shippable model, a fresh interpreter imports webapp.app under a Python audit hook, sends a real
photo to all three endpoints (/classify, /crop, /preview) and reports every file under the repo it
opened. Anything outside webapp/deploy/release_manifest.py fails the test: a release built from that
list would crash, or silently fall back, in production (docs/ops1_release_isolation_plan.html §4.2).
Needs the DVC-tracked .pt files, the DINOv3 backbone and one raw photo; skips with a reason otherwise.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "webapp" / "deploy"))
from release_manifest import SEG_LIBRARY, SEG_VENDORED, all_files, manifest  # noqa: E402

# allrigs_dino3b16_seg_s7: the ML-2 segmenter model (P6); skipped until it is shipped.
MODELS = ("allrigs_dino3b16_s123", "allrigs_cam_s123", "allrigs_dino3b16_seg_s7", "allrigs_dino3b16_seg_country_s123")
# Shipped beside the code they describe, never opened by it.
DOCS = {"third_party/efficientvit/LICENSE", "third_party/efficientvit/PATCHES.md"}
# The deploy's smoke photo, from the same list the deploy uses.
PHOTO = REPO / next(line.split()[2] for line in (REPO / "webapp/deploy/fixtures.txt").read_text().splitlines()
                    if line.startswith("smoke "))

TRACE = r"""
import json, sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
photo = Path(sys.argv[2]).read_bytes()          # read before the hook: the test's file, not the app's
opened = set()

def hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        try:
            rel = p.resolve().relative_to(repo)
        except (ValueError, OSError):
            return
        parts = rel.parts
        if len(parts) >= 2 and parts[-2] == "__pycache__":     # bytecode stands for its module
            rel = Path(*parts[:-2], parts[-1].split(".", 1)[0] + ".py")
        opened.add(rel.as_posix())

sys.addaudithook(hook)
import io
import webapp.app as app
client = app.app.test_client()
status = {}
for endpoint in ("/classify", "/crop", "/preview"):
    r = client.post(endpoint, data={"photo": (io.BytesIO(photo), "photo.jpg")},
                    content_type="multipart/form-data")
    status[endpoint] = r.status_code
print("TRACE " + json.dumps({"opened": sorted(opened), "status": status}))
"""


def _have(model: str) -> str | None:
    if not (REPO / f"models/{model}.json").is_file():
        return f"models/{model}.json is not present"
    m = manifest(model, lambda p: (REPO / p).read_text(), lambda p: (REPO / p).is_file())
    for p in [*m["dvc"], *m["pretrained"]]:
        if not (REPO / p).is_file():
            return f"{p} is not present"
    return None if PHOTO.is_file() else f"{PHOTO} is not present"


class TestReleaseManifest(unittest.TestCase):
    def test_every_opened_file_is_in_the_manifest(self):
        for model in MODELS:
            with self.subTest(model=model):
                missing = _have(model)
                if missing:
                    self.skipTest(missing)
                m = manifest(model, lambda p: (REPO / p).read_text(), lambda p: (REPO / p).is_file())
                with tempfile.TemporaryDirectory() as logs:
                    env = {**os.environ, "COFFEE_CV_CHECKPOINT": f"models/{model}.pt",
                           "COFFEE_CV_LOG_DIR": logs, "PYTHONDONTWRITEBYTECODE": "1"}
                    out = subprocess.run([sys.executable, "-c", TRACE, str(REPO), str(PHOTO)], cwd=REPO,
                                         env=env, capture_output=True, text=True, timeout=900)
                self.assertEqual(out.returncode, 0, out.stderr[-3000:])
                trace = json.loads(next(line[6:] for line in out.stdout.splitlines()
                                        if line.startswith("TRACE ")))
                self.assertEqual(trace["status"], {"/classify": 200, "/crop": 200, "/preview": 200})
                outside = sorted(set(trace["opened"]) - all_files(m))
                self.assertEqual(outside, [], f"{model} opened repo files outside the manifest")
                # The other direction: the manifest names no code or model file the app never needs.
                # (pyproject/uv.lock build the venv and index.html is nginx's; neither is opened in-process.)
                unused = sorted(all_files(m) - set(trace["opened"]) - set(m["generated"])
                                - {"webapp/pyproject.toml", "webapp/uv.lock", "webapp/static/index.html"} - DOCS)
                self.assertEqual(unused, [], f"{model}: manifest lists files the app never opened")


SEG_TRACE = r"""
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
repo = Path(sys.argv[1]).resolve()
opened = set()

def hook(event, args):
    if event == "open" and isinstance(args[0], (str, bytes, Path)):
        p = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        try:
            rel = p.resolve().relative_to(repo)
        except (ValueError, OSError):
            return
        parts = rel.parts
        if len(parts) >= 2 and parts[-2] == "__pycache__":
            rel = Path(*parts[:-2], parts[-1].split(".", 1)[0] + ".py")
        opened.add(rel.as_posix())

from coffeecv.config import RunConfig
cfg = replace(RunConfig.from_params_yaml(), crop_method="segment")
sys.addaudithook(hook)
from coffeecv import infer
infer.segmenter_for(cfg)
rgb = (np.random.default_rng(0).random((600, 800, 3)) * 255).astype(np.uint8)
infer.segment_bean_region(rgb, cfg)
print("TRACE " + json.dumps(sorted(opened)))
"""


def _seg_card(**over) -> dict:
    from coffeecv.config import RunConfig
    cfg = RunConfig.from_params_yaml()
    training = {"model_name": "dinov3_vitb16", "crop_method": "segment", "seg_weights": cfg.seg_weights,
                "seg_weights_sha256": cfg.seg_weights_sha256, "seg_decoder": cfg.seg_decoder,
                "seg_decoder_sha256": cfg.seg_decoder_sha256, **over}
    return {"training_config": training, "dino": {"weights": "dinov3/w.pth"}}


def _seg_manifest(card: dict) -> dict:
    pretrained = json.loads((REPO / "models_pretrained/manifest.json").read_text())
    pretrained.append({"path": "dinov3/w.pth", "sha256": "d" * 64})
    files = {"models/m.pt.dvc": "outs:\n- md5: 0\n", "models/m.json": json.dumps(card),
             "models_pretrained/manifest.json": json.dumps(pretrained)}
    return manifest("m", files.__getitem__, files.__contains__)


class TestSegmenterRelease(unittest.TestCase):
    """Ticket ML-2 P6: a crop_method "segment" model's release adds the segmenter's code, the vendored
    EfficientViT files L0 imports, and the encoder and decoder weights its card pins by sha256."""

    def test_shape(self):
        from coffeecv.config import RunConfig
        cfg = RunConfig.from_params_yaml()
        m = _seg_manifest(_seg_card())
        self.assertTrue(set(SEG_LIBRARY + SEG_VENDORED) <= set(m["git"]))
        self.assertEqual(m["pretrained"][f"models_pretrained/{cfg.seg_weights}"], cfg.seg_weights_sha256)
        self.assertEqual(m["pretrained"][cfg.seg_decoder], cfg.seg_decoder_sha256)
        self.assertEqual(m["pretrained"]["models_pretrained/dinov3/w.pth"], "d" * 64)

    def test_encoder_digest_must_match_the_pretrained_manifest(self):
        with self.assertRaisesRegex(ValueError, "pins"):
            _seg_manifest(_seg_card(seg_weights_sha256="0" * 64))

    def test_unknown_crop_method_refused(self):
        with self.assertRaisesRegex(ValueError, "no release shape"):
            _seg_manifest(_seg_card(crop_method="mystery"))

    def test_tray_model_unchanged(self):
        for model in ("allrigs_cam_s123", "allrigs_dino3b16_s123"):
            m = manifest(model, lambda p: (REPO / p).read_text(), lambda p: (REPO / p).is_file())
            self.assertFalse(any(p.startswith("third_party/") for p in m["git"]), model)

    def test_segmenter_opens_only_manifest_files(self):
        """The import trace behind SEG_VENDORED, rerun: building L0 and segmenting one photo opens no
        repo file the segmenter release lacks. Runs before any segmenter model is shipped."""
        from coffeecv.config import RunConfig
        cfg = RunConfig.from_params_yaml()
        for p in (f"models_pretrained/{cfg.seg_weights}", cfg.seg_decoder):
            if not (REPO / p).is_file():
                self.skipTest(f"{p} is not present")
        out = subprocess.run([sys.executable, "-c", SEG_TRACE, str(REPO)], cwd=REPO, capture_output=True,
                             text=True, timeout=900, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        opened = set(json.loads(next(line[6:] for line in out.stdout.splitlines() if line.startswith("TRACE "))))
        m = _seg_manifest(_seg_card())
        self.assertEqual(sorted(opened - all_files(m)), [])
        self.assertEqual(sorted(set(SEG_VENDORED) - opened), [], "SEG_VENDORED lists files L0 never imports")


if __name__ == "__main__":
    unittest.main()
