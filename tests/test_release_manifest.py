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
from release_manifest import all_files, manifest  # noqa: E402

MODELS = ("allrigs_dino3b16_s123", "allrigs_cam_s123")
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
                                - {"webapp/pyproject.toml", "webapp/uv.lock", "webapp/static/index.html"})
                self.assertEqual(unused, [], f"{model}: manifest lists files the app never opened")


class TestSegmenterModelsRefused(unittest.TestCase):
    """Ticket ML-2: until P6 defines a segmenter model's release (its modules, vendored files, weights and
    webapp env), the manifest refuses one rather than stage a release that cannot boot."""

    def test_refused(self):
        card = {"training_config": {"model_name": "dinov3_vitb16", "crop_method": "segment"}}
        files = {"models/m.pt.dvc": "outs:\n- md5: 0\n", "models/m.json": json.dumps(card)}
        with self.assertRaisesRegex(ValueError, "P6"):
            manifest("m", files.__getitem__, files.__contains__)


if __name__ == "__main__":
    unittest.main()
