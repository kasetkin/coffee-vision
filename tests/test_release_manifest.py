"""The release manifest is complete: serving never opens a repo file outside it. Plain unittest.

    python -m unittest discover -s tests -p 'test_release_manifest.py'

For each shippable model, a fresh interpreter imports webapp.app under a Python audit hook, sends a real
photo to /analyze, reads its whole streamed answer, and reports every file under the repo it
opened. Anything outside webapp/deploy/release_manifest.py fails the test: a release built from that
list would crash, or silently fall back, in production (docs/ops1_release_isolation_plan.html §4.2).
Needs the DVC-tracked .pt, the model's LiteRT export and the smoke photo; skips with a reason otherwise. The app runs
its graphs on the CPU here (COFFEE_CV_LITERT_ON_CPU), which opens the same files the GPU would.
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
from release_manifest import LITERT_RUNTIME, VK_ICD, all_files, manifest  # noqa: E402

from tests._tiers import real_data  # noqa: E402

# The model the live release serves, the one with a LiteRT export (dvc.yaml's litert_export, ticket ML-7 P9).
MODELS = ("dino3b16_segxl0v5_country_aug8_s42",)
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
r = client.post("/analyze", data={"photo": (io.BytesIO(photo), "photo.jpg")}, content_type="multipart/form-data")
parts = [json.loads(line)["part"] for line in r.get_data(as_text=True).splitlines()]   # runs the whole stream
print("TRACE " + json.dumps({"opened": sorted(opened), "status": {"/analyze": [r.status_code, parts]}}))
"""


def _have(model: str) -> str | None:
    if not (REPO / f"models/{model}.json").is_file():
        return f"models/{model}.json is not present"
    if not (REPO / f"models/{model}.litert/manifest.json").is_file():
        return f"models/{model}.litert/ is not present: dvc pull it, or dvc repro -s litert_export@{model} (ML-7 P9)"
    m = manifest(model, lambda p: (REPO / p).read_text(), lambda p: (REPO / p).is_file())
    for p in [*m["dvc"], *m["pinned"]]:
        if not (REPO / p).is_file():
            return f"{p} is not present"
    return None


class TestReleaseManifest(unittest.TestCase):
    @real_data
    def test_every_opened_file_is_in_the_manifest(self):
        for model in MODELS:
            with self.subTest(model=model):
                missing = _have(model)
                if missing:
                    self.skipTest(missing)
                m = manifest(model, lambda p: (REPO / p).read_text(), lambda p: (REPO / p).is_file())
                with tempfile.TemporaryDirectory() as logs:
                    env = {**os.environ, "COFFEE_CV_CHECKPOINT": f"models/{model}.pt", "COFFEE_CV_LOG_DIR": logs,
                           "PYTHONDONTWRITEBYTECODE": "1", "COFFEE_CV_LITERT_ON_CPU": "1",
                           "COFFEE_CV_WARMUP_PHOTO": m["env"]["COFFEE_CV_WARMUP_PHOTO"]}
                    out = subprocess.run([sys.executable, "-c", TRACE, str(REPO), str(PHOTO)], cwd=REPO,
                                         env=env, capture_output=True, text=True, timeout=900)
                self.assertEqual(out.returncode, 0, out.stderr[-3000:])
                trace = json.loads(next(line[6:] for line in out.stdout.splitlines()
                                        if line.startswith("TRACE ")))
                self.assertEqual(trace["status"], {"/analyze": [200, ["preview", "classification"]]})
                outside = sorted(set(trace["opened"]) - all_files(m))
                self.assertEqual(outside, [], f"{model} opened repo files outside the manifest")
                # The other direction: the manifest names no code or model file the app never needs.
                # (pyproject/uv.lock build the venv and index.html is nginx's; neither is opened in-process.)
                unused = sorted(all_files(m) - set(trace["opened"]) - set(m["generated"])
                                - {"webapp/pyproject.toml", "webapp/uv.lock", "webapp/static/index.html"})
                self.assertEqual(unused, [], f"{model}: manifest lists files the app never opened")


LOCK = """stages:
  litert_export@m:
    cmd: convert
    deps:
    - path: models/m.pt
      hash: md5
      md5: 1
    outs:
    - path: models/m.litert
      hash: md5
      md5: abc.dir
      nfiles: 7
  other:
    cmd: x
"""
LISTING = [{"relpath": f, "md5": f"md5-{f}"} for f in
           ("check.json", "convert.json", "decoder.tflite", "dino_b8.tflite", "encoder.tflite", "head.npz",
            "manifest.json")]
FIXTURES_TXT = "# comment\nsmoke 0123 dataset/s/smoke.jpg\ncompare 4567 dataset/s/other.jpg\n"


def _synthetic(card: dict, accelerators=("gpu", "gpu", "gpu"), lock: str = LOCK) -> dict:
    """The manifest of a synthetic model `m`, with its export's listing and converted manifest as DVC objects."""
    files = {"models/m.pt.dvc": "outs:\n- md5: 0\n", "models/m.json": json.dumps(card), "dvc.lock": lock,
             "webapp/deploy/fixtures.txt": FIXTURES_TXT}
    converted = {"graphs": {g: {"accelerator": a} for g, a in zip(("encoder", "decoder", "dino"), accelerators)}}
    objects = {"abc.dir": json.dumps(LISTING), "md5-manifest.json": json.dumps(converted)}
    return manifest("m", files.__getitem__, files.__contains__, objects.__getitem__)


CARD = {"training_config": {"model_name": "dinov3_vitb16", "crop_method": "segment"}}


class TestLiteRtRelease(unittest.TestCase):
    """ML-7 P10: a release holds the serving code, the checkpoint and its LiteRT export, and the warm-up photo; no
    torch-side module, no vendored segmenter and no pretrained weights."""

    def test_shape(self):
        m = _synthetic(CARD)
        self.assertFalse(any(p.startswith(("third_party/", "models_pretrained/")) for p in m["git"]))
        self.assertFalse(any(p.endswith(("backbones.py", "dino_classifier.py", "model.py", "transforms.py",
                                         "dataset.py")) for p in m["git"]))
        self.assertEqual(sorted(p for p in m["dvc"] if ".litert/" in p),
                         [f"models/m.litert/{f}" for f in ("decoder.tflite", "dino_b8.tflite", "encoder.tflite",
                                                           "head.npz", "manifest.json")])
        self.assertEqual(m["dvc"]["models/m.litert/encoder.tflite"], "md5-encoder.tflite")
        self.assertEqual(m["pinned"], {"dataset/s/smoke.jpg": "0123"})

    def test_a_gpu_release_names_its_runtime_and_driver(self):
        self.assertEqual(_synthetic(CARD)["env"], {"COFFEE_CV_WARMUP_PHOTO": "dataset/s/smoke.jpg",
                                                   "COFFEE_CV_LITERT_RUNTIME": LITERT_RUNTIME,
                                                   "VK_ICD_FILENAMES": VK_ICD})

    def test_a_cpu_release_names_neither(self):
        self.assertEqual(_synthetic(CARD, ("cpu", "cpu", "cpu"))["env"],
                         {"COFFEE_CV_WARMUP_PHOTO": "dataset/s/smoke.jpg"})

    def test_a_model_without_an_export_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no litert_export@m stage"):
            _synthetic(CARD, lock="stages:\n  other:\n    cmd: x\n")

    def test_models_the_service_cannot_run_are_refused(self):
        for name, card in (("tray resnet18", {"training_config": {"model_name": "resnet18"}}),
                           ("tray dinov3", {"training_config": {"model_name": "dinov3_vitb16"}})):
            with self.subTest(name), self.assertRaisesRegex(ValueError, "no release shape"):
                _synthetic(card)


if __name__ == "__main__":
    unittest.main()
