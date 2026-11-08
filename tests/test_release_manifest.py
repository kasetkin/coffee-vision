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
from release_manifest import LITERT_RUNTIME, VK_ICD, all_files, front_manifest, manifest, unexpected  # noqa: E402

from coffeecv import canonical  # noqa: E402
from tests._tiers import real_data  # noqa: E402

# The model the live release serves, the one with a LiteRT export (dvc.yaml's litert_export, ticket ML-7 P9).
MODELS = ("dino3b16_segxl0v7r_country_aug8_s42",)
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
r = client.post("/analyze", data={"photo": (io.BytesIO(photo), "photo.jxl")}, content_type="multipart/form-data")
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
                    # The host takes canonical photos only (ML-19 D12): the photo crosses as the front sends it,
                    # made here so that the traced process opens only what the app opens.
                    canon = Path(logs) / "smoke.jxl"
                    canon.write_bytes(canonical.canonicalize_file(PHOTO))
                    out = subprocess.run([sys.executable, "-c", TRACE, str(REPO), str(canon)], cwd=REPO,
                                         env=env, capture_output=True, text=True, timeout=900)
                self.assertEqual(out.returncode, 0, out.stderr[-3000:])
                trace = json.loads(next(line[6:] for line in out.stdout.splitlines()
                                        if line.startswith("TRACE ")))
                self.assertEqual(trace["status"], {"/analyze": [200, ["preview", "classification"]]})
                outside = sorted(set(trace["opened"]) - all_files(m))
                self.assertEqual(outside, [], f"{model} opened repo files outside the manifest")
                # The other direction: the manifest names no code or model file the app never needs.
                # (pyproject/uv.lock build the venv and the static files are nginx's; none is opened in-process.)
                unused = sorted(f for f in all_files(m) - set(trace["opened"]) - set(m["generated"])
                                - {"webapp/pyproject.toml", "webapp/uv.lock"} if not f.startswith("webapp/static/"))
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



class TestTwoReleases(unittest.TestCase):
    """OPS-15 (ADR 0031): one release ID, two trees. The front's holds the page, the agent files and the check service;
    the inference host's holds the app and the model, and its own nginx site, rendered at build."""

    def test_the_page_ships_to_the_front_not_the_inference_host(self):
        static = {p.relative_to(REPO).as_posix() for p in (REPO / "webapp/static").rglob("*") if p.is_file()}
        front = front_manifest()
        self.assertEqual(static - set(front["git"]), set())
        self.assertEqual(static & set(_synthetic(CARD)["git"]), set())

    def test_the_front_holds_the_check_service_and_nothing_of_the_models(self):
        front = front_manifest()
        self.assertTrue({"webapp/front/check_service.py", "webapp/front/upload_check.py", "webapp/front/pyproject.toml",
                         "webapp/front/uv.lock"} <= set(front["git"]))
        # Since ticket ML-19 P9 the front canonicalizes uploads: coffeecv's canonicalization and nothing else of it.
        self.assertEqual({p for p in front["git"] if p.startswith("coffeecv/")},
                         {"coffeecv/__init__.py", "coffeecv/canonical.py", "coffeecv/jxl_encode.py",
                          "coffeecv/jxl_header.py", "coffeecv/data/rec2020_bt709.icc"})
        self.assertFalse(any(p.startswith("models/") or p == "webapp/app.py" for p in front["git"]))
        for p in front["git"]:
            with self.subTest(path=p):
                self.assertTrue((REPO / p).is_file(), f"{p} is not in the repo")

    def test_the_fronts_coffeecv_modules_import_only_each_other(self):
        """What check_service imports from coffeecv, followed module by module, is all in the front's release."""
        import ast
        shipped = {p for p in front_manifest()["git"] if p.endswith(".py")}
        todo, seen = ["webapp/front/check_service.py"], set()
        while todo:
            path = todo.pop()
            if path in seen:
                continue
            seen.add(path)
            for node in ast.walk(ast.parse((REPO / path).read_text())):
                names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else \
                    [a.name for a in node.names] if isinstance(node, ast.Import) else []
                for name in names:
                    if name.startswith("coffeecv"):
                        mod = name if name != "coffeecv" else "coffeecv." + node.names[0].name
                        todo.append(mod.replace(".", "/") + ".py")
        self.assertTrue(seen - {"webapp/front/check_service.py"})
        self.assertEqual(seen - shipped, set())

    def test_each_release_carries_its_own_site(self):
        self.assertIn("inference.nginx.conf", _synthetic(CARD)["generated"])
        self.assertEqual(set(front_manifest()["generated"]),
                         {"front.nginx.conf", "front.env", "webapp/BUILD_INFO.json"})

    def test_a_staged_front_is_checked_like_a_staged_inference_release(self):
        front = front_manifest()
        listing = sorted(all_files(front)) + [".venv/bin/python", ".complete",
                                              "webapp/front/__pycache__/check_service.cpython-312.pyc"]
        self.assertEqual(unexpected(front, listing), [])
        self.assertEqual(unexpected(front, listing + ["webapp/app.py"]), ["webapp/app.py"])


if __name__ == "__main__":
    unittest.main()
