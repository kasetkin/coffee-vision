"""Tests for coffeecv.litert, serving's LiteRT backend (ticket ML-7 plan P8), and the serving path without torch.
Plain unittest.

Run from the repo root:  python -m unittest tests.test_litert -v

The fast tests need no model: the delegate's report is parsed from lines LiteRT printed on yappy's GPU (ML-7 P4 and
P7's logs), and the graphs are stand-ins. The real-data tests run ML-7 P7's served files (DVC:
analysis/ml7_gpu/artifacts) on the devcontainer's CPU, XNNPACK at fp32, against the PyTorch model on deploy
fixtures, within D7's fp32 bars (ADR 0020). The GPU's own checks run on the serving host (plan P11).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from coffeecv import infer, litert, model_io
from coffeecv.config import REPO_ROOT

from tests._tiers import real_data

ARTIFACTS = REPO_ROOT / "analysis" / "ml7_gpu" / "artifacts"
SERVED = {"encoder.tflite": ARTIFACTS / "tflite" / "encoder.tflite",
          "decoder.tflite": ARTIFACTS / "p7_whole" / "decoder.tflite",
          "dino_b8.tflite": ARTIFACTS / "p7_fp16w" / "dino_b8.tflite"}
CHECKPOINT = REPO_ROOT / "models" / "allrigs_dino3b16_segxl0v3_country_s42.pt"
FIXTURES = [ln.split() for ln in (REPO_ROOT / "webapp/deploy/fixtures.txt").read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]
# The smoke photo and three compare photos: two cameras' JPEGs and a HEIC.
FIXTURE_PHOTOS = [FIXTURES[0][2]] + [p for _, _, p in FIXTURES[1:] if p.lower().endswith(".heic")][:1] \
    + [p for _, _, p in FIXTURES[1:] if not p.lower().endswith(".heic")][:2]

# What LiteRT printed on yappy (ML-7 P7, P4): a graph the delegate took whole, and one it split.
WHOLE_LOG = """\
INFO: [delegate_webgpu.cc:363] Created a WebGPU environment.
INFO: [environment.cc:47] Adding options to the existing LiteRT environment
INFO: [gpu_environment.h:155] Created LiteRT GpuEnvironment.
INFO: [environment.cc:47] Adding options to the existing LiteRT environment
I0000 00:00:1792002825.070402   76081 delegate_kernel.cc:984] Initializing WebGPU-based API from graph.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
"""
SPLIT_LOG = """\
INFO: [gpu_environment.h:155] Created LiteRT GpuEnvironment.
INFO: [environment.cc:47] Adding options to the existing LiteRT environment
ERROR: Following operations are not supported by GPU delegate:
SELECT_V2: Cond must be float or bool type, if, else tensors must be either be same the shape as output or constant, scalar.
336 operations will run on the GPU, and the remaining 38 operations will run on the CPU.
I0000 00:00:1791968486.108140   62865 delegate_kernel.cc:984] Initializing WebGPU-based API from graph.
"""
NO_GPU_LOG = "INFO: Created TensorFlow Lite XNNPACK delegate for CPU.\n"

# The serving modules, imported with torch and the model libraries made unimportable.
NO_TORCH = r"""
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("torch", "torchvision", "timm", "segment_anything", "efficientvit"):
            raise ImportError("serving imported " + name)
sys.meta_path.insert(0, Block())
"""


class TestDelegateReport(unittest.TestCase):
    def test_whole_graph_on_the_gpu(self):
        r = litert.delegate_report(WHOLE_LOG)
        self.assertEqual(r, {"gpu_environment": True, "refused": [], "gpu_ops": None, "cpu_ops": None,
                             "all_on_gpu": True})

    def test_split_graph(self):
        r = litert.delegate_report(SPLIT_LOG)
        self.assertFalse(r["all_on_gpu"])
        self.assertEqual((r["gpu_ops"], r["cpu_ops"]), (336, 38))
        self.assertEqual(len(r["refused"]), 1)
        self.assertTrue(r["refused"][0].startswith("SELECT_V2: "))

    def test_no_gpu_environment_is_not_on_the_gpu(self):
        self.assertFalse(litert.delegate_report(NO_GPU_LOG)["all_on_gpu"])


class TestGpuProof(unittest.TestCase):
    """ADR 0020's proof beyond the delegate's report: the libraries mapped, the render node, the i915 counters."""
    ON_INTEL = {"libraries": ["libLiteRtWebGpuAccelerator.so", "libvulkan_intel.so"],
                "render_nodes": ["/dev/dri/renderD128"]}

    def test_intels_driver_busy_is_proven(self):
        p = litert.gpu_proof(self.ON_INTEL, idle_ms=3000, wall_ms=10000)
        self.assertEqual((p["ok"], p["why_not"], p["busy_frac"]), (True, [], 0.7))

    def test_lavapipe_is_refused_even_beside_a_hardware_driver(self):
        mapped = {**self.ON_INTEL, "libraries": self.ON_INTEL["libraries"] + ["libvulkan_lvp.so"]}
        self.assertIn("libvulkan_lvp (Mesa's CPU renderer) mapped", litert.gpu_proof(mapped, 3000, 10000)["why_not"])

    def test_lavapipe_alone_is_no_hardware_driver(self):
        mapped = {**self.ON_INTEL, "libraries": ["libLiteRtWebGpuAccelerator.so", "libvulkan_lvp.so"]}
        self.assertIn("no hardware Vulkan driver mapped", litert.gpu_proof(mapped, 3000, 10000)["why_not"])

    def test_no_accelerator_library(self):
        mapped = {**self.ON_INTEL, "libraries": ["libvulkan_intel.so"]}
        self.assertEqual(litert.gpu_proof(mapped, 3000, 10000)["why_not"],
                         ["libLiteRtWebGpuAccelerator.so not mapped"])

    def test_no_render_node(self):
        p = litert.gpu_proof({**self.ON_INTEL, "render_nodes": []}, 3000, 10000)
        self.assertEqual(p["why_not"], ["no render node open"])

    def test_an_idle_gpu_is_refused(self):
        p = litert.gpu_proof(self.ON_INTEL, idle_ms=9900, wall_ms=10000)
        self.assertFalse(p["ok"])
        self.assertEqual(p["busy_frac"], 0.01)

    def test_no_counters_is_refused(self):
        self.assertEqual(litert.gpu_proof(self.ON_INTEL, None, 10000)["why_not"], ["no i915 counters"])

    def test_this_process_maps_no_vulkan(self):
        """gpu_mapped reads this process: the test runner opens no GPU."""
        mapped = litert.gpu_mapped()
        self.assertEqual(set(mapped), {"libraries", "render_nodes"})
        self.assertFalse(any(n.startswith("libvulkan_") for n in mapped["libraries"]))


class FakeGraph:
    """Stands in for litert.Graph: prints `log` to fd 2 as LiteRT does from C++, and returns zeros."""
    log = WHOLE_LOG

    def __init__(self, path, accelerator, precision, runtime_dir=None, threads=4):
        os.write(2, self.log.encode())
        self.input_shapes, self.output_shapes = [(1, 4)], [(1, 2)]

    def __call__(self, *arrays):
        return [np.zeros(s, np.float32) for s in self.output_shapes]


class TestOpenGraph(unittest.TestCase):
    SPEC = {"file": "g.tflite", "accelerator": "gpu", "precision": "fp32", "inputs": [[1, 4]], "outputs": [[1, 2]]}

    def open(self, log: str, **spec):
        with mock.patch.object(litert, "Graph", type("G", (FakeGraph,), {"log": log})):
            return litert.open_graph(Path("/nonexistent"), {**self.SPEC, **spec}, Path("/rt"))

    def test_a_whole_gpu_graph_loads(self):
        _, report = self.open(WHOLE_LOG)
        self.assertTrue(report["all_on_gpu"])

    def test_a_split_gpu_graph_is_refused(self):
        with self.assertRaisesRegex(litert.LiteRtLoadError, "not wholly on the GPU.*336 ops on the GPU and 38"):
            self.open(SPLIT_LOG)

    def test_a_gpu_graph_without_a_gpu_environment_is_refused(self):
        with self.assertRaisesRegex(litert.LiteRtLoadError, "not wholly on the GPU"):
            self.open(NO_GPU_LOG)

    def test_a_cpu_graph_needs_no_gpu_report(self):
        _, report = self.open(NO_GPU_LOG, accelerator="cpu")
        self.assertFalse(report["gpu_environment"])

    def test_shapes_other_than_the_manifests_are_refused(self):
        with self.assertRaisesRegex(litert.LiteRtLoadError, "not the manifest's"):
            self.open(WHOLE_LOG, inputs=[[8, 4]])


class TestManifest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        files = {}
        for name in ("encoder.tflite", "decoder.tflite", "dino_b8.tflite", "head.npz"):
            (self.dir / name).write_bytes(name.encode())
            files[name] = hashlib.sha256(name.encode()).hexdigest()
        def graph(f):
            return {"file": f, "sha256": files[f], "accelerator": "gpu", "precision": "fp32", "batch": 1,
                    "inputs": [], "outputs": [], "forms": {}}

        self.m = {"format": litert.MANIFEST_FORMAT, "checkpoint_sha256": "0" * 64,
                  "segmenter": {"weights_sha256": "a" * 64, "decoder_sha256": "b" * 64},
                  "backbone": {"name": "dinov3_vitb16", "weights_sha256": "c" * 64},
                  "head": {"file": "head.npz", "sha256": files["head.npz"], "readout": "cls_mean"},
                  "graphs": {"encoder": graph("encoder.tflite"), "decoder": graph("decoder.tflite"),
                             "dino": {**graph("dino_b8.tflite"), "precision": "fp16"}},
                  "versions": {}}

    def load(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.m))
        return litert.load_manifest(self.dir)

    def test_a_good_manifest_loads(self):
        self.assertEqual(self.load()["graphs"]["dino"]["precision"], "fp16")

    def test_no_manifest(self):
        with self.assertRaisesRegex(litert.LiteRtLoadError, "no manifest"):
            litert.load_manifest(self.dir)

    def test_another_format(self):
        self.m["format"] = "coffeecv.litert/0"
        with self.assertRaisesRegex(litert.LiteRtLoadError, "format"):
            self.load()

    def test_a_missing_graph(self):
        del self.m["graphs"]["decoder"]
        with self.assertRaisesRegex(litert.LiteRtLoadError, "no decoder graph"):
            self.load()

    def test_a_file_that_is_not_the_manifests(self):
        (self.dir / "dino_b8.tflite").write_bytes(b"another file")
        with self.assertRaisesRegex(litert.LiteRtLoadError, "dino_b8.tflite: sha256"):
            self.load()

    def test_a_missing_file(self):
        (self.dir / "head.npz").unlink()
        with self.assertRaisesRegex(litert.LiteRtLoadError, "head.npz: missing"):
            self.load()

    def test_an_unknown_accelerator(self):
        self.m["graphs"]["encoder"]["accelerator"] = "npu"
        with self.assertRaisesRegex(litert.LiteRtLoadError, "accelerator 'npu'"):
            self.load()

    def test_fp16_on_the_cpu_is_refused(self):
        self.m["graphs"]["dino"]["accelerator"] = "cpu"
        with self.assertRaisesRegex(litert.LiteRtLoadError, "fp16 on the CPU"):
            self.load()


class BatchGraph:
    """A dino graph of batch 8 whose readout is each patch's mean, repeated: [8, 3, 2, 2] -> [8, 4]."""
    input_shapes = [(8, 3, 2, 2)]

    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return [np.repeat(x.mean(axis=(1, 2, 3))[:, None], 4, axis=1)]


class TestClassifier(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.head = {"weight": rng.normal(size=(3, 4)).astype(np.float32),
                     "bias": rng.normal(size=3).astype(np.float32)}
        self.graph = BatchGraph()
        self.clf = litert.LiteRtClassifier(self.graph, self.head)
        self.x = rng.normal(size=(40, 3, 2, 2)).astype(np.float32)

    def test_forty_patches_are_five_calls_of_eight(self):
        probs, embeds = self.clf.probs_and_embeddings(self.x)
        self.assertEqual(self.graph.calls, 5)
        self.assertEqual((probs.shape, embeds.shape), ((40, 3), (40, 4)))
        np.testing.assert_allclose(embeds[:, 0], self.x.mean(axis=(1, 2, 3)), rtol=1e-6)
        logits = embeds @ self.head["weight"].T + self.head["bias"]
        want = np.exp(logits - logits.max(1, keepdims=True))
        np.testing.assert_allclose(probs, want / want.sum(1, keepdims=True), rtol=1e-6)
        self.assertEqual(self.clf.embedding_dim, 4)

    def test_tta_is_refused(self):
        with self.assertRaisesRegex(ValueError, "TTA"):
            self.clf.probs_and_embeddings(self.x, tta=True)

    def test_a_count_not_a_multiple_of_the_batch_is_refused(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            self.clf.probs_and_embeddings(self.x[:39])


class TestSamEngine(unittest.TestCase):
    def test_only_the_box_prompt_multimask_is_served(self):
        sam = litert.LiteRtSam(None, None, "a", "b")
        box = np.zeros((1, 4), np.float32)
        for kwargs in ({"box": box, "points": np.zeros((1, 1, 2)), "labels": np.ones((1, 1)), "multimask": True},
                       {"box": box, "points": None, "labels": None, "multimask": False},
                       {"box": None, "points": None, "labels": None, "multimask": True}):
            with self.subTest(**{k: v is not None if k != "multimask" else v for k, v in kwargs.items()}):
                with self.assertRaisesRegex(ValueError, "box prompt alone"):
                    sam.decode(**kwargs)


class TestWarmUp(unittest.TestCase):
    """ADR 0021: the boot's warm-up fails the boot on an error or a non-finite answer."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.photo = Path(tmp.name) / "photo.png"
        from PIL import Image
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(self.photo)
        self.args = (None, ["a", "b"], {"a": "A", "b": "B"}, object(), None, None, None)

    def warm(self, entry: dict, photo: Path | None = None):
        with mock.patch.object(infer, "classify_one", return_value=entry), \
                mock.patch.object(infer, "crop_to_bean_region", return_value=(None, None)):
            return infer.warm_up(photo or self.photo, *self.args)

    def test_a_good_answer_boots(self):
        out = self.warm({"verdict": "predicted", "ranked": [("a", "A", 0.9), ("b", "B", 0.1)],
                         "ood": {"score": 0.01, "threshold": 0.5}})
        self.assertEqual((out["photo"], out["verdict"]), ("photo.png", "predicted"))

    def test_a_non_finite_answer_fails_the_boot(self):
        for entry in ({"verdict": "predicted", "ranked": [("a", "A", float("nan")), ("b", "B", 0.1)], "ood": None},
                      {"verdict": "predicted", "ranked": [("a", "A", 0.9)], "ood": {"score": float("inf")}}):
            with self.subTest(entry=entry), self.assertRaisesRegex(SystemExit, "non-finite"):
                self.warm(entry)

    def test_no_answer_fails_the_boot(self):
        with self.assertRaisesRegex(SystemExit, "no answer"):
            self.warm({"verdict": "REFUSED (unmeasurable)", "error": "too small"})

    def test_a_missing_photo_fails_the_boot(self):
        with self.assertRaisesRegex(SystemExit, "missing"):
            self.warm({}, self.photo.with_name("absent.png"))


class TestServingImportsNoTorch(unittest.TestCase):
    def test_serving_modules(self):
        code = NO_TORCH + "\n".join(f"import {m}" for m in (
            "coffeecv.config", "coffeecv.photo_io", "coffeecv.model_io", "coffeecv.infer", "coffeecv.segment_beans",
            "coffeecv.litert", "coffeecv.class_list"))
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])


def _have_served() -> str | None:
    for f in (*SERVED.values(), CHECKPOINT, *(REPO_ROOT / p for p in FIXTURE_PHOTOS)):
        if not f.exists():
            return f"{f.relative_to(REPO_ROOT)} missing (dvc pull analysis/ml7_gpu/artifacts.dvc, dataset)"
    return None


def build_cpu_dir(out: Path) -> dict:
    """A converted-model directory of P7's served files, every graph on the CPU at fp32, with the checkpoint's head:
    what P9's export will write for a CPU host."""
    from coffeecv.class_list import load_classes
    for name, src in SERVED.items():
        (out / name).symlink_to(src)
    cfg, _ = infer.config_for_checkpoint(CHECKPOINT, None)
    model, head = infer.load_model(CHECKPOINT, cfg.model_name, len(load_classes(REPO_ROOT / cfg.classes_file).keys),
                                   cfg.dropout)
    litert.save_head(out / "head.npz", head.weight.detach().numpy(), head.bias.detach().numpy())
    def cpu(f):
        return {"file": f, "accelerator": "cpu", "precision": "fp32", "forms": {}}

    return litert.write_manifest(
        out, {"encoder": cpu("encoder.tflite"), "decoder": cpu("decoder.tflite"), "dino": cpu("dino_b8.tflite")},
        {"file": "head.npz", "readout": model.readout}, hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        {"weights_sha256": cfg.seg_weights_sha256, "decoder_sha256": cfg.seg_decoder_sha256},
        {"name": cfg.model_name, "weights_sha256": model.backbone.weights_sha256}, {"ai_edge_litert": "2.3.0"})


@real_data
class TestAgainstPyTorch(unittest.TestCase):
    """ADR 0020's fp32 bars on the devcontainer's CPU: the LiteRT path and the PyTorch path through one classify_one,
    on deploy fixtures. The served DINOv3 file carries the fp16 forms, exact in real arithmetic, so at fp32 it
    meets the fp32 bar."""

    @classmethod
    def setUpClass(cls):
        missing = _have_served()
        if missing:
            raise unittest.SkipTest(missing)
        from coffeecv.class_list import load_classes
        from coffeecv.segment_beans import BeanSegmenter, seg_params
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        build_cpu_dir(Path(tmp.name))
        cls.models = litert.load_models(Path(tmp.name))
        cls.cfg, _ = infer.config_for_checkpoint(CHECKPOINT, None)
        classes = load_classes(REPO_ROOT / cls.cfg.classes_file)
        cls.ids, cls.labels = list(classes.keys), dict(classes.labels)
        cls.torch = infer.TorchClassifier(*infer.load_model(CHECKPOINT, cls.cfg.model_name, len(cls.ids),
                                                            cls.cfg.dropout))
        cls.ref = infer.load_ood_reference(CHECKPOINT, cls.torch.embedding_dim)
        cls.probe = infer.load_ood_probe(CHECKPOINT, embedding_dim=cls.torch.embedding_dim)
        cls.seg = {"litert": BeanSegmenter(seg_params(cls.cfg), engine=cls.models.sam),
                   "torch": infer.segmenter_for(cls.cfg)}
        cls.clf = {"litert": cls.models.classifier, "torch": cls.torch}

    def test_cpu_graphs_report_no_gpu(self):
        self.assertFalse(any(r["gpu_environment"] for r in self.models.reports.values()))

    def test_masks_within_d7(self):
        from coffeecv.photo_io import load_rgb_image
        for p in FIXTURE_PHOTOS:
            rgb = load_rgb_image(REPO_ROOT / p)
            a, b = (self.seg[k].predict_mask(rgb) for k in ("litert", "torch"))
            with self.subTest(photo=Path(p).name):
                union = np.count_nonzero(a | b)
                self.assertGreaterEqual(np.count_nonzero(a & b) / union if union else 1.0, 0.999)

    def test_readouts_within_d7(self):
        """Every patch's readout cosine >= 0.99999, on the PyTorch path's own patches."""
        for k, p in enumerate(FIXTURE_PHOTOS):
            patches, _ = infer.patches_for_photo(REPO_ROOT / p, self.cfg, 40, [42, k], segmenter=self.seg["torch"])
            x = np.stack([model_io.eval_patch(q, self.cfg.patch_resize) for q in patches])
            _, a = self.clf["litert"].probs_and_embeddings(x)
            _, b = self.clf["torch"].probs_and_embeddings(x)
            cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
            with self.subTest(photo=Path(p).name):
                self.assertGreaterEqual(float(cos.min()), 0.99999)

    def test_answers_match(self):
        for k, p in enumerate(FIXTURE_PHOTOS):
            e = {b: infer.classify_one(REPO_ROOT / p, self.cfg, self.ids, self.labels, self.clf[b], self.ref,
                                       seed_key=[42, k], tta=False, probe=self.probe, segmenter=self.seg[b])
                 for b in ("litert", "torch")}
            with self.subTest(photo=Path(p).name):
                self.assertEqual(e["litert"]["verdict"], e["torch"]["verdict"])
                self.assertEqual(e["litert"]["ranked"][0][0], e["torch"]["ranked"][0][0])
                np.testing.assert_allclose([v for *_, v in e["litert"]["ranked"]],
                                           [v for *_, v in e["torch"]["ranked"]], atol=1e-3)


@real_data
class TestServiceBoots(unittest.TestCase):
    """webapp.app on LiteRT with torch unimportable: it boots, warms up (ADR 0021) and answers /analyze."""

    def test_boot_and_answer(self):
        missing = _have_served()
        if missing:
            self.skipTest(missing)
        with tempfile.TemporaryDirectory() as tmp:
            build_cpu_dir(Path(tmp))
            code = NO_TORCH + r"""
import io, json, sys
from pathlib import Path
import webapp.app as app
photo = Path(sys.argv[1]).read_bytes()
out = {"warm_up": app.WARM_UP}
r = app.app.test_client().post("/analyze", data={"photo": (io.BytesIO(photo), "photo.jpg")},
                               content_type="multipart/form-data")
out["/analyze"] = [r.status_code, [json.loads(line)["part"] for line in r.get_data(as_text=True).splitlines()]]
print("RESULT " + json.dumps(out))
"""
            env = {**os.environ, "COFFEE_CV_CHECKPOINT": str(CHECKPOINT.relative_to(REPO_ROOT)),
                   "COFFEE_CV_LITERT_DIR": tmp, "COFFEE_CV_LOG_DIR": tmp, "PYTHONDONTWRITEBYTECODE": "1"}
            out = subprocess.run([sys.executable, "-c", code, str(REPO_ROOT / FIXTURE_PHOTOS[1])], cwd=REPO_ROOT,
                                 env=env, capture_output=True, text=True, timeout=900)
        self.assertEqual(out.returncode, 0, out.stderr[-3000:])
        res = json.loads(next(ln[7:] for ln in out.stdout.splitlines() if ln.startswith("RESULT ")))
        self.assertEqual(res["warm_up"]["photo"], Path(FIXTURE_PHOTOS[0]).name)
        for k in ("boot_s", "warm_up_s", "peak_rss_mb", "mem_available_mb"):
            self.assertIn(k, res["warm_up"])
        self.assertEqual(res["/analyze"], [200, ["preview", "classification"]])


if __name__ == "__main__":
    unittest.main()
