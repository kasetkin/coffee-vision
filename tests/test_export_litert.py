"""Tests for coffeecv.export_litert, the converted model a release serves (ticket ML-7 D6, plan P9). Plain unittest.

Run from the repo root:  python -m unittest tests.test_export_litert -v

No model and no litert-torch: the conversion's graphs, its forms and the converter are stand-ins, so what is tested is
which form each graph gets and what a failed check leaves behind. The real export is the dvc.yaml stage litert_export,
and its check is D7's (check.json in the converted directory).
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from coffeecv import export_litert, litert, litert_forms  # noqa: F401 -- torch loads here, not inside patch.dict(sys.modules)


class TestProfiles(unittest.TestCase):
    def test_every_profile_names_every_graph(self):
        for name, profile in export_litert.PROFILES.items():
            with self.subTest(profile=name):
                self.assertEqual(set(profile), set(litert.GRAPHS))
                for acc, prec in profile.values():
                    self.assertIn(acc, litert.ACCELERATORS)
                    self.assertIn(prec, litert.PRECISIONS)

    def test_fp16_only_on_the_gpu(self):
        """ADR 0019: fp16 on a CPU needs hardware fp16, measured on that host first; load_manifest refuses it."""
        for name, profile in export_litert.PROFILES.items():
            for graph, (acc, prec) in profile.items():
                if prec == "fp16":
                    self.assertEqual(acc, "gpu", f"{name}: {graph}")

    def test_yappys_profile_is_d4(self):
        self.assertEqual(export_litert.PROFILES["gpu_split"],
                         {"encoder": ("gpu", "fp32"), "decoder": ("gpu", "fp32"), "dino": ("gpu", "fp16")})

    def test_graph_files(self):
        self.assertEqual(set(export_litert.GRAPH_FILES), set(litert.GRAPHS))


class TestConvert(unittest.TestCase):
    """Which forms each graph gets: litert_gpu_forms on all three, fp16_forms on the fp16 ones only."""

    def run_convert(self, profile: str) -> tuple[dict, list]:
        calls = []

        class Module:
            def __init__(self, name):
                self.name = name

            def eval(self):
                return self

        class Example:
            def __init__(self, batch):
                self.shape = (batch,)

        graphs = {"encoder": (Module("encoder"), (Example(1),)), "decoder": (Module("decoder"), (Example(1),)),
                  "dino_b8": (Module("dino"), (Example(8),)), "dino_b40": (Module("dino"), (Example(40),))}

        def converted(module, example):
            def export(path):
                Path(path).write_bytes(module.name.encode())
            return types.SimpleNamespace(export=export)

        fake_litert_torch = types.SimpleNamespace(convert=converted)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(sys.modules, {"litert_torch": fake_litert_torch}), \
                mock.patch("coffeecv.litert_forms.prepare", return_value=(graphs, [])), \
                mock.patch("coffeecv.litert_forms.litert_gpu_forms",
                           side_effect=lambda m, batch: calls.append(("gpu", m.name, batch)) or {"n": 1}), \
                mock.patch("coffeecv.litert_forms.fp16_forms",
                           side_effect=lambda m: calls.append(("fp16", m.name)) or {"n": 2}), \
                mock.patch.object(export_litert, "version", return_value="0.9.4"):
            out = export_litert.convert(Path(tmp) / "m.pt", profile)
            record = json.loads((out / "convert.json").read_text())
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["convert.json", "decoder.tflite", "dino_b8.tflite", "encoder.tflite"])
        return record, calls

    def test_gpu_split(self):
        record, calls = self.run_convert("gpu_split")
        self.assertEqual(calls, [("gpu", "encoder", 1), ("gpu", "decoder", 1), ("gpu", "dino", 8), ("fp16", "dino")])
        self.assertEqual(record["forms"]["dino"], {"litert_gpu_forms": {"n": 1}, "fp16_forms": {"n": 2}})
        self.assertEqual(record["profile"], "gpu_split")

    def test_cpu_gets_no_fp16_forms(self):
        _, calls = self.run_convert("cpu")
        self.assertNotIn("fp16", [c[0] for c in calls])


class TestFinish(unittest.TestCase):
    def test_a_failed_check_removes_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m.pt"
            ckpt.write_bytes(b"checkpoint")
            out = litert.litert_dir_for(ckpt)
            out.mkdir()
            (out / "convert.json").write_text(json.dumps({"profile": "gpu_split", "torch": "2.13", "litert_torch": "0.9.4",
                                                          "forms": {g: {} for g in litert.GRAPHS}}))
            cfg = types.SimpleNamespace(classes_file="classes.txt", model_name="dinov3_vitb16", dropout=0.0,
                                        seg_weights_sha256="a" * 64, seg_decoder_sha256="b" * 64)
            head = types.SimpleNamespace(weight=mock.MagicMock(), bias=mock.MagicMock())
            model = types.SimpleNamespace(readout="cls_mean", backbone=types.SimpleNamespace(weights_sha256="c" * 64))
            failing = {"passed": False, "photos": 1, "rows": [{"pass": False}]}
            with mock.patch("coffeecv.infer.config_for_checkpoint", return_value=(cfg, "")), \
                    mock.patch("coffeecv.class_list.load_classes", return_value=types.SimpleNamespace(keys=["a"])), \
                    mock.patch("coffeecv.infer.load_model", return_value=(model, head)), \
                    mock.patch("coffeecv.litert.save_head"), \
                    mock.patch("coffeecv.litert.write_manifest",
                               side_effect=lambda d, *a: (Path(d) / "manifest.json").write_text("{}")), \
                    mock.patch.object(export_litert, "check", return_value=failing):
                with self.assertRaisesRegex(SystemExit, "check failed on 1 photos"):
                    export_litert.finish(ckpt, "gpu_split", [])
            self.assertFalse((out / "manifest.json").exists())
            self.assertTrue((out / "check.json").exists())

    def test_another_profile_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "m.pt"
            out = litert.litert_dir_for(ckpt)
            out.mkdir()
            (out / "convert.json").write_text(json.dumps({"profile": "cpu"}))
            with self.assertRaisesRegex(SystemExit, "converted for profile cpu"):
                export_litert.finish(ckpt, "gpu_split", [])


if __name__ == "__main__":
    unittest.main()
