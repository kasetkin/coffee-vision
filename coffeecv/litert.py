"""Serving's models on LiteRT (ticket ML-7, ADR 0019): the segmenter's encoder and decoder, and DINOv3 with its head,
from the converted files a release carries beside its checkpoint. No torch.

The release's converted model is a directory beside the checkpoint, `<checkpoint stem>.litert/`, with a
manifest.json (MANIFEST_FORMAT) that P9's export writes:

    checkpoint_sha256          the .pt the files were converted from
    segmenter                  {weights_sha256, decoder_sha256}: the segmenter's identity, as BeanSegmenter checks it
    backbone                   {name, weights_sha256}: the DINOv3 weights the dino graph was converted from
    head                       {file, sha256, readout}: the head's weight and bias as an .npz, on DINOv3's readout
    graphs                     {encoder, decoder, dino}: each {file, sha256, accelerator, precision, batch, inputs,
                               outputs, forms}; inputs and outputs are the shapes in signature order
    versions                   {torch, litert_torch, ai_edge_litert}

The accelerator and each graph's precision are the manifest's, chosen for the serving host (ADR 0019): "gpu" is
LiteRT's WebGPU accelerator (Dawn on Vulkan), "cpu" XNNPACK. A GPU graph is trusted only when the delegate's own
report puts every op on the GPU (ADR 0020); is_fully_accelerated() is not read, since it once said true for a graph
the delegate had split (ML-7 P2). LiteRT reports from C++, so each graph's load and first call run under
CapturedStderr. Lifted from analysis/ml7_gpu/run_litert.py, where P2-P7 measured it.

On the GPU, LiteRT is given `runtime_dir`, a directory holding libLiteRt.so and the WebGPU accelerator library only:
pointed at the wheel's whole directory, it tries each GPU library it finds (ML-7 P2).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coffeecv import model_io

MANIFEST_FORMAT = "coffeecv.litert/1"
GRAPHS = ("encoder", "decoder", "dino")
ACCELERATORS = ("gpu", "cpu")
PRECISIONS = ("fp32", "fp16")
THREADS = 4                        # the release's thread count (ML-7 P0), for XNNPACK and LiteRT's CPU work


class LiteRtLoadError(RuntimeError):
    """A converted model that must not serve: a bad manifest or file, or an accelerator that did not take a graph."""


def litert_dir_for(checkpoint: Path) -> Path:
    """Where a checkpoint's converted model lives: beside it, named after it, as its OOD files are."""
    return checkpoint.with_suffix(".litert")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest(model_dir: Path) -> dict:
    """The manifest, checked: its format, every graph present with a known accelerator and precision, and every
    file's sha256. Refuses rather than serving a file that is not the one the manifest names."""
    path = model_dir / "manifest.json"
    if not path.exists():
        raise LiteRtLoadError(f"{path}: no manifest")
    m = json.loads(path.read_text())
    if m.get("format") != MANIFEST_FORMAT:
        raise LiteRtLoadError(f"{path}: format {m.get('format')!r}, not {MANIFEST_FORMAT}")
    missing = [g for g in GRAPHS if g not in m.get("graphs", {})]
    if missing:
        raise LiteRtLoadError(f"{path}: no {', '.join(missing)} graph")
    for name in GRAPHS:
        g = m["graphs"][name]
        if g.get("accelerator") not in ACCELERATORS or g.get("precision") not in PRECISIONS:
            raise LiteRtLoadError(f"{path}: {name} has accelerator {g.get('accelerator')!r} and precision "
                                  f"{g.get('precision')!r}; expected one of {ACCELERATORS} and {PRECISIONS}")
        if g["accelerator"] == "cpu" and g["precision"] != "fp32":
            raise LiteRtLoadError(f"{path}: {name} is fp16 on the CPU, which XNNPACK does not run here; a CPU "
                                  f"host's precisions are measured there (ADR 0019)")
    for entry in (*m["graphs"].values(), m["head"]):
        f = model_dir / entry["file"]
        if not f.exists():
            raise LiteRtLoadError(f"{f}: missing")
        if _sha256(f) != entry["sha256"]:
            raise LiteRtLoadError(f"{f}: sha256 {_sha256(f)[:12]}, not the manifest's {entry['sha256'][:12]}")
    return m


SPLIT = re.compile(r"(\d+) operations will run on the GPU, and the remaining (\d+) operations will run on the CPU")
REFUSED_HEADER = "ERROR: Following operations are not supported by GPU delegate:"


def delegate_report(log: str) -> dict:
    """The GPU delegate's own placement report, read from LiteRT's log: the ops it refused and its split. No
    split line after a GPU environment was created means the delegate took the whole graph."""
    lines = log.splitlines()
    refused = []
    if REFUSED_HEADER in lines:
        for line in lines[lines.index(REFUSED_HEADER) + 1:]:
            if SPLIT.search(line) or not re.match(r"[A-Z0-9_]+: ", line):
                break
            refused.append(line)
    m = SPLIT.search(log)
    gpu_env = "Created LiteRT GpuEnvironment" in log
    return {"gpu_environment": gpu_env, "refused": refused,
            "gpu_ops": int(m.group(1)) if m else None, "cpu_ops": int(m.group(2)) if m else None,
            "all_on_gpu": gpu_env and m is None and not refused}


class CapturedStderr:
    """fd 2 into a temporary file while open, then written on to the real stderr; .text holds it. LiteRT reports
    from C++, below Python's sys.stderr."""

    def __enter__(self):
        sys.stderr.flush()
        self._saved = os.dup(2)
        self._tmp = tempfile.TemporaryFile()
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode(errors="replace")
        self._tmp.close()
        sys.stderr.write(self.text)
        sys.stderr.flush()


class Graph:
    """One .tflite on one accelerator. Called with numpy arrays in signature order; returns float32 arrays.

    LiteRT's options are passed in full: the hardware_accel= shortcut silently sets one CPU thread (ML-7 P1)."""

    def __init__(self, path: Path, accelerator: str, precision: str, runtime_dir: Path | None = None,
                 threads: int = THREADS):
        from ai_edge_litert.compiled_model import (CompiledModel, CpuOptions, Environment, EnvironmentOptions,
                                                   GpuOptions, HardwareAccelerator, Options)
        gpu = accelerator == "gpu"
        if gpu and runtime_dir is None:
            raise LiteRtLoadError(f"{path.name}: the GPU needs runtime_dir, LiteRT's WebGPU accelerator directory")
        hw = HardwareAccelerator.GPU | HardwareAccelerator.CPU if gpu else HardwareAccelerator.CPU
        options = Options(hardware_accelerators=hw, cpu_options=CpuOptions(num_threads=threads),
                          gpu_options=GpuOptions(enforce_f32=precision == "fp32") if gpu else None)
        env = Environment.create(options=EnvironmentOptions(runtime_path=str(runtime_dir))) if gpu else None
        self.name, self.accelerator, self.precision = path.stem, accelerator, precision
        self.model = CompiledModel.from_file(str(path), environment=env, options=options)
        sig = self.model.get_signature_by_index(0)
        key = sig.get("key") or list(self.model.get_signature_list())[0]
        out_details = self.model.get_output_tensor_details(key)
        in_details = self.model.get_input_tensor_details(key)
        self.input_shapes = [tuple(int(d) for d in in_details[n]["shape"]) for n in sig["inputs"]]
        self.output_shapes = [tuple(int(d) for d in out_details[n]["shape"]) for n in sig["outputs"]]
        self._inputs = self.model.create_input_buffers(0)
        self._outputs = self.model.create_output_buffers(0)

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        for buf, a, shape in zip(self._inputs, arrays, self.input_shapes):
            if a.shape != shape:
                raise ValueError(f"{self.name}: input {a.shape}, the graph takes {shape}")
            buf.write(np.ascontiguousarray(a, dtype=np.float32))
        self.model.run_by_index(0, self._inputs, self._outputs)
        return [buf.read(int(np.prod(s)), np.float32).reshape(s) for buf, s in zip(self._outputs, self.output_shapes)]


def open_graph(model_dir: Path, spec: dict, runtime_dir: Path | None = None) -> tuple[Graph, dict]:
    """(the graph, its delegate report), the graph loaded and called once on zeros under CapturedStderr, so the
    delegate has placed it. A GPU graph with any op off the GPU is refused (ADR 0020)."""
    with CapturedStderr() as err:
        g = Graph(model_dir / spec["file"], spec["accelerator"], spec["precision"], runtime_dir)
        g(*(np.zeros(s, np.float32) for s in g.input_shapes))
    if [list(s) for s in g.input_shapes] != spec["inputs"] or [list(s) for s in g.output_shapes] != spec["outputs"]:
        raise LiteRtLoadError(f"{spec['file']}: shapes {g.input_shapes} -> {g.output_shapes}, not the manifest's "
                              f"{spec['inputs']} -> {spec['outputs']}")
    report = delegate_report(err.text)
    if spec["accelerator"] == "gpu" and not report["all_on_gpu"]:
        raise LiteRtLoadError(f"{spec['file']}: not wholly on the GPU (environment {report['gpu_environment']}, "
                              f"{report['gpu_ops']} ops on the GPU and {report['cpu_ops']} on the CPU, refused "
                              f"{report['refused']})")
    return g, report


class LiteRtSam:
    """The segmenter's encoder and decoder graphs behind BeanSegmenter's engine interface (segment_beans.TorchSam's
    docstring). The decoder graph takes the box prompt alone and returns the three multimask outputs, as the served
    segmenter (box prompt, multi3) uses it: points, and the single-mask output, are refused."""

    frame = image_side = 1024          # XL0: the encoder's input and the prompt frame are both 1024

    def __init__(self, encoder: Graph, decoder: Graph, weights_sha256: str, decoder_sha256: str | None):
        self.encoder, self.decoder = encoder, decoder
        self.weights_sha256, self.decoder_sha256 = weights_sha256, decoder_sha256
        self.features = None

    def set_image(self, x: np.ndarray) -> None:
        self.features = self.encoder(x)[0]

    def decode(self, box, points, labels, multimask: bool) -> tuple[np.ndarray, np.ndarray]:
        if points is not None or box is None or not multimask:
            raise ValueError("the LiteRT decoder takes the box prompt alone, multimask")
        low, iou = self.decoder(self.features, np.asarray(box, np.float32))
        return low[0], iou[0]


class LiteRtClassifier:
    """DINOv3's readout on LiteRT at the graph's batch, then the head and softmax in numpy (model_io)."""

    def __init__(self, dino: Graph, head: dict):
        self.dino = dino
        self.weight, self.bias = head["weight"], head["bias"]
        self.batch = dino.input_shapes[0][0]
        self.embedding_dim = int(self.weight.shape[1])

    def readouts(self, x: np.ndarray) -> np.ndarray:
        """float32 [N, 3, 224, 224] -> [N, embedding_dim], N a multiple of the graph's batch."""
        if len(x) % self.batch:
            raise ValueError(f"{len(x)} patches is not a multiple of the graph's batch {self.batch}")
        return np.concatenate([self.dino(x[i:i + self.batch])[0] for i in range(0, len(x), self.batch)])

    def probs_and_embeddings(self, x: np.ndarray, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if tta:
            raise ValueError("TTA is not served: the frozen DINOv3 model is served without it")
        embeds = self.readouts(x)
        return model_io.softmax(model_io.head_logits(embeds, self.weight, self.bias)), embeds


@dataclass
class LiteRtModels:
    """A release's converted model, loaded: the segmenter's engine, the classifier, the manifest and each graph's
    delegate report."""
    sam: LiteRtSam
    classifier: LiteRtClassifier
    manifest: dict
    reports: dict


def save_head(path: Path, weight: np.ndarray, bias: np.ndarray) -> None:
    """The frozen model's Linear head as serving reads it: weight [classes, embedding_dim] and bias, float32."""
    np.savez(path, weight=np.asarray(weight, np.float32), bias=np.asarray(bias, np.float32))


def write_manifest(model_dir: Path, graphs: dict[str, dict], head: dict, checkpoint_sha256: str, segmenter: dict,
                   backbone: dict, versions: dict) -> dict:
    """Write model_dir/manifest.json for files already in model_dir. `graphs` maps each of GRAPHS to {file,
    accelerator, precision, forms}; each file's sha256, shapes and batch are read here, the shapes by loading it
    on the CPU. `head` is {file, readout}. Returns the manifest, as load_manifest will check it."""
    out = {"format": MANIFEST_FORMAT, "checkpoint_sha256": checkpoint_sha256, "segmenter": segmenter,
           "backbone": backbone, "head": {**head, "sha256": _sha256(model_dir / head["file"])},
           "graphs": {}, "versions": versions}
    for name in GRAPHS:
        spec = graphs[name]
        g = Graph(model_dir / spec["file"], "cpu", "fp32")
        out["graphs"][name] = {**spec, "sha256": _sha256(model_dir / spec["file"]),
                               "batch": g.input_shapes[0][0], "inputs": [list(s) for s in g.input_shapes],
                               "outputs": [list(s) for s in g.output_shapes]}
        del g
    (model_dir / "manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def load_models(model_dir: Path, runtime_dir: Path | None = None, on_cpu: bool = False) -> LiteRtModels:
    """Every graph the manifest names, on its accelerator at its precision, checked (load_manifest, open_graph).
    `on_cpu` puts every graph on XNNPACK at fp32 whatever the manifest names, for checks off the serving host
    (export_litert's finish): exact forms, so an fp16 graph's file at fp32 is held to the fp32 bars."""
    m = load_manifest(model_dir)
    graphs, reports = {}, {}
    for name in GRAPHS:
        spec = {**m["graphs"][name], "accelerator": "cpu", "precision": "fp32"} if on_cpu else m["graphs"][name]
        graphs[name], reports[name] = open_graph(model_dir, spec, runtime_dir)
    with np.load(model_dir / m["head"]["file"]) as z:
        head = {"weight": z["weight"].astype(np.float32), "bias": z["bias"].astype(np.float32)}
    seg = m["segmenter"]
    return LiteRtModels(LiteRtSam(graphs["encoder"], graphs["decoder"], seg["weights_sha256"], seg.get("decoder_sha256")),
                        LiteRtClassifier(graphs["dino"], head), m, reports)
