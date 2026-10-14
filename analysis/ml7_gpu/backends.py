"""ML-7 P4: every cell's graphs behind one call, so the comparison run (p4_graph.py, p4_replay.py) treats them
alike. open_graph(cell, graph, ...) returns a Model whose __call__ takes numpy arrays and returns numpy arrays:

  dino_b8, dino_b40   patches [B, 3, 224, 224] -> readouts [B, 1536]
  encoder             image [1, 3, 1024, 1024] -> embedding [1, 256, 64, 64]
  decoder             embedding [1, 256, 64, 64], box [1, 4] -> (low_masks [1, 3, 256, 256], ious [1, 3])

The cells (plan P4) and the information-only variants (ticket D4, Q4), each run by its own venv:

  torch        the release's PyTorch modules, unrewritten (graphs.load_shipped; ML7_REPO is the release, and
               COFFEE_CV_CHECKPOINT its checkpoint, as its service sets it); the release's venv
  litert_cpu   LiteRT, XNNPACK                          litert_gpu   LiteRT, Vulkan (WebGPU), fp32
  ort_cpu      ORT's CPU provider                       webgpu       ORT's WebGPU plugin (Vulkan), fp32
  ovcpu        ORT's OpenVINO provider, CPU             ovgpu        ORT's OpenVINO provider, GPU (OpenCL), fp32
  variants     ort_cpu_fused (DINOv3's ORT-fused graphs), ort_cpu_int8 and ovcpu_int8 (the int8 encoder),
               litert_gpu_fp16, webgpu_fp16 (the mixed fp16 copies), webgpu_fp16blanket (DINOv3 b8 with
               nothing kept in fp32), ovgpu_fp16 (OpenVINO's f16 hint), litert_gpu_split (P7: the segmenter's
               encoder and decoder in fp32, DINOv3 in fp16, in one process; LiteRT sets the precision per
               compiled model)

Every cell runs at --threads CPU threads; the GPU cells refuse CPU fallback as in P2 and P3.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import numpy as np

GRAPHS = ("dino_b8", "dino_b40", "encoder", "decoder")


class Cell(NamedTuple):
    runtime: str                  # torch, litert, ort
    device: str = "cpu"           # litert: cpu or gpu; ort: run_ort.py's --provider
    precision: str = "fp32"       # litert's GPU (enforce_f32) and OpenVINO's hint
    variant: str = ""             # the .onnx file's: "", "fp16", "fp16_blanket", "opt", "int8"
    graphs: tuple = GRAPHS        # the graphs the cell has a file for
    path: str | None = None       # the GPU path cellrun.gpu_proof checks
    fp16_graphs: tuple = ()       # the graphs run in fp16 whatever `precision` says (litert_gpu_split)

    def precision_of(self, graph: str) -> str:
        return "fp16" if graph in self.fp16_graphs else self.precision


CELLS = {
    "torch": Cell("torch"),
    "litert_cpu": Cell("litert"),
    "litert_gpu": Cell("litert", "gpu", path="vulkan"),
    "ort_cpu": Cell("ort", "cpu"),
    "ovcpu": Cell("ort", "openvino_cpu"),
    "webgpu": Cell("ort", "webgpu", path="vulkan"),
    "ovgpu": Cell("ort", "openvino_gpu", path="openvino"),
    "ort_cpu_fused": Cell("ort", "cpu", variant="opt", graphs=("dino_b8", "dino_b40")),
    "ort_cpu_int8": Cell("ort", "cpu", variant="int8", graphs=("encoder",)),
    "ovcpu_int8": Cell("ort", "openvino_cpu", variant="int8", graphs=("encoder",)),
    "litert_gpu_fp16": Cell("litert", "gpu", "fp16", path="vulkan"),
    "litert_gpu_split": Cell("litert", "gpu", path="vulkan", fp16_graphs=("dino_b8", "dino_b40")),
    "webgpu_fp16": Cell("ort", "webgpu", variant="fp16", path="vulkan"),
    "ovgpu_fp16": Cell("ort", "openvino_gpu", "fp16", path="openvino"),
    "webgpu_fp16blanket": Cell("ort", "webgpu", variant="fp16_blanket", graphs=("dino_b8",), path="vulkan"),
}
# The comparison run's cells, and the variants it runs once for information (plan P4; p4_run.sh reads these).
P4_CELLS = ("torch", "litert_cpu", "ort_cpu", "ovcpu", "litert_gpu", "webgpu", "ovgpu")
P4_VARIANTS = ("litert_gpu_fp16", "webgpu_fp16", "webgpu_fp16blanket", "ovgpu_fp16", "ort_cpu_int8", "ovcpu_int8")


def onnx_name(graph: str, variant: str) -> str:
    return f"{graph}.{variant}.onnx" if variant else f"{graph}.onnx"


class Model:
    """One graph of one cell; call it with numpy arrays. .info says what ran it."""

    def __init__(self, fn, info: dict):
        self._fn, self.info = fn, info

    def __call__(self, *arrays: np.ndarray):
        return self._fn(*arrays)


def _torch_modules(threads: int) -> dict:
    import torch
    torch.set_num_threads(threads)
    import graphs
    checkpoint = os.environ.get("COFFEE_CV_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("backends: the torch cell needs COFFEE_CV_CHECKPOINT (the release's release.env)")
    model, seg, _ = graphs.load_shipped(graphs.repo_path(Path(checkpoint)))
    readout = graphs.DinoReadout(model.backbone, model.readout).eval()
    return {"dino": readout, "encoder": graphs.SegEncoder(seg.model).eval(),
            "decoder": graphs.SegDecoder(seg.model).eval(), "torch": torch}


def open_graph(cell_name: str, graph: str, model_dir: Path, threads: int, runtime_dir: Path | None = None,
               cache_dir: Path | None = None, _torch: dict | None = None) -> Model:
    """model_dir holds onnx/ and tflite/ (P2's layout); runtime_dir is LiteRT's GPU library (rt_vulkan);
    cache_dir OpenVINO's. _torch: the torch cell's modules, loaded once per process (_torch_modules) when a
    process opens several of its graphs."""
    cell = CELLS[cell_name]
    if graph not in cell.graphs:
        raise SystemExit(f"backends: {cell_name} has no {graph}")
    kind = graph.split("_")[0]
    precision = cell.precision_of(graph)
    info = {"cell": cell_name, "graph": graph, "runtime": cell.runtime, "device": cell.device,
            "precision": precision, "variant": cell.variant}
    if cell.runtime == "torch":
        mods = _torch or _torch_modules(threads)
        torch, module = mods["torch"], mods[kind]
        info["file"] = "release PyTorch modules"

        def run(*arrays):
            with torch.inference_mode():
                out = module(*(torch.from_numpy(np.ascontiguousarray(a)) for a in arrays))
            return tuple(o.numpy() for o in out) if isinstance(out, tuple) else out.numpy()
        return Model(run, info)
    if cell.runtime == "litert":
        from run_litert import Graph
        if cell.device == "gpu" and runtime_dir is None:
            raise SystemExit("backends: LiteRT's GPU needs runtime_dir (rt_vulkan)")
        g = Graph(model_dir / "tflite" / f"{graph}.tflite", threads, cell.device, precision == "fp32",
                  runtime_dir if cell.device == "gpu" else None)
        info.update(file=f"{graph}.tflite", batch=g.input_shapes[0][0],
                    fully_accelerated=bool(g.model.is_fully_accelerated()))
        if kind == "decoder":
            return Model(lambda e, b: tuple(g(e, b)), info)
        return Model(lambda x: g(x)[0], info)
    from run_ort import make_session
    name = onnx_name(graph, cell.variant)
    s = make_session(model_dir / "onnx" / name, cell.device, threads, False, precision, cache_dir)
    info.update(file=name, batch=s.get_inputs()[0].shape[0], providers=s.get_providers())
    if kind == "decoder":
        return Model(lambda e, b: tuple(s.run(None, {"embedding": e, "box": b})), info)
    feed = "patches" if kind == "dino" else "image"
    return Model(lambda x: s.run(None, {feed: x})[0], info)
