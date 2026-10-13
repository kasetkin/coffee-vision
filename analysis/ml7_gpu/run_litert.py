"""ML-7: LiteRT on make_refs.py's inputs, one graph, one accelerator, one precision per process (plan P2); the
outputs for parity.py and a record of the cell (cellrun.measure). The run_litert environment (ai-edge-litert 2.3.0):

    <venv>/bin/python run_litert.py --model-dir <dir> --inputs inputs.npz --refs refs.npz --graph dino \\
        --file dino_b8.tflite --accelerator gpu --runtime-dir ~/ml7/rt_vulkan --precision fp32 \\
        --threads 4 --out <name>.npz

  --accelerator cpu   LiteRT's CPU accelerator (XNNPACK underneath); P1
  --accelerator gpu   GPU with CPU fallback for ops the GPU lacks; is_fully_accelerated() says whether any fell
                      back. The GPU library comes from --runtime-dir, which holds libLiteRt.so and exactly one GPU
                      library (rt_vulkan: the wheel's libLiteRtWebGpuAccelerator.so; plan P2). --precision fp32
                      sets GpuOptions(enforce_f32=True): without it LiteRT's GPU computes in fp16 (ticket 2.4, D12).
                      The record keeps the delegate's own report (delegate_report): the ops it refused, its
                      GPU/CPU split.

Options are passed in full: the hardware_accel= shortcut silently sets one CPU thread. --graph dino runs each
photo's 40 patches at the file's batch; encoder each photo's image; decoder each photo's PyTorch embedding
(refs) with its box. Inputs and outputs follow the signature's order, which litert-torch writes in the wrapper's
argument and return order.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cellrun  # noqa: E402


class Graph:
    def __init__(self, path: Path, threads: int, accelerator: str, fp32: bool, runtime_dir: Path | None):
        from ai_edge_litert.compiled_model import (CompiledModel, CpuOptions, Environment, EnvironmentOptions,
                                                   GpuOptions, HardwareAccelerator, Options)
        hw = HardwareAccelerator.CPU if accelerator == "cpu" else HardwareAccelerator.GPU | HardwareAccelerator.CPU
        options = Options(hardware_accelerators=hw, cpu_options=CpuOptions(num_threads=threads),
                          gpu_options=GpuOptions(enforce_f32=fp32) if accelerator == "gpu" else None)
        env = (Environment.create(options=EnvironmentOptions(runtime_path=str(runtime_dir)))
               if runtime_dir else None)
        self.model = CompiledModel.from_file(str(path), environment=env, options=options)
        sig = self.model.get_signature_by_index(0)
        key = sig.get("key") or list(self.model.get_signature_list())[0]
        self.outputs = self.model.get_output_tensor_details(key)
        self.out_names = sig["outputs"]
        details = self.model.get_input_tensor_details(key)
        self.input_shapes = [[int(d) for d in details[name]["shape"]] for name in sig["inputs"]]
        self.inputs = self.model.create_input_buffers(0)
        self.output_buffers = self.model.create_output_buffers(0)

    def __call__(self, *arrays: np.ndarray) -> list[np.ndarray]:
        for buf, a in zip(self.inputs, arrays):
            buf.write(np.ascontiguousarray(a, dtype=np.float32))
        self.model.run_by_index(0, self.inputs, self.output_buffers)
        out = []
        for name, buf in zip(self.out_names, self.output_buffers):
            shape = [int(d) for d in self.outputs[name]["shape"]]
            out.append(buf.read(int(np.prod(shape)), np.float32).reshape(shape))
        return out


SPLIT = re.compile(r"(\d+) operations will run on the GPU, and the remaining (\d+) operations will run on the CPU")


def delegate_report(log: str) -> dict:
    """The GPU delegate's own placement report, read from LiteRT's log: the ops it refused and its split. No
    split line after a GPU environment was created means the delegate took the whole graph. is_fully_accelerated()
    is not trusted: it said True for a graph the delegate had split (probed on yappy, plan P2)."""
    lines = log.splitlines()
    refused = []
    if "ERROR: Following operations are not supported by GPU delegate:" in lines:
        for line in lines[lines.index("ERROR: Following operations are not supported by GPU delegate:") + 1:]:
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
        self._saved, self._tmp = os.dup(2), tempfile.TemporaryFile()
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


def gpu_path(runtime_dir: Path) -> str:
    """Which GPU path a runtime directory holds: "vulkan" (the wheel's WebGPU library) or "opencl" (P6's build)."""
    libs = {p.name for p in runtime_dir.iterdir()}
    if "libLiteRtWebGpuAccelerator.so" in libs:
        return "vulkan"
    if libs & {"libLiteRtOpenClAccelerator.so", "libLiteRtGpuAccelerator.so"}:
        return "opencl"
    raise SystemExit(f"run_litert: no GPU library in {runtime_dir}")


def items_and_call(graph: str, inputs: dict, refs: dict | None):
    if graph == "dino":
        def call(g, p):
            b = g.input_shapes[0][0]
            return np.concatenate([g(p[i:i + b])[0] for i in range(0, len(p), b)])
        return list(inputs["patches"]), call
    if graph == "encoder":
        return list(inputs["images"]), lambda g, x: g(x)[0]
    if refs is None:
        raise SystemExit("run_litert: --graph decoder needs --refs for its PyTorch embeddings")
    return list(zip(refs["embeddings"], inputs["boxes"])), lambda g, eb: g(eb[0], eb[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None)
    ap.add_argument("--graph", choices=("dino", "encoder", "decoder"), required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--accelerator", choices=("cpu", "gpu"), required=True)
    ap.add_argument("--runtime-dir", type=Path, default=None, help="the GPU library's directory (rt_vulkan)")
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32", help="the GPU's (enforce_f32)")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import ai_edge_litert

    if args.accelerator == "gpu" and args.runtime_dir is None:
        ap.error("--accelerator gpu needs --runtime-dir")
    path = gpu_path(args.runtime_dir) if args.accelerator == "gpu" else None
    inputs = dict(np.load(args.inputs, allow_pickle=False))
    refs = dict(np.load(args.refs, allow_pickle=False)) if args.refs else None
    items, call = items_and_call(args.graph, inputs, refs)
    sampler = cellrun.Sampler()
    sampler.start()
    graphs = []

    def make():
        graphs.append(Graph(args.model_dir / args.file, args.threads, args.accelerator, args.precision == "fp32",
                            args.runtime_dir))
        return graphs[0]
    try:
        with CapturedStderr() as err:
            outs, record = cellrun.measure(make, call, items, sampler, path)
    finally:
        sampler.stop()
    if args.graph == "dino":
        out = {"readouts": np.stack(outs)}
    elif args.graph == "encoder":
        out = {"embeddings": np.stack(outs)}
    else:
        out = {"low_masks": np.stack([o[0][0] for o in outs]), "ious": np.stack([o[1][0] for o in outs])}
    record.update(runtime="litert", ai_edge_litert=getattr(ai_edge_litert, "__version__", "?"),
                  accelerator=args.accelerator, path=path, precision=args.precision,
                  runtime_dir=str(args.runtime_dir) if args.runtime_dir else None,
                  fully_accelerated=bool(graphs[0].model.is_fully_accelerated()),
                  delegate=delegate_report(err.text) if args.accelerator == "gpu" else None,
                  graph=args.graph, file=args.file,
                  batch=graphs[0].input_shapes[0][0], threads=args.threads)
    np.savez(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    t = record["timing"]
    print(json.dumps({"file": args.file, "accelerator": args.accelerator, "precision": args.precision,
                      "cold_s": t["cold_s"], "warm_median_s": t["warm"]["median"],
                      "fully_accelerated": record["fully_accelerated"],
                      "delegate_all_on_gpu": (record["delegate"] or {}).get("all_on_gpu"), "cpu_over_wall": record["cpu_over_wall"],
                      "gpu_proof": record.get("gpu_proof", {}).get("ok")}))


if __name__ == "__main__":
    main()
