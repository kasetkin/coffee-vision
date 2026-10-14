"""ML-7: ONNX Runtime on make_refs.py's inputs, one graph, one provider, one file per process (plan P2); the
outputs for parity.py and a record of the cell (cellrun.measure). Run with the venv of the provider:

    <venv>/bin/python run_ort.py --onnx-dir <dir> --inputs inputs.npz --refs refs.npz --graph dino \\
        --file dino_b8.onnx --provider webgpu --threads 4 --out <name>.npz

  --provider cpu      ORT's default CPU provider (convert_onnx's venv; P1)
  --provider webgpu   the onnxruntime-ep-webgpu plugin: Dawn on Vulkan (run_ort_webgpu's venv; P2).
  --provider openvino_gpu, openvino_cpu
                      the OpenVINO execution provider (run_ort_openvino's venv; P3): Intel's GPU plugin on NEO's
                      OpenCL, or its CPU plugin at --threads, computing at --precision (INFERENCE_PRECISION_HINT;
                      fp16 is the GPU's information run, ticket D12). --cache-dir is OpenVINO's compiled-model
                      cache: give an empty one to time a cold start that compiles everything.

On every provider but cpu the session fails rather than half-running on ORT's CPU provider
(session.disable_cpu_ep_fallback=1) unless --fallback is given, for diagnosis only, with ORT's verbose log on so
the log names the nodes placed on the CPU; thread spinning is off, so a
GPU run does not take the N100's shared power budget for busy waits.

--graph dino runs each photo's 40 patches at the file's batch; encoder each photo's image; decoder each photo's
PyTorch embedding (refs) with its box. <name>.json holds the record: providers, timing (cold, warm), CPU time
against wall time, peak RSS, the i915 and CPU counters, the libraries and render nodes the process opened, and on
a GPU provider the proof that the GPU computed (cellrun.gpu_proof).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cellrun  # noqa: E402

GPU_PATH = {"webgpu": "vulkan", "openvino_gpu": "openvino"}
_REGISTERED: set[str] = set()                # plugin execution-provider libraries registered in this process


def make_session(path: Path, provider: str, threads: int, fallback: bool, precision: str = "fp32",
                 cache_dir: Path | None = None):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    if provider == "cpu":
        return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "0" if fallback else "1")
    if fallback:                             # diagnosis: ORT's verbose log names every node and its provider
        so.log_severity_level, so.log_verbosity_level = 0, 0
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if provider == "webgpu":
        import onnxruntime_ep_webgpu as webgpu
        if "webgpu" not in _REGISTERED:      # once per process: a second registration fails, and ORT then aborts
            ort.register_execution_provider_library("webgpu", webgpu.get_library_path())
            _REGISTERED.add("webgpu")
        device = next((d for d in ort.get_ep_devices() if d.ep_name == webgpu.get_ep_name()), None)
        if device is None:
            raise SystemExit("run_ort: no WebGPU device")
        so.add_provider_for_devices([device], {})
        return ort.InferenceSession(str(path), sess_options=so)
    if provider in ("openvino_gpu", "openvino_cpu"):
        device = provider.split("_")[1].upper()
        options = {"device_type": device, "precision": precision.upper(), "num_of_threads": str(threads),
                   "load_config": json.dumps({device: {"INFERENCE_PRECISION_HINT": precision.replace("fp", "f")}})}
        if cache_dir is not None:
            options["cache_dir"] = str(cache_dir)
        return ort.InferenceSession(str(path), so, providers=[("OpenVINOExecutionProvider", options)])
    raise SystemExit(f"run_ort: unknown provider {provider}")


def items_and_call(graph: str, inputs: dict, refs: dict | None):
    if graph == "dino":
        def call(s, p):
            b = s.get_inputs()[0].shape[0]          # the batch the file was exported at
            return np.concatenate([s.run(None, {"patches": p[i:i + b]})[0] for i in range(0, len(p), b)])
        return list(inputs["patches"]), call
    if graph == "encoder":
        return list(inputs["images"]), lambda s, x: s.run(None, {"image": x})[0]
    if refs is None:
        raise SystemExit("run_ort: --graph decoder needs --refs for its PyTorch embeddings")
    return (list(zip(refs["embeddings"], inputs["boxes"])),
            lambda s, eb: s.run(None, {"embedding": eb[0], "box": eb[1]}))


def print_photo(photos):
    """A log line per photo, as its call ends."""
    def progress(i: int, n: int, phase: str, seconds: float) -> None:
        print(f"photo {i + 1}/{n} {photos[i % len(photos)]} {phase} {seconds:.3f} s", flush=True)
    return progress


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None)
    ap.add_argument("--graph", choices=("dino", "encoder", "decoder"), required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--provider", choices=("cpu", "webgpu", "openvino_gpu", "openvino_cpu"), required=True)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32", help="OpenVINO's (the file's otherwise)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="OpenVINO's compiled-model cache")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--fallback", action="store_true", help="allow CPU fallback (diagnosis only, never a result)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import onnxruntime as ort

    inputs = dict(np.load(args.inputs, allow_pickle=False))
    refs = dict(np.load(args.refs, allow_pickle=False)) if args.refs else None
    path = args.onnx_dir / args.file
    items, call = items_and_call(args.graph, inputs, refs)
    sampler = cellrun.Sampler()
    sampler.start()
    sessions = []

    def make():
        sessions.append(make_session(path, args.provider, args.threads, args.fallback, args.precision,
                                     args.cache_dir))
        return sessions[0]
    try:
        outs, record = cellrun.measure(make, call, items, sampler, GPU_PATH.get(args.provider),
                                       progress=print_photo(inputs["photos"]))
    finally:
        sampler.stop()
    if args.graph == "dino":
        out = {"readouts": np.stack(outs)}
    elif args.graph == "encoder":
        out = {"embeddings": np.stack(outs)}
    else:
        out = {"low_masks": np.stack([o[0][0] for o in outs]), "ious": np.stack([o[1][0] for o in outs])}
    record.update(runtime="onnxruntime", onnxruntime=ort.__version__, provider=args.provider,
                  providers=sessions[0].get_providers(), fallback=args.fallback, graph=args.graph, file=args.file,
                  precision_hint=args.precision if args.provider.startswith("openvino") else None,
                  cache_dir=str(args.cache_dir) if args.cache_dir else None,
                  batch=sessions[0].get_inputs()[0].shape[0], threads=args.threads)
    np.savez(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    t = record["timing"]
    print(json.dumps({"file": args.file, "provider": args.provider, "precision_hint": record["precision_hint"],
                      "cold_s": t["cold_s"],
                      "warm_median_s": t["warm"]["median"], "cpu_over_wall": record["cpu_over_wall"],
                      "gpu_proof": record.get("gpu_proof", {}).get("ok"), "providers": record["providers"]}))


if __name__ == "__main__":
    main()
