"""One cell of the GPU option test (research_quantization.md §5.4): one graph of the live release on `yappy`'s GPU
with one set of LiteRT GpuOptions, in a process of its own. It records the cold start, the warm time per photo,
the delegate's placement report, ADR 0020's GPU proof and the memory the process took, and saves the outputs.

    <release>/.venv/bin/python gpu_quantized_ops.py --release /opt/coffee-cv/current --inputs inputs.npz \\
        --graph dino --cell quant --out out/r1_dino_quant

The cells:
    base         the release's own options: enforce_f32 as the manifest's precision says, nothing else
    share        base + constant_tensor_sharing
    quant        base + constant_tensor_sharing + allow_src_quantized_fc_conv_ops (LiteRT's doc: sharing is required)
    quant_only   base + allow_src_quantized_fc_conv_ops, without sharing

Run from the release's venv with the release's coffeecv on the path, so the runtime, the graphs and the proof code
are the ones that serve. The inputs are ML-7's (make_refs.py): 13 photos, 40 patches each, and each photo's
1024 px image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

CELLS = {
    "base": {},
    "share": {"constant_tensor_sharing": True},
    "quant": {"constant_tensor_sharing": True, "allow_src_quantized_fc_conv_ops": True},
    "quant_only": {"allow_src_quantized_fc_conv_ops": True},
}
WARMUPS = 2


def mem_available_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    raise RuntimeError("no MemAvailable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--graph", choices=("dino", "encoder"), required=True)
    ap.add_argument("--cell", choices=tuple(CELLS), required=True)
    ap.add_argument("--photos", type=int, default=13)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    release = args.release.resolve()
    sys.path.insert(0, str(release))
    from ai_edge_litert.compiled_model import (CompiledModel, CpuOptions, Environment, EnvironmentOptions,
                                               GpuOptions, HardwareAccelerator, Options)

    from coffeecv import litert

    model_dir = next((release / "models").glob("*.litert"))
    spec = json.loads((model_dir / "manifest.json").read_text())["graphs"][args.graph]
    runtime_dir = release / ".venv/lib/python3.12/site-packages/ai_edge_litert"
    gpu_options = {"enforce_f32": spec["precision"] == "fp32", **CELLS[args.cell]}

    z = np.load(args.inputs)
    if args.graph == "dino":
        batch = spec["batch"]
        photos = [[p[i:i + batch] for i in range(0, len(p), batch)] for p in z["patches"][:args.photos]]
    else:
        photos = [[img] for img in z["images"][:args.photos]]

    mem0, idle0, t0 = mem_available_mb(), litert.gpu_idle_ms(), time.perf_counter()
    with litert.CapturedStderr() as err:
        options = Options(hardware_accelerators=HardwareAccelerator.GPU | HardwareAccelerator.CPU,
                          cpu_options=CpuOptions(num_threads=litert.THREADS), gpu_options=GpuOptions(**gpu_options))
        env = Environment.create(options=EnvironmentOptions(runtime_path=str(runtime_dir)))
        model = CompiledModel.from_file(str(model_dir / spec["file"]), environment=env, options=options)
        sig = model.get_signature_by_index(0)
        key = sig.get("key") or list(model.get_signature_list())[0]
        out_details = model.get_output_tensor_details(key)
        out_shape = tuple(int(d) for d in out_details[sig["outputs"][0]]["shape"])
        inputs, outputs = model.create_input_buffers(0), model.create_output_buffers(0)

        def call(x: np.ndarray) -> np.ndarray:
            inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
            model.run_by_index(0, inputs, outputs)
            return outputs[0].read(int(np.prod(out_shape)), np.float32).reshape(out_shape)

        call(photos[0][0])
        cold_s = time.perf_counter() - t0
    report = litert.delegate_report(err.text)
    mem_loaded = mem_available_mb()

    for _ in range(WARMUPS):
        call(photos[0][0])
    per_photo, outs = [], []
    idle1, t1 = litert.gpu_idle_ms(), time.perf_counter()
    for calls in photos:
        t = time.perf_counter()
        outs.append(np.concatenate([call(x) for x in calls]))
        per_photo.append(time.perf_counter() - t)
    wall_ms = (time.perf_counter() - t1) * 1000
    idle_ms = None if idle1 is None else litert.gpu_idle_ms() - idle1
    proof = litert.gpu_proof(litert.gpu_mapped(), idle_ms, wall_ms)
    out = np.stack(outs)

    record = {
        "graph": args.graph, "cell": args.cell, "file": spec["file"], "file_sha256": spec["sha256"],
        "gpu_options": gpu_options, "photos": len(photos), "calls_per_photo": len(photos[0]),
        "cold_s": round(cold_s, 3), "per_photo_s": [round(t, 4) for t in per_photo],
        "median_s": round(float(np.median(per_photo)), 4), "p90_s": round(float(np.percentile(per_photo, 90)), 4),
        "mem_available_mb": {"before": mem0, "loaded": mem_loaded, "after": mem_available_mb()},
        "delegate": report, "gpu_proof": proof,
        "nonfinite": int((~np.isfinite(out)).sum()),
        "output_sha256": hashlib.sha256(out.tobytes()).hexdigest(),
        "log_lines": [ln for ln in err.text.splitlines() if "uantiz" in ln or "ERROR" in ln or "WARNING" in ln][:40],
        "idle_before_load_ms": idle0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out.with_suffix(".npy"), out.astype(np.float32))
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=1))
    args.out.with_suffix(".log").write_text(err.text)
    print(f"CELL {args.graph} {args.cell}: median {record['median_s']} s, p90 {record['p90_s']} s, cold "
          f"{record['cold_s']} s, all_on_gpu {report['all_on_gpu']}, proof {proof['ok']} {proof['why_not']}, "
          f"nonfinite {record['nonfinite']}, mem {mem0}->{mem_loaded} MB", flush=True)


if __name__ == "__main__":
    main()
