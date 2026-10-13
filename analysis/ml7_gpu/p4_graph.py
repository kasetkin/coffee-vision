"""ML-7 P4: one cell, one graph, in a fresh process (plan P4): the session's creation and one cold call, 3
warm-ups, then 20 timed calls, each one call of the graph: a batch of patches (dino_b8: 8, dino_b40: 40), one
image (encoder), one embedding and box (decoder; refs' PyTorch embeddings). --timed 100 on the cell's faster
DINOv3 batch is the stability run. Writes <out>.json: backends.Model.info, the record of cellrun.measure (timing,
CPU time against wall time, the calls with a NaN or Inf, peak RSS and the MemAvailable drop, the counters, the
GPU proof on a GPU cell) and, for LiteRT's GPU, the delegate's report. Run by p4_run.sh with the cell's venv:

    <venv>/bin/python p4_graph.py --cell webgpu --graph dino_b8 --model-dir p2 --inputs p2/inputs.npz \\
        --refs p2/refs_emb.npz --threads 4 --warmup 3 --timed 20 --out p4/out/r1/webgpu__dino_b8.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends  # noqa: E402
import cellrun  # noqa: E402
from run_litert import CapturedStderr, delegate_report  # noqa: E402


def items_for(graph: str, inputs: dict, refs: dict) -> list:
    """Every call's inputs, in order: one per batch, image or embedding."""
    if graph.startswith("dino"):
        b = int(graph.split("_b")[1])
        patches = inputs["patches"].reshape(-1, *inputs["patches"].shape[2:])
        return [(patches[i:i + b],) for i in range(0, len(patches), b)]
    if graph == "encoder":
        return [(x,) for x in inputs["images"]]
    return list(zip(refs["embeddings"], inputs["boxes"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=sorted(backends.CELLS), required=True)
    ap.add_argument("--graph", choices=backends.GRAPHS, required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="onnx/ and tflite/ (P2's p2/)")
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--refs", type=Path, required=True, help="refs' PyTorch embeddings, for the decoder")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--warmup", type=int, required=True, help="3 in plan P4 (p4_run.sh holds the numbers)")
    ap.add_argument("--timed", type=int, required=True, help="20, or 100 for the stability run")
    ap.add_argument("--runtime-dir", type=Path, default=None, help="LiteRT's GPU library (rt_vulkan)")
    ap.add_argument("--cache-dir", type=Path, default=None, help="OpenVINO's cache; emptied by the caller")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inputs = dict(np.load(args.inputs, allow_pickle=False))
    refs = dict(np.load(args.refs, allow_pickle=False))
    items = items_for(args.graph, inputs, refs)
    cell = backends.CELLS[args.cell]
    args.out.parent.mkdir(parents=True, exist_ok=True)          # the capture's file lives beside the outputs
    sampler = cellrun.Sampler()
    sampler.start()
    opened = []

    def make():
        opened.append(backends.open_graph(args.cell, args.graph, args.model_dir, args.threads, args.runtime_dir,
                                          args.cache_dir))
        return opened[0]
    try:
        with CapturedStderr(args.out.parent / (args.out.name + ".stderr")) as err:
            _, record = cellrun.measure(make, lambda m, item: m(*item), items, sampler, cell.path,
                                        warmup=args.warmup, timed=args.timed)
    finally:
        sampler.stop()
    record.update(opened[0].info, threads=args.threads, calls_per_photo=len(items) // len(inputs["photos"]))
    if cell.runtime == "litert" and cell.device == "gpu":
        record["delegate"] = delegate_report(err.text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    t = record["timing"]
    print(json.dumps({"cell": args.cell, "graph": args.graph, "cold_s": t["cold_s"],
                      "warm_median_s": t["warm"]["median"], "warm_p90_s": t["warm"]["p90"],
                      "nonfinite_calls": record["nonfinite_calls"], "cpu_over_wall": record["cpu_over_wall"],
                      "gpu_proof": record.get("gpu_proof", {}).get("ok")}))


if __name__ == "__main__":
    main()
