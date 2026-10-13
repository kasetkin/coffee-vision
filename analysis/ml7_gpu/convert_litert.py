"""ML-7 P1: the four graphs to LiteRT. The convert_litert environment (torch 2.13 and litert-torch), from the repo root:

    analysis/ml7_gpu/convert_litert/.venv/bin/python analysis/ml7_gpu/convert_litert.py \\
        --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt --out-dir analysis/ml7_gpu/artifacts/tflite

The same wrappers and rewrites as the ONNX export (graphs.prepare), at the same fixed shapes: dino_b8, dino_b40,
encoder, decoder, each <name>.tflite. The encoder also takes graphs.litert_gpu_forms (plan P2), the forms LiteRT's
WebGPU delegate places on the GPU. manifest.json records, per file: the op histogram, every tensor above 4-D
(these fall back to the CPU on LiteRT's GPU, ticket 2.4), the file size and sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graphs  # noqa: E402


def describe(path: Path, graph: str) -> dict:
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    ops = Counter(op["op_name"] for op in it._get_ops_details())
    over4 = [{"name": t["name"], "shape": [int(d) for d in t["shape"]]} for t in it.get_tensor_details()
             if len(t["shape"]) > 4]
    return {"file": path.name, "graph": graph, "ops": dict(sorted(ops.items())), "tensors_above_4d": over4,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--only", nargs="*", default=None, help="these graphs only; default all four")
    args = ap.parse_args()
    import litert_torch

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update(torch=torch.__version__, litert_torch=litert_torch.__version__)
    g, _ = graphs.prepare(args.checkpoint)
    names = args.only or [f"dino_b{b}" for b in graphs.DINO_BATCHES] + ["encoder", "decoder"]
    for name in names:
        module, ex = g[name]
        forms = graphs.litert_gpu_forms(module) if name == "encoder" else {}
        path = out / f"{name}.tflite"
        litert_torch.convert(module.eval(), ex).export(str(path))
        manifest[path.name] = describe(path, name) | ({"litert_gpu_forms": forms} if forms else {})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        d = manifest[path.name]
        print(f"{path.name}: {d['bytes'] / 2**20:.0f} MiB, {sum(d['ops'].values())} ops, "
              f"{len(d['tensors_above_4d'])} tensors above 4-D", flush=True)


if __name__ == "__main__":
    main()
