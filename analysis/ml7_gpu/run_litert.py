"""ML-7: LiteRT on make_refs.py's inputs, one cell per process; the outputs for parity.py. P1 runs the CPU (XNNPACK);
P2 adds the GPU paths to this same script. The run_litert environment (ai-edge-litert 2.3.0, plan P2's runtime):

    analysis/ml7_gpu/run_litert/.venv/bin/python analysis/ml7_gpu/run_litert.py --model-dir <dir> \\
        --inputs inputs.npz --dino dino_b8.tflite --encoder encoder.tflite --decoder decoder.tflite \\
        --threads 4 --out outputs_litert_cpu.npz

Options are passed in full: the hardware_accel= shortcut silently sets one CPU thread (plan P2). Any of --dino,
--encoder and --decoder may be left out; the decoder reads this run's encoder output when there is one, else
the PyTorch embeddings in --refs. A graph's inputs and outputs are taken in its signature's order, which
litert-torch writes in the wrapper's argument and return order.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


class Graph:
    def __init__(self, path: Path, threads: int):
        from ai_edge_litert.compiled_model import CompiledModel, CpuOptions, HardwareAccelerator, Options
        self.path = path
        self.model = CompiledModel.from_file(str(path), options=Options(
            hardware_accelerators=HardwareAccelerator.CPU, cpu_options=CpuOptions(num_threads=threads)))
        sig = self.model.get_signature_by_index(0)
        key = sig.get("key") or list(self.model.get_signature_list())[0]
        self.outputs = self.model.get_output_tensor_details(key)
        self.out_names = sig["outputs"]
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

    @property
    def fully_accelerated(self) -> bool:
        return bool(self.model.is_fully_accelerated())


def run(args) -> tuple[dict, dict]:
    import ai_edge_litert
    inputs = dict(np.load(args.inputs, allow_pickle=False))
    out, info = {}, {"ai_edge_litert": getattr(ai_edge_litert, "__version__", "?"), "threads": args.threads,
                     "accelerator": "CPU (XNNPACK)"}
    if args.dino:
        g = Graph(args.model_dir / args.dino, args.threads)
        batch = int(g.model.get_input_tensor_details(list(g.model.get_signature_list())[0])
                    [g.model.get_signature_by_index(0)["inputs"][0]]["shape"][0])
        t0 = time.perf_counter()
        out["readouts"] = np.stack([np.concatenate([g(p[i:i + batch])[0] for i in range(0, len(p), batch)])
                                    for p in inputs["patches"]])
        info["dino"] = {"file": args.dino, "batch": batch, "seconds": round(time.perf_counter() - t0, 1),
                        "fully_accelerated": g.fully_accelerated}
    if args.encoder:
        g = Graph(args.model_dir / args.encoder, args.threads)
        t0 = time.perf_counter()
        out["embeddings"] = np.stack([g(x)[0] for x in inputs["images"]])
        info["encoder"] = {"file": args.encoder, "seconds": round(time.perf_counter() - t0, 1),
                           "fully_accelerated": g.fully_accelerated}
    if args.decoder:
        emb = out.get("embeddings")
        if emb is None:
            emb = np.load(args.refs, allow_pickle=False)["embeddings"]
            info["decoder_input"] = "PyTorch embeddings (refs)"
        g = Graph(args.model_dir / args.decoder, args.threads)
        res = [g(e, b) for e, b in zip(emb, inputs["boxes"])]
        out["low_masks"] = np.stack([r[0][0] for r in res])
        out["ious"] = np.stack([r[1][0] for r in res])
        info["decoder"] = {"file": args.decoder, "fully_accelerated": g.fully_accelerated}
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None, help="for the decoder's input when --encoder is not given")
    ap.add_argument("--dino")
    ap.add_argument("--encoder")
    ap.add_argument("--decoder")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, info = run(args)
    np.savez(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
