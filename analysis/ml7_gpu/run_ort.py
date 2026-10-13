"""ML-7: ONNX Runtime on make_refs.py's inputs, one cell per process; the outputs for parity.py. P1 runs the CPU
provider; P2 and P3 add the GPU providers to this same script.

    <venv>/bin/python analysis/ml7_gpu/run_ort.py --onnx-dir <dir> --inputs inputs.npz --dino dino_b8.onnx \\
        --encoder encoder.onnx --decoder decoder.onnx --threads 4 --out outputs_ort_cpu.npz

Any of --dino, --encoder and --decoder may be left out. The decoder reads this run's own encoder output when
--encoder is given (as serving chains them), else the PyTorch embeddings in --refs. DINOv3 runs at the batch its
file was exported with, over each photo's 40 patches.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def session(path: Path, threads: int, provider: str = "CPUExecutionProvider"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=[provider])


def run(args) -> tuple[dict, dict]:
    import onnxruntime as ort
    inputs = dict(np.load(args.inputs, allow_pickle=False))
    out, info = {}, {"onnxruntime": ort.__version__, "threads": args.threads, "provider": args.provider}
    if args.dino:
        s = session(args.onnx_dir / args.dino, args.threads, args.provider)
        batch = s.get_inputs()[0].shape[0]
        t0 = time.perf_counter()
        out["readouts"] = np.stack([np.concatenate([s.run(None, {"patches": p[i:i + batch]})[0]
                                                    for i in range(0, len(p), batch)]) for p in inputs["patches"]])
        info["dino"] = {"file": args.dino, "batch": batch, "seconds": round(time.perf_counter() - t0, 1)}
    if args.encoder:
        s = session(args.onnx_dir / args.encoder, args.threads, args.provider)
        t0 = time.perf_counter()
        out["embeddings"] = np.stack([s.run(None, {"image": x})[0] for x in inputs["images"]])
        info["encoder"] = {"file": args.encoder, "seconds": round(time.perf_counter() - t0, 1)}
    if args.decoder:
        emb = out.get("embeddings")
        if emb is None:
            if args.refs is None:
                raise SystemExit("run_ort: --decoder needs --encoder or --refs for its input")
            emb = np.load(args.refs, allow_pickle=False)["embeddings"]
            info["decoder_input"] = "PyTorch embeddings (refs)"
        s = session(args.onnx_dir / args.decoder, args.threads, args.provider)
        res = [s.run(None, {"embedding": e, "box": b}) for e, b in zip(emb, inputs["boxes"])]
        out["low_masks"] = np.stack([r[0][0] for r in res])
        out["ious"] = np.stack([r[1][0] for r in res])
        info["decoder"] = {"file": args.decoder}
    return out, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--refs", type=Path, default=None, help="for the decoder's input when --encoder is not given")
    ap.add_argument("--dino")
    ap.add_argument("--encoder")
    ap.add_argument("--decoder")
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, info = run(args)
    np.savez(args.out, **out)
    args.out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
