"""ML-7 P1: the four graphs to ONNX, and the ONNX variants plan P1 names. The convert_onnx environment (torch 2.14,
the training lock's), from the repo root:

    UV_PROJECT_ENVIRONMENT=analysis/ml7_gpu/convert_onnx/.venv uv run --locked --project analysis/ml7_gpu/convert_onnx \\
        python analysis/ml7_gpu/export_onnx.py --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt \\
        --out-dir analysis/ml7_gpu/artifacts/onnx --calib analysis/ml7_gpu/scratch/calib_inputs.npz

Base graphs (graphs.prepare, rewrites applied): dino_b8, dino_b40, encoder, decoder, each <name>.onnx with its
weights in <name>.onnx.data beside it (the two travel together), by torch.onnx.export(dynamo=True) at its
default opset, unoptimised (export_base says why). Variants:

  <name>.fp16.onnx        mixed fp16, inputs and outputs kept fp32 (keep_io_types), for ORT's WebGPU cell (ticket
                          D12): the converter's default op block list plus Softmax and LayerNormalization, and
                          the attention-score MatMuls (the q k^T feeding each Softmax) by node name.
  dino_b8.fp16_blanket.onnx  nothing kept in fp32: does DINOv3 really overflow (ticket D12)?
  dino_b<B>.opt.onnx      ORT's transformer fusions (optimize_model, model_type "vit"), for ORT's CPU provider.
  encoder.int8.onnx       static QDQ int8, per-channel weights, calibrated on make_refs.py's calibration inputs;
                          for information only (ticket D4).

manifest.json records, per file: graph, variant, opset, op histogram, file sizes and sha256, and the fusions
that applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graphs  # noqa: E402

IO = {"dino": (["patches"], ["readout"]), "encoder": (["image"], ["embedding"]),
      "decoder": (["embedding", "box"], ["low_masks", "ious"])}
ATTN = {"num_heads": 12, "hidden_size": 768}          # DINOv3 ViT-B/16


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(path: Path, graph: str, variant: str, **extra) -> dict:
    import onnx
    m = onnx.load(str(path), load_external_data=False)
    files = [path] + ([d] if (d := path.with_name(path.name + ".data")).exists() else [])
    return {"file": path.name, "graph": graph, "variant": variant,
            "opset": {o.domain or "ai.onnx": o.version for o in m.opset_import},
            "ops": dict(sorted(Counter(n.op_type for n in m.graph.node).items())),
            "files": {f.name: {"bytes": f.stat().st_size, "sha256": _sha(f)} for f in files}, **extra}


def clear(path: Path) -> None:
    """Remove a graph and its .data before writing it again: onnx's external-data writer appends to an existing
    .data file, so a rerun would carry the old tensors along."""
    for f in (path, path.with_name(path.name + ".data")):
        f.unlink(missing_ok=True)


def save(model, path: Path) -> None:
    import onnx
    data = path.name + ".data"
    clear(path)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=data)


def export_base(module, args, path: Path, kind: str) -> None:
    ins, outs = IO[kind]
    clear(path)
    # optimize=False: torch 2.14's default optimisation pass (onnxscript 0.7.2) drops `x + 1e-15` as a no-op, and
    # the XL0 encoder's linear attention (LiteMLA) divides by exactly that where its denominator is 0, so the
    # optimised graph returns NaN (found in P1). ORT optimises the graph at session load instead (ticket R4).
    prog = torch.onnx.export(module, args, dynamo=True, input_names=ins, output_names=outs, optimize=False)
    prog.save(str(path), external_data=True)
    slice_sequences(path)


def slice_sequences(path: Path) -> int:
    """sequences_to_slices on a saved .onnx, in place: the graph is rewritten and its .data left byte-identical,
    so a host that has the weights needs only the new .onnx. The number of SequenceAt replaced."""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    n = sequences_to_slices(model)
    if n:
        onnx.save_model(model, str(path))
    return n


def repair_casts(path: Path) -> int:
    """retype_casts on a saved .onnx, in place, the graph only (as slice_sequences)."""
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    n = len(retype_casts(model))
    if n:
        onnx.save_model(model, str(path))
    return n


def _constant_ints(model, name: str) -> list[int] | int | None:
    """An INT64 constant's value (a Constant node's or an initializer's): an int for a scalar, else a list."""
    from onnx import numpy_helper
    for n in model.graph.node:
        if n.op_type == "Constant" and n.output[0] == name:
            a = numpy_helper.to_array(n.attribute[0].t)
            return int(a) if a.ndim == 0 else [int(v) for v in a]
    for t in model.graph.initializer:
        if t.name == name:
            a = numpy_helper.to_array(t)
            return int(a) if a.ndim == 0 else [int(v) for v in a]
    return None


def sequences_to_slices(model) -> int:
    """Each SplitToSequence (the exporter's chunk and tensor_split) and the SequenceAt nodes that read it, as one
    Slice per SequenceAt: the same tensors, in ops ORT's WebGPU provider has (it has no sequence kernels; plan
    P2). In place, the graph only: initializers and their external data are untouched. Returns the number of
    SequenceAt replaced; refuses a split or position that is not a constant."""
    from onnx import TensorProto, helper
    splits = {n.output[0]: n for n in model.graph.node if n.op_type == "SplitToSequence"}
    nodes, done = [], 0
    for n in model.graph.node:
        if n.op_type == "SplitToSequence":
            continue
        if n.op_type != "SequenceAt" or n.input[0] not in splits:
            if set(n.input) & splits.keys():
                raise ValueError(f"{n.name}: a {n.op_type} reads a sequence; only SequenceAt is rewritten")
            nodes.append(n)
            continue
        split = splits[n.input[0]]
        axis = next((a.i for a in split.attribute if a.name == "axis"), 0)
        size = _constant_ints(model, split.input[1]) if len(split.input) > 1 else None
        p = _constant_ints(model, n.input[1])
        if size is None or not isinstance(p, int) or p < 0:
            raise ValueError(f"{split.name} -> {n.name}: split {size}, position {p}; constants needed")
        bounds = [0]
        for s in ([size] * (p + 1) if isinstance(size, int) else size[:p + 1]):
            bounds.append(bounds[-1] + s)
        out = n.output[0]
        for part, v in (("starts", bounds[p]), ("ends", bounds[p + 1]), ("axes", axis)):
            nodes.append(helper.make_node("Constant", [], [f"{out}_{part}"],
                                          value=helper.make_tensor(f"{out}_{part}", TensorProto.INT64, [1], [v])))
        nodes.append(helper.make_node("Slice", [split.input[0], f"{out}_starts", f"{out}_ends", f"{out}_axes"],
                                      [out], name=f"{out}_slice"))
        done += 1
    used = {i for n in nodes for i in n.input} | {o.name for o in model.graph.output}
    nodes = [n for n in nodes if n.op_type != "Constant" or n.output[0] in used]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return done


SCALE_OPS = {"Div", "Mul", "Add", "Cast"}     # what may sit between q k^T and its Softmax (SAM divides by sqrt(d))


def attention_score_matmuls(model) -> list[str]:
    """The MatMul nodes a Softmax reads, directly or through a scale (SCALE_OPS): q k^T in every attention block,
    DINOv3's (MatMul -> Softmax) and SAM's decoder's (MatMul -> Div -> Softmax)."""
    by_output = {o: n for n in model.graph.node for o in n.output}
    found = set()
    for n in model.graph.node:
        if n.op_type != "Softmax":
            continue
        src = by_output.get(n.input[0])
        while src is not None and src.op_type in SCALE_OPS:
            src = by_output.get(src.input[0])
        if src is not None and src.op_type == "MatMul":
            found.add(src.name)
    return sorted(found)


def fp16_copy(src: Path, dst: Path, blanket: bool) -> dict:
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(str(src))
    if blanket:
        kept_ops, kept_nodes = [], []
    else:
        # Softmax and LayerNorm stay fp32 (plan P1), whole (LayerNormalization) or decomposed: the encoder's and
        # decoder's LayerNorm2d is ReduceMean, Sub, Pow, Sqrt and Div. Every op of those types stays fp32, not only
        # the LayerNorm ones, which also keeps the XL0 encoder's linear-attention division (LiteMLA) in fp32.
        kept_ops = list(float16.DEFAULT_OP_BLOCK_LIST) + ["Softmax", "LayerNormalization",
                                                           "ReduceMean", "Sub", "Pow", "Sqrt", "Div"]
        kept_nodes = attention_score_matmuls(m)
    out = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=kept_ops, node_block_list=kept_nodes)
    retyped = retype_casts(out)
    save(out, dst)
    loads(dst)
    return {"kept_fp32_ops": sorted(set(kept_ops) & {n.op_type for n in m.graph.node}),
            "kept_fp32_attention_matmuls": len(kept_nodes), "retyped_casts": len(retyped)}


def retype_casts(model) -> list[str]:
    """Each Cast whose output the graph types otherwise, cast to that type: onnxconverter-common types a Cast's
    output fp16 for its consumers and leaves the Cast at float (the fp16 decoder's scalar_tensor Casts of 1024,
    2 and 1, exact in fp16), and the model does not load (plan P2). In place; the outputs retyped."""
    typed = {v.name: v.type.tensor_type.elem_type for v in model.graph.value_info}
    retyped = []
    for n in model.graph.node:
        if n.op_type != "Cast" or n.output[0] not in typed:
            continue
        to = next(a for a in n.attribute if a.name == "to")
        if to.i != typed[n.output[0]]:
            to.i = typed[n.output[0]]
            retyped.append(n.output[0])
    return retyped


def loads(path: Path) -> None:
    """An ORT CPU session is created from the file: its types check out (P1 had not loaded the fp16 copies)."""
    import onnxruntime as ort
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def fuse(src: Path, dst: Path) -> dict:
    from onnxruntime.transformers import optimizer
    m = optimizer.optimize_model(str(src), model_type="vit", **ATTN)
    stats = dict(m.get_fused_operator_statistics())       # zeros kept: a fusion that did not apply is reported
    clear(dst)
    m.save_model_to_file(str(dst), use_external_data_format=True)
    return {"fusions": stats}


class _Calib:
    """onnxruntime's CalibrationDataReader over make_refs.py's calibration inputs."""

    def __init__(self, images: np.ndarray):
        self._it = iter([{"image": x} for x in images])

    def get_next(self):
        return next(self._it, None)


def int8_encoder(src: Path, dst: Path, calib: Path) -> dict:
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process
    import tempfile
    images = np.load(calib)["images"]
    # The pre-processed model, and the per-tensor files its save leaves beside it, stay in a scratch directory.
    clear(dst)
    with tempfile.TemporaryDirectory(prefix="ml7_quant_") as tmp:
        pre = Path(tmp) / "encoder.preproc.onnx"
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True, save_as_external_data=True,
                          external_data_location=pre.name + ".data")
        quantize_static(str(pre), str(dst), _Calib(images), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, use_external_data_format=True)
    return {"calibration_photos": int(len(images)), "format": "QDQ", "activations": "uint8",
            "weights": "int8 per-channel"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--calib", type=Path, default=None, help="make_refs.py calib's output (the int8 step)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="these steps only (base fp16 fused int8); default all. repair: the P2 fixes, graph only, on "
                         "files written before them (sequences_to_slices on DINOv3's, retype_casts on the fp16 "
                         "copies), never in the default")
    args = ap.parse_args()
    steps = set(args.only or ["base", "fp16", "fused", "int8"])
    if "int8" in steps and args.calib is None:
        ap.error("the int8 step needs --calib")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["torch"] = torch.__version__

    def record(path: Path, graph: str, variant: str, **extra):
        manifest[path.name] = describe(path, graph, variant, **extra)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"{path.name}: {sum(f['bytes'] for f in manifest[path.name]['files'].values()) / 2**20:.0f} MiB, "
              f"opset {manifest[path.name]['opset']}, {sum(manifest[path.name]['ops'].values())} nodes "
              + (json.dumps(extra) if extra else ""), flush=True)

    names = [f"dino_b{b}" for b in graphs.DINO_BATCHES] + ["encoder", "decoder"]
    if "base" in steps:
        g, _ = graphs.prepare(args.checkpoint)
        for name in names:
            module, ex = g[name]
            export_base(module, ex, out / f"{name}.onnx", name.split("_")[0])
            record(out / f"{name}.onnx", name, "fp32")
    if "fp16" in steps:
        for name in names:
            record(out / f"{name}.fp16.onnx", name, "fp16 mixed",
                   **fp16_copy(out / f"{name}.onnx", out / f"{name}.fp16.onnx", blanket=False))
        record(out / "dino_b8.fp16_blanket.onnx", "dino_b8", "fp16 blanket",
               **fp16_copy(out / "dino_b8.onnx", out / "dino_b8.fp16_blanket.onnx", blanket=True))
    if "fused" in steps:
        for b in graphs.DINO_BATCHES:
            record(out / f"dino_b{b}.opt.onnx", f"dino_b{b}", "fp32 ORT-fused",
                   **fuse(out / f"dino_b{b}.onnx", out / f"dino_b{b}.opt.onnx"))
    if "repair" in steps:
        for f in [f for f, m in manifest.items() if isinstance(m, dict) and m.get("variant", "").startswith("fp")]:
            m = manifest[f]
            kept = {k: v for k, v in m.items() if k not in ("file", "graph", "variant", "opset", "ops", "files")}
            if f.startswith("dino") and "ORT-fused" not in m["variant"]:
                kept["sequence_slices"] = slice_sequences(out / f) or kept.get("sequence_slices", 0)
            if "fp16" in m["variant"]:
                kept["retyped_casts"] = repair_casts(out / f)
            loads(out / f)
            record(out / f, m["graph"], m["variant"], **kept)
    if "int8" in steps:
        record(out / "encoder.int8.onnx", "encoder", "int8 QDQ",
               **int8_encoder(out / "encoder.onnx", out / "encoder.int8.onnx", args.calib))


if __name__ == "__main__":
    main()
