"""The converted model a release serves (ticket ML-7 D6, plan P9): a shipped checkpoint's three graphs as LiteRT files
beside it, in models/<name>.litert/, with the head and the manifest coffeecv.litert reads. Two steps, each in the
environment it needs; the dvc.yaml stage litert_export runs both:

    convert   in convert/.venv (torch 2.13, litert-torch 0.9.4): the encoder, the decoder and DINOv3 at batch 8 in
              their exact forms (coffeecv.litert_forms, ADR 0022) to .tflite, and convert.json, what was applied
    finish    in the training environment (torch 2.14, ai-edge-litert 2.3.0): the head as head.npz, manifest.json, and
              D7's check (ADR 0020) on the CPU against PyTorch on the deploy fixtures, written to check.json. A model
              that fails loses its manifest, so coffeecv.litert refuses it.

    convert/.venv/bin/python -m coffeecv.export_litert convert --checkpoint models/<name>.pt --profile gpu_split
    python -m coffeecv.export_litert finish --checkpoint models/<name>.pt --profile gpu_split

A profile is the serving host's accelerator and precision per graph (ADR 0019), measured on that host: gpu_split is
yappy's (ML-7 D2, D4). A graph served in fp16 gets fp16_forms; every graph gets litert_gpu_forms, which are exact,
so harmless on a CPU (ADR 0019). The GPU's own check runs on the serving host (plan P11); this one holds the files
to the fp32 bars on the CPU, where an fp16 graph's file runs at fp32.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np

GRAPH_FILES = {"encoder": "encoder.tflite", "decoder": "decoder.tflite", "dino": "dino_b8.tflite"}
DINO_BATCH = 8                     # 40 patches as 5 calls of 8 (ML-7 P4, P7: batch 8 beat 40 on yappy's GPU)
PROFILES = {
    # yappy, the Intel N100's iGPU through WebGPU on Vulkan: the segmenter fp32, DINOv3 fp16 (ADR 0019).
    "gpu_split": {"encoder": ("gpu", "fp32"), "decoder": ("gpu", "fp32"), "dino": ("gpu", "fp16")},
    # A host without a usable GPU, until its precisions are measured there (ADR 0019).
    "cpu": {"encoder": ("cpu", "fp32"), "decoder": ("cpu", "fp32"), "dino": ("cpu", "fp32")},
}
# ADR 0020's bars for a graph in fp32, which the check holds every file to on the CPU.
READOUT_COSINE = 0.99999
MASK_IOU = 0.999


def litert_dir(checkpoint: Path) -> Path:
    from coffeecv.litert import litert_dir_for
    return litert_dir_for(checkpoint)


def convert(checkpoint: Path, profile: str) -> Path:
    """The three graphs to .tflite, in a fresh directory beside the checkpoint. Returns it."""
    import litert_torch
    import torch

    from coffeecv import litert_forms
    out = litert_dir(checkpoint)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    graphs, _ = litert_forms.prepare(checkpoint)
    record = {"profile": profile, "torch": torch.__version__, "litert_torch": version("litert-torch"), "forms": {}}
    for name, key in (("encoder", "encoder"), ("decoder", "decoder"), ("dino", f"dino_b{DINO_BATCH}")):
        module, example = graphs[key]
        forms = {"litert_gpu_forms": litert_forms.litert_gpu_forms(module, batch=example[0].shape[0])}
        if PROFILES[profile][name][1] == "fp16":
            forms["fp16_forms"] = litert_forms.fp16_forms(module)
        litert_torch.convert(module.eval(), example).export(str(out / GRAPH_FILES[name]))
        record["forms"][name] = forms
        print(f"{GRAPH_FILES[name]}: {(out / GRAPH_FILES[name]).stat().st_size / 2**20:.0f} MiB, forms {forms}", flush=True)
    (out / "convert.json").write_text(json.dumps(record, indent=2) + "\n")
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return 1.0 if union == 0 else np.count_nonzero(a & b) / union


def check(model_dir: Path, checkpoint: Path, photos: list[Path]) -> dict:
    """D7's fp32 bars on the CPU against PyTorch, through one classify_one: per photo the mask's IoU, every patch's
    readout cosine on PyTorch's patches, the verdict, the top-1 and the largest class-probability change."""
    from coffeecv import infer, litert, model_io
    from coffeecv.class_list import load_classes
    from coffeecv.config import REPO_ROOT
    from coffeecv.photo_io import load_rgb_image
    from coffeecv.segment_beans import BeanSegmenter, seg_params

    cfg, _ = infer.config_for_checkpoint(checkpoint, None)
    classes = load_classes(REPO_ROOT / cfg.classes_file)
    ids, labels = list(classes.keys), dict(classes.labels)
    torch_clf = infer.TorchClassifier(*infer.load_model(checkpoint, cfg.model_name, len(ids), cfg.dropout))
    ref = infer.load_ood_reference(checkpoint, torch_clf.embedding_dim)
    probe = infer.load_ood_probe(checkpoint, embedding_dim=torch_clf.embedding_dim)
    models = litert.load_models(model_dir, on_cpu=True)
    seg = {"litert": BeanSegmenter(seg_params(cfg), engine=models.sam), "torch": infer.segmenter_for(cfg)}
    clf = {"litert": models.classifier, "torch": torch_clf}
    rows = []
    for k, photo in enumerate(photos):
        rgb = load_rgb_image(photo)
        iou = _iou(*(seg[b].predict_mask(rgb) for b in ("litert", "torch")))
        patches, _ = infer.patches_for_photo(photo, cfg, 40, [42, k], segmenter=seg["torch"])
        x = np.stack([model_io.eval_patch(p, cfg.patch_resize) for p in patches])
        a, b = (clf[n].probs_and_embeddings(x)[1] for n in ("litert", "torch"))
        cosine = float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min())
        e = {n: infer.classify_one(photo, cfg, ids, labels, clf[n], ref, seed_key=[42, k], tta=False, probe=probe,
                                   segmenter=seg[n]) for n in ("litert", "torch")}
        probs = {n: {c: p for c, _, p in e[n]["ranked"]} for n in e}
        row = {"photo": photo.name, "mask_iou": iou, "least_readout_cosine": cosine,
               "verdict_same": e["litert"]["verdict"] == e["torch"]["verdict"],
               "top1_same": e["litert"]["ranked"][0][0] == e["torch"]["ranked"][0][0],
               "max_prob_change": max(abs(probs["litert"][c] - probs["torch"][c]) for c in probs["torch"])}
        row["pass"] = (iou >= MASK_IOU and cosine >= READOUT_COSINE and row["verdict_same"] and row["top1_same"])
        rows.append(row)
        print(f"{photo.name}: IoU {iou:.5f}, cosine {cosine:.7f}, verdict and top-1 "
              f"{'same' if row['verdict_same'] and row['top1_same'] else 'CHANGED'}", flush=True)
    return {"bars": {"mask_iou": MASK_IOU, "readout_cosine": READOUT_COSINE}, "photos": len(rows),
            "passed": all(r["pass"] for r in rows), "rows": rows}


def finish(checkpoint: Path, profile: str, photos: list[Path]) -> dict:
    """The head, the manifest, and the check; a model that fails the check loses its manifest."""
    from coffeecv import infer, litert
    from coffeecv.class_list import load_classes
    from coffeecv.config import REPO_ROOT

    out = litert_dir(checkpoint)
    record = json.loads((out / "convert.json").read_text())
    if record["profile"] != profile:
        raise SystemExit(f"{out} was converted for profile {record['profile']}, not {profile}")
    cfg, _ = infer.config_for_checkpoint(checkpoint, None)
    model, head = infer.load_model(checkpoint, cfg.model_name, len(load_classes(REPO_ROOT / cfg.classes_file).keys),
                                   cfg.dropout)
    litert.save_head(out / "head.npz", head.weight.detach().numpy(), head.bias.detach().numpy())
    graphs = {name: {"file": GRAPH_FILES[name], "accelerator": acc, "precision": prec, "forms": record["forms"][name]}
              for name, (acc, prec) in PROFILES[profile].items()}
    litert.write_manifest(
        out, graphs, {"file": "head.npz", "readout": model.readout}, hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        {"weights_sha256": cfg.seg_weights_sha256, "decoder_sha256": cfg.seg_decoder_sha256},
        {"name": cfg.model_name, "weights_sha256": model.backbone.weights_sha256},
        {"torch": record["torch"], "litert_torch": record["litert_torch"], "ai_edge_litert": version("ai-edge-litert"),
         "profile": profile})
    result = check(out, checkpoint, photos)
    (out / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    if not result["passed"]:
        (out / "manifest.json").unlink()
        raise SystemExit(f"{out}: D7's check failed on {sum(not r['pass'] for r in result['rows'])} photos "
                         f"(check.json); the manifest is removed, so it cannot serve")
    return result


def fixture_photos() -> list[Path]:
    """The deploy's fixed photos (webapp/deploy/fixtures.txt): the smoke photo and the compare set."""
    from coffeecv.config import REPO_ROOT
    lines = (REPO_ROOT / "webapp" / "deploy" / "fixtures.txt").read_text().splitlines()
    return [REPO_ROOT / ln.split()[2] for ln in lines if ln.strip() and not ln.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("convert", "finish"))
    ap.add_argument("--checkpoint", type=Path, required=True, help="a shipped models/<name>.pt, repo-relative")
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    args = ap.parse_args()
    from coffeecv.config import REPO_ROOT
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else REPO_ROOT / args.checkpoint
    if args.step == "convert":
        convert(checkpoint, args.profile)
    else:
        result = finish(checkpoint, args.profile, fixture_photos())
        print(f"D7's check passed on {result['photos']} fixtures")


if __name__ == "__main__":
    main()
