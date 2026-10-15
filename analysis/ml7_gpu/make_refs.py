"""ML-7 P1: the inputs every runtime gets, PyTorch's answers to them, and the check of the export rewrites.
Training environment (torch 2.14), from the repo root:

    python analysis/ml7_gpu/make_refs.py refs --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt \\
        --out-dir analysis/ml7_gpu/artifacts
    python analysis/ml7_gpu/make_refs.py pick-calib --out analysis/ml7_gpu/calibration_photos.txt
    python analysis/ml7_gpu/make_refs.py calib --checkpoint models/allrigs_dino3b16_segxl0v3_country_s42.pt \\
        --out analysis/ml7_gpu/scratch/calib_inputs.npz

refs: for each photo of webapp/deploy/fixtures.txt, the serving path's own preprocessed tensors, saved to
inputs.npz so every path gets the same bytes and preprocessing stays outside the comparison: the 40 patches
[40, 3, 224, 224] as patches_for_photo and the eval transform make them, the encoder input [1, 3, 1024, 1024]
and the transformed whole-image box [1, 4]. Then PyTorch's outputs for those inputs, to refs.npz: the readouts,
the encoder embedding, the three low-res masks and their IoUs, the final multi3 mask after the segmenter's own
post-processing and its D18 verdict, the mean class probabilities, the top-1, and the guard's score and verdict. classify_one runs beside them: the answers
rebuilt from the readouts (parity._answers) and the mask must be its own, or this stops. Last, the rewritten
graphs (graphs.prepare) run on the same inputs against these references, each within REWRITE_TOL of the output's
largest magnitude (plan P1), or this stops; the result goes to rewrite_check.json.

pick-calib: the int8 encoder's calibration photos (plan P1): two from each segcrop session in dvc.yaml, so every
camera is in, none a fixture, fixed by a seed. calib: their encoder inputs, for export_onnx.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import graphs  # noqa: E402  (puts the repo and the vendored efficientvit on sys.path)
import parity  # noqa: E402

REPO = graphs.REPO
FIXTURES = REPO / "webapp" / "deploy" / "fixtures.txt"
REWRITE_TOL = 1e-5          # plan P1: max-abs under 1e-5 of the output's largest magnitude
CALIB_PER_SESSION = 2
SCALARS = ("fallback", "top1", "guard_score", "refused")     # one value per photo in refs.npz
CALIB_SEED = 7


@dataclass
class Context:
    """What serving loads for a checkpoint (webapp/app.py), to rebuild answers from readouts and masks from logits."""
    checkpoint: Path
    cfg: object
    class_ids: list
    class_labels: dict
    model: torch.nn.Module
    head_module: torch.nn.Module
    head: dict
    ref: dict | None
    probe: dict | None
    seg: object

    @classmethod
    def load(cls, checkpoint: Path) -> "Context":
        from coffeecv.class_list import load_classes
        from coffeecv.infer import (config_for_checkpoint, embedding_dim_of, load_model, load_ood_probe,
                                    load_ood_reference, segmenter_for)

        checkpoint = graphs.repo_path(checkpoint)
        cfg, _ = config_for_checkpoint(checkpoint, None)
        classes = load_classes(REPO / cfg.classes_file)
        model, head = load_model(checkpoint, cfg.model_name, len(classes.keys), cfg.dropout)
        weights = {"weight": head.weight.detach().numpy(), "bias": head.bias.detach().numpy()}
        return cls(checkpoint, cfg, list(classes.keys), dict(classes.labels), model, head, weights,
                   load_ood_reference(checkpoint, embedding_dim_of(head)),
                   load_ood_probe(checkpoint, embedding_dim=embedding_dim_of(head)), segmenter_for(cfg))


def read_list(list_file: Path) -> list[Path]:
    """Every photo of a `<role> <sha256> <repo path>` list (fixtures.txt, calibration_photos.txt), from the repo,
    checked against its sha256."""
    out = []
    for line in list_file.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        _, sha, rel = line.split()
        path = REPO / rel
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"make_refs: {rel} does not match its sha256 in {list_file.name}")
        out.append(path)
    return out


def encoder_input(seg, rgb: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """The segmenter's own preprocessing (BeanSegmenter._encode): the transformed photo and its input size."""
    from segment_anything.utils.transforms import ResizeLongestSide
    x = seg.model.transform(rgb).unsqueeze(0)
    return x, ResizeLongestSide.get_preprocess_shape(*rgb.shape[:2], seg.model.image_size[0])


def pack_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Bool masks of different sizes as one uint8 array [P, bytes], packed bits zero-padded; refs' original_size
    gives each back (parity.unpack_mask)."""
    packed = [np.packbits(m.ravel()) for m in masks]
    n = max(len(p) for p in packed)
    return np.stack([np.pad(p, (0, n - len(p))) for p in packed])


@torch.no_grad()
def make_refs(ctx: Context, photos: list[Path]) -> tuple[dict, dict]:
    from coffeecv.dataset import load_rgb_image
    from coffeecv.infer import (TorchClassifier, classify_one, forward_with_embeddings, inference_tta_for,
                                patches_for_photo, sig12)
    from coffeecv.segment_beans import segment_and_crop
    from coffeecv.transforms import build_eval_transform

    transform = build_eval_transform(ctx.cfg.patch_resize)
    pr = ctx.seg.predictor
    tta = inference_tta_for(ctx.checkpoint, ctx.cfg.model_name)
    if tta:   # the graphs are one plain pass over the patches; a TTA model would serve eight
        raise SystemExit("make_refs: this checkpoint is served with dihedral TTA; the graphs do not cover it")
    inp = {k: [] for k in ("patches", "images", "boxes", "input_size", "original_size")}
    ref = {k: [] for k in ("readouts", "embeddings", "low_masks", "ious", "masks", "probs", "fallback", "top1",
                           "guard_score", "refused")}
    for path in photos:
        entry = classify_one(path, ctx.cfg, ctx.class_ids, ctx.class_labels, TorchClassifier(ctx.model, ctx.head_module),
                             ctx.ref,
                             n_patches=40, tta=tta, probe=ctx.probe)
        patches, _ = patches_for_photo(path, ctx.cfg, 40, [42, 0])     # classify_one's own seed key
        tensors = torch.stack([transform(p) for p in patches])
        _, readouts = forward_with_embeddings(ctx.model, ctx.head_module, tensors)
        mine = parity._answers(readouts, ctx.head, ctx.ref, ctx.probe)
        ood = entry.get("ood") or {}
        if ctx.class_ids[mine["top1"]] != entry["ranked"][0][0] or ("score" in mine and (
                sig12(mine["score"]) != ood.get("score")
                or mine["refused"] != (entry["verdict"] == "REFUSED (out of distribution)"))):
            raise SystemExit(f"make_refs: the answers rebuilt from {path.name}'s readouts are not classify_one's")

        rgb = load_rgb_image(path)
        h, w = rgb.shape[:2]
        image, input_size = encoder_input(ctx.seg, rgb)
        crop = segment_and_crop(rgb, ctx.seg)                    # serving's segmenter step; leaves the encoding set
        low, iou = ctx.seg._decode(h, w, multimask=True)
        mask = ctx.seg._upsample(low[2])                         # multi3
        if not np.array_equal(mask, crop.photo_mask):
            raise SystemExit(f"make_refs: {path.name}: multi3's post-processed mask is not segment_and_crop's")
        if not torch.equal(image, ctx.seg.model.transform(rgb).unsqueeze(0)):
            raise SystemExit(f"make_refs: {path.name}: the encoder input is not reproducible")
        box = pr.apply_boxes_torch(torch.tensor([ctx.seg.prompt_geometry(h, w)["box"]], dtype=torch.float))

        inp["patches"].append(tensors.numpy())
        inp["images"].append(image.numpy())
        inp["boxes"].append(box.numpy())
        inp["input_size"].append(input_size)
        inp["original_size"].append((h, w))
        ref["readouts"].append(readouts)
        ref["embeddings"].append(pr.features.numpy())
        ref["low_masks"].append(low.numpy())
        ref["ious"].append(iou.numpy())
        ref["masks"].append(mask)
        ref["fallback"].append(bool(crop.info["fallback"]))
        ref["top1"].append(mine["top1"])
        ref["probs"].append(mine["probs"])
        ref["guard_score"].append(mine.get("score", np.nan))
        ref["refused"].append(bool(mine.get("refused", False)))
        print(f"{path.name}: {entry['ranked'][0][0]}, {entry['verdict']}, mask {mask.mean():.1%}"
              f"{' (D18 fallback)' if crop.info['fallback'] else ''}", flush=True)

    meta = {"photos": np.array([p.name for p in photos]),
            "paths": np.array([str(p.relative_to(REPO)) for p in photos]),
            "input_size": np.array(inp.pop("input_size")), "original_size": np.array(inp.pop("original_size"))}
    inputs = {**meta, **{k: np.stack(v) for k, v in inp.items()}}
    masks = ref.pop("masks")
    refs = {**meta, "mask_bits": pack_masks(masks),
            **{k: np.array(v) for k, v in ref.items() if k in SCALARS},
            **{k: np.stack(v) for k, v in ref.items() if k not in SCALARS}}
    return inputs, refs


@torch.no_grad()
def check_rewrites(checkpoint: Path, inputs: dict, refs: dict) -> dict:
    """Each rewritten graph against PyTorch's own outputs on the fixtures: the largest absolute difference over the
    output's largest magnitude, per graph, within REWRITE_TOL."""
    g, neck = graphs.prepare(checkpoint)
    report = {"neck": [{**r, "in_hw": list(r["in_hw"]), "out_hw": list(r["out_hw"])} for r in neck],
              "tolerance": REWRITE_TOL, "graphs": {}}

    def rel(got, want):
        return float(np.abs(got - want).max() / np.abs(want).max())

    for b in graphs.DINO_BATCHES:
        module = g[f"dino_b{b}"][0]
        got = np.stack([np.concatenate([module(torch.from_numpy(p[i:i + b])).numpy() for i in range(0, 40, b)])
                        for p in inputs["patches"]])
        report["graphs"][f"dino_b{b}"] = rel(got, refs["readouts"])
    enc = g["encoder"][0]
    report["graphs"]["encoder"] = rel(np.stack([enc(torch.from_numpy(x)).numpy() for x in inputs["images"]]),
                                      refs["embeddings"])
    dec = g["decoder"][0]
    outs = [dec(torch.from_numpy(e), torch.from_numpy(b)) for e, b in zip(refs["embeddings"], inputs["boxes"])]
    report["graphs"]["decoder_low_masks"] = rel(np.stack([o[0][0].numpy() for o in outs]), refs["low_masks"])
    report["graphs"]["decoder_ious"] = rel(np.stack([o[1][0].numpy() for o in outs]), refs["ious"])
    report["pass"] = all(v <= REWRITE_TOL for v in report["graphs"].values())
    return report


def pick_calibration(fixtures: list[Path]) -> list[Path]:
    """CALIB_PER_SESSION photos from each segcrop session (dvc.yaml), none a fixture, seeded."""
    import yaml
    from coffeecv.segcrop_session import RAW_ROOT, _find_images

    sessions = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]["segcrop"]["foreach"]
    taken = {p.name for p in fixtures}
    rng = random.Random(CALIB_SEED)
    out = []
    for s in sessions:
        photos = [p for d in sorted((RAW_ROOT / s).iterdir()) if d.is_dir() and d.name.startswith("class_")
                  for p in _find_images(d) if p.name not in taken]
        out += sorted(rng.sample(photos, CALIB_PER_SESSION))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refs")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--out-dir", type=Path, required=True)
    pc = sub.add_parser("pick-calib")
    pc.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("calib")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--list", type=Path, default=HERE / "calibration_photos.txt")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)

    if args.cmd == "refs":
        ctx = Context.load(args.checkpoint)
        inputs, refs = make_refs(ctx, read_list(FIXTURES))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(args.out_dir / "inputs.npz", **inputs)
        np.savez(args.out_dir / "refs.npz", **refs)
        check = check_rewrites(args.checkpoint, inputs, refs)
        (args.out_dir / "rewrite_check.json").write_text(json.dumps(check, indent=2) + "\n")
        print("rewrites: " + ", ".join(f"{k} {v:.1e}" for k, v in check["graphs"].items())
              + f" (of the largest magnitude; tolerance {REWRITE_TOL:.0e})")
        if not check["pass"]:
            raise SystemExit("make_refs: a rewritten graph is not the original at float rounding; stop (plan P1)")
    elif args.cmd == "pick-calib":
        photos = pick_calibration(read_list(FIXTURES))
        lines = [f"calib {hashlib.sha256(p.read_bytes()).hexdigest()} {p.relative_to(REPO)}" for p in photos]
        args.out.write_text("# ML-7 P1: the int8 XL0 encoder's calibration photos, from make_refs.py pick-calib:\n"
                            f"# {CALIB_PER_SESSION} per segcrop session in dvc.yaml, none a fixture, seed "
                            f"{CALIB_SEED}. One line each: calib <sha256> <path under the repo>.\n"
                            + "\n".join(lines) + "\n")
        print(f"{len(photos)} photos -> {args.out}")
    else:
        from coffeecv.dataset import load_rgb_image
        from coffeecv.segment_beans import BeanSegmenter, seg_params
        from coffeecv.infer import config_for_checkpoint
        cfg, _ = config_for_checkpoint(graphs.repo_path(args.checkpoint), None)
        seg = BeanSegmenter(seg_params(cfg))
        xs = [encoder_input(seg, load_rgb_image(p))[0].numpy() for p in read_list(args.list)]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, images=np.stack(xs))
        print(f"{len(xs)} encoder inputs -> {args.out}")


if __name__ == "__main__":
    main()
