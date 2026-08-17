"""Classify new, unlabeled photos -- and refuse when the photo is outside what
the model was trained on.

**This file must mirror how the training/eval dataset builds patches.** It is the
inference half of the parity rule that the rest of the pipeline is built around:
whatever measurement training depends on has to run the same way here, or every
reported metric describes a lab that the field does not resemble. The previous
version of this file violated that rule outright -- it sized patches from a
`--scale-ratio` the *user* supplied, asking a person pointing a phone to know how
much to rescale for their camera. That is the fragility bean-unit sizing exists
to remove, and it predated the sizing change by five days without being updated.

So patches here are sized exactly as `MultiPhotoPatchDataset._extract_photo`
sizes them: bean pitch measured per photo by `bean_scale.estimate_bean_pitch`,
patch side = B * pitch with B log-uniform over the trained range, sampled inside
`compute_valid_region_rect`, stored at `patch_store_size` and then resized to
`patch_resize` by the eval transform. The two-step resize is deliberate -- going
straight to 224 from full resolution is not the same interpolation the model was
trained and scored on.

Two independent refusals, because a wrong answer stated confidently is worse than
no answer (Phase 9: on an out-of-rig photo the model gave p=0.755 to a class that
was wrong, and flipped to a different wrong class under a 15% brightness change):

1. **Scale.** If the measured pitch says the frame cannot supply patches in the
   trained bean range, the photo is framed wrong and there is nothing to fix
   downstream. This is the interpretable form of the guard -- "move the camera
   back" is advice a user can act on.
2. **Out-of-distribution.** Penultimate-embedding distance to the nearest
   training-class centroid, normalized by that class's own spread. Phase 9
   measured 0.97 mean / 1.24 max over held-out training photos against 1.92-1.94
   for a photo from an unseen rig, and softmax confidence was *useless* at this
   (0.97 on a good photo, indistinguishable from in-distribution 0.997). Needs a
   reference file from `build_ood_reference.py`; without one this guard is
   reported as unavailable rather than silently skipped.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv.bean_scale import estimate_bean_pitch
from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT, RunConfig
from coffeecv.dataset import load_class_labels, load_rgb_image
from coffeecv.geometry import compute_valid_region_rect, sample_bean_unit_patch_boxes
from coffeecv.model import build_model
from coffeecv.transforms import build_eval_transform

DEVICE = torch.device("cpu")

# Phase 10's calibration: above the 1.24 in-distribution maximum, below the 1.92
# observed on a genuinely out-of-rig photo. Recorded as a number with a basis
# rather than a tuned constant -- one positive example is thin evidence, so the
# margin is reported alongside every verdict.
OOD_THRESHOLD = 1.4


def grayscale_like_training(rgb: np.ndarray) -> np.ndarray:
    """The exact luminance conversion `_extract_photo` feeds to the estimator.

    Spelled out rather than delegated to cv2/PIL because the pitch estimate is a
    parity-critical quantity: a different set of luma weights would shift it, and
    a shifted pitch silently rescales every patch relative to training.
    """
    gray = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return gray.astype(np.uint8)


def reference_path_for(checkpoint: Path) -> Path:
    """Where this checkpoint's OOD reference lives: beside it, named after it.

    Phase 10's requirement was that the guard travels with the model, and a fixed
    global path cannot do that -- it would leave one file to be silently
    overwritten by whichever checkpoint was processed last, which is how a guard
    ends up describing weights nobody is running. Deriving the path also means a
    reference for a transient `outputs/` checkpoint lands under the gitignored
    outputs tree, while one built for a shipped `models/*.pt` sits next to it and
    gets committed alongside.
    """
    return checkpoint.with_suffix(".ood_reference.json")


def config_for_checkpoint(checkpoint: Path, explicit: str | None) -> tuple[RunConfig, str]:
    """The config that *this checkpoint* was trained with, not whatever params.yaml
    happens to say now.

    params.yaml is a moving target -- it is rewritten per fold during a sweep and
    reset to the adopted values afterwards -- so pairing it with a checkpoint from
    some earlier run silently mismatches patch geometry, and patch geometry is the
    one thing inference must get right. Each run archives its own config.json
    beside its outputs, so prefer that when it is sitting next to the checkpoint.
    """
    card = checkpoint.with_suffix(".json")          # shipped model: models/<name>.json
    run_cfg = checkpoint.parent.parent / "config.json"  # live run: outputs/config.json
    if explicit:
        raw = json.load(open(explicit))
        source = explicit
    elif card.exists():
        # A shipped model carries its own card, and the card's training_config is
        # the authoritative record of how it was fitted -- the model travels out of
        # this repo, so it cannot depend on a params.yaml it will not have.
        raw = json.load(open(card)).get("training_config", {})
        source = f"{card} (training_config)"
    elif run_cfg.exists():
        raw = json.load(open(run_cfg))
        source = str(run_cfg)
    else:
        return RunConfig.from_params_yaml(), "params.yaml (no config found beside the checkpoint)"
    known = RunConfig.__dataclass_fields__
    return RunConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                        for k, v in raw.items() if k in known}), source


def load_model(checkpoint: Path, model_name: str, num_classes: int, dropout: float):
    model, head = build_model(model_name, num_classes=num_classes, freeze_mode="none", dropout=dropout)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, head


@torch.no_grad()
def forward_with_embeddings(model, head, tensors: torch.Tensor, batch_size: int = 32):
    """(probs, embeddings) in one pass.

    The embedding is the head's *input* -- the pooled penultimate feature, 512-d
    for resnet18 -- captured with a pre-hook rather than by rebuilding the model
    headless, so the thing measured is exactly what this checkpoint feeds its
    classifier and cannot drift from it.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach().cpu())

    handle = head.register_forward_pre_hook(hook)
    try:
        probs = []
        for i in range(0, len(tensors), batch_size):
            logits = model(tensors[i:i + batch_size].to(DEVICE))
            probs.append(F.softmax(logits, dim=1).cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(probs, axis=0), torch.cat(captured).numpy()


def patches_for_photo(path: Path, cfg: RunConfig, n_patches: int, seed_key: list[int]):
    """Sample patches the way training does. Returns (patches, diagnostics)."""
    rgb = load_rgb_image(path)
    h, w = rgb.shape[:2]
    region = compute_valid_region_rect(h, w, cfg.safety_margin)
    pitch = estimate_bean_pitch(grayscale_like_training(rgb))

    rng = np.random.default_rng(seed_key)
    boxes, clamped = sample_bean_unit_patch_boxes(
        rng, region, n_patches, pitch, cfg.patch_beans_min, cfg.patch_beans_max,
        # Rotation jitter is a *training* augmentation; eval splits never get it,
        # so neither does inference.
        0.0,
    )

    patches = []
    for box, angle, side in boxes:
        patch = Image.fromarray(rgb[box.y0:box.y1, box.x0:box.x1])
        if cfg.patch_store_size and patch.size[0] != cfg.patch_store_size:
            patch = patch.resize((cfg.patch_store_size, cfg.patch_store_size), Image.BILINEAR)
        patches.append(patch)

    room = min(region.width, region.height)
    return patches, {
        "bean_pitch_px": round(pitch, 1),
        # How many beans span the usable short side -- the quantity that decides
        # whether the trained patch range is reachable at all in this frame.
        "beans_across": round(room / pitch, 2),
        "patches_clamped": clamped,
        "clamp_rate": round(clamped / max(len(boxes), 1), 3),
    }


def scale_verdict(diag: dict, cfg: RunConfig) -> tuple[bool, str]:
    """Can this frame supply patches in the range the model was trained on?"""
    across = diag["beans_across"]
    if across < cfg.patch_beans_min:
        return False, (f"frame spans only {across:.1f} beans; the model was trained on patches of "
                       f"{cfg.patch_beans_min:.0f}-{cfg.patch_beans_max:.0f} beans, so not even the "
                       f"smallest fits. Move the camera back or zoom out.")
    if across < cfg.patch_beans_max:
        return True, (f"frame spans {across:.1f} beans; patches above {across:.1f} are clamped "
                      f"({diag['clamp_rate']:.0%} of them). Usable, but framing wider would give the "
                      f"scale variety the model expects.")
    return True, f"frame spans {across:.1f} beans, covering the trained {cfg.patch_beans_min:.0f}-{cfg.patch_beans_max:.0f} range."


def ood_scores(embeddings: np.ndarray, ref: dict) -> tuple[np.ndarray, list[str]]:
    """Per-patch distance to the nearest class centroid, in units of that class's spread."""
    cids = sorted(ref["classes"])
    cents = np.array([ref["classes"][c]["centroid"] for c in cids])
    spreads = np.array([ref["classes"][c]["spread"] for c in cids])
    d = np.linalg.norm(embeddings[:, None, :] - cents[None, :, :], axis=2) / spreads[None, :]
    nearest = d.argmin(axis=1)
    return d.min(axis=1), [cids[i] for i in nearest]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None,
                   help="a run's archived config.json; defaults to params.yaml. The checkpoint and the "
                        "patch geometry have to come from the same run or the parity this file exists "
                        "to preserve is broken at the first step.")
    p.add_argument("--ood-reference", default=None,
                   help="defaults to <checkpoint>.ood_reference.json, so the guard travels with the model")
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg, cfg_source = config_for_checkpoint(Path(args.checkpoint), args.config)
    print(f"config: {cfg_source}")

    if not cfg.patch_beans_max:
        raise SystemExit(
            "This checkpoint's config has bean-unit patch sizing disabled (patch_beans_max=0), so there "
            "is no way to size patches from the photo alone -- the other sizing modes need to be told how "
            "the rig was framed. Inference is only defined for bean-unit runs."
        )

    class_labels = load_class_labels(REPO_ROOT / cfg.classes_file)
    class_ids = sorted(class_labels)
    model, head = load_model(Path(args.checkpoint), cfg.model_name, len(class_ids), cfg.dropout)

    ref_path = Path(args.ood_reference) if args.ood_reference else reference_path_for(Path(args.checkpoint))
    ref = json.load(open(ref_path)) if ref_path.exists() else None
    if ref is None:
        print(f"OOD guard UNAVAILABLE: no reference at {ref_path}. Build one with "
              f"`python -m coffeecv.build_ood_reference`. Predictions below are unguarded.")
    elif ref.get("checkpoint_sha") and ref["checkpoint_sha"] != _sha(Path(args.checkpoint)):
        raise SystemExit(
            f"OOD reference {ref_path} was built from a different checkpoint. Centroids live in the "
            f"embedding space of one specific set of weights and mean nothing against another -- "
            f"rebuild it for this checkpoint."
        )

    transform = build_eval_transform(cfg.patch_resize)
    images = sorted(q for q in Path(args.images_dir).iterdir()
                    if q.suffix.lower() in {".jpg", ".jpeg", ".png"} and not q.name.endswith("__mask.png"))
    if not images:
        raise FileNotFoundError(f"No images in {args.images_dir}")

    results = {}
    for idx, path in enumerate(images):
        # A user photo can be anything -- too small for the estimator's analysis
        # window, unreadable, not really beans. That is a refusal for this photo,
        # not a reason to abandon the batch, and it must not surface as a stack
        # trace to someone holding a phone.
        try:
            patches, diag = patches_for_photo(path, cfg, args.n_patches, [args.seed, idx])
        except (ValueError, OSError) as exc:
            print(f"\n{path.name}\n  -> REFUSED: cannot measure bean scale ({exc})")
            results[path.name] = {"verdict": "REFUSED (unmeasurable)", "error": str(exc)}
            continue
        ok_scale, scale_msg = scale_verdict(diag, cfg)

        entry = {**diag, "scale_ok": ok_scale, "scale_note": scale_msg}
        print(f"\n{path.name}")
        print(f"  scale: {scale_msg}")

        if not ok_scale:
            entry["verdict"] = "REFUSED (scale)"
            results[path.name] = entry
            print("  -> REFUSED: photo is outside the trained patch scale, not predicting.")
            continue

        probs, embeds = forward_with_embeddings(model, head, torch.stack([transform(x) for x in patches]))
        mean = probs.mean(axis=0)
        ranked = sorted(zip(class_ids, mean), key=lambda t: -t[1])
        entry["ranked"] = [(c, class_labels[c], float(v)) for c, v in ranked]

        if ref is None:
            entry["verdict"] = "UNGUARDED"
            entry["ood"] = None
        else:
            scores, _ = ood_scores(embeds, ref)
            median = float(np.median(scores))
            # Two bands, and the gap between them is the honest part. REFUSE at
            # Phase 9's 1.4, which was calibrated against a wildly different rig
            # and only catches photos that far out. WARN above the training set's
            # own 95th percentile, because measurement on this repo's held-out rig
            # showed the interesting failure sits *between* the two: sony_cam
            # photos score ~1.24 against training's ~0.99 while top-1 accuracy
            # halves, yet only 6% cross 1.4. A score in the warn band means the
            # model is working outside what it saw, and its answer is worth less
            # than the probability next to it suggests.
            warn_at = (ref.get("photo_scores") or {}).get("p95")
            entry["ood"] = {"median": round(median, 3),
                            "frac_patches_over_threshold": round(float((scores > OOD_THRESHOLD).mean()), 3),
                            "threshold": OOD_THRESHOLD, "warn_above": warn_at}
            margin = median - OOD_THRESHOLD
            print(f"  OOD: median distance {median:.2f} vs threshold {OOD_THRESHOLD} "
                  f"({'over' if margin > 0 else 'under'} by {abs(margin):.2f})")
            if warn_at and OOD_THRESHOLD >= median > warn_at:
                entry["ood"]["warned"] = True
                print(f"       WARNING: above the training distribution's 95th percentile ({warn_at:.2f}). "
                      f"Not refused, but treat the answer below as unreliable --\n"
                      f"       on this repo's held-out rig, photos in this band lost about half their "
                      f"top-1 accuracy while staying under the refusal threshold.")
            if median > OOD_THRESHOLD:
                entry["verdict"] = "REFUSED (out of distribution)"
                results[path.name] = entry
                print("  -> REFUSED: this photo does not resemble the training distribution.")
                print("     Not offering a best guess: on such photos the embedding sits roughly")
                print("     equidistant from every class, so a ranked list would be false precision.")
                continue
            entry["verdict"] = "predicted"

        cid, label, pr = entry["ranked"][0]
        print(f"  -> {cid} {label}  p={pr:.3f}" + ("" if ref else "   (unguarded)"))
        results[path.name] = entry

    n_ref = sum(1 for r in results.values() if r["verdict"].startswith("REFUSED"))
    print(f"\n{len(results) - n_ref}/{len(results)} photos predicted, {n_ref} refused.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint),
            "patch_beans": [cfg.patch_beans_min, cfg.patch_beans_max],
            "patch_store_size": cfg.patch_store_size,
            "patch_resize": cfg.patch_resize,
            "n_patches_per_image": args.n_patches,
            "ood_reference": str(ref_path) if ref else None,
            "per_image": results,
        }, indent=2))
        print(f"Wrote {out}")


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


if __name__ == "__main__":
    main()
