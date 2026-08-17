"""Compute the OOD reference a checkpoint needs to be able to refuse.

Phase 10 specified this and it was never built, so the one validated result of
Phase 9 -- that embedding distance cleanly separates in-rig from out-of-rig
photos where softmax confidence does not -- has been sitting unshipped ever
since. `infer.py` reports the guard as unavailable without the file this writes.

For each class: the centroid of the training-split patch embeddings, and that
class's own spread (mean distance of its patches to their centroid). A photo's
score is then distance to the nearest centroid *divided by that class's spread*,
so a tight class and a diffuse one are judged on the same scale rather than the
tight one flagging everything.

The reference is written beside the checkpoint and stamped with its hash.
Centroids live in the embedding space of one specific set of weights; against
different weights the numbers are not wrong so much as meaningless, so `infer.py`
refuses to pair them rather than reporting a confident distance in a space the
checkpoint never defined.

    python -m coffeecv.build_ood_reference
    python -m coffeecv.build_ood_reference --checkpoint outputs/checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT, RunConfig
from coffeecv.dataset import MultiPhotoPatchDataset, discover_classes_multi, load_class_labels, resolve_rigs
from coffeecv.infer import (_sha, config_for_checkpoint, forward_with_embeddings, load_model,
                            reference_path_for)
from coffeecv.transforms import build_eval_transform


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None,
                   help="defaults to the config.json archived beside the checkpoint; see infer.py")
    p.add_argument("--out", default=None,
                   help="defaults to <checkpoint>.ood_reference.json, beside the weights it describes")
    args = p.parse_args()

    # Same rule as infer.py: the geometry must come from the run that produced
    # these weights, not from whatever params.yaml currently holds.
    cfg, cfg_source = config_for_checkpoint(Path(args.checkpoint), args.config)
    print(f"config: {cfg_source}")
    train_rig_dirs, _, classes_file = cfg.resolve_paths()
    train_rigs = resolve_rigs(train_rig_dirs)
    class_labels = load_class_labels(classes_file)
    # Discovered from the rig, exactly as train_baseline does, so the label->index
    # mapping the centroids are keyed by is the one the checkpoint was fitted with.
    class_ids = discover_classes_multi(train_rigs[0].cropped_dir)
    print(f"train rigs: {[r.name for r in train_rigs]}")

    # The *train* split specifically: the reference describes what the model was
    # fitted on. Building it from val or test would measure how far new data sits
    # from data the model also had to generalize to, which is a different and much
    # softer question than the one the guard asks.
    ds = MultiPhotoPatchDataset(
        split="train",
        transform=build_eval_transform(cfg.patch_resize),
        rigs=train_rigs,
        classes_file=classes_file,
        class_ids=class_ids,
        seed=cfg.seed,
        crop_size=cfg.patch_crop_size,
        resize=cfg.patch_resize,
        safety_margin=cfg.safety_margin,
        patches_per_class={"train": cfg.train_patches_per_class,
                           "val": cfg.val_patches_per_class,
                           "test": cfg.test_patches_per_class,
                           "all": cfg.xrig_patches_per_class},
        photos_per_split={"train": cfg.train_photos_per_class,
                          "val": cfg.val_photos_per_class,
                          "test": cfg.test_photos_per_class},
        patch_store_size=cfg.patch_store_size or None,
        patch_scale_frac=((cfg.patch_scale_frac_min, cfg.patch_scale_frac_max)
                          if cfg.patch_scale_frac_max > 0 else None),
        patch_beans=((cfg.patch_beans_min, cfg.patch_beans_max)
                     if cfg.patch_beans_max > 0 else None),
    )
    print(f"train patches: {len(ds)}")

    model, head = load_model(Path(args.checkpoint), cfg.model_name, len(class_ids), cfg.dropout)
    tensors = torch.stack([ds[i][0] for i in range(len(ds))])
    _, embeds = forward_with_embeddings(model, head, tensors)
    labels = np.array([ds[i][1] for i in range(len(ds))])

    classes = {}
    for i, cid in enumerate(class_ids):
        e = embeds[labels == i]
        if len(e) == 0:
            raise SystemExit(f"class {cid} has no training patches; cannot build a reference")
        centroid = e.mean(axis=0)
        spread = float(np.linalg.norm(e - centroid, axis=1).mean())
        classes[cid] = {"centroid": centroid.tolist(), "spread": spread, "n": int(len(e))}
        print(f"  {cid} {class_labels[cid]:<26} n={len(e):<5} spread={spread:.3f}")

    # Self-scores: what the guard reports on the data it was built from. These are
    # optimistic by construction (same patches), so they are recorded as context
    # for reading a live score, never as the calibration itself -- Phase 9's
    # threshold came from *held-out* photos scoring 0.97 mean / 1.24 max against
    # 1.92 for an out-of-rig photo.
    cents = np.array([classes[c]["centroid"] for c in class_ids])
    spreads = np.array([classes[c]["spread"] for c in class_ids])
    d = (np.linalg.norm(embeds[:, None, :] - cents[None, :, :], axis=2) / spreads[None, :]).min(axis=1)

    # Per-*photo* medians as well, because that is the unit infer.py scores: one
    # photo's patches get pooled to a median before any verdict. Patch-level and
    # photo-level spreads are not interchangeable (patch max 1.5 here vs photo max
    # ~1.2), and quoting the wrong one would put the warning band in the wrong
    # place. Derived only from training data, so the band never gets tuned on the
    # held-out rig it is supposed to be judged against.
    by_photo: dict[str, list[float]] = {}
    for meta, score in zip(ds._meta, d):
        by_photo.setdefault(f"{meta.rig_name}/{meta.photo_name}", []).append(float(score))
    photo_med = np.array([np.median(v) for v in by_photo.values()])

    out = Path(args.out) if args.out else reference_path_for(Path(args.checkpoint))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha": _sha(Path(args.checkpoint)),
        "model_name": cfg.model_name,
        "patch_beans": [cfg.patch_beans_min, cfg.patch_beans_max],
        "train_rigs": list(cfg.train_rigs),
        "heldout_rig": cfg.heldout_rig,
        "embedding_dim": int(embeds.shape[1]),
        "classes": classes,
        "self_scores": {"mean": round(float(d.mean()), 3),
                        "p95": round(float(np.percentile(d, 95)), 3),
                        "max": round(float(d.max()), 3),
                        "note": "per-patch, on the training patches themselves; optimistic, context only"},
        # The band infer.py warns at. Photo-level because that is the decision
        # unit; p95 rather than max so one outlier photo cannot widen it away.
        "photo_scores": {"n_photos": len(photo_med),
                         "median": round(float(np.median(photo_med)), 3),
                         "p95": round(float(np.percentile(photo_med, 95)), 3),
                         "max": round(float(photo_med.max()), 3),
                         "note": "per-photo medians over training photos; infer.py warns above p95"},
    }, indent=2))
    print(f"\nper-patch  self-scores: mean {d.mean():.2f}  p95 {np.percentile(d, 95):.2f}  max {d.max():.2f}")
    print(f"per-photo  self-scores: median {np.median(photo_med):.2f}  p95 {np.percentile(photo_med, 95):.2f}  "
          f"max {photo_med.max():.2f}   ({len(photo_med)} photos)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
