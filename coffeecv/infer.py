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

One refusal, because a wrong answer stated confidently is worse than no answer
(Phase 9: on an out-of-rig photo the model gave p=0.755 to a class that was
wrong, and flipped to a different wrong class under a 15% brightness change):

**Out-of-distribution.** Penultimate-embedding distance to the nearest
   training-class centroid, normalized by that class's own spread. Phase 9
   measured 0.97 mean / 1.24 max over held-out training photos against 1.92-1.94
   for a photo from an unseen rig, and softmax confidence was *useless* at this
   (0.97 on a good photo, indistinguishable from in-distribution 0.997). Needs a
reference file from `build_ood_reference.py`; without one this guard is
reported as unavailable rather than silently skipped.

There was a second refusal here until 2026-09-03 -- a framing/scale guard that
refused when the frame could not supply patches in the trained bean range. It
was removed because it could not fire: `beans_across` has a hard floor of
`2.055 * _K_LO` = 8.22 while its thresholds sat at 4.0 and 7.0, so every branch
below OK was unreachable, and there was no too-far bound at all. Measured over
15 photos on 5 rigs it returned OK every time. Deleting it changed no behaviour;
keeping it meant this docstring advertised a refusal that had never once
happened. `beans_across` is still measured per photo and carried in the diag
dict (and logged per request by the web service) so a future guard can be built
from the real distribution rather than from theory, which is what made the
original unreachable. Design work: docs/scale_guard_plan.md.

**A too-far photo therefore gets a confident answer with no framing warning.**
That was already true while the dead guard was in place; the OOD check and the
tray-crop detector are partial backstops, but this is "no worse than before",
not "handled".
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv.bean_scale import estimate_bean_pitch, pitch_kwargs
from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT, RunConfig
from coffeecv.crop_tray import locate_bean_crop
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


def classes_path_for(checkpoint: Path) -> Path:
    """Where this checkpoint's frozen class list lives: beside it, named after it.

    dataset/classes.txt is expected to grow over time as new bean types are
    added ahead of the next training run, but a shipped checkpoint's classifier
    head is a fixed size -- it must keep reading the class list it was trained
    against, not whatever classes.txt says today (that would size the head
    wrong and crash `load_state_dict`, or worse, silently misalign class ids).
    Same reasoning as `reference_path_for`: the artifact that defines a
    checkpoint's output space has to travel with the checkpoint, not live at
    one global path every checkpoint shares. Falls back to
    `cfg.classes_file` (the live file) in `config_for_checkpoint` below when no
    snapshot exists beside a checkpoint -- true for every checkpoint shipped
    before this snapshot convention existed.
    """
    return checkpoint.with_suffix(".classes.txt")


def config_for_checkpoint(checkpoint: Path, explicit: str | None) -> tuple[RunConfig, str]:
    """The config that *this checkpoint* was trained with, not whatever params.yaml
    happens to say now.

    params.yaml is a moving target -- it is rewritten per fold during a sweep and
    reset to the adopted values afterwards -- so pairing it with a checkpoint from
    some earlier run silently mismatches patch geometry, and patch geometry is the
    one thing inference must get right. Each run archives its own config.json
    beside its outputs, so prefer that when it is sitting next to the checkpoint.

    `classes_file` is overridden separately, below, to a frozen snapshot beside
    the checkpoint when one exists -- see `classes_path_for`.
    """
    card = checkpoint.with_suffix(".json")          # shipped model: models/<name>.json
    run_cfg = checkpoint.parent.parent / "config.json"  # live run: outputs/config.json
    if explicit:
        raw, source = json.load(open(explicit)), explicit
    elif card.exists():
        # A shipped model carries its own card, and the card's training_config is
        # the authoritative record of how it was fitted -- the model travels out of
        # this repo, so it cannot depend on a params.yaml it will not have.
        raw, source = json.load(open(card)).get("training_config", {}), f"{card} (training_config)"
    elif run_cfg.exists():
        raw, source = json.load(open(run_cfg)), str(run_cfg)
    else:
        raw, source = None, "params.yaml (no config found beside the checkpoint)"

    if raw is None:
        cfg = RunConfig.from_params_yaml()
    else:
        known = RunConfig.__dataclass_fields__
        cfg = RunConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                           for k, v in raw.items() if k in known})

    frozen_classes = classes_path_for(checkpoint)
    if frozen_classes.exists():
        # Absolute rather than relative-to-REPO_ROOT: `checkpoint` itself may be
        # relative (e.g. the CLI's default) or absolute, and every consumer joins
        # this back onto REPO_ROOT, which pathlib resolves to the right-hand side
        # unchanged when it's already absolute -- so resolving here is safe either way.
        cfg = replace(cfg, classes_file=str(frozen_classes.resolve()))
    return cfg, source


def load_model(checkpoint: Path, model_name: str, num_classes: int, dropout: float):
    model, head = build_model(model_name, num_classes=num_classes, freeze_mode="none", dropout=dropout)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model, head


@torch.no_grad()
def forward_with_embeddings(model, head, tensors: torch.Tensor, batch_size: int = 32,
                            tta: bool = False):
    """(probs, embeddings) in one pass.

    The embedding is the head's *input* -- the pooled penultimate feature, 512-d
    for resnet18 -- captured with a pre-hook rather than by rebuilding the model
    headless, so the thing measured is exactly what this checkpoint feeds its
    classifier and cannot drift from it.

    With `tta`, probabilities are averaged over the 8 dihedral orientations. This
    is not a generic trick bolted on: training samples exactly this group
    (RandomRightAngleRotation + both flips), so averaging over it at inference
    asks the model the same question in the orientations it was taught are
    equivalent, and averages away the disagreement it should not have had.
    Measured on this repo's own folds it is worth +0.0235 cross-rig macro-F1 for
    8x inference cost and no retraining.

    **Embeddings always come from the untransformed pass**, even under TTA. The
    OOD reference defines a specific embedding space built from plain forward
    passes; averaging embeddings over orientations would move points inside that
    space and silently invalidate every distance measured against it.
    """
    captured: list[torch.Tensor] = []

    def hook(_module, inputs):
        captured.append(inputs[0].detach().cpu())

    handle = head.register_forward_pre_hook(hook)
    try:
        probs = []
        for i in range(0, len(tensors), batch_size):
            probs.append(F.softmax(model(tensors[i:i + batch_size].to(DEVICE)), dim=1).cpu().numpy())
    finally:
        handle.remove()
    probs = np.concatenate(probs, axis=0)
    embeds = torch.cat(captured).numpy()

    if tta:
        acc = probs.copy()
        n = 1
        for k in range(4):
            for flip in (False, True):
                if k == 0 and not flip:
                    continue  # the identity, already accumulated above
                out = []
                for i in range(0, len(tensors), batch_size):
                    b = torch.rot90(tensors[i:i + batch_size], k, dims=(2, 3))
                    if flip:
                        b = torch.flip(b, dims=(3,))
                    out.append(F.softmax(model(b.to(DEVICE)), dim=1).cpu().numpy())
                acc += np.concatenate(out, axis=0)
                n += 1
        probs = acc / n
    return probs, embeds


def crop_to_bean_region(rgb: np.ndarray) -> tuple[np.ndarray, dict | None]:
    """Crop `rgb` to the detected bean-filled region -- the live-inference
    mirror of the offline crop stage (coffeecv.crop_session, driven by
    coffeecv.crop_tray) that box_pictures/iphone/oneplus's training data
    already went through before training ever saw it. Returns (possibly
    cropped rgb, crop_info): crop_info is None for passthrough (no tray
    found -- the common, correct outcome for a frame-filling photo, same as
    pixel_cam/sony_cam's raw captures), or a dict describing the crop.

    BGR conversion happens here, once: locate_tray_rough's texture stage
    reads the wrong luma weights if fed RGB unconverted (verified: mean abs
    diff 15.9/255 on a real photo), so this must run before locate_bean_crop,
    not be left to it.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    result = locate_bean_crop(bgr)
    if result is None:
        return rgb, None
    x, y, w, h = result.box
    info = {"box": [x, y, w, h], "needs_review": result.needs_review, "note": None}
    if result.needs_review:
        info["note"] = (
            "A tray/background region was found and cropped out, but the framing was "
            "unusual enough that the crop's confidence is lower than normal. Not "
            "refused -- the sampled patches may include a thin sliver of background "
            "or miss a few edge beans.")
    return rgb[y:y + h, x:x + w], info


def patches_for_photo(path: Path, cfg: RunConfig, n_patches: int, seed_key: list[int],
                       skip_crop: bool = False):
    """Sample patches the way training does. Returns (patches, diagnostics).

    skip_crop is a user override, not an automatic decision: the live crop
    detector is a heuristic (coffeecv.crop_tray.locate_bean_crop), and a
    person looking at the framing preview may see a box they don't trust --
    this lets them fall back to the pre-crop-fix behavior (whole frame, same
    as an undetected passthrough) for that one photo, deliberately, rather
    than silently living with a bad detection.
    """
    t0 = time.monotonic()
    rgb = load_rgb_image(path)
    t1 = time.monotonic()
    decoded_h, decoded_w = rgb.shape[:2]
    crop_info = None
    if not skip_crop:
        rgb, crop_info = crop_to_bean_region(rgb)
    t2 = time.monotonic()
    h, w = rgb.shape[:2]
    region = compute_valid_region_rect(h, w, cfg.safety_margin)
    pitch = estimate_bean_pitch(grayscale_like_training(rgb), **pitch_kwargs(cfg))

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
        "crop": crop_info,
        "bean_pitch_px": round(pitch, 1),
        # How many beans span the usable short side -- the quantity that decides
        # whether the trained patch range is reachable at all in this frame.
        "beans_across": round(room / pitch, 2),
        "patches_clamped": clamped,
        "clamp_rate": round(clamped / max(len(boxes), 1), 3),
        "decoded_wh": [decoded_w, decoded_h],
        # Diagnostic only, for the web service's request logging -- never used
        # for any pass/fail decision, so it can't become a parity concern.
        "timing_ms": {"decode": round((t1 - t0) * 1000), "crop_detect": round((t2 - t1) * 1000)},
    }


def ood_scores(embeddings: np.ndarray, ref: dict) -> tuple[np.ndarray, list[str]]:
    """Per-patch distance to the nearest class centroid, in units of that class's spread."""
    cids = sorted(ref["classes"])
    cents = np.array([ref["classes"][c]["centroid"] for c in cids])
    spreads = np.array([ref["classes"][c]["spread"] for c in cids])
    d = np.linalg.norm(embeddings[:, None, :] - cents[None, :, :], axis=2) / spreads[None, :]
    nearest = d.argmin(axis=1)
    return d.min(axis=1), [cids[i] for i in nearest]


def classify_one(path: Path, cfg: RunConfig, class_ids: list[str], class_labels: dict[str, str],
                  model, head, ref: dict | None, n_patches: int = 40,
                  seed_key: list[int] | None = None, tta: bool = True,
                  skip_crop: bool = False) -> dict:
    """Classify a single photo. One call = exactly one iteration of the CLI's batch
    loop below, extracted so the CLI and any other caller (the web service) are
    provably running one code path rather than two that can silently drift
    -- the same parity concern this whole file exists for.

    No printing here -- that's presentation, kept in `_print_cli_verdict` below so a
    non-CLI caller isn't stuck with stdout lines meant for a terminal.
    """
    if seed_key is None:
        seed_key = [42, 0]

    try:
        patches, diag = patches_for_photo(path, cfg, n_patches, seed_key, skip_crop=skip_crop)
    except (ValueError, OSError) as exc:
        return {"verdict": "REFUSED (unmeasurable)", "error": str(exc)}

    # beans_across stays in the diag (and is logged per request by the web
    # service) even though nothing gates on it -- see the module docstring.
    entry = {**diag}

    transform = build_eval_transform(cfg.patch_resize)
    t_infer0 = time.monotonic()
    probs, embeds = forward_with_embeddings(
        model, head, torch.stack([transform(x) for x in patches]), tta=tta)
    entry["timing_ms"]["inference"] = round((time.monotonic() - t_infer0) * 1000)
    mean = probs.mean(axis=0)
    ranked = sorted(zip(class_ids, mean), key=lambda t: -t[1])
    entry["ranked"] = [(c, class_labels[c], float(v)) for c, v in ranked]

    if ref is None:
        entry["verdict"] = "UNGUARDED"
        entry["ood"] = None
        return entry

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
    # Message text lives here, once, rather than in the CLI printer or the web
    # response separately -- both callers show the exact same words for the exact
    # same verdict.
    if warn_at and OOD_THRESHOLD >= median > warn_at:
        entry["ood"]["warned"] = True
        entry["ood"]["note"] = (
            f"Above the training distribution's 95th percentile ({warn_at:.2f}). Not refused, but "
            f"treat the answer below as unreliable -- on this repo's held-out rig, photos in this "
            f"band lost about half their top-1 accuracy while staying under the refusal threshold.")
    if median > OOD_THRESHOLD:
        entry["verdict"] = "REFUSED (out of distribution)"
        entry["ood"]["note"] = (
            "This photo does not resemble the training distribution. Not offering a best guess: on "
            "such photos the embedding sits roughly equidistant from every class, so a ranked list "
            "would be false precision.")
        return entry
    entry["verdict"] = "predicted"
    return entry


def _print_cli_verdict(name: str, entry: dict, ref: dict | None) -> None:
    """Reproduces exactly the per-photo stdout of the pre-refactor inline loop."""
    if entry["verdict"] == "REFUSED (unmeasurable)":
        print(f"\n{name}\n  -> REFUSED: cannot measure bean scale ({entry['error']})")
        return

    print(f"\n{name}")
    crop = entry.get("crop")
    if crop and crop.get("needs_review"):
        print(f"  crop: WARNING: {crop['note']}")
    print(f"  scale: frame spans {entry['beans_across']:.1f} beans (measured, not enforced)")

    if entry["verdict"] == "REFUSED (scale)":
        print("  -> REFUSED: photo is outside the trained patch scale, not predicting.")
        return

    ood = entry.get("ood")
    if ood is not None:
        median = ood["median"]
        margin = median - ood["threshold"]
        print(f"  OOD: median distance {median:.2f} vs threshold {ood['threshold']} "
              f"({'over' if margin > 0 else 'under'} by {abs(margin):.2f})")
        # `note` is set by classify_one() -- same string the web response uses, so
        # the two callers can't drift on what this actually says.
        if ood.get("warned"):
            print(f"       WARNING: {ood['note']}")
        if entry["verdict"] == "REFUSED (out of distribution)":
            print(f"  -> REFUSED: {ood['note']}")
            return

    cid, label, pr = entry["ranked"][0]
    print(f"  -> {cid} {label}  p={pr:.3f}" + ("" if ref else "   (unguarded)"))


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
    p.add_argument("--no-tta", action="store_true",
                   help="disable dihedral test-time augmentation. TTA is ON by default: it is worth "
                        "+0.0235 cross-rig macro-F1 on this repo's folds, costs only inference time, "
                        "and averages over the same symmetry group training augments with.")
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
        entry = classify_one(path, cfg, class_ids, class_labels, model, head, ref,
                              n_patches=args.n_patches, seed_key=[args.seed, idx], tta=not args.no_tta)
        results[path.name] = entry
        _print_cli_verdict(path.name, entry, ref)

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
            "tta": not args.no_tta,
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
