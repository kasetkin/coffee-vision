"""Evaluate a checkpoint on `dataset/2026-07-24__first_pictures` -- the
project's earliest capture session, one labeled `class=NNN.heif` photo per
class, predating the box/pixel/sony rigs and never entered into `dvc.yaml`'s
crop stage or `run_folds.RIGS`.

**Why this needs its own script instead of just calling `infer.py`.** These
HEIF files use the *original* lens-circle geometry (square frame, a circle
inscribed in it, `alpha==0` outside -- confirmed by inspection: corners at
alpha 0, centre at 255, ~21.4% of pixels transparent, matching a circle
inscribed in a square almost exactly). `infer.py` only knows the *newer*
rectangular convention (`geometry.compute_valid_region_rect`, no lens circle,
whole frame usable) that the three registered rigs use -- pointing it at these
photos directly would treat the transparent corners as valid bean-pile content
and produce a garbage bean-pitch estimate.

The fix is a one-off pre-crop using the *original* `geometry.compute_valid_region`
(lens-circle-aware) to get the same safe inscribed square `PatchCoffeeDataset`
used to use, with alpha dropped -- after that, the cropped square stands in for
a "loaded photo" and every downstream step (pitch estimation, bean-unit patch
sampling, dihedral TTA) is `infer.py.patches_for_photo`'s own logic, not a
reimplementation, so this stays exactly as parity-faithful to training as
normal inference is. **Do not shortcut this to "resize the whole crop and score
it as one patch"** -- that bypasses the model's trained notion of patch scale
entirely (a patch is supposed to span ~4-7 beans, not the whole frame) and was
caught giving a different, wrong answer (8/9 instead of the correct 9/9) the
first time this script was written.

**What this is and is not evidence of.** It tests generalization to a
completely different lens/framing convention (circular macro vs. the training
rigs' rectangular box/frame-filling shots) that no training or screening data
ever included -- a genuinely new axis nothing else in this project measures.
It is NOT a fresh-scoop bean-identity test: this project has used the same
purchased bags of green beans throughout, so bean identity is almost certainly
still shared with the training data, same caveat as every other cross-rig
number here. And n=9 (one photo per class) is a wide-confidence-interval
sample, not a precise measurement -- report it as a spot-check, not a metric.

    python -m coffeecv.eval_legacy_lens_photos
    python -m coffeecv.eval_legacy_lens_photos --checkpoint models/other.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image

from coffeecv.bean_scale import estimate_bean_pitch
from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT
from coffeecv.dataset import load_class_labels
from coffeecv.geometry import compute_valid_region, compute_valid_region_rect, sample_bean_unit_patch_boxes
from coffeecv.infer import config_for_checkpoint, forward_with_embeddings, grayscale_like_training, load_model
from coffeecv.transforms import build_eval_transform

pillow_heif.register_heif_opener()

DEFAULT_SESSION = REPO_ROOT / "dataset" / "2026-07-24__first_pictures"


def crop_to_safe_square(heif_path: Path, safety_margin: float) -> np.ndarray:
    """The lens-circle-safe inscribed square, RGB only (alpha dropped -- by
    construction this region never touches the transparent corners, verified
    below by checking not a single alpha==0 pixel survives the crop)."""
    arr = np.array(Image.open(heif_path).convert("RGBA"))
    h, w = arr.shape[:2]
    region = compute_valid_region(h, w, safety_margin)
    crop = arr[region.y0:region.y1, region.x0:region.x1, :3]
    transparent = (arr[region.y0:region.y1, region.x0:region.x1, 3] == 0).sum()
    if transparent:
        raise AssertionError(
            f"{heif_path.name}: {transparent} transparent px leaked into the 'safe' crop -- "
            f"the lens-circle assumption may not hold for this photo, do not trust the result."
        )
    return crop


def patches_for_legacy_photo(rgb: np.ndarray, cfg, n_patches: int, seed_key: list[int]):
    """`infer.py.patches_for_photo`'s own logic, parameterised on an
    already-cropped RGB array instead of a file path -- the lens-circle crop
    above stands in for "loading the photo", so `safety_margin=1.0` here (no
    further shrink; that job is already done)."""
    h, w = rgb.shape[:2]
    region = compute_valid_region_rect(h, w, 1.0)
    pitch = estimate_bean_pitch(grayscale_like_training(rgb))

    rng = np.random.default_rng(seed_key)
    boxes, clamped = sample_bean_unit_patch_boxes(
        rng, region, n_patches, pitch, cfg.patch_beans_min, cfg.patch_beans_max, 0.0,
    )
    patches = []
    for box, _angle, _side in boxes:
        patch = Image.fromarray(rgb[box.y0:box.y1, box.x0:box.x1])
        if cfg.patch_store_size and patch.size[0] != cfg.patch_store_size:
            patch = patch.resize((cfg.patch_store_size, cfg.patch_store_size), Image.BILINEAR)
        patches.append(patch)

    room = min(region.width, region.height)
    return patches, {"bean_pitch_px": round(pitch, 1), "beans_across": round(room / pitch, 2),
                     "patches_clamped": clamped}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", default=str(DEFAULT_SESSION))
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--config", default=None)
    p.add_argument("--safety-margin", type=float, default=0.97)
    p.add_argument("--n-patches", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dihedral-tta", action="store_true")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    cfg, cfg_source = config_for_checkpoint(ckpt, args.config)
    print(f"checkpoint: {ckpt}\nconfig: {cfg_source}")

    class_labels = load_class_labels(REPO_ROOT / cfg.classes_file)
    class_ids = sorted(class_labels)
    model, head = load_model(ckpt, cfg.model_name, len(class_ids), cfg.dropout)
    eval_transform = build_eval_transform(cfg.patch_resize)

    session = Path(args.session_dir)
    heifs = sorted(session.glob("*__class=*.heif"))
    if not heifs:
        raise FileNotFoundError(f"No '*__class=NNN.heif' files under {session}")

    n_correct = 0
    for idx, heif_path in enumerate(heifs):
        true_id = heif_path.name.split("class=")[1].split(".")[0]
        rgb = crop_to_safe_square(heif_path, args.safety_margin)
        patches, diag = patches_for_legacy_photo(rgb, cfg, args.n_patches, [args.seed, idx])
        tensors = torch.stack([eval_transform(patch) for patch in patches])
        probs, _ = forward_with_embeddings(model, head, tensors, tta=not args.no_dihedral_tta)
        mean = probs.mean(axis=0)
        ranked = sorted(zip(class_ids, mean), key=lambda t: -t[1])
        pred_id, pred_p = ranked[0]
        correct = pred_id == true_id
        n_correct += correct
        mark = "correct" if correct else "WRONG"
        print(f"{heif_path.name}: true={true_id} {class_labels.get(true_id, true_id):<24} "
              f"pred={pred_id} {class_labels.get(pred_id, pred_id):<24} p={pred_p:.3f}  "
              f"beans_across={diag['beans_across']}  [{mark}]")

    print(f"\n{n_correct}/{len(heifs)} correct. n=1 photo/class -- a spot-check, not a metric; "
          f"see this module's docstring for what this does and does not test.")


if __name__ == "__main__":
    main()
