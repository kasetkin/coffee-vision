"""Run the trained checkpoint over new, unlabeled photos.

Unlike PatchCoffeeDataset (which crops from the fixed circular macro-lens
photos used for training), this script accepts arbitrary already-cropped
RGB images -- e.g. a rectangular top-down photo of a bean tray -- and
patch-samples them the same way (random crop_size x crop_size boxes,
resized to patch_resize) before running them through the classifier.

`crop_size` here is NOT necessarily the same value as training's
patch_crop_size: if the new photos were taken at a different camera
distance/zoom, the apparent bean size in pixels differs, and crop_size
should be rescaled so beans appear at roughly the same size in the
resized patch that the network was trained on. See --scale-ratio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from coffeecv.config import CHECKPOINTS_DIR, REPO_ROOT
from coffeecv.dataset import load_class_labels
from coffeecv.geometry import Region, sample_patch_boxes
from coffeecv.model import build_model
from coffeecv.transforms import build_eval_transform

DEVICE = torch.device("cpu")


def mask_path_for(img_path: Path) -> Path | None:
    """Sibling `<stem>__mask.png` for a `<stem>__bbox.jpg`, if crop_tray_sam.py was
    run with --masked. Returns None (not an error) for ordinary `__cropped.jpg`
    crops, which are already clean on their own and have no mask to pair with."""
    if not img_path.name.endswith("__bbox.jpg"):
        return None
    candidate = img_path.with_name(img_path.name.replace("__bbox.jpg", "__mask.png"))
    return candidate if candidate.exists() else None


def valid_patch_origins(mask01: np.ndarray, crop_size: int) -> np.ndarray:
    """1 at (y, x) iff the crop_size x crop_size box anchored there (top-left) is
    fully inside mask01. Computed via erosion: a pixel survives erosion by a
    crop_size x crop_size all-ones kernel only if every pixel under that kernel
    was 1 to begin with, which is exactly the "does this patch fit" test."""
    kernel = np.ones((crop_size, crop_size), np.uint8)
    return cv2.erode(mask01, kernel, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=0)


def patches_for_image_masked(
    img_path: Path, mask_path: Path, crop_size: int, n_patches: int, seed_key: list[int]
) -> tuple[list[Image.Image], int]:
    """Like patches_for_image, but only accepts patches fully inside the mask
    instead of anywhere in the rectangular image. Returns (patches, n_valid_origins)
    -- the latter is a direct measure of how much usable area this photo had,
    comparable across photos regardless of crop shape."""
    img = Image.open(img_path).convert("RGB")
    mask = np.array(Image.open(mask_path).convert("L"))
    mask01 = (mask > 127).astype(np.uint8)
    w, h = img.size
    if crop_size > min(w, h):
        raise ValueError(f"crop_size={crop_size} too large for {img_path.name} ({w}x{h})")

    valid = valid_patch_origins(mask01, crop_size)
    ys, xs = np.where(valid > 0)
    if len(ys) == 0:
        raise ValueError(f"No position fits crop_size={crop_size} inside the mask for {img_path.name}")

    rng = np.random.default_rng(seed_key)
    idx = rng.integers(0, len(ys), size=n_patches)
    arr = np.array(img)
    patches = [Image.fromarray(arr[ys[i]:ys[i] + crop_size, xs[i]:xs[i] + crop_size]) for i in idx]
    return patches, len(ys)


def load_model(checkpoint_path: Path, model_name: str, num_classes: int, dropout: float = 0.2):
    model, _ = build_model(model_name, num_classes=num_classes, freeze_mode="none", dropout=dropout)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def patches_for_image(img_path: Path, crop_size: int, n_patches: int, seed_key: list[int]) -> list[Image.Image]:
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    if crop_size > min(w, h):
        raise ValueError(f"crop_size={crop_size} too large for {img_path.name} ({w}x{h})")
    region = Region(y0=0, y1=h, x0=0, x1=w)
    rng = np.random.default_rng(seed_key)
    boxes = sample_patch_boxes(rng, region, n_patches, crop_size)
    return [img.crop((b.x0, b.y0, b.x1, b.y1)) for b in boxes]


@torch.no_grad()
def predict_probs(model, patches: list[Image.Image], resize: int, batch_size: int = 32) -> np.ndarray:
    transform = build_eval_transform(resize)
    tensors = torch.stack([transform(p) for p in patches])
    all_probs = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size].to(DEVICE)
        logits = model(batch)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description="Classify new (already-cropped) photos with the trained checkpoint.")
    p.add_argument("--images-dir", required=True, help="Directory of cropped RGB images to classify.")
    p.add_argument("--checkpoint", default=str(CHECKPOINTS_DIR / "best.pt"))
    p.add_argument("--model-name", default="resnet18")
    p.add_argument("--classes-file", default="dataset/classes.txt")
    p.add_argument("--crop-size", type=int, default=700, help="Patch crop size in source pixels.")
    p.add_argument("--patch-resize", type=int, default=224)
    p.add_argument("--n-patches", type=int, default=24, help="Patches sampled per image.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="Optional path to write JSON results.")
    args = p.parse_args()

    class_labels = load_class_labels(REPO_ROOT / args.classes_file)
    class_ids = sorted(class_labels.keys())

    model = load_model(Path(args.checkpoint), args.model_name, num_classes=len(class_ids))

    images_dir = Path(args.images_dir)
    excluded = {"contact_sheet.jpg"}
    image_paths = sorted(
        p for p in list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))
        if p.name not in excluded and not p.name.endswith("__mask.png")
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    per_image_results = {}
    all_probs = []
    for idx, img_path in enumerate(image_paths):
        mpath = mask_path_for(img_path)
        if mpath is not None:
            patches, n_valid_origins = patches_for_image_masked(
                img_path, mpath, args.crop_size, args.n_patches, seed_key=[args.seed, idx]
            )
        else:
            patches = patches_for_image(img_path, args.crop_size, args.n_patches, seed_key=[args.seed, idx])
            n_valid_origins = None
        probs = predict_probs(model, patches, args.patch_resize)
        mean_probs = probs.mean(axis=0)
        all_probs.append(probs)
        ranked = sorted(zip(class_ids, mean_probs), key=lambda t: -t[1])
        per_image_results[img_path.name] = {
            "mean_probs": {cid: float(pr) for cid, pr in zip(class_ids, mean_probs)},
            "ranked": [(cid, class_labels[cid], float(pr)) for cid, pr in ranked],
            "n_valid_origins": n_valid_origins,
        }
        top_cid, top_label, top_p = per_image_results[img_path.name]["ranked"][0]
        mode = f"masked, n_valid_origins={n_valid_origins}" if mpath is not None else "rectangle"
        print(f"{img_path.name} [{mode}]: top1={top_cid} {top_label} p={top_p:.3f}")

    pooled = np.concatenate(all_probs, axis=0).mean(axis=0)
    ranked_overall = sorted(zip(class_ids, pooled), key=lambda t: -t[1])
    print("\n=== Overall (pooled across all images/patches) ===")
    for cid, pr in ranked_overall:
        print(f"  {cid} {class_labels[cid]:25s} {pr:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "crop_size": args.crop_size,
            "patch_resize": args.patch_resize,
            "n_patches_per_image": args.n_patches,
            "per_image": per_image_results,
            "overall_ranked": [(cid, class_labels[cid], float(pr)) for cid, pr in ranked_overall],
        }, indent=2))
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
