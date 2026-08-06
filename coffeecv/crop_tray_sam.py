"""Prototype: locate the bean-only region using MobileSAM instead of the
saturation-threshold heuristic in crop_tray.py.

Rationale (see crop_tray.py's module docstring for the background): the
beans-vs-rim boundary is a genuine object/material-boundary problem, and
color thresholds kept needing hand-tuned safety margins to route around
the tray's reflective inner wall. A promptable segmentation model should
find that boundary directly instead of us approximating it by saturation.

Stage 1 (find the tray's rough bounding box in the whole photo) is left
untouched -- reuses crop_tray.locate_tray_rough, which already works fine
with plain texture detection and doesn't need a model. Only stage 2
(bean region vs. rim, within that box) is replaced here.

Key finding while prototyping: a single point prompt at the tray center
snaps to ONE bean (SAM finds the tightest coherent boundary at that
point, and individual beans are their own closed contours), not the
whole pile. A grid of foreground points scattered across the tray
interior fixes this -- with multiple points, SAM finds one mask
consistent with all of them, i.e. roughly "the granular pile," not a
single grain.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

from coffeecv.crop_tray import _max_rectangle, locate_tray_rough


@dataclass
class SamCropResult:
    box: tuple[int, int, int, int]
    rough_method: str
    mask_area_frac: float  # SAM mask area / rough box area
    retained_area_frac: float  # final rectangle area / rough box area
    n_prompt_points: int
    inference_seconds: float
    needs_review: bool


def _prompt_grid(rough_box: tuple[int, int, int, int], n: int = 3) -> list[list[int]]:
    x, y, w, h = rough_box
    offsets = np.linspace(0.2, 0.8, n)
    return [[int(x + fx * w), int(y + fy * h)] for fy in offsets for fx in offsets]


def _run_sam_mask(
    model: SAM, img_path: str, rough_box: tuple[int, int, int, int], grid_n: int
) -> tuple[np.ndarray, float]:
    """Returns (full-image-sized binary mask, inference seconds)."""
    points = _prompt_grid(rough_box, grid_n)
    labels = [1] * len(points)

    t0 = time.time()
    results = model(img_path, points=[points], labels=[labels], verbose=False)
    elapsed = time.time() - t0

    r = results[0]
    if r.masks is None or len(r.masks.data) == 0:
        raise RuntimeError(f"SAM returned no mask for {img_path}")
    mask = (r.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8)
    return mask, elapsed


def locate_bean_region_sam(
    model: SAM,
    img_bgr: np.ndarray,
    img_path: str,
    rough_box: tuple[int, int, int, int],
    grid_n: int = 3,
    min_size_px: int = 120,
) -> SamCropResult:
    x, y, bw, bh = rough_box
    mask, elapsed = _run_sam_mask(model, img_path, rough_box, grid_n)

    # Restrict to the rough box before taking the max rectangle -- SAM's mask can
    # extend slightly past it (e.g. if beans spill visually into shadow at the box
    # edge), and we don't want the inscribed rectangle reaching past our own prior.
    sub_mask = mask[y:y + bh, x:x + bw]
    mask_area_frac = float(sub_mask.mean())

    _, rx0, ry0, rx1, ry1 = _max_rectangle(sub_mask)
    final_w, final_h = rx1 - rx0, ry1 - ry0
    needs_review = bool(
        final_w < min_size_px
        or final_h < min_size_px
        or mask_area_frac < 0.15  # SAM likely latched onto something much smaller than the tray
    )

    return SamCropResult(
        box=(int(x + rx0), int(y + ry0), int(final_w), int(final_h)),
        rough_method="",
        mask_area_frac=mask_area_frac,
        retained_area_frac=float((final_w * final_h) / max(bw * bh, 1)),
        n_prompt_points=grid_n * grid_n,
        inference_seconds=elapsed,
        needs_review=needs_review,
    )


@dataclass
class SamMaskResult:
    bbox: tuple[int, int, int, int]  # loose bbox around the mask, in original-image coords
    rough_method: str
    mask_area_frac: float  # SAM mask area / bbox area -- how much of the loose crop is usable
    n_prompt_points: int
    inference_seconds: float
    needs_review: bool


def locate_bean_bbox_and_mask_sam(
    model: SAM,
    img_path: str,
    rough_box: tuple[int, int, int, int],
    grid_n: int = 3,
    bbox_pad_frac: float = 0.01,
    min_size_px: int = 120,
) -> tuple[SamMaskResult, np.ndarray]:
    """Like locate_bean_region_sam, but instead of collapsing the irregular mask
    down to one inscribed rectangle, returns a loose bounding box around the
    whole mask plus the mask itself (cropped to that box) -- for callers that
    sample patches directly against the mask instead of a single rectangle."""
    x, y, bw, bh = rough_box
    mask, elapsed = _run_sam_mask(model, img_path, rough_box, grid_n)
    sub_mask = mask[y:y + bh, x:x + bw]

    ys, xs = np.where(sub_mask > 0)
    if len(ys) == 0:
        raise RuntimeError(f"SAM mask is empty within the rough box for {img_path}")
    mx0, mx1 = xs.min(), xs.max() + 1
    my0, my1 = ys.min(), ys.max() + 1
    px, py = int((mx1 - mx0) * bbox_pad_frac), int((my1 - my0) * bbox_pad_frac)
    bx0, by0 = max(0, mx0 - px), max(0, my0 - py)
    bx1, by1 = min(bw, mx1 + px), min(bh, my1 + py)

    mask_crop = sub_mask[by0:by1, bx0:bx1]
    mask_area_frac = float(mask_crop.mean())
    final_w, final_h = bx1 - bx0, by1 - by0
    needs_review = bool(
        final_w < min_size_px
        or final_h < min_size_px
        or mask_area_frac < 0.15
    )

    result = SamMaskResult(
        bbox=(int(x + bx0), int(y + by0), int(final_w), int(final_h)),
        rough_method="",
        mask_area_frac=mask_area_frac,
        n_prompt_points=grid_n * grid_n,
        inference_seconds=elapsed,
        needs_review=needs_review,
    )
    return result, mask_crop


def crop_dataset_sam_masked(
    images_dir: Path,
    out_dir: Path,
    weights: str = "mobile_sam.pt",
    grid_n: int = 3,
) -> list[dict]:
    """Saves a loose RGB bbox crop + a paired binary mask per image, instead of
    one inscribed rectangle. Consumers (see infer.py's mask-aware sampling)
    sample patches directly against the mask, using more of the irregular
    region than a single axis-aligned rectangle can cover."""
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg/jpeg images found in {images_dir}")

    model = SAM(weights)
    reports = []
    for p in image_paths:
        img = cv2.imread(str(p))
        entry = {"file": p.name}
        try:
            rough_box, method = locate_tray_rough(img)
            result, mask_crop = locate_bean_bbox_and_mask_sam(model, str(p), rough_box, grid_n=grid_n)
            result.rough_method = method

            bx, by, bw, bh = result.bbox
            crop = img[by:by + bh, bx:bx + bw]
            stem = p.name.replace(".jpg", "").replace(".jpeg", "")
            # "__bbox" (not "__cropped"): this is a loose box around the mask, rim
            # included on purpose -- the paired __mask.png is what's actually clean.
            # Naming it "__cropped" caused real confusion (looked like a bad crop).
            crop_path = out_dir / f"{stem}__bbox.jpg"
            mask_path = out_dir / f"{stem}__mask.png"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(mask_path), mask_crop * 255)

            entry.update(asdict(result))
            entry["rough_box"] = list(rough_box)
            entry["out_path"] = str(crop_path)
            entry["mask_path"] = str(mask_path)
            entry["error"] = None
        except Exception as e:
            entry["error"] = str(e)
            entry["needs_review"] = True
        reports.append(entry)

    n_flagged = sum(1 for r in reports if r.get("needs_review"))
    total_t = sum(r.get("inference_seconds", 0) for r in reports)
    print(f"Cropped {len(reports)} images (masked), {n_flagged} flagged, {total_t:.1f}s SAM inference total.")
    (out_dir / "crop_report.json").write_text(json.dumps(reports, indent=2))
    return reports


def crop_dataset_sam(
    images_dir: Path,
    out_dir: Path,
    weights: str = "mobile_sam.pt",
    grid_n: int = 3,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg/jpeg images found in {images_dir}")

    model = SAM(weights)
    reports = []
    for p in image_paths:
        img = cv2.imread(str(p))
        entry = {"file": p.name}
        try:
            rough_box, method = locate_tray_rough(img)
            result = locate_bean_region_sam(model, img, str(p), rough_box, grid_n=grid_n)
            result.rough_method = method

            bx, by, bw, bh = result.box
            crop = img[by:by + bh, bx:bx + bw]
            out_path = out_dir / p.name.replace(".jpg", "__cropped.jpg").replace(".jpeg", "__cropped.jpg")
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            entry.update(asdict(result))
            entry["rough_box"] = list(rough_box)
            entry["out_path"] = str(out_path)
            entry["error"] = None
        except Exception as e:
            entry["error"] = str(e)
            entry["needs_review"] = True
        reports.append(entry)

    n_flagged = sum(1 for r in reports if r.get("needs_review"))
    total_t = sum(r.get("inference_seconds", 0) for r in reports)
    print(f"Cropped {len(reports)} images, {n_flagged} flagged, {total_t:.1f}s SAM inference total.")
    (out_dir / "crop_report.json").write_text(json.dumps(reports, indent=2))
    return reports


def main() -> None:
    p = argparse.ArgumentParser(description="Locate and crop the bean tray using MobileSAM.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--weights", default="mobile_sam.pt")
    p.add_argument("--grid-n", type=int, default=3, help="NxN grid of foreground prompt points.")
    p.add_argument(
        "--masked", action="store_true",
        help="Save a loose bbox crop + paired mask instead of one inscribed rectangle "
             "(for mask-aware patch sampling in infer.py, rather than uniform box sampling).",
    )
    args = p.parse_args()

    driver = crop_dataset_sam_masked if args.masked else crop_dataset_sam
    reports = driver(Path(args.images_dir), Path(args.out_dir), args.weights, args.grid_n)
    for r in reports:
        if r.get("needs_review"):
            print(f"  REVIEW: {r['file']} -> {r.get('error') or r}")


if __name__ == "__main__":
    main()
