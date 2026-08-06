"""Locate and crop the bean-filled tray in a top-down box photo.

Unlike the circular macro-lens rig (see geometry.py), these photos have no
alpha channel marking valid pixels -- the tray has to be found by content.
Two stages:

  1. `locate_tray_rough`: find the tray's approximate bounding box by texture
     (the bean pile is highly textured; the table and tray rim are not).
  2. `locate_bean_region`: within that box, separate bean pixels from the
     metal rim by saturation (beans are visibly more saturated/brown than
     the rim or table), then take the largest all-bean axis-aligned
     rectangle, plus a safety trim.

Known failure mode: the tray's slanted inner wall can catch a warm
reflection off the beans, which fools plain saturation thresholding into
scoring rim pixels as "bean". The safety trim exists specifically to
route around this; `locate_bean_region` also self-checks contamination
(fraction of low-saturation pixels left in the final box) and grows the
trim if needed, but this is a heuristic, not a guarantee -- see the
`needs_review` flag in the crop report and use `build_contact_sheet` to
actually look at the results rather than trusting the numbers blind.

Designed to tolerate the tray appearing at different sizes in frame (i.e.
photos taken from an approximate, not fixed, distance): thresholds are
computed per-image (Otsu) rather than hardcoded, and morphological kernel
sizes scale with the detected tray size rather than being fixed pixel
counts.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# ---- Stage 1: rough tray localization -------------------------------------

# Fixed thresholds tried only if per-image Otsu thresholding fails to yield
# a plausible candidate (defense in depth against unusual lighting).
_TEXTURE_THRESHOLD_LADDER = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def _texture_map(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    lap = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
    texture = cv2.GaussianBlur(np.abs(lap), (0, 0), 15)
    return texture / (texture.max() + 1e-6)


def _odd(n: int, minimum: int = 3) -> int:
    n = max(int(n), minimum)
    return n if n % 2 == 1 else n + 1


def _rect_candidates(
    mask: np.ndarray,
    img_h: int,
    img_w: int,
    aspect_range: tuple[float, float],
    area_frac_range: tuple[float, float],
    center_frac: tuple[float, float],
) -> list[tuple[float, int, int, int, int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lo_c, hi_c = center_frac
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        area_frac = area / (img_h * img_w)
        if not (area_frac_range[0] < area_frac < area_frac_range[1]):
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw / 2, y + bh / 2
        if not (lo_c * img_w < cx < hi_c * img_w):
            continue
        if not (lo_c * img_h < cy < hi_c * img_h):
            continue
        aspect = bw / bh
        if not (aspect_range[0] < aspect < aspect_range[1]):
            continue
        candidates.append((area, x, y, bw, bh))
    return candidates


def locate_tray_rough(
    img_bgr: np.ndarray,
    aspect_range: tuple[float, float] = (0.5, 1.3),
    area_frac_range: tuple[float, float] = (0.01, 0.6),
    center_frac: tuple[float, float] = (0.15, 0.85),
) -> tuple[tuple[int, int, int, int], str]:
    """Returns ((x, y, w, h), method) where method is "otsu" or "ladder:<thr>",
    letting callers flag ladder-fallback detections for closer review."""
    h, w = img_bgr.shape[:2]
    texture = _texture_map(img_bgr)
    texture_u8 = (texture * 255).astype(np.uint8)
    close_k = np.ones((_odd(0.02 * min(h, w)),) * 2, np.uint8)
    open_k = np.ones((_odd(0.02 * min(h, w)),) * 2, np.uint8)

    _, otsu_mask = cv2.threshold(texture_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, close_k)
    otsu_mask = cv2.morphologyEx(otsu_mask, cv2.MORPH_OPEN, open_k)
    candidates = _rect_candidates(otsu_mask, h, w, aspect_range, area_frac_range, center_frac)
    if candidates:
        candidates.sort(reverse=True)
        _, x, y, bw, bh = candidates[0]
        return (x, y, bw, bh), "otsu"

    for thr in _TEXTURE_THRESHOLD_LADDER:
        mask = (texture > thr).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        candidates = _rect_candidates(mask, h, w, aspect_range, area_frac_range, center_frac)
        if candidates:
            candidates.sort(reverse=True)
            _, x, y, bw, bh = candidates[0]
            return (x, y, bw, bh), f"ladder:{thr}"

    raise RuntimeError("locate_tray_rough: no plausible tray region found")


# ---- Stage 2: precise bean-region extraction -------------------------------

def _max_rectangle(mask01: np.ndarray) -> tuple[int, int, int, int, int]:
    """Largest all-ones axis-aligned rectangle. Returns (area, x0, y0, x1, y1)."""
    rows, cols = mask01.shape
    height = np.zeros(cols, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for r in range(rows):
        height = np.where(mask01[r] == 1, height + 1, 0)
        stack: list[tuple[int, int]] = []
        for c in range(cols + 1):
            h = height[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, sh = stack.pop()
                area = sh * (c - s)
                if area > best[0]:
                    best = (area, s, r - sh + 1, c, r + 1)
                start = s
            stack.append((start, h))
    return best


@dataclass
class CropResult:
    box: tuple[int, int, int, int]  # x, y, w, h in original-image coordinates
    rough_method: str
    s_thresh: float
    contamination: float
    extra_trim_iters: int
    retained_area_frac: float  # final crop area / rough box area
    needs_review: bool


def locate_bean_region(
    img_bgr: np.ndarray,
    rough_box: tuple[int, int, int, int],
    pad_frac: float = 0.03,
    base_trim_frac: float = 0.05,
    border_band_frac: float = 0.035,
    border_max_contamination: float = 0.22,
    trim_step_frac: float = 0.015,
    max_iters_per_side: int = 8,
    min_size_px: int = 120,
) -> CropResult:
    """Whole-box average "contamination" (low-saturation pixel fraction) turned out to be
    a poor rim-leak detector: coffee beans' own pale crease line is low-saturation too, so
    it's present at ~5-10% almost everywhere, and averaging over the whole box can dilute a
    real, localized leak (verified: a corner-only rim sliver still measured ~5% average,
    under a naive whole-box threshold, and only showed up on a full-resolution look, not the
    contact-sheet thumbnail). So this checks each of the 4 edges independently via a thin
    border band and trims only the sides that actually need it -- `border_max_contamination`
    is set well above the ambient bean-crease noise floor (~5-10%) so it triggers on real
    leaks (which run much hotter along the whole edge) without chasing bean texture."""
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = rough_box
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    sub = img_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    s_thresh, otsu_mask = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (otsu_mask > 0).astype(np.uint8)

    short_side = min(sub.shape[:2])
    close_k = np.ones((_odd(0.045 * short_side),) * 2, np.uint8)
    open_k = np.ones((_odd(0.01 * short_side),) * 2, np.uint8)
    erode_k = np.ones((_odd(0.01 * short_side),) * 2, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.erode(mask, erode_k)

    _, rx0, ry0, rx1, ry1 = _max_rectangle(mask)
    rw, rh = rx1 - rx0, ry1 - ry0
    cur_x0, cur_y0 = rx0 + int(rw * base_trim_frac), ry0 + int(rh * base_trim_frac)
    cur_x1, cur_y1 = rx1 - int(rw * base_trim_frac), ry1 - int(rh * base_trim_frac)

    def band_contamination(bx0, by0, bx1, by1) -> float:
        region = S[by0:by1, bx0:bx1]
        return float((region <= s_thresh).mean()) if region.size else 1.0

    total_iters = 0
    # Full-width top/bottom bands first (also covers the top/bottom half of each corner),
    # then full-height left/right bands over the now-updated row range (covers the rest).
    for _ in range(max_iters_per_side):
        band_h = max(2, int((cur_y1 - cur_y0) * border_band_frac))
        if band_contamination(cur_x0, cur_y0, cur_x1, cur_y0 + band_h) <= border_max_contamination:
            break
        cur_y0 += max(1, int((cur_y1 - cur_y0) * trim_step_frac))
        total_iters += 1
        if cur_y1 - cur_y0 < min_size_px:
            break
    for _ in range(max_iters_per_side):
        band_h = max(2, int((cur_y1 - cur_y0) * border_band_frac))
        if band_contamination(cur_x0, cur_y1 - band_h, cur_x1, cur_y1) <= border_max_contamination:
            break
        cur_y1 -= max(1, int((cur_y1 - cur_y0) * trim_step_frac))
        total_iters += 1
        if cur_y1 - cur_y0 < min_size_px:
            break
    for _ in range(max_iters_per_side):
        band_w = max(2, int((cur_x1 - cur_x0) * border_band_frac))
        if band_contamination(cur_x0, cur_y0, cur_x0 + band_w, cur_y1) <= border_max_contamination:
            break
        cur_x0 += max(1, int((cur_x1 - cur_x0) * trim_step_frac))
        total_iters += 1
        if cur_x1 - cur_x0 < min_size_px:
            break
    for _ in range(max_iters_per_side):
        band_w = max(2, int((cur_x1 - cur_x0) * border_band_frac))
        if band_contamination(cur_x1 - band_w, cur_y0, cur_x1, cur_y1) <= border_max_contamination:
            break
        cur_x1 -= max(1, int((cur_x1 - cur_x0) * trim_step_frac))
        total_iters += 1
        if cur_x1 - cur_x0 < min_size_px:
            break

    final_w, final_h = cur_x1 - cur_x0, cur_y1 - cur_y0
    whole_box_contamination = band_contamination(cur_x0, cur_y0, cur_x1, cur_y1)
    needs_review = (
        final_w < min_size_px
        or final_h < min_size_px
        or total_iters >= 4 * max_iters_per_side  # every side maxed out without converging
    )

    return CropResult(
        box=(int(x0 + cur_x0), int(y0 + cur_y0), int(final_w), int(final_h)),
        rough_method="",  # filled in by caller
        s_thresh=float(s_thresh),
        contamination=whole_box_contamination,
        extra_trim_iters=total_iters,
        retained_area_frac=(final_w * final_h) / max(bw * bh, 1),
        needs_review=needs_review,
    )


# ---- Batch driver + QA tooling ---------------------------------------------

def crop_dataset(
    images_dir: Path,
    out_dir: Path,
    aspect_range: tuple[float, float] = (0.5, 1.3),
    area_frac_range: tuple[float, float] = (0.01, 0.6),
    center_frac: tuple[float, float] = (0.15, 0.85),
    border_max_contamination: float = 0.22,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.jpeg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg/jpeg images found in {images_dir}")

    reports = []
    for p in image_paths:
        img = cv2.imread(str(p))
        entry = {"file": p.name}
        try:
            rough_box, method = locate_tray_rough(img, aspect_range, area_frac_range, center_frac)
            result = locate_bean_region(img, rough_box, border_max_contamination=border_max_contamination)
            result.rough_method = method
            if method != "otsu":
                result.needs_review = True  # ladder fallback = unusual lighting, flag it

            x, y, w, h = result.box
            crop = img[y:y + h, x:x + w]
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
    print(f"Cropped {len(reports)} images, {n_flagged} flagged for review.")
    (out_dir / "crop_report.json").write_text(json.dumps(reports, indent=2))
    return reports


def build_contact_sheet(out_dir: Path, thumb_size: int = 260, cols: int = 5) -> Path:
    """Tile all `*__cropped.jpg` / `*__bbox.jpg` images in out_dir into one grid
    with filename captions, so review is one glance instead of one file open per
    image. `__bbox.jpg` files (crop_tray_sam.py --masked) are loose boxes with a
    paired `__mask.png` defining what's actually usable -- shown here with the
    mask overlaid (green) rather than the raw box, so the sheet reflects what
    sampling will actually use, not what the rim-including box looks like."""
    crop_paths = sorted(out_dir.glob("*__cropped.jpg")) + sorted(out_dir.glob("*__bbox.jpg"))
    if not crop_paths:
        raise FileNotFoundError(f"No cropped images found in {out_dir}")

    report = {}
    report_path = out_dir / "crop_report.json"
    if report_path.exists():
        report = {r["file"]: r for r in json.loads(report_path.read_text())}

    rows = (len(crop_paths) + cols - 1) // cols
    label_h = 40
    sheet = np.full((rows * (thumb_size + label_h), cols * thumb_size, 3), 30, dtype=np.uint8)

    for i, p in enumerate(crop_paths):
        img = cv2.imread(str(p))
        mask_path = p.with_name(p.name.replace("__bbox.jpg", "__mask.png"))
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            m = mask > 127
            img = img.copy()
            img[m] = (img[m] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

        h, w = img.shape[:2]
        scale = thumb_size / max(h, w)
        thumb = cv2.resize(img, (int(w * scale), int(h * scale)))
        th, tw = thumb.shape[:2]
        r, c = divmod(i, cols)
        y0 = r * (thumb_size + label_h) + (thumb_size - th) // 2
        x0 = c * thumb_size + (thumb_size - tw) // 2
        sheet[y0:y0 + th, x0:x0 + tw] = thumb

        src_name = p.name.replace("__cropped.jpg", ".jpg").replace("__bbox.jpg", ".jpg")
        flagged = report.get(src_name, {}).get("needs_review", False)
        color = (0, 0, 255) if flagged else (0, 200, 0)
        label = (src_name[:22] + ("!REVIEW" if flagged else ""))
        cv2.putText(
            sheet, label, (c * thumb_size + 4, r * (thumb_size + label_h) + thumb_size + 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    out_path = out_dir / "contact_sheet.jpg"
    cv2.imwrite(str(out_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Detect and crop the bean tray in top-down box photos.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--border-max-contamination", type=float, default=0.22)
    p.add_argument("--contact-sheet", action="store_true")
    args = p.parse_args()

    reports = crop_dataset(
        Path(args.images_dir), Path(args.out_dir), border_max_contamination=args.border_max_contamination
    )
    for r in reports:
        if r.get("needs_review"):
            print(f"  REVIEW: {r['file']} -> {r.get('error') or r}")

    if args.contact_sheet:
        sheet_path = build_contact_sheet(Path(args.out_dir))
        print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
