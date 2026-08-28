"""Validate `coffeecv.crop_tray.locate_bean_crop` -- the exact function that will
run live at inference (see coffeecv.infer.crop_to_bean_region) -- against real
photos from all 5 rigs, before it's wired into classify_one().

Two different questions, both answered by the same pass:
  - box_pictures/iphone/oneplus (real tray-in-frame photos): does the detected
    box track the known-good fixed_trim crop that rig's training data was
    actually built from (iou_vs_fixed_trim)?
  - pixel_cam/sony_cam (frame-filling photos, nothing to detect): how often
    does locate_bean_crop wrongly find a "tray" anyway and crop a perfectly
    good photo down to an arbitrary slice of itself? locate_tray_rough alone
    was found to do this on every single test photo (pure texture noise, no
    real boundary); _saturation_gap's job is to catch that, and this is where
    its false-positive rate against real data actually gets measured
    (1 - passthrough_rate, since these two rigs should be ~100% passthrough).

Phase A of the live-crop plan: land detection, validate it against real photos
with a human looking at the contact sheet, and only then (Phase B) make
inference depend on it. Never touches data/cropped/ -- output goes to
outputs/qa_live_crop/<session>/, entirely separate from the training data tree.

    python -m coffeecv.qa_live_crop --session 2026-08-07__box_pictures_all_classes
    python -m coffeecv.qa_live_crop --all
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import yaml

from coffeecv.config import REPO_ROOT
from coffeecv.crop_tray import (
    _find_images,
    _imread_bgr,
    build_contact_sheet,
    locate_bean_crop,
    locate_tray_rough,
    resolve_trim,
)

RAW_ROOT = REPO_ROOT / "dataset"
QA_ROOT = REPO_ROOT / "outputs" / "qa_live_crop"

# All 5 active rigs. box_pictures/iphone/oneplus (fixed_trim) have real
# background and a known-good ground-truth crop to compare against;
# pixel_cam/sony_cam (method: none) are frame-filling and should score ~100%
# passthrough -- their purpose here is measuring the false-positive rate.
SESSIONS = [
    "2026-08-07__box_pictures_all_classes",
    "2026-08-25__iphone",
    "2026-08-25__oneplus",
    "2026-08-09__pixel_cam",
    "2026-08-09__sony_cam",
]


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _fixed_trim_box(img_bgr, trim: dict) -> tuple[int, int, int, int] | None:
    """Recompute the box crop_dataset_fixed_trim would have produced for this
    photo -- same stage-1 rough box, same per-side trim math as
    crop_tray.crop_dataset_fixed_trim -- so locate_bean_crop's adaptive box can
    be compared against the box that method actually used to build training
    data, on the rigs where that ground truth exists."""
    try:
        (x, y, w, h), _ = locate_tray_rough(img_bgr)
    except Exception:
        return None
    tl, tr = int(w * trim["left"]), int(w * trim["right"])
    tt, tb = int(h * trim["top"]), int(h * trim["bottom"])
    return (x + tl, y + tt, w - tl - tr, h - tt - tb)


def qa_session(session: str, out_root: Path = QA_ROOT) -> dict:
    raw_session = RAW_ROOT / session
    out_session = out_root / session
    class_dirs = sorted(p for p in raw_session.iterdir() if p.is_dir() and p.name.startswith("class_"))
    if not class_dirs:
        raise FileNotFoundError(f"No class_* directories under {raw_session}")

    fixed_trim = None
    crop_cfg_path = RAW_ROOT / f"{session}.crop.yaml"
    if crop_cfg_path.exists():
        cfg = yaml.safe_load(crop_cfg_path.read_text()) or {}
        if cfg.get("method") == "fixed_trim":
            fixed_trim = resolve_trim(cfg.get("trim", cfg.get("trim_frac")))

    all_reports = []
    elapsed_all = []
    for class_dir in class_dirs:
        out_dir = out_session / class_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        reports = []
        for photo_path in _find_images(class_dir):
            img = _imread_bgr(photo_path)
            entry = {"file": photo_path.name}
            t0 = time.perf_counter()
            result = locate_bean_crop(img)
            elapsed = time.perf_counter() - t0
            entry["elapsed_s"] = round(elapsed, 3)
            elapsed_all.append(elapsed)

            if result is None:
                entry.update({"passthrough": True, "needs_review": False, "box": None})
                out_img = img
            else:
                x, y, w, h = result.box
                out_img = img[y:y + h, x:x + w]
                entry.update({
                    "passthrough": False,
                    "box": list(result.box),
                    "rough_method": result.rough_method,
                    "needs_review": result.needs_review,
                })
                if fixed_trim is not None:
                    fbox = _fixed_trim_box(img, fixed_trim)
                    entry["iou_vs_fixed_trim"] = round(_iou(result.box, fbox), 3) if fbox else None

            out_path = out_dir / (photo_path.stem + "__cropped.jpg")
            cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            entry["out_path"] = str(out_path)
            reports.append(entry)

        (out_dir / "crop_report.json").write_text(json.dumps(reports, indent=2))
        all_reports.extend(reports)

    n = len(all_reports)
    n_review = sum(1 for r in all_reports if r["needs_review"])
    n_passthrough = sum(1 for r in all_reports if r["passthrough"])
    ious = sorted(r["iou_vs_fixed_trim"] for r in all_reports if r.get("iou_vs_fixed_trim") is not None)
    elapsed_sorted = sorted(elapsed_all)

    summary = {
        "session": session,
        "n_photos": n,
        "needs_review_rate": round(n_review / n, 3) if n else None,
        "passthrough_rate": round(n_passthrough / n, 3) if n else None,
        "elapsed_mean_s": round(statistics.mean(elapsed_all), 2) if elapsed_all else None,
        "elapsed_p95_s": round(elapsed_sorted[int(0.95 * len(elapsed_sorted))], 2) if elapsed_sorted else None,
        "iou_vs_fixed_trim_mean": round(statistics.mean(ious), 3) if ious else None,
        "iou_vs_fixed_trim_median": round(statistics.median(ious), 3) if ious else None,
        "iou_vs_fixed_trim_p05": round(ious[int(0.05 * len(ious))], 3) if ious else None,
    }
    out_session.mkdir(parents=True, exist_ok=True)
    (out_session / "qa_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    for class_dir in class_dirs:
        out_dir = out_session / class_dir.name
        try:
            sheet_path = build_contact_sheet(out_dir)
            print(f"  contact sheet: {sheet_path}")
        except FileNotFoundError:
            pass
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", choices=SESSIONS, help="run one session")
    p.add_argument("--all", action="store_true", help="run all 5 sessions")
    args = p.parse_args()
    if not args.session and not args.all:
        raise SystemExit("pass --session <name> or --all")

    sessions = SESSIONS if args.all else [args.session]
    summaries = [qa_session(s) for s in sessions]
    QA_ROOT.mkdir(parents=True, exist_ok=True)
    (QA_ROOT / "qa_summary_all.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
