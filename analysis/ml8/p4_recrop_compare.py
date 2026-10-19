"""Ticket ML-8 P4 (R3): what the re-crop of 2026-10-19 (XL0-v3 with today's code, ML-7's ported mask code
included) changed against ML-6 P8's crops (0675f40), on the photos both have.

    python analysis/ml8/p4_recrop_compare.py OLD_ROOT [--out analysis/ml8/p4_recrop_report.txt]

OLD_ROOT holds P8's segcrop outputs, one dir per session, as data/segcropped/<session> held them before the
re-crop (a hard-linked copy, or `dvc checkout` at 0675f40). Per photo, from each class's segcrop_report.json:
the crop box, the D18 fallback, the full-frame mask's sha256, the pixels that differ in the crop mask training
reads (filled and cropped; compared in photo coordinates from each crop's mask and box), and the patches the D17 rule places: SERVING_PATCHES
bean-unit boxes drawn on each crop with one fixed RNG key, as training draws them (dataset._extract_photo),
counted where they differ. Photos only one side has are listed apart (the re-crop's new photos).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from coffeecv.bean_scale import estimate_bean_pitch, pitch_kwargs  # noqa: E402
from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.dataset import SPLIT_SEED_COMPONENT, bean_share_rule, load_bean_mask, load_rgb_image  # noqa: E402
from coffeecv.geometry import compute_valid_region_rect, sample_masked_bean_unit_boxes  # noqa: E402

NEW_ROOT = REPO_ROOT / "data" / "segcropped"
SERVING_PATCHES = 40


def reports(session_dir: Path) -> dict[tuple[str, str], dict]:
    """{(class dir, raw file): its segcrop_report.json row}."""
    out = {}
    for rep in sorted(session_dir.glob("class_*/segcrop_report.json")):
        for r in json.loads(rep.read_text()):
            out[(rep.parent.name, r["file"])] = r
    return out


def boxes(crop: Path, cfg: RunConfig) -> list[tuple]:
    rgb = load_rgb_image(crop)
    h, w = rgb.shape[:2]
    gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype(np.uint8)
    pitch = estimate_bean_pitch(gray, **pitch_kwargs(cfg))
    mask = load_bean_mask(crop, rgb.shape[:2])
    rng = np.random.default_rng([cfg.seed, 0, 0, SPLIT_SEED_COMPONENT["all"]])
    got, _, _ = sample_masked_bean_unit_boxes(rng, compute_valid_region_rect(h, w, cfg.safety_margin),
                                              SERVING_PATCHES, pitch, cfg.patch_beans_min, cfg.patch_beans_max,
                                              mask, *bean_share_rule(cfg))
    return [(b.y0, b.y1, b.x0, b.x1) for b, _, _ in got]


def mask_pixels_changed(old: Path, new: Path, old_row: dict, new_row: dict) -> int | None:
    """Pixels where the two full-frame masks differ, each crop's mask placed at its box ([x0, y0, w, h] in the
    photo, segment_beans); None if either is a D18 fallback (no mask, no box)."""
    placed = []
    for crop, row in ((old, old_row), (new, new_row)):
        m = load_bean_mask(crop, load_rgb_image(crop).shape[:2])
        if m is None or row["box"] is None:
            return None
        placed.append((m.astype(bool), row["box"]))
    H = max(y0 + h for _, (x0, y0, w, h) in placed)
    W = max(x0 + w for _, (x0, y0, w, h) in placed)
    canvases = []
    for m, (x0, y0, w, h) in placed:
        c = np.zeros((H, W), bool)
        c[y0:y0 + h, x0:x0 + w] = m
        canvases.append(c)
    return int((canvases[0] ^ canvases[1]).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_root", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    cfg = RunConfig.from_params_yaml()
    lines, totals = [], Counter()
    changed_rows = []
    only_new, only_old = [], []
    for old_dir in sorted(d for d in args.old_root.iterdir() if d.is_dir()):
        session = old_dir.name
        old, new = reports(old_dir), reports(NEW_ROOT / session)
        only_new += [f"{session}/{c}/{f}" for c, f in sorted(set(new) - set(old))]
        only_old += [f"{session}/{c}/{f}" for c, f in sorted(set(old) - set(new))]
        s = Counter()
        for key in sorted(set(old) & set(new)):
            o, n = old[key], new[key]
            crop_name = Path(o["out_path"]).name
            oc, nc = old_dir / key[0] / crop_name, NEW_ROOT / session / key[0] / crop_name
            s["photos"] += 1
            box = o["box"] != n["box"]
            fb = o["fallback"] != n["fallback"]
            msha = o["mask_sha256"] != n["mask_sha256"]
            s["box_changed"] += box
            s["fallback_changed"] += fb
            s["mask_changed"] += msha
            px = mask_pixels_changed(oc, nc, o, n) if msha else 0
            nb = 0
            if msha or box or fb:
                bo, bn = boxes(oc, cfg), boxes(nc, cfg)
                nb = sum(a != b for a, b in zip(bo, bn)) + abs(len(bo) - len(bn))
                s["patches_moved"] += nb
                s["photos_with_patches_moved"] += nb > 0
                changed_rows.append((session, key[0], key[1], box, fb, px, nb))
            if px:
                s["mask_px_max"] = max(s["mask_px_max"], px)
        totals.update({k: v for k, v in s.items() if k != "mask_px_max"})
        totals["mask_px_max"] = max(totals["mask_px_max"], s["mask_px_max"])
        lines.append(f"{session:<40} {s['photos']:>5} {s['box_changed']:>5} {s['fallback_changed']:>5} "
                     f"{s['mask_changed']:>5} {s['mask_px_max']:>8} {s['photos_with_patches_moved']:>5} "
                     f"{s['patches_moved']:>6}")
    head = (f"{'session':<40} {'both':>5} {'box':>5} {'fb':>5} {'mask':>5} {'max px':>8} {'p.mvd':>5} "
            f"{'boxes':>6}")
    out = ["ML-8 P4 (R3): the re-crop of 2026-10-19 against ML-6 P8's crops (0675f40), photos both have",
           "both: photos in both; box / fb / mask: crop box, D18 fallback, full-frame mask (its sha256) changed;",
           "max px: the most pixels changed in one photo's crop mask (the mask training reads: filled and cropped,",
           f"compared in photo coordinates); p.mvd / boxes: photos whose {SERVING_PATCHES} D17 patch boxes (one fixed",
           "RNG key, as dataset._extract_photo draws them) move, and how many boxes", "", head, "-" * len(head), *lines, "-" * len(head),
           f"{'all':<40} {totals['photos']:>5} {totals['box_changed']:>5} {totals['fallback_changed']:>5} "
           f"{totals['mask_changed']:>5} {totals['mask_px_max']:>8} {totals['photos_with_patches_moved']:>5} "
           f"{totals['patches_moved']:>6}", "",
           f"photos only in the re-crop: {len(only_new)}", *(f"  {p}" for p in only_new),
           f"photos only in P8's crops: {len(only_old)}", *(f"  {p}" for p in only_old), "",
           "photos that changed (session, class, file, box changed, fallback changed, mask px, boxes moved):",
           *(f"  {r[0]}/{r[1]}/{r[2]}  box={r[3]} fallback={r[4]} px={r[5]} boxes={r[6]}" for r in changed_rows)]
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
