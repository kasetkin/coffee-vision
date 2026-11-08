"""Write data/ml5_labels/pass14 from pass13: the segmenter's labels in their canonical photos' geometry (ticket ML-19
D9, D19, D28, D47; plan P5). Labels are carried, not judged again; pass13 stays as recorded.

    python scripts/make_pass14.py [--check]
    python scripts/make_pass14.py --review-items      # labels/ml19/pass14_review_items.csv, from pass14
    python -m coffeecv.review_masks --items labels/ml19/pass14_review_items.csv --name ml19_pass14 --canonical

Every row keeps its key, the original's path and sha256 (D33). A row whose photo canonicalization turns (a JPEG or PNG
with an EXIF rotate tag) or caps (over 60 MP) gets its mask, include and exclude points, fence and brush strokes
mapped by that turn and cap (coffeecv.label_geometry), and its mask_sha256 recomputed; every other row and mask is
carried as it is. Checks, every run: each turned mask turned back is pass13's to the pixel (a capped one is compared
by its area instead), and each mask's size is its canonical photo's. pass14/turned.csv lists the rows that changed,
for the owner's review on the labelling page (D47). A photo pass13 names that has since moved (ML-16 D3 moved the
example photos to dataset/example_photos/) is found there by its sha256. --check writes nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from coffeecv import canonical as cn  # noqa: E402
from coffeecv import label_geometry as lg  # noqa: E402
from coffeecv.photo_io import DATASET_MAX_PIXELS  # noqa: E402
from coffeecv.segment_beans import mask_sha256  # noqa: E402

PASS13 = REPO / "data/ml5_labels/pass13"
PASS14 = REPO / "data/ml5_labels/pass14"
REVIEW_ITEMS = REPO / "labels/ml19/pass14_review_items.csv"
MOVED_TO = REPO / "dataset/example_photos"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_photo(row: dict) -> tuple[Path, bytes]:
    path = REPO / row["path"]
    candidates = [path] if path.exists() else sorted(p for p in MOVED_TO.iterdir() if cn.is_photo(p))
    for p in candidates:
        data = p.read_bytes()
        if sha256(data) == row["photo_sha256"]:
            return p, data
    raise SystemExit(f"{row['id']}: no photo with sha256 {row['photo_sha256'][:16]} at {row['path']} or in {MOVED_TO}")


def stored_size(data: bytes, name: str) -> tuple[int, int]:
    """The photo's width and height as its decoder stores them, before any turn, from its header."""
    fmt = cn.accepted_format(data, name)
    if fmt.decoder == "pillow_heif":
        import pillow_heif
        return pillow_heif.open_heif(__import__("io").BytesIO(data)).size
    if fmt.decoder == "imagecodecs.jpegxl":
        header = cn.read_header(data)
        return header.width, header.height
    if fmt.decoder == "rawpy":
        import io

        import rawpy
        with rawpy.imread(io.BytesIO(data)) as raw:   # rawpy's postprocess turns by the raw's flip: as decoded
            s = raw.sizes
            return (s.width, s.height) if s.flip in (0, 3) else (s.height, s.width)
    return cn._pillow_metadata(data, cn.pillow_plugin(fmt))[3]


def carry(row: dict, write: bool) -> tuple[dict, dict | None]:
    """pass14's row, and the turned.csv row if it changed."""
    photo, data = find_photo(row)
    orientation = cn.applied_orientation(data, photo.name)
    w, h = stored_size(data, photo.name)
    cw, ch = cn.capped_size(w, h)
    old_mask = REPO / row["mask"]
    new_mask = PASS14 / old_mask.relative_to(PASS13)
    mask = np.asarray(Image.open(old_mask)).astype(bool)
    if mask.shape != (h, w):
        raise SystemExit(f"{row['id']}: pass13's mask is {mask.shape[::-1]}, its photo {w} x {h}")
    out = {**row, "mask": str(new_mask.relative_to(REPO))}
    if orientation == 1 and (cw, ch) == (w, h):
        if write:
            new_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_mask, new_mask)
        return out, None
    turned = lg.canonical_mask(mask, orientation)
    up_w, up_h = (ch, cw) if orientation >= 5 else (cw, ch)
    if turned.shape != (up_h, up_w):
        raise SystemExit(f"{row['id']}: the turned mask is {turned.shape[::-1]}, its canonical photo {up_w} x {up_h}")
    back = lg.stored_mask(turned, orientation)
    if (cw, ch) == (w, h):
        if not np.array_equal(back, mask):
            raise SystemExit(f"{row['id']}: the turned mask turned back is not pass13's")
        check = "turned back: pass13's to the pixel"
    else:
        if abs(back.mean() - mask.mean()) > 0.002:
            raise SystemExit(f"{row['id']}: the capped mask's area {back.mean():.4f} is not pass13's {mask.mean():.4f}")
        check = f"capped {w}x{h} -> {cw}x{ch}: area {mask.mean():.4f} -> {back.mean():.4f}"
    scale = cw / w

    def points(field: str) -> str:
        return json.dumps([[lg.round4(v) for v in p] for p in lg.turn_points(json.loads(row[field]), orientation)]) \
            if row[field] else ""

    out.update(mask_sha256=mask_sha256(turned), include=points("include"), exclude=points("exclude"),
               fence=json.dumps([lg.round4(v) for v in lg.turn_fence(json.loads(row["fence"]), orientation)])
               if row["fence"] else "",
               strokes=json.dumps([{**s, "r": round(s["r"], 2), "path": [[lg.round4(v) for v in p] for p in s["path"]]}
                                   for s in lg.turn_strokes(json.loads(row["strokes"]), orientation, scale)])
               if row["strokes"] else "")
    if write:
        new_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(turned).convert("1").save(new_mask)    # as seg_review saves a mask
    return out, {"id": row["id"], "path": row["path"], "status": row["status"], "split": row["split"],
                 "orientation": orientation, "stored_wh": f"{w}x{h}", "canonical_wh": f"{up_w}x{up_h}", "check": check}


def review_items() -> None:
    """The rows pass14 turned or capped, as coffeecv.review_masks items over their canonical photos (D47): each
    item's path is its original's (found by sha256 if it moved), its mask pass14's."""
    with open(PASS14 / "labels.csv", newline="") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    with open(PASS14 / "turned.csv", newline="") as f:
        turned = [r["id"] for r in csv.DictReader(f)]
    REVIEW_ITEMS.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEW_ITEMS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "path", "mask", "mask_sha256", "photo_sha256"])
        w.writeheader()
        for i in turned:
            r = rows[i]
            photo, _ = find_photo(r)
            w.writerow({"item": i, "path": str(photo.relative_to(REPO)), "mask": r["mask"],
                        "mask_sha256": r["mask_sha256"], "photo_sha256": r["photo_sha256"]})
    print(f"{len(turned)} items -> {REVIEW_ITEMS.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="check every row, write nothing")
    ap.add_argument("--review-items", action="store_true", help="write the owner's review list from pass14, only")
    args = ap.parse_args()
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS
    if args.review_items:
        review_items()
        return
    with open(PASS13 / "labels.csv", newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    if not args.check and PASS14.exists():
        raise SystemExit(f"{PASS14} exists: remove it first (it is made whole from pass13)")
    out, turned = [], []
    for row in rows:
        new, change = carry(row, write=not args.check)
        out.append(new)
        if change:
            turned.append(change)
    print(f"pass14: {len(out)} rows, {len(turned)} turned or capped "
          f"({sum(t['status'] == 'accepted' for t in turned)} accepted, {sum(t['status'] == 'negative' for t in turned)}"
          " negative); every check passed")
    if args.check:
        return
    with open(PASS14 / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    with open(PASS14 / "turned.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(turned[0]))
        w.writeheader()
        w.writerows(turned)
    review_items()


if __name__ == "__main__":
    main()
