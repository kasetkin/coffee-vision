"""The 27 photos of labels/ml3/pool_exclude.csv as items for the owner's review page, on their XL0-v4 masks
(owner, 2026-10-24). They were declined on the masks of older segmenters (ft_s123, XL0-v3, XL0-v4's first crops):
the 8 "D18 fallback" rows were fallbacks then and are not under XL0-v4 (coffeecv.find_tiny_masks found none).

    python analysis/ml3/pool_exclude_rereview_items.py [--out outputs/pool_exclude_rereview]
    python -m coffeecv.review_masks --items outputs/pool_exclude_rereview/items.csv --name pool_exclude_rereview

Each item's mask is the segmenter's full-frame mask, as the live site draws it and as analysis/ml8/p4_review_items.py
made them: the photo through segcrop's segmenter (params.yaml seg_params, the stage's 4 threads, which change mask
bits), checked against the mask_sha256 the segcrop stage recorded. Items are numbered in the file's order; the old
reason stays out of the item name, so the review is blind to it. Writes only under --out (gitignored: the masks are
of DVC photos).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.segcrop_session import crop_photo  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, mask_sha256, seg_params  # noqa: E402
from mask_review import POOL_EXCLUDE  # noqa: E402

SEGCROPPED = REPO_ROOT / "data" / "segcropped"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "pool_exclude_rereview")
    args = ap.parse_args()
    torch.set_num_threads(4)
    seg = BeanSegmenter(seg_params(RunConfig.from_params_yaml()))
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, r in enumerate(csv.DictReader(POOL_EXCLUDE.read_text().splitlines()), 1):
        report = json.loads((SEGCROPPED / r["session"] / r["class_folder"] / "segcrop_report.json").read_text())
        rep = next(x for x in report if x["file"] == r["file"])
        photo = Path("dataset") / r["session"] / r["class_folder"] / r["file"]
        full = crop_photo(REPO_ROOT / photo, seg).photo_mask.astype(bool)
        if mask_sha256(full) != rep["mask_sha256"]:
            raise SystemExit(f"{photo}: the segmenter's mask is not the one segcrop recorded")
        item = f"{n:02d}__{r['session']}__{r['class_folder'].split('__')[0]}__{Path(r['file']).stem}"
        mask_path = args.out / "masks" / f"{item}.png"
        Image.fromarray(full.astype(np.uint8) * 255).save(mask_path)
        rows.append({"item": item, "path": str(photo), "mask": str(mask_path.relative_to(REPO_ROOT)),
                     "mask_sha256": rep["mask_sha256"], "session": r["session"], "fallback": int(rep["fallback"]),
                     "area_frac": rep["mask_area_frac"], "old_reason": r["reason"]})
        print(f"  {item}  area {rep['mask_area_frac']:.4f}", flush=True)
    with open(args.out / "items.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} items -> {args.out.relative_to(REPO_ROOT)}/items.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
