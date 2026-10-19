"""Ticket ML-8 P4 (D11): the 20 photos new to the re-crop as items for the owner's review page,
coffeecv.review_masks, as ML-3 P5 reviewed its new sessions.

    python analysis/ml8/p4_review_items.py [--sheet outputs/ml8_p4] [--out outputs/ml8_p4/review_photo]
    python -m coffeecv.review_masks --items outputs/ml8_p4/review_photo/items.csv --name ml8_p4_new_crops

The items are the rows of p4_new_crops_sheet.py's index.csv. Each one's mask is the segmenter's full-frame mask, as
the live site draws it: the photo through segcrop's segmenter (params.yaml seg_params, the stage's 4 threads, which
change mask bits), checked against the mask_sha256 the segcrop stage recorded. Not the crop's mask placed back in
the photo: D4's crop cuts up to (1 - keep_frac) / 4 of the bean pixels from each side, so that mask has straight
edges along the crop box (the owner's catch, 2026-10-19). A D18 fallback's mask is the one too small to crop by.
Writes only under --out (gitignored: the masks are of DVC photos). Declines go to labels/ml3/pool_exclude.csv.
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
from p4_recrop_compare import NEW_ROOT, reports  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", type=Path, default=REPO_ROOT / "outputs" / "ml8_p4")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "ml8_p4" / "review_photo")
    args = ap.parse_args()
    torch.set_num_threads(4)
    seg = BeanSegmenter(seg_params(RunConfig.from_params_yaml()))
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for r in csv.DictReader((args.sheet / "index.csv").read_text().splitlines()):
        rep = reports(NEW_ROOT / r["session"])[(r["class_folder"], r["file"])]
        photo = Path("dataset") / r["session"] / r["class_folder"] / r["file"]
        full = crop_photo(REPO_ROOT / photo, seg).photo_mask.astype(bool)
        if mask_sha256(full) != rep["mask_sha256"]:
            raise SystemExit(f"{photo}: the segmenter's mask is not the one segcrop recorded")
        item = f"{int(r['n']):02d}__{r['session']}__{r['class_folder'].split('__')[0]}__{Path(r['file']).stem}"
        mask_path = args.out / "masks" / f"{item}.png"
        Image.fromarray(full.astype(np.uint8) * 255).save(mask_path)
        rows.append({"item": item, "path": str(photo), "mask": str(mask_path.relative_to(REPO_ROOT)),
                     "mask_sha256": mask_sha256(full), "session": r["session"], "fallback": r["fallback"],
                     "area_frac": r["mask_area_frac"]})
    with open(args.out / "items.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} items -> {args.out}/items.csv")
    print(json.dumps([r["item"] for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
