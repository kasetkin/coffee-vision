"""The photos new to a session's crops as items for the owner's review page (owner, 2026-10-25): those in its DVC
listing at HEAD and not at --since, a commit before they were added.

    python analysis/ml3/new_crops_items.py --session random_date_raccoon --since 26ccde6 --name new_crops_2026-10-25
    python -m coffeecv.review_masks --items outputs/new_crops_2026-10-25/items.csv --name new_crops_2026-10-25

Each item's mask is the segmenter's full-frame mask, as the live site draws it and as
analysis/ml3/pool_exclude_rereview_items.py made them: the photo through segcrop's segmenter (params.yaml
seg_params, the stage's 4 threads, which change mask bits), checked against the mask_sha256 the segcrop stage
recorded, so run it after `dvc repro -s segcrop@<session>`. Writes only under outputs/<name> (gitignored: the
masks are of DVC photos).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.segcrop_session import crop_photo  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, mask_sha256, seg_params  # noqa: E402

SEGCROPPED = REPO_ROOT / "data" / "segcropped"
CACHE = REPO_ROOT / ".dvc" / "cache" / "files" / "md5"


def listing(session: str, rev: str) -> set[str]:
    """The relpaths in dataset/<session>.dvc's .dir as of `rev`, read from the local DVC cache."""
    dvc = subprocess.run(["git", "show", f"{rev}:dataset/{session}.dvc"], cwd=REPO_ROOT, capture_output=True,
                         text=True, check=True).stdout
    h = yaml.safe_load(dvc)["outs"][0]["md5"]
    return {e["relpath"] for e in json.loads((CACHE / h[:2] / h[2:]).read_text())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--since", required=True, help="a commit before the new photos were added")
    ap.add_argument("--name", required=True, help="the review's name: outputs/<name>/items.csv")
    args = ap.parse_args()
    new = sorted(listing(args.session, "HEAD") - listing(args.session, args.since))
    if not new:
        sys.exit(f"no photos in dataset/{args.session} at HEAD that were not there at {args.since}")
    out = REPO_ROOT / "outputs" / args.name
    torch.set_num_threads(4)
    seg = BeanSegmenter(seg_params(RunConfig.from_params_yaml()))
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, rel in enumerate(new, 1):
        class_folder, file = rel.split("/")
        report = json.loads((SEGCROPPED / args.session / class_folder / "segcrop_report.json").read_text())
        rep = next((x for x in report if x["file"] == file), None)
        if rep is None:
            sys.exit(f"{rel}: not in the segcrop report; run dvc repro -s segcrop@{args.session} first")
        photo = Path("dataset") / args.session / rel
        full = crop_photo(REPO_ROOT / photo, seg).photo_mask.astype(bool)
        if mask_sha256(full) != rep["mask_sha256"]:
            sys.exit(f"{photo}: the segmenter's mask is not the one segcrop recorded")
        item = f"{n:02d}__{args.session}__{class_folder.split('__')[0]}__{Path(file).stem}"
        mask_path = out / "masks" / f"{item}.png"
        Image.fromarray(full.astype(np.uint8) * 255).save(mask_path)
        rows.append({"item": item, "path": str(photo), "mask": str(mask_path.relative_to(REPO_ROOT)),
                     "mask_sha256": rep["mask_sha256"], "session": args.session, "fallback": int(rep["fallback"]),
                     "area_frac": rep["mask_area_frac"]})
        print(f"  {item}  area {rep['mask_area_frac']:.4f}{'  D18 fallback' if rep['fallback'] else ''}", flush=True)
    with open(out / "items.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} items -> {(out / 'items.csv').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
