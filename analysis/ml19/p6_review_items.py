"""Ticket ML-19 P6: the photos the owner judges before ML-16 trains (docs/agents/experiments.md: a photo new to the
crops is judged first), as items for coffeecv.review_masks, as ML-8 P4 did (analysis/ml8/p4_review_items.py).

    python analysis/ml19/p6_review_items.py [--out outputs/ml19_p6/review]
    python -m coffeecv.review_masks --items outputs/ml19_p6/review/items.csv --name ml19_p6_new_crops --canonical

The items, from every segcrop session's segcrop_report.json:
  - new: a photo whose original (by sha256) no session's crops held before P6's re-crop. The crops before are the
    segcrop outputs of the dvc.lock at BEFORE, read from the DVC cache; a renamed photo is the same file, so it is
    not new. A crop before is identified by its original's path then: the file at that path today, or, for a photo
    renamed since, labels/ml3/strip_manifest.csv's sha256_after for that path (or RENAMED_UNRECORDED's new name).
  - fallback: a D18 fallback today (its mask too small to crop by).
  - the P7 test photos wrong on every patch (analysis/ml19/p7_baseline.md).
Photos already in labels/ml3/pool_exclude.csv are left out. Each item's mask is the segmenter's full-frame mask on the
canonical photo, as the segcrop stage made it (params.yaml seg_params, the stage's 4 threads, which change mask
bits), checked against the mask_sha256 the stage recorded; its path is the original's, shown canonical by
--canonical. Writes only under --out (gitignored: the masks are of DVC photos). Declines go to
labels/ml3/pool_exclude.csv.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from coffeecv.config import RunConfig  # noqa: E402
from coffeecv.repo_files import sha256_file  # noqa: E402
from coffeecv.segcrop_session import crop_photo  # noqa: E402
from coffeecv.segment_beans import BeanSegmenter, mask_sha256, seg_params  # noqa: E402

BEFORE = "a0b03ca~1"   # the last dvc.lock before P6's re-crop
CACHE = REPO / ".dvc" / "cache" / "files" / "md5"
SEGCROPPED = REPO / "data" / "segcropped"
WRONG_IN_P7 = {"20261103_110934_Pixel_9_Pro_Ethiopia", "20261103_110947_Pixel_9_Pro_Ethiopia"}
# Crops before whose photo was renamed after the strip manifest was written, so neither records its sha256. Matched
# by the time in both names (a Pixel's PXL_ name is UTC, its new name +05:00) and checked by mask area: the same box
# and size, or (G8441, D28's orientation now applied) the box turned.
RENAMED_UNRECORDED = {
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PIC_20261026_122926.JPG": "20261026_122926_G8441_Brazil.JPG",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261026_054309193.jpg":
        "20261026_104309_Pixel_9_Pro_Brazil.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261026_103148849.jpg":
        "20261026_103148_OnePlus_Brazil.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261026_103200996.jpg":
        "20261026_103200_OnePlus_Brazil.jpg",
    "dataset/random_date_raccoon/class_013__Brazil_SulDeMinas/PXL_20261026_103458326.jpg":
        "20261026_103458_OnePlus_Brazil.jpg",
}


def cached(md5: str) -> bytes:
    return (CACHE / md5[:2] / md5[2:]).read_bytes()


def reports_before() -> list[tuple[str, str, str]]:
    """(session, class folder, original's file name) of every crop the segcrop stages held at BEFORE."""
    lock = yaml.safe_load(subprocess.run(["git", "show", f"{BEFORE}:dvc.lock"], cwd=REPO, check=True,
                                         capture_output=True, text=True).stdout)["stages"]
    out = []
    for name, stage in lock.items():
        if not name.startswith("segcrop@"):
            continue
        session = name.split("@", 1)[1]
        for entry in json.loads(cached(stage["outs"][0]["md5"])):
            rel = Path(entry["relpath"])
            if rel.name == "segcrop_report.json":
                out += [(session, rel.parent.name, r["file"]) for r in json.loads(cached(entry["md5"]))]
    return out


def sha_before(rows: list[tuple[str, str, str]]) -> set[str]:
    stripped = {}
    with open(REPO / "labels" / "ml3" / "strip_manifest.csv", newline="") as f:
        for r in csv.DictReader(line for line in f if not line.startswith("#")):
            stripped[r["path"]] = r["sha256_after"]
    shas, missing = set(), []
    for session, cls, file in rows:
        rel = f"dataset/{session}/{cls}/{file}"
        if (REPO / rel).is_file():
            shas.add(sha256_file(REPO / rel))
        elif rel in stripped:
            shas.add(stripped[rel])
        elif rel in RENAMED_UNRECORDED:
            shas.add(sha256_file(REPO / Path(rel).parent / RENAMED_UNRECORDED[rel]))
        else:
            missing.append(rel)
    if missing:   # a photo moved out of the sets (dataset/example_photos) is not in the pool today either
        print(f"{len(missing)} crops before with no file today and no strip-manifest row: {missing}", file=sys.stderr)
    return shas


def excluded() -> set[tuple[str, str, str]]:
    with open(REPO / "labels" / "ml3" / "pool_exclude.csv", newline="") as f:
        return {(r["session"], r["class_folder"], r["file"]) for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "ml19_p6" / "review")
    args = ap.parse_args()
    before = sha_before(reports_before())
    skip = excluded()
    picked = []
    for rep in sorted(SEGCROPPED.glob("*/class_*/segcrop_report.json")):
        session, cls = rep.parent.parent.name, rep.parent.name
        if session == "pool":
            continue
        for r in json.loads(rep.read_text()):
            if (session, cls, r["file"]) in skip:
                continue
            photo = Path("dataset") / session / cls / r["file"]
            why = [w for w, hit in (("new", sha256_file(REPO / photo) not in before), ("fallback", r["fallback"]),
                                    ("wrong in P7", Path(r["file"]).stem in WRONG_IN_P7)) if hit]
            if why:
                picked.append((photo, r, why))
    print(f"{len(picked)} items: {sum('new' in w for *_, w in picked)} new, "
          f"{sum('fallback' in w for *_, w in picked)} fallbacks", flush=True)

    torch.set_num_threads(4)
    seg = BeanSegmenter(seg_params(RunConfig.from_params_yaml()))
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for n, (photo, r, why) in enumerate(picked, 1):
        full = crop_photo(REPO / r["canonical"], seg).photo_mask.astype(bool)
        if mask_sha256(full) != r["mask_sha256"]:
            raise SystemExit(f"{photo}: the segmenter's mask is not the one segcrop recorded")
        item = f"{n:02d}__{photo.parts[1]}__{photo.parent.name.split('__')[0]}__{photo.stem}"
        mask = args.out / "masks" / f"{item}.png"
        Image.fromarray(full.astype(np.uint8) * 255).save(mask)
        rows.append({"item": item, "path": str(photo), "mask": str(mask.relative_to(REPO)),
                     "mask_sha256": r["mask_sha256"], "why": " + ".join(why), "fallback": r["fallback"],
                     "area_frac": round(r["mask_area_frac"], 4)})
        print(f"  {item}: {' + '.join(why)}", flush=True)
    with open(args.out / "items.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} items -> {(args.out / 'items.csv').relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
