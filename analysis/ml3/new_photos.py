"""Ticket ML-3 P2 (Q4): fix the set of new photos before anything is stripped, and check it.

    python analysis/ml3/new_photos.py            # writes labels/ml3/new_photos.csv, prints counts; exit 1 on a failed check

One row per photo in dataset_new_ignored/: session, class folder, file, camera (EXIF Make and Model),
capture time (DateTimeOriginal with its offset) and sha256 of the bytes as they are now, before the strip.
Paths are session-relative, so they stay valid when P4 moves the sessions into dataset/. P4 re-counts
against this file; the Q8 backup's test extraction lists every row.

Checks: every class folder is one of the fourteen in D15 (a new coffee needs its own classes.txt line
first); every session's photos come from the camera of the pool it joins (D6) -- the (Make, Model) pairs
of that pool's existing sessions, read from dvc.yaml's merge_segcam_* stages; random_date_raccoon joins
cam_pixel, so it must be the Pixel too.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW = REPO_ROOT / "dataset_new_ignored"
OUT = REPO_ROOT / "labels" / "ml3" / "new_photos.csv"
D15_IDS = {f"{i:03d}" for i in range(1, 15)}
POOL_OF = {"pixel": "cam_pixel", "sony": "cam_sony", "oneplus": "cam_oneplus", "random_date_raccoon": "cam_pixel"}
FIELDS = ["session", "class_folder", "file", "camera", "capture_time", "sha256"]


def exif(paths: list[Path]) -> dict[Path, dict]:
    out = subprocess.run(["exiftool", "-j", "-Make", "-Model", "-DateTimeOriginal", "-OffsetTimeOriginal",
                          *map(str, paths)], capture_output=True, text=True, check=True).stdout
    return {Path(d["SourceFile"]).resolve(): d for d in json.loads(out)}


def pool_cameras() -> dict[str, set[tuple[str, str]]]:
    """(Make, Model) pairs of every photo already in each segmenter pool."""
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())["stages"]
    cameras = {}
    for name, body in stages.items():
        if not name.startswith("merge_segcam_"):
            continue
        parts = body["cmd"].split()
        sessions = parts[parts.index("--sessions") + 1:]
        photos = [p for s in sessions for p in sorted((REPO_ROOT / "dataset" / s).glob("class_*/*")) if p.is_file()]
        cameras[parts[parts.index("--name") + 1]] = {(d.get("Make"), d.get("Model")) for d in exif(photos).values()}
    return cameras


def main() -> int:
    photos = sorted(p for p in NEW.glob("*/class_*/*") if p.is_file())
    meta = exif(photos)
    pools = pool_cameras()
    rows, problems = [], []
    for p in photos:
        session, folder = p.parts[-3], p.parts[-2]
        d = meta[p.resolve()]
        m = re.match(r"^class_(\d+)__", folder)
        if not m or m.group(1) not in D15_IDS:
            problems.append(f"{session}/{folder}: not one of D15's fourteen class folders")
        pool = POOL_OF.get(session.split("__", 1)[-1]) or POOL_OF.get(session)
        camera = (d.get("Make"), d.get("Model"))
        if pool is None:
            problems.append(f"{session}: no pool for this session (D6)")
        elif camera not in pools[pool]:
            problems.append(f"{session}/{folder}/{p.name}: camera {camera} is not {pool}'s {sorted(pools[pool])}")
        rows.append({"session": session, "class_folder": folder, "file": p.name,
                     "camera": " ".join(c for c in camera if c),
                     "capture_time": f"{d.get('DateTimeOriginal', '')}{d.get('OffsetTimeOriginal', '')}",
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        f.write("# ML-3 P2: the new photos, fixed by the owner on 2026-10-05 (Q4); written by "
                "analysis/ml3/new_photos.py. sha256 is before the strip.\n")
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    counts = Counter((r["session"], r["class_folder"]) for r in rows)
    for (s, c), n in sorted(counts.items()):
        print(f"{s:<22} {c:<28} {n:>4}")
    print(f"{'total':<51} {len(rows):>4}  -> {OUT.relative_to(REPO_ROOT)}")
    for p in problems:
        print(f"FAIL {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
