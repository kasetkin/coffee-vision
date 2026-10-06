"""Ticket ML-3 P4 pre-checks, run on dataset_new_ignored/ before its sessions move into dataset/.

    python analysis/ml3/premerge_check.py        # prints counts; exit 1 on a failed check

  - the set is the one fixed in P2: every (session, class folder, file) of labels/ml3/new_photos.csv is on
    disk and nothing else is, and each photo's bytes are the stripped ones (labels/ml3/strip_manifest.csv's
    sha256_after for that row's sha256_before), so nothing was added, lost or rewritten since the strip;
  - the camera EXIF matches the pool each session joins (D6), as new_photos.py checks it;
  - no new photo duplicates one under dataset/: no sha256 match, and no file-name stem match (the stems are
    capture timestamps, PXL_20260911_103514697), as checked for ood_positives on 2026-09-30.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from new_photos import NEW, OUT as NEW_PHOTOS, POOL_OF, REPO_ROOT, exif, pool_cameras  # noqa: E402

STRIP_MANIFEST = REPO_ROOT / "labels" / "ml3" / "strip_manifest.csv"
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".dng", ".webp"}


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    problems = []
    fixed = read_csv(NEW_PHOTOS)
    after_of = {r["sha256_before"]: r["sha256_after"] for r in read_csv(STRIP_MANIFEST)}
    on_disk = {p.relative_to(NEW).as_posix(): p for p in NEW.glob("*/class_*/*") if p.is_file()}
    want = {f"{r['session']}/{r['class_folder']}/{r['file']}": r for r in fixed}
    problems += [f"{k}: in new_photos.csv, not on disk" for k in sorted(want.keys() - on_disk.keys())]
    problems += [f"{k}: on disk, not in new_photos.csv" for k in sorted(on_disk.keys() - want.keys())]
    stray = sorted(p.relative_to(NEW).as_posix() for p in NEW.rglob("*") if p.is_file()
                   and p.parent != NEW and p.relative_to(NEW).as_posix() not in on_disk)
    problems += [f"{k}: a file outside class_*/ folders" for k in stray]

    new_sha = {}
    for k in sorted(want.keys() & on_disk.keys()):
        sha = new_sha[k] = sha256_file(on_disk[k])
        expected = after_of.get(want[k]["sha256"])
        if expected is None:
            problems.append(f"{k}: its pre-strip hash is not in strip_manifest.csv")
        elif sha != expected:
            problems.append(f"{k}: bytes are not the stripped ones")

    pools, meta = pool_cameras(), exif(list(on_disk.values()))
    for k, p in sorted(on_disk.items()):
        session = k.split("/", 1)[0]
        pool = POOL_OF.get(session.split("__", 1)[-1]) or POOL_OF.get(session)
        camera = (meta[p.resolve()].get("Make"), meta[p.resolve()].get("Model"))
        if pool is None or camera not in pools[pool]:
            problems.append(f"{k}: camera {camera} does not fit pool {pool}")

    old = [p for p in (REPO_ROOT / "dataset").rglob("*") if p.is_file() and p.suffix.lower() in PHOTO_EXT]
    old_sha = {sha256_file(p): p for p in old}
    old_stem = {p.stem.lower(): p for p in old}
    for k, p in sorted(on_disk.items()):
        if new_sha.get(k) in old_sha:
            problems.append(f"{k}: same bytes as {old_sha[new_sha[k]].relative_to(REPO_ROOT)}")
        if p.stem.lower() in old_stem:
            problems.append(f"{k}: same stem as {old_stem[p.stem.lower()].relative_to(REPO_ROOT)}")
    dup_stems = [s for s, n in Counter(p.stem.lower() for p in on_disk.values()).items() if n > 1]
    problems += [f"stem {s} appears more than once among the new photos" for s in dup_stems]

    counts = Counter(tuple(k.split("/")[:2]) for k in on_disk)
    for (s, c), n in sorted(counts.items()):
        print(f"{s:<22} {c:<28} {n:>4}")
    print(f"{'total':<51} {len(on_disk):>4}  (new_photos.csv: {len(fixed)}; compared with {len(old)} photos under dataset/)")
    for p in problems:
        print(f"FAIL {p}")
    print("OK" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
