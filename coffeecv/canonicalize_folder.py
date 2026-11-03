"""The canonical store, one dataset folder at a time (ticket ML-19 D18, D33, D49; ADR 0033): every photo under
dataset/<folder> canonicalized to data/canonical/<folder>/.../<its file name>.jxl. The originals stay untouched. No
manifest: a record keys a photo by its original's path and sha256, and coffeecv.canonical.canonical_path maps that path
to its canonical photo.

    python -m coffeecv.canonicalize_folder <folder> [--jobs 4] [--threads 2]

Run as the canonicalize@<folder> stages of dvc.yaml, on the store's machine only: libjxl's output depends on the CPU
(D51), and this machine's and the front's agree. A photo that cannot be read fails the stage, by name. Files that are
not photos (manifests, READMEs) are left out. coffeecv.canonical refuses any imagecodecs but its own: a new version is
a new store (ADR 0033).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from coffeecv import canonical as cn
from coffeecv.photo_io import DATASET_MAX_PIXELS


def photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and cn.is_photo(p))


def _one(job: tuple[Path, Path, int]) -> tuple[str, int, int, float, str]:
    """Canonicalize one photo into place: (its path, original bytes, canonical bytes, seconds, error)."""
    original, target, threads = job
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = DATASET_MAX_PIXELS   # the owner's 200 MP photos, as the dataset tools read them
    t = time.perf_counter()
    try:
        data = original.read_bytes()
        canon = cn.encode(cn.decode(data, original.name), numthreads=threads)
    except cn.Unreadable as exc:
        return str(original), 0, 0, 0.0, str(exc)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(canon)
    os.replace(tmp, target)
    return str(original), len(data), len(canon), time.perf_counter() - t, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="a folder under dataset/, e.g. segmenter_positives")
    ap.add_argument("--jobs", type=int, default=4, help="photos at a time (default 4)")
    ap.add_argument("--threads", type=int, default=cn.THREADS, help="libjxl threads per photo (no effect on output)")
    ap.add_argument("--dataset", type=Path, default=cn.DATASET)
    ap.add_argument("--store", type=Path, default=cn.STORE)
    args = ap.parse_args()
    source = args.dataset / args.folder
    if not source.is_dir():
        raise SystemExit(f"{source} is not a folder")
    todo = [(p, cn.canonical_path(p, args.dataset, args.store), args.threads) for p in photos(source)]
    t0, done, failed, size_in, size_out = time.perf_counter(), 0, [], 0, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for name, n_in, n_out, seconds, error in pool.map(_one, todo, chunksize=1):
            if error:
                failed.append(f"{name}: {error}")
                continue
            done += 1
            size_in, size_out = size_in + n_in, size_out + n_out
            if done % 25 == 0:
                print(f"{done}/{len(todo)} photos, {time.perf_counter() - t0:.0f} s", flush=True)
    print(f"{args.folder}: {done} photos canonicalized, {size_in / 2**20:.0f} MB -> {size_out / 2**20:.0f} MB, "
          f"{time.perf_counter() - t0:.0f} s, {args.jobs} jobs")
    if failed:
        print(f"{len(failed)} photos could not be read:\n  " + "\n  ".join(failed), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
