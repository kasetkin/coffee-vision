"""The canonical store, one dataset folder at a time (ticket ML-19 D18, D33, D49; ADR 0033): every photo under
dataset/<folder> canonicalized to data/canonical/<folder>/.../<its file name>.jxl. The originals stay untouched. No
manifest: a record keys a photo by its original's path and sha256, and coffeecv.canonical.canonical_path maps that path
to its canonical photo.

    python -m coffeecv.canonicalize_folder <folder> [--jobs 4] [--threads 2] [--full]

Run as the canonicalize@<folder> stages of dvc.yaml, on the store's machine only: libjxl's output depends on the CPU
(D51), and this machine's and the front's agree. A photo that cannot be read fails the stage, by name. A file whose
extension is in no accepted format (coffeecv/data/accepted_formats.csv: a manifest, a README, a TIFF) is reported and
left out, as intake reports one (ticket OPS-18 D16; ADR 0034). coffeecv.canonical refuses any imagecodecs but its own: a new version is
a new store (ADR 0033).

Since ticket ML-26 the stage processes only the photos new or changed since its last run, when nothing else changed
(coffeecv.incremental reads dvc.lock): a canonical photo on disk as dvc.lock recorded it stays, and one whose original
is gone is deleted. Anything else, or --full, rebuilds the folder's store from nothing. The output is the same bytes
either way: a photo's canonical photo depends on that photo alone.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from coffeecv import canonical as cn
from coffeecv import incremental as inc

# coffeecv.photo_io.DATASET_MAX_PIXELS (tests/test_canonicalize_folder.py holds them equal), not imported: photo_io
# registers pillow-avif-plugin with Pillow, and the front, which decodes with Pillow's own AVIF, must decode as the
# store does (D35).
DATASET_MAX_PIXELS = 2**28


def photos_and_others(folder: Path) -> tuple[list[Path], list[Path]]:
    """(the files under `folder` in an accepted format, by extension; every other file), each sorted."""
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    return [p for p in files if cn.is_photo(p)], [p for p in files if not cn.is_photo(p)]


def canonicalize_one(job: tuple[Path, Path, int]) -> tuple[str, int, int, float, str]:
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


def plan(folder: str, found: list[Path], dataset: Path, store: Path, full: bool) -> tuple[list[Path], int, int, str]:
    """(the photos to canonicalize, how many stay, how many stale canonical photos were deleted, how). Incremental only
    for a repository's own store (<root>/dataset into <root>/data/canonical, with a dvc.lock); the rest of the time,
    and on --full, the folder's store is emptied and every photo is done."""
    root, out_dir = dataset.resolve().parent, store / folder
    record, why = None, "--full"
    if not full:
        record, why = ((None, "not a repository's store")
                       if store.resolve() != root / "data" / "canonical" or not (root / "dvc.lock").exists()
                       else inc.recorded(f"canonicalize@{folder}", f"dataset/{folder}", f"data/canonical/{folder}", root))
    if record is None:
        shutil.rmtree(out_dir, ignore_errors=True)
        return found, 0, 0, f"full rebuild ({why})"
    want = {cn.canonical_path(p, dataset, store).relative_to(out_dir).as_posix(): p for p in found}
    keep = {rel for rel, p in want.items()
            if p.relative_to(dataset / folder).as_posix() in record.inputs and inc.keeps(record, out_dir, rel)}
    removed = 0
    for f in sorted(out_dir.rglob("*")) if out_dir.is_dir() else []:
        if f.is_file() and f.relative_to(out_dir).as_posix() not in want:
            f.unlink()
            removed += 1
    return [p for rel, p in want.items() if rel not in keep], len(keep), removed, "incremental"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", help="a folder under dataset/, e.g. segmenter_positives")
    ap.add_argument("--jobs", type=int, default=4, help="photos at a time (default 4)")
    ap.add_argument("--threads", type=int, default=cn.THREADS, help="libjxl threads per photo (no effect on output)")
    ap.add_argument("--dataset", type=Path, default=cn.DATASET)
    ap.add_argument("--store", type=Path, default=cn.STORE)
    ap.add_argument("--full", action="store_true", help="rebuild the folder's store from nothing (ticket ML-26)")
    args = ap.parse_args()
    source = args.dataset / args.folder
    if not source.is_dir():
        raise SystemExit(f"{source} is not a folder")
    found, others = photos_and_others(source)
    if others:
        print(f"{len(others)} files not in an accepted format, left alone:\n  "
              + "\n  ".join(p.relative_to(args.dataset).as_posix() for p in others), flush=True)
    found, kept, removed, how = plan(args.folder, found, args.dataset, args.store, args.full)
    print(f"{args.folder}: {how}: {len(found)} photos to do, {kept} kept, {removed} stale removed", flush=True)
    todo = [(p, cn.canonical_path(p, args.dataset, args.store), args.threads) for p in found]
    t0, done, failed, size_in, size_out = time.perf_counter(), 0, [], 0, 0
    # spawn, not fork: rawpy's OpenMP can deadlock in a forked worker (rawpy's own warning, seen 2026-11-03)
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for name, n_in, n_out, seconds, error in pool.map(canonicalize_one, todo, chunksize=1):
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
