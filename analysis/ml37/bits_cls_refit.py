"""ML-37 bits for the classifier, level 4 (features): every pool photo's training and test patches, per seed, per arm.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_cls_refit.py --arms u8 f16 \
        [--seeds 42 123 7] [--train-per-class 600] [--limit N] [--threads 8]

Not a recorded run and not fit_frozen_head: no experiments/ entry, nothing under data/ or models/ written. The split
and the boxes are the fit's: dataset.pooled_class_photos over data/segcropped/pool, split_photos_by_class per seed,
each split's per-class budget spread over its photos as MultiPhotoPatchDataset does (train --train-per-class, test
params.yaml's 160), and each photo's boxes drawn by _extract_photo's rule (bits_cls_common.Geometry, checked in
bits_cls_check_boxes.txt). Val is skipped: the head is fitted at the fixed C = 0.1 (fit_frozen_head.DEFAULT_C), so
val chooses nothing. Every photo is read once for all seeds, its float crop made (bits_cls_common.float_crop), and each
box's model input built per arm and embedded by dinov3_vitb16 (cls_mean) in float32. No augmented views: the heads
fitted from these features are the unaugmented arm of ADR 0025's recipe (see bits_cls_refit_eval.py).

A photo's features go to tmp/ml37/bits_cls/refit/<arms>/<photo stem>.npz (a rerun skips photos done). The next
two photos' inputs are prepared on --prep threads (one each) while the current one embeds on the remaining threads.
Log to stdout.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from analysis.ml37 import bits_cls_common as bc
from analysis.ml37.bits_cls_embed import embed
from coffeecv.backbones import build_backbone
from coffeecv.class_list import load_classes
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import pooled_class_photos, resolve_captures, split_photos_by_class

SPLITS = ("train", "test")


def jobs(seeds: list[int], budget: dict[str, int], frac: dict[str, float]) -> dict[str, list[tuple]]:
    """Photo name -> [(seed, split, class_idx, photo_idx, n_patches, class key)] for the splits in SPLITS."""
    classes = load_classes(REPO_ROOT / "dataset" / "classes.txt")
    captures = resolve_captures([REPO_ROOT / "data" / "segcropped" / "pool"])
    out = defaultdict(list)
    for ci, key in enumerate(classes.keys):
        pool, _ = pooled_class_photos(captures, classes.folders[key])
        for seed in seeds:
            split = split_photos_by_class(pool, seed, ci, frac)
            for s in SPLITS:
                base, extra = divmod(budget[s], len(split[s]))
                for pi, ph in enumerate(split[s]):
                    out[ph.name].append((seed, s, ci, pi, base + (1 if pi < extra else 0), key))
    return out


def prepare(name: str, todo: list[tuple], entry: dict, geo: bc.Geometry, arms: list[str]) -> dict:
    x, mask = bc.float_crop(entry, threads=1, rec=None)
    u8 = np.rint(x * 255).astype(np.uint8)
    pitch = None
    rows, inputs = [], {a: [] for a in arms}
    for seed, split, ci, pi, n, key in todo:
        if n == 0:
            continue
        boxes, pitch = geo.boxes(u8, mask, seed, ci, pi, split, n, pitch)
        for b, _, _ in boxes:
            rows.append((seed, SPLITS.index(split), ci, b.y0, b.y1, b.x0, b.x1))
            px = bc.cut(x, b)
            for a in arms:
                inputs[a].append(bc.model_input(px, a))
    return {"name": name, "rows": np.array(rows, np.int64).reshape(-1, 7), "inputs": inputs, "pitch": pitch}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, choices=bc.ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--train-per-class", type=int, default=None, help="default: params.yaml's")
    ap.add_argument("--limit", type=int, default=None, help="only the first N photos (a toy run)")
    ap.add_argument("--threads", type=int, default=bc.THREADS)
    ap.add_argument("--prep", type=int, default=2, help="threads preparing the next photos' inputs")
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads - a.prep))
    geo = bc.Geometry.from_params()
    c = geo.cfg
    budget = {"train": a.train_per_class or c.train_patches_per_class, "test": c.test_patches_per_class}
    frac = {"train": c.train_photo_frac, "val": c.val_photo_frac, "test": c.test_photo_frac}
    work = jobs(a.seeds, budget, frac)
    entries = bc.pool_entries()
    out_dir = bc.SCRATCH / "refit" / "_".join(a.arms)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in sorted(work) if not (out_dir / (n.removesuffix(".jxl") + ".npz")).exists()][:a.limit]
    total = sum(len(work[n]) for n in names)
    print(f"{len(names)} photos to do ({len(work)} in the pool), budget {budget}, arms {a.arms}, seeds {a.seeds}, "
          f"{total} (photo, seed) jobs", flush=True)
    bb = build_backbone("dinov3_vitb16")
    t0, done_patches = time.time(), 0
    with ThreadPoolExecutor(a.prep) as ex:
        ahead = {j: ex.submit(prepare, names[j], work[names[j]], entries[names[j]], geo, a.arms)
                 for j in range(min(a.prep, len(names)))}
        for i, name in enumerate(names):
            p = ahead.pop(i).result()
            j = i + a.prep
            if j < len(names):
                ahead[j] = ex.submit(prepare, names[j], work[names[j]], entries[names[j]], geo, a.arms)
            n = len(p["rows"])
            feats = {}
            if n:
                f = embed(bb, [v for arm in a.arms for v in p["inputs"][arm]]).reshape(len(a.arms), n, -1)
                feats = {arm: f[j] for j, arm in enumerate(a.arms)}
            np.savez(out_dir / (name.removesuffix(".jxl") + ".npz"), rows=p["rows"], pitch=p["pitch"], **feats)
            p = None
            done_patches += n
            if i % 10 == 0 or i + 1 == len(names):
                el = time.time() - t0
                print(f"[{time.strftime('%H:%M:%S')}] {i + 1}/{len(names)} photos, {done_patches} patches, "
                      f"{el / 60:.1f} min, {el / max(1, done_patches) / len(a.arms):.3f} s/embedding, "
                      f"rss {bc.rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    main()
