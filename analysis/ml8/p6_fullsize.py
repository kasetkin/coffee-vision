"""Ticket ML-8 P6: peak RAM and time of the augmented head fit at full size, before the three-seed run (R4).

    python analysis/ml8/p6_fullsize.py [--seed 42] [--views-subset 256] [--out analysis/ml8/p6_fullsize.json]

The smoke's 6 training patches per class cannot show either. This holds one full training split in memory (as
fit_one_seed does, D8's 600 patches per class), embeds every original, and the views of `--views-subset` patches
(the time per view, all in and in the backbone), then releases the patches and fits both heads at full size:
the unaugmented one on the originals, the augmented one on 1 + views rows per patch at weight 1/(1 + views), the
diagnostic C grid included. The views of the other patches are stood in for by their original plus Gaussian
noise matched, per dimension, to the measured views' spread, so the fits see the full row count and a
realistic spread. Peak RSS (VmHWM) is read after each step. Nothing is archived; the patches are never written.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from coffeecv import fit_frozen_head as ffh  # noqa: E402
from coffeecv.backbones import build_backbone  # noqa: E402
from coffeecv.fold_data import build_fold_datasets  # noqa: E402
from coffeecv.transforms import build_eval_transform  # noqa: E402


def rss_gb() -> dict[str, float]:
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines() if ":" in line)
    return {k: round(int(status[k].split()[0]) / 1e6, 2) for k in ("VmRSS", "VmHWM")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--views-subset", type=int, default=256)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rec: dict = {"seed": args.seed}

    def step(name: str, t0: float, **extra) -> None:
        rec[name] = {"seconds": round(time.perf_counter() - t0, 1), **rss_gb(), **extra}
        print(f"{name}: {rec[name]}", flush=True)

    bb = build_backbone("dinov3_vitb16")
    cfg = ffh.frozen_config(bb.name, args.seed, smoke=False)
    aug = ffh.HeadAugment.from_config(cfg)
    eval_tf = build_eval_transform(cfg.patch_resize)
    t = time.perf_counter()
    fold = build_fold_datasets(cfg, eval_tf, eval_tf, only=("train",))
    ds = fold.train
    step("train_split_built", t, patches=len(ds), patch_bytes=int(sum(p.nbytes for p in ds._patches)))
    t = time.perf_counter()
    X, y, ms = ffh.embed(ds, bb, "cls_mean")
    step("originals_embedded", t, backbone_ms_per_patch=round(ms, 1))
    sub = np.linspace(0, len(ds) - 1, min(args.views_subset, len(ds))).round().astype(int)
    t = time.perf_counter()
    V_sub, ms_view = ffh.embed_views(ds._patches, sub, bb, "cls_mean", aug, args.seed)
    wall = time.perf_counter() - t
    n_views = V_sub.shape[0] * V_sub.shape[1]
    step("subset_views_embedded", t, views=int(V_sub.shape[0] * V_sub.shape[1]),
         backbone_ms_per_view=round(ms_view, 1), all_in_ms_per_view=round(1000 * wall / n_views, 1))
    n = len(ds)
    val_test = (cfg.val_patches_per_class + cfg.test_patches_per_class) / cfg.train_patches_per_class * n
    est_h = (n * aug.views * wall / n_views + (n + val_test) * ms / 1000) / 3600   # views, then every original
    del fold, ds
    gc.collect()
    rec["patches_released"] = rss_gb()

    # Stand-in views for the patches not in the subset: the original plus noise of the measured per-dim spread.
    spread = (V_sub - X[sub][:, None, :]).reshape(-1, X.shape[1]).std(axis=0)
    rng = np.random.default_rng(0)
    views = (X[:, None, :] + rng.normal(0, 1, (n, aug.views, X.shape[1])).astype(np.float32) * spread).astype(np.float32)
    views[sub] = V_sub
    rec["views_cosine_to_original_measured"] = round(float(np.mean(
        np.sum(X[sub][:, None] * V_sub, axis=2) / np.linalg.norm(X[sub], axis=1)[:, None]
        / np.linalg.norm(V_sub, axis=2))), 4)
    Xv = {"val": X[: n // 4], "test": X[n // 4: n // 2]}          # stand-ins: only the fits' cost matters here
    yv = {"val": y[: n // 4], "test": y[n // 4: n // 2]}
    t = time.perf_counter()
    ffh.fit_arm(X, y, None, Xv, yv, ffh.DEFAULT_C, list(cfg_classes(cfg)), {c: c for c in cfg_classes(cfg)})
    step("unaugmented_fit_with_grid", t, rows=n)
    t = time.perf_counter()
    X_all, y_all, w = ffh.augmented_rows(X, views, y)
    del views
    gc.collect()
    ffh.fit_arm(X_all, y_all, w, Xv, yv, ffh.DEFAULT_C, list(cfg_classes(cfg)), {c: c for c in cfg_classes(cfg)})
    step("augmented_fit_with_grid", t, rows=int(len(X_all)))
    rec["estimated_hours_of_embedding_per_seed"] = round(est_h, 2)
    print(json.dumps(rec, indent=1))
    if args.out:
        args.out.write_text(json.dumps(rec, indent=1) + "\n")
    return 0


def cfg_classes(cfg) -> list[str]:
    from coffeecv.class_list import load_classes
    return list(load_classes(REPO_ROOT / cfg.classes_file).keys)


if __name__ == "__main__":
    sys.exit(main())
