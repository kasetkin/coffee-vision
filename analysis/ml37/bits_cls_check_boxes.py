"""ML-37 bits for the classifier: check the box replication against dataset._extract_photo, and time each step.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/bits_cls_check_boxes.py [--n 10]

On --n photos of exp290's test split (the live head's, seed 123), spread over the classes: the stored 8-bit crop
through MultiPhotoPatchDataset._extract_photo itself (a stub instance carrying the fit's attributes), against
bits_cls_common's Geometry.boxes on round(float crop * 255). Reports: the share of samples where round(float * 255)
equals the stored crop, the pitch of each, whether the boxes are identical, the largest difference between the today
arm's model input and the one the stored crop's own uint8 patches give (eval_patch), and the time of each step at 8
threads.
Writes analysis/ml37/bits_cls_check_boxes.txt.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from analysis.ml37 import bits_cls_common as bc
from coffeecv import crop_io
from coffeecv.backbones import build_backbone
from coffeecv.bean_scale import pitch_kwargs
from coffeecv.class_list import load_classes
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import MultiPhotoPatchDataset, bean_share_rule

EXP290 = REPO_ROOT / "experiments" / "exp290__dino3b16_frozen_clsmean_segcrop_country_aug8_s123_ml29gm"
OUT = Path(__file__).with_suffix(".txt")


def exp290_test() -> list[dict]:
    """exp290's test photos: class, class_idx, photo_idx (rank among the class's test photos), patches."""
    classes = load_classes(REPO_ROOT / "dataset" / "classes.txt")
    rows = list(csv.DictReader(open(EXP290 / "predictions_test.csv")))
    per = defaultdict(lambda: defaultdict(int))
    for r in rows:
        per[r["true_label"]][r["photo"]] += 1
    out = []
    for ci, key in enumerate(classes.keys):
        for pi, name in enumerate(sorted(per[key])):
            out.append({"class": key, "class_idx": ci, "photo_idx": pi, "photo": name, "n": per[key][name]})
    return out


def stub_dataset(cfg) -> MultiPhotoPatchDataset:
    ds = object.__new__(MultiPhotoPatchDataset)
    ds.split, ds.patch_beans = "test", (cfg.patch_beans_min, cfg.patch_beans_max)
    ds.pitch_geometry, ds.bean_share_rule = pitch_kwargs(cfg), bean_share_rule(cfg)
    ds.pitch_by_photo, ds.fallback_photos, ds.n_below_share, ds.n_clamped = {}, [], 0, 0
    ds.rotation_jitter_degrees, ds.patch_store_size, ds.patch_scale_frac = 0.0, cfg.patch_store_size, None
    ds._patches, ds._meta = [], []
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    torch.set_num_threads(bc.THREADS)
    geo = bc.Geometry.from_params()
    entries = bc.pool_entries()
    test = exp290_test()
    pick = [test[i] for i in np.linspace(0, len(test) - 1, a.n).astype(int)]
    lines = [f"{len(pick)} photos of exp290's test split (seed 123); {bc.THREADS} threads"]
    bb = build_backbone("dinov3_vitb16")
    t_embed = []
    for t in pick:
        e = entries[t["photo"]]
        t0 = time.perf_counter()
        x, mask = bc.float_crop(e)
        t_float = time.perf_counter() - t0
        t0 = time.perf_counter()
        stored = crop_io.load_crop(e["pool_path"])
        t_stored = time.perf_counter() - t0
        u8 = np.rint(x * 255).astype(np.uint8)
        equal = float(np.mean(u8 == stored))
        ds = stub_dataset(geo.cfg)
        ds._extract_photo(e["pool_path"], t["n"], 123, t["class_idx"], t["photo_idx"], t["class"], "", "pool",
                          geo.cfg.patch_crop_size, geo.cfg.safety_margin)
        t0 = time.perf_counter()
        mine, pitch = geo.boxes(u8, mask, 123, t["class_idx"], t["photo_idx"], "test", t["n"])
        t_boxes = time.perf_counter() - t0
        same = [m.box for m in ds._meta] == [b for b, _, _ in mine]
        t0 = time.perf_counter()
        inputs = [bc.model_input(bc.cut(x, b), arm) for b, _, _ in mine for arm in bc.ARMS]
        t_inputs = (time.perf_counter() - t0) / len(inputs)
        t0 = time.perf_counter()
        with torch.no_grad():
            bb.features(torch.from_numpy(np.stack(inputs[:32])))
        t_embed.append((time.perf_counter() - t0) / min(32, len(inputs)))
        # today's arm on the stored crop's own patches (_extract_photo's uint8 448 px) equals the arm on u8?
        today_diff = max(float(np.abs((bc.model_input(bc.cut(x, b), "today")
                                       - bc.eval_patch(bc.Image.fromarray(p), bc.RESIZE)) * bc.STD * 255).max())
                         for (b, _, _), p in zip(mine, ds._patches))
        lines.append(f"{t['photo'][:60]:60s} {x.shape[1]}x{x.shape[0]} fallback={mask is None} "
                     f"u8==stored {equal:.7f} pitch stored {ds.pitch_by_photo[e['pool_path'].name]:.3f} "
                     f"float {pitch:.3f} boxes identical {same} ({len(mine)}) today-arm input vs the stored crop's "
                     f"patches, max |diff| {today_diff:.2f} levels | float crop {t_float:.2f} s, stored decode {t_stored:.2f} s, boxes "
                     f"{t_boxes:.2f} s, input {t_inputs * 1000:.0f} ms/arm-patch")
        print(lines[-1], flush=True)
    lines.append(f"embed: {np.mean(t_embed):.3f} s per image (batches of up to 32, float32, {bc.THREADS} threads); "
                 f"peak RSS {bc.rss_mb():.0f} MB")
    OUT.write_text("\n".join(lines) + "\n")
    print(lines[-1])


if __name__ == "__main__":
    main()
