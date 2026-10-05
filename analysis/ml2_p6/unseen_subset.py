"""ML-2 P6 (ticket D21): each head seed's val/test macro-F1 on the photos the segmenter never trained on.

The segmenter's lists come from their own split (seed 239), so 194-204 of each head seed's 288 val/test
photos are in seg_train or seg_val. This scores both P4 arms on the rest, next to the full split. The
archived predictions are per patch with no photo, so each run's split is rebuilt (boxes are drawn once,
from generators keyed on the photo, so the rebuild is the fitted one) and every row's true label is
checked against the rebuilt patch's class before it is used.

    python analysis/ml2_p6/unseen_subset.py --pairs 258:255 259:256 260:257 > analysis/ml2_p6/unseen_subset.txt

Needs data/segcropped (the VM), and data/cropped for the tray-heuristic arms (255-257): those pools were
retired in ticket ML-3 P2b, so run it at its own commit (dbb4e62).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from coffeecv.compare_experiments import _find
from coffeecv.config import REPO_ROOT
from coffeecv.fold_data import build_fold_datasets
from coffeecv.infer import config_for_checkpoint
from coffeecv.metrics import compute_split_metrics
from coffeecv.transforms import build_eval_transform

LISTS = REPO_ROOT / "labels/ml2/photo_lists.yaml"


def seen_photos() -> set[tuple[str, str]]:
    """(cam_* pool, photo stem) of every photo the segmenter trained or selected on."""
    lists = yaml.safe_load(LISTS.read_text())["lists"]
    seen = set()
    for name in ("seg_train", "seg_val"):
        for e in lists[name]:
            crop = Path(e["crop"])
            seen.add((crop.parts[-3], crop.name.split("__", 1)[0]))
    return seen


def split_scores(exp: str, split: str, seen: set) -> dict:
    exp_dir = _find(exp)
    cfg, _ = config_for_checkpoint(Path("unused.pt"), str(exp_dir / "config.json"))
    tf = build_eval_transform(cfg.patch_resize)
    ds = getattr(build_fold_datasets(cfg, tf, tf, only=(split,)), split)
    with open(exp_dir / f"predictions_{split}.csv") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(ds._meta):
        raise SystemExit(f"exp{exp} {split}: {len(rows)} predictions but {len(ds._meta)} rebuilt patches")
    for r, m in zip(rows, ds._meta):
        if r["true_label"] != m.class_id:
            raise SystemExit(f"exp{exp} {split}: rebuilt patch order does not match the predictions")
    ids = ds.class_ids
    y = np.array([ids.index(r["true_label"]) for r in rows])
    p = np.array([ids.index(r["pred_label"]) for r in rows])
    unseen = np.array([(m.capture, m.photo_name.split("__", 1)[0]) not in seen for m in ds._meta])
    photos = {(m.capture, m.photo_name) for m, u in zip(ds._meta, unseen) if u}
    out = {}
    for name, keep in (("full", np.ones_like(unseen)), ("unseen", unseen)):
        present = sorted(set(y[keep].tolist()))
        sm = compute_split_metrics(y[keep], p[keep], np.zeros(int(keep.sum())), ids, ds.class_labels,
                                   macro_labels=present)
        out[name] = {"macro_f1": sm["macro_f1"], "n_patches": int(keep.sum()), "n_classes": len(present)}
    out["unseen"]["n_photos"] = len(photos)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pairs", nargs="+", required=True, help="new:baseline experiment ids, one per seed")
    args = ap.parse_args()
    seen = seen_photos()
    result = {}
    print(f"{'seed':>5} {'split':>5} {'subset':>7} {'photos':>6} {'patches':>7}  {'base':>7} {'new':>7} {'Δ':>8}")
    for pair in args.pairs:
        new, base = pair.split(":")
        seed = json.loads((_find(new) / "config.json").read_text())["seed"]
        for split in ("val", "test"):
            n, b = split_scores(new, split, seen), split_scores(base, split, seen)
            result[f"s{seed}_{split}"] = {"new": n, "baseline": b}
            for subset in ("full", "unseen"):
                if n[subset]["n_patches"] != b[subset]["n_patches"]:
                    raise SystemExit(f"seed {seed} {split} {subset}: the arms hold different patch counts")
                photos = n[subset].get("n_photos", "")
                bv, nv = b[subset]["macro_f1"], n[subset]["macro_f1"]
                print(f"{seed:>5} {split:>5} {subset:>7} {photos:>6} {n[subset]['n_patches']:>7}  "
                      f"{bv:>7.4f} {nv:>7.4f} {nv - bv:>+8.4f}", flush=True)
    print("JSON " + json.dumps(result))


if __name__ == "__main__":
    main()
