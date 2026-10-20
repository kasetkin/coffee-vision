"""Ticket ML-8 P7: ADR 0002's paired print of the augmented and unaugmented heads (P6's two arms of each run,
equal total sample weight, C = 0.1), with the previous shipping fits beside them, unpaired.

    python analysis/ml8/p7_paired.py [--runs 270 271 272] [--before 267 268 269] [--out analysis/ml8/p7_paired.txt]

Per seed: val and test macro-F1 of both arms and the delta (augmented minus unaugmented), then their mean and
range; each arm's diagnostic val-picked C. Per class: each arm's F1 as the mean over seeds, the delta, and the
class's val and test photo counts per seed (distinct photos in predictions_<split>.csv; the arms share them).
Beside it, exp267-269 (ML-6 P8's shipping fits: four camera pools, 150/40/40 patches per class, no augmentation),
unpaired (ticket R1: new photos, one pool, a new budget, new crops). Read it with R2: the split is by single photo,
so near-duplicate shots of one tray can sit in train and test.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
ARMS = ("unaugmented", "augmented")
SPLITS = ("val", "test")


def run_dir(exp: int) -> Path:
    [d] = list(EXPERIMENTS.glob(f"exp{exp}__*"))
    return d


def photos(d: Path, split: str) -> dict[str, int]:
    """{class: distinct photos in the split}."""
    seen: dict[str, set] = {}
    with open(d / f"predictions_{split}.csv") as f:
        for r in csv.DictReader(f):
            seen.setdefault(r["true_label"], set()).add((r["folder_id"], r["photo"]))
    return {c: len(s) for c, s in seen.items()}


def seed_of(d: Path) -> int:
    return json.loads((d / "config.json").read_text())["seed"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, nargs="+", default=[270, 271, 272])
    ap.add_argument("--before", type=int, nargs="+", default=[267, 268, 269])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out: list[str] = []
    runs = [(e, run_dir(e)) for e in args.runs]
    metrics = {e: json.loads((d / "metrics.json").read_text()) for e, d in runs}
    classes = metrics[args.runs[0]]["class_ids"]
    for e in args.runs:
        if metrics[e]["class_ids"] != classes:
            raise SystemExit(f"exp{e} has another class list: {metrics[e]['class_ids']}")

    out += ["ML-8 P7: augmented (8 views, weight 1/9 per row) vs unaugmented head, one run each seed, C = 0.1",
            "", f"{'run':<8}{'seed':>5}  {'split':<5}{'unaug':>8}{'aug':>8}{'delta':>9}   val-picked C (unaug, aug)"]
    deltas = {s: [] for s in SPLITS}
    for e, d in runs:
        a = metrics[e]["arms"]
        for s in SPLITS:
            u, g = (a[arm]["splits"][s]["macro_f1"] for arm in ARMS)
            deltas[s].append(g - u)
            pick = (f"   {a['unaugmented']['diagnostic_val_selected_C']:g}, {a['augmented']['diagnostic_val_selected_C']:g}"
                    if s == "val" else "")
            out.append(f"exp{e:<5}{seed_of(d):>5}  {s:<5}{u:>8.4f}{g:>8.4f}{g - u:>+9.4f}{pick}")
    out.append("")
    for s in SPLITS:
        v = np.array(deltas[s])
        out.append(f"{s:<5} delta: mean {v.mean():+.4f}, range [{v.min():+.4f}, {v.max():+.4f}], "
                   f"{int((v > 0).sum())} of {len(v)} seeds positive")

    out += ["", "Per class, F1 as the mean over seeds (photos: per seed, val / test)", "",
            f"{'class':<11}{'val unaug':>10}{'aug':>8}{'delta':>8}{'test unaug':>12}{'aug':>8}{'delta':>8}   photos"]
    counts = {e: {s: photos(d, s) for s in SPLITS} for e, d in runs}
    for c in classes:
        row = f"{c:<11}"
        for s in SPLITS:
            u, g = (np.mean([metrics[e]["arms"][arm]["splits"][s]["per_class"][c]["f1"] for e in args.runs])
                    for arm in ARMS)
            row += f"{u:>10.4f}{g:>8.4f}{g - u:>+8.4f}" if s == "val" else f"{u:>12.4f}{g:>8.4f}{g - u:>+8.4f}"
        row += "   " + ", ".join(f"{counts[e]['val'].get(c, 0)}/{counts[e]['test'].get(c, 0)}" for e in args.runs)
        out.append(row)

    out += ["", "Beside it, unpaired (R1): the previous shipping fits", "",
            f"{'run':<8}{'seed':>5}{'val':>8}{'test':>8}   name"]
    for e in args.before:
        d = run_dir(e)
        m = json.loads((d / "metrics.json").read_text())
        out.append(f"exp{e:<5}{seed_of(d):>5}{m['splits']['val']['macro_f1']:>8.4f}"
                   f"{m['splits']['test']['macro_f1']:>8.4f}   {d.name.split('__', 1)[1]}")
    best = max(args.runs, key=lambda e: metrics[e]["arms"]["augmented"]["splits"]["val"]["macro_f1"])
    out += ["", f"augmented seed with the highest val macro-F1 (the one to ship, Q1): exp{best}, "
            f"seed {seed_of(run_dir(best))}"]
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
