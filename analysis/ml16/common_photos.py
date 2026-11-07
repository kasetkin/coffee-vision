"""Ticket ML-16: exp280-282 (xl0_v7r_s123's crops, the 87 new photos) against exp279/277/278 (xl0_v6_s42's crops),
read beside coffeecv.compare_experiments.

The two arms' test sets are not the same photos. A class's split is a seeded shuffle of its photos sorted by name
(coffeecv.dataset.split_photos_by_class), so it is redrawn whenever the class gains, loses or renames a photo: D10's
87 photos, the owner's 99 renames of 2026-11-03 (80c1674) and 31 Colombia renames (fd490ce), and the exclusions.
This prints, per seed, which classes kept identical test photos, and the macro and per-class test F1 of both arms
on the photos in both (matched by name without the suffix: .jpg crops then, .jxl now). Those photos' patches still
differ, since the crops do.

    python analysis/ml16/common_photos.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coffeecv.config import REPO_ROOT  # noqa: E402

PAIRS = ((42, 280, 279), (123, 281, 277), (7, 282, 278))  # seed, new, baseline


def predictions(exp: int) -> list[dict]:
    (path,) = (REPO_ROOT / "experiments").glob(f"exp{exp}__*/predictions_test.csv")
    with path.open() as f:
        return list(csv.DictReader(f))


def key(row: dict) -> tuple[str, str]:
    return row["capture"], Path(row["photo"]).stem


def main() -> None:
    per_class = {}
    labels = None
    for seed, new, base in PAIRS:
        rows = {"new": predictions(new), "base": predictions(base)}
        photos = {arm: defaultdict(set) for arm in rows}
        for arm, rs in rows.items():
            for r in rs:
                photos[arm][r["true_label"]].add(key(r))
        labels = labels or list(dict.fromkeys(r["true_label"] for r in rows["base"]))
        same = [c for c in labels if photos["new"][c] == photos["base"][c]]
        common = set().union(*(photos["new"][c] & photos["base"][c] for c in labels))
        print(f"seed {seed} (exp{new} vs exp{base}): identical test photos in {', '.join(same)}")
        f1 = {}
        for arm, rs in rows.items():
            kept = [r for r in rs if key(r) in common]
            y, p = [r["true_label"] for r in kept], [r["pred_label"] for r in kept]
            f1[arm] = f1_score(y, p, labels=labels, average=None)
            print(f"  {arm:4s} on the common photos: macro-F1 {f1[arm].mean():.4f}, {len(kept)} patches")
        per_class[seed] = f1
    print("\nper-class test F1 on the common photos, baseline -> new")
    print(f"  {'class':10s}" + "".join(f"{'s' + str(s):>16s}" for s, _, _ in PAIRS))
    for i, c in enumerate(labels):
        cells = "".join(f"{per_class[s]['base'][i]:>9.3f}->{per_class[s]['new'][i]:.3f}" for s, _, _ in PAIRS)
        print(f"  {c:10s}{cells}")


if __name__ == "__main__":
    main()
