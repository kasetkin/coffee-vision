"""ML-19 P7 (D14, D48): the live classifier on canonical crops against its own test predictions, on the photos the two
test splits share.

    python analysis/ml19/p7/paired.py > analysis/ml19/p7/paired.txt

Reads the source experiment's predictions_test.csv (the originals' crops, JPEG q95) and baseline.py's (the canonical
photos' crops, lossless JXL). A photo is paired by its class and its original's stem. On the shared photos, each side
is scored by patch (macro F1, accuracy) and by photo (the photo's mean probability's top class); the photos whose
vote changed are listed. The rest of each split is counted, not scored.
"""
from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def read(path: Path) -> dict[tuple[str, str], list[dict]]:
    """Patches by (class, the original's stem)."""
    by_photo = collections.defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_photo[(row["true_label"], row["photo"].rsplit("__cropped.", 1)[0])].append(row)
    return by_photo


def scores(by_photo: dict, photos: list, classes: list[str]) -> dict:
    rows = [r for p in photos for r in by_photo[p]]
    y, pred = [r["true_label"] for r in rows], [r["pred_label"] for r in rows]
    votes = {p: vote(by_photo[p], classes) for p in photos}
    return {"patches": len(rows), "macro_f1": f1_score(y, pred, labels=classes, average="macro", zero_division=0),
            "accuracy": accuracy_score(y, pred),
            "photo_accuracy": float(np.mean([votes[p] == p[0] for p in photos])), "votes": votes}


def vote(rows: list[dict], classes: list[str]) -> str:
    probs = np.array([[float(r[f"p_{k}"]) for k in classes] for r in rows]).mean(0)
    return classes[int(probs.argmax())]


def main() -> None:
    exp = json.loads((HERE / "baseline.json").read_text())["source_experiment"]
    today = read(REPO / "experiments" / exp / "predictions_test.csv")
    canon = read(HERE / "predictions_test.csv")
    classes = json.loads((REPO / "experiments" / exp / "metrics.json").read_text())["class_ids"]
    shared = sorted(set(today) & set(canon))
    print(f"test photos: {len(today)} in {exp}, {len(canon)} on canonical crops, {len(shared)} shared; "
          f"{len(set(today) - set(canon))} only in the old split, {len(set(canon) - set(today))} only in the new")
    a, b = scores(today, shared, classes), scores(canon, shared, classes)
    print(f"\non the {len(shared)} shared photos   {'originals':>10} {'canonical':>10}")
    print(f"patches                     {a['patches']:10d} {b['patches']:10d}")
    for k in ("macro_f1", "accuracy", "photo_accuracy"):
        print(f"{k:27} {a[k]:10.4f} {b[k]:10.4f}")
    print("\nper class, photo votes right (originals -> canonical):")
    for c in classes:
        ps = [p for p in shared if p[0] == c]
        print(f"  {c:10} {sum(a['votes'][p] == c for p in ps):3d} -> {sum(b['votes'][p] == c for p in ps):3d} of {len(ps)}")
    changed = [p for p in shared if a["votes"][p] != b["votes"][p]]
    print(f"\nphotos whose vote changed: {len(changed)}")
    for p in changed:
        print(f"  {p[0]:10} {p[1]}: {a['votes'][p]} -> {b['votes'][p]}")
    only = [p for p in canon if p not in today]
    if only:
        s = scores(canon, only, classes)
        print(f"\nthe {len(only)} photos only in the new split: patch accuracy {s['accuracy']:.4f}, "
              f"photo accuracy {s['photo_accuracy']:.4f}")
        dated = re.compile(r"\d{8}_\d{6}_")   # random_date_raccoon's names: renamed since exp279, or new to the pool
        for label, ps in (("dated names", [p for p in only if dated.match(p[1])]),
                          ("the older names", [p for p in only if not dated.match(p[1])])):
            s = scores(canon, ps, classes)
            wrong = round(s["patches"] * (1 - s["accuracy"]))
            print(f"  {label}: {len(ps)} photos, patch accuracy {s['accuracy']:.4f} ({wrong} of {s['patches']} wrong), "
                  f"photo accuracy {s['photo_accuracy']:.4f}")


if __name__ == "__main__":
    main()
