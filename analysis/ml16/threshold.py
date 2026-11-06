"""Ticket ML-16 D9: the tiny-mask threshold (seg_min_area_frac) re-derived for xl0_v7r_s123, the segmenter D8 ships,
by ML-2 D18's rule as ML-5 D21 applies it: half the smallest non-empty test-positive mask the owner accepted. The owner
judges that one model's masks (accept or decline, coffeecv.review_masks --canonical), not a blind pair: D8 chose the
model on its validation scores.

    python analysis/ml16/threshold.py items     # analysis/ml16/threshold_items.csv, for the review page
    python -m coffeecv.review_masks --items analysis/ml16/threshold_items.csv --name ml16_d9_xl0_v7r_s123 --canonical
    python analysis/ml16/threshold.py report    # the new value, the test positives under it, the negatives' rates

The masks are seg_ml19_predict@xl0_v7r_s123's (data/seg_masks_ml19/xl0_v7r_s123); the rule itself is
coffeecv.seg_review.tiny_threshold, which P10's paired report runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from coffeecv.config import REPO_ROOT, RunConfig  # noqa: E402
from coffeecv.repo_files import read_csv  # noqa: E402
from coffeecv.seg_dataset import load_seg_dataset, photo_id  # noqa: E402
from coffeecv.seg_review import tiny_threshold  # noqa: E402

MODEL = "xl0_v7r_s123"
MASKS = REPO_ROOT / "data" / "seg_masks_ml19" / MODEL
NAME = f"ml16_d9_{MODEL}"
ITEMS = REPO_ROOT / "analysis" / "ml16" / "threshold_items.csv"
DECISIONS = REPO_ROOT / "labels" / "ml2" / "review" / f"{NAME}.decisions.jsonl"
REPORT = REPO_ROOT / "analysis" / "ml16" / "threshold_report.json"


def test_positives() -> list[dict]:
    """Every test positive of the segmenter dataset with its mask, matched by the photo's sha256."""
    index = {r["photo_sha256"]: r for r in read_csv(MASKS / "index.csv")}
    out, missing = [], []
    for e in load_seg_dataset():
        if e["split"] != "test" or e["source"] == "negative":
            continue
        r = index.get(e["sha256"])
        if r is None:
            missing.append(e["path"])
            continue
        out.append({"item": photo_id(e["path"]), "path": e["path"], "mask": str((MASKS / f"{r['id']}.png")
                    .relative_to(REPO_ROOT)), "mask_sha256": r["mask_sha256"], "photo_sha256": e["sha256"]})
    if missing:
        raise SystemExit(f"{len(missing)} test positives have no mask, e.g. {missing[0]}")
    return out


def decisions(items: list[dict]) -> dict[str, bool]:
    """{item: accepted}, the last row per item made for its mask; every item must be judged."""
    last = {}
    for line in DECISIONS.read_text().splitlines():
        row = json.loads(line)
        last[row["item"]] = row
    want = {it["item"]: it["mask_sha256"] for it in items}
    judged = {k: r["decision"] == "accept" for k, r in last.items() if want.get(k) == r.get("mask_sha256")}
    missing = [k for k in want if k not in judged]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(want)} masks not judged yet, e.g. {missing[0]}")
    return judged


def report() -> dict:
    cfg = RunConfig.from_params_yaml()
    items = test_positives()
    accepted = decisions(items)
    areas = {it["item"]: float((np.array(Image.open(REPO_ROOT / it["mask"])) > 0).mean()) for it in items}
    rep = tiny_threshold(items, {k: {MODEL: v} for k, v in accepted.items()}, MODEL, areas, cfg.seg_min_area_frac)
    rep["accepted"], rep["declined"] = sum(accepted.values()), sum(not v for v in accepted.values())
    rep["declined_items"] = sorted(it["path"] for it in items if not accepted[it["item"]])
    negs = [np.array(Image.open(MASKS / f"{r['id']}.png")) > 0
            for r in read_csv(MASKS / "index.csv") if r.get("source") == "negative"]
    for key, t in (("negatives_old", rep["old"]), ("negatives_new", rep["new"])):
        rep[key] = {"n": len(negs), "empty": sum(not m.any() for m in negs),
                    "empty_or_tiny": sum(bool(not m.any() or (t is not None and m.mean() < t)) for m in negs)}
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["items", "report"])
    args = ap.parse_args()
    if args.command == "items":
        items = test_positives()
        ITEMS.parent.mkdir(parents=True, exist_ok=True)
        with ITEMS.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["item", "path", "mask", "mask_sha256", "photo_sha256"])
            w.writeheader()
            w.writerows(items)
        print(f"{len(items)} test positives -> {ITEMS.relative_to(REPO_ROOT)}")
        return
    rep = report()
    REPORT.write_text(json.dumps(rep, indent=2) + "\n")
    print(f"{MODEL}: {rep['accepted']} accepted, {rep['declined']} declined of {rep['accepted'] + rep['declined']} "
          f"test positives ({rep['accepted_empty']} accepted empty, not counted)")
    print(f"tiny-mask threshold: new {rep['new']}, old {rep['old']}; test positives under old "
          f"{len(rep['under_old'])}, under new {len(rep['under_new'])}")
    for p in sorted(set(rep["under_old"]) | set(rep["under_new"])):
        print(f"  under {'old and new' if p in rep['under_old'] and p in rep['under_new'] else 'old' if p in rep['under_old'] else 'new'}: {p}")
    for key in ("negatives_old", "negatives_new"):
        n = rep[key]
        print(f"test negatives ({key.split('_')[1]}): {n['empty']}/{n['n']} empty, {n['empty_or_tiny']}/{n['n']} empty or tiny")
    print(f"-> {REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
