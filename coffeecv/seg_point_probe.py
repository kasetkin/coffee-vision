"""The point probe (ticket ML-5 P6, D19, D23): does a segmenter follow point prompts?

For each accepted reference mask with the points that made it, and each segmenter and decoder output:

  replay   the whole-image box plus the stored points; IoU against the reference, and whether it is empty
  clicks   the box alone, then 3 simulated corrective clicks, each at the distance-transform maximum of the
           largest error region (include on a missed region, exclude on a false one), the box always kept

IoU and error regions at 1024 px on the long side. Printed per (reference set, model, output): the share at
IoU >= 0.9 after the replay and after each click, the empty share of the replay, and the median IoU per click.
Built from analysis/ml5_point_probe/point_probe.py, so that P8 could rerun it on a fine-tune against a
session's accepts before choosing the next session's redraw model.

    python -m coffeecv.seg_point_probe --models xl0_v3 --outputs multi3 --refs ml5:pass3

Reference sets:
  ml5:<session>    a review session's labels accepted in round 2 or 3 (coffeecv/seg_review.py)

ML-2's sets (ml2_base_points, ml2_labels) went with ML-2's records on 2026-10-21; the analysis's rows.csv and
summary.txt, made on them, stay in analysis/ml5_point_probe.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from PIL import Image

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import load_rgb_image
from coffeecv.segment_beans import THREADS, BeanSegmenter, named_params

SMALL = 1024                      # long side for error regions and IoU
OUTPUTS = ("single", "multi1", "multi2", "multi3")
CLICKS = 3
CUT = 0.9
ROW_FIELDS = ["set", "path", "model", "output", "n_points", "replay_iou", "replay_empty",
              *(f"click{i}" for i in range(CLICKS + 1))]


class Ref(NamedTuple):
    set: str
    path: str                     # repo-relative photo
    mask: Path                    # the accepted reference mask, full resolution
    include: list                 # [x, y] fractions of the photo
    exclude: list


def small(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    s = SMALL / max(h, w)
    return cv2.resize(mask.astype(np.uint8), (round(w * s), round(h * s)),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def next_click(pred: np.ndarray, ref: np.ndarray) -> tuple[str, list[float]] | None:
    """("inc" | "exc", [x, y] as fractions) at the deepest point of the largest error region, or None if the
    masks agree."""
    best = None
    for kind, region in (("inc", ref & ~pred), ("exc", pred & ~ref)):
        n, comp, stats, _ = cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
        if n > 1:
            j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if best is None or stats[j, cv2.CC_STAT_AREA] > best[0]:
                best = (stats[j, cv2.CC_STAT_AREA], kind, comp == j)
    if best is None:
        return None
    _, kind, region = best
    depth = cv2.distanceTransform(np.pad(region, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(depth.argmax()), depth.shape)
    h, w = region.shape
    return kind, [x / (w - 1), y / (h - 1)]


# ---------------------------------------------------------------- reference sets

def _ml5_session(session: str) -> list[Ref]:
    from coffeecv import seg_review
    return [Ref(f"ml5:{session}", r["path"], REPO_ROOT / r["mask"], r["include"], r["exclude"])
            for r in seg_review.accepted_labels(session) if int(r["round"]) >= 2]


def reference_items(refs: str) -> list[Ref]:
    if refs.startswith("ml5:"):
        return _ml5_session(refs.removeprefix("ml5:"))
    raise ValueError(f"reference set {refs!r}: expected ml5:<session>")


# ---------------------------------------------------------------- the probe

def _decode(seg: BeanSegmenter, inc: list, exc: list, output: str) -> np.ndarray:
    return small(seg.decode_points(inc, exc, output)[0])


def probe(models: list[str], outputs: list[str], items: list[Ref], threads: int = THREADS) -> list[dict]:
    """One row per (item, model, output), ROW_FIELDS. Segmenters on the same weights share one encoding."""
    torch.set_num_threads(threads)
    segs = {name: BeanSegmenter(named_params(name)) for name in models}
    rows = []
    t0 = time.perf_counter()
    for n, it in enumerate(items):
        rgb = load_rgb_image(REPO_ROOT / it.path)
        ref_full = np.array(Image.open(it.mask)) > 0
        if ref_full.shape != rgb.shape[:2]:
            raise ValueError(f"{it.path}: reference {ref_full.shape} does not match photo {rgb.shape[:2]}")
        ref = small(ref_full)
        encodings = {}
        for name, seg in segs.items():
            if seg.weights_sha256 not in encodings:
                encodings[seg.weights_sha256] = seg.encode(rgb)
            else:
                seg.set_encoding(encodings[seg.weights_sha256])
            for output in outputs:
                replay = _decode(seg, it.include, it.exclude, output)
                clicks = {"inc": [], "exc": []}
                pred = _decode(seg, [], [], output)
                curve = [iou(pred, ref)]
                for _ in range(CLICKS):
                    if c := next_click(pred, ref):
                        clicks[c[0]].append(c[1])
                    pred = _decode(seg, clicks["inc"], clicks["exc"], output)
                    curve.append(iou(pred, ref))
                rows.append({"set": it.set, "path": it.path, "model": name, "output": output,
                             "n_points": len(it.include) + len(it.exclude), "replay_iou": iou(replay, ref),
                             "replay_empty": int(replay.sum() == 0),
                             **{f"click{i}": v for i, v in enumerate(curve)}})
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(items)} {time.perf_counter() - t0:.0f}s", flush=True)
    return rows


def _share(rows: list[dict], col: str) -> float:
    return sum(float(r[col]) >= CUT for r in rows) / len(rows)


def summary(rows: list[dict]) -> str:
    """The table analysis/ml5_point_probe/summary.py printed, one line per (set, model, output)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["set"], r["model"], r["output"])].append(r)
    w_set = max([3, *(len(k[0]) for k in groups)])
    w_model = max([14, *(len(k[1]) for k in groups)])
    lines = [f"{'set':{w_set}} {'model':{w_model}} {'out':7} {'n':>4}  replay>=.9 empty  | box-only>=.9  click1  "
             f"click2  click3 | median IoU c0..c3"]
    for key in sorted(groups):
        rs = groups[key]
        empty = sum(int(r["replay_empty"]) for r in rs) / len(rs)
        medians = " ".join(f"{statistics.median(float(r[f'click{i}']) for r in rs):.3f}" for i in range(CLICKS + 1))
        lines.append(f"{key[0]:{w_set}} {key[1]:{w_model}} {key[2]:7} {len(rs):4}  {_share(rs, 'replay_iou'):9.0%} "
                     f"{empty:6.0%} | {_share(rs, 'click0'):10.0%} {_share(rs, 'click1'):7.0%} "
                     f"{_share(rs, 'click2'):7.0%} {_share(rs, 'click3'):7.0%} | {medians}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True,
                    help="segmenter names (segment_beans.NAMED_SEGMENTERS) or WEIGHTS:DECODER")
    ap.add_argument("--outputs", nargs="+", choices=OUTPUTS, required=True, help="decoder outputs (D23)")
    ap.add_argument("--refs", nargs="+", required=True, help="ml5:<session>")
    ap.add_argument("--out", type=Path, help="also write every row as CSV")
    args = ap.parse_args(argv)
    items = [it for refs in args.refs for it in reference_items(refs)]
    missing = [it.path for it in items if not (REPO_ROOT / it.path).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} photos missing (dvc checkout): {missing[:5]}")
    print(f"{len(items)} reference masks: " + ", ".join(f"{s} {sum(i.set == s for i in items)}"
                                                        for s in dict.fromkeys(i.set for i in items)), flush=True)
    rows = probe(args.models, args.outputs, items)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=ROW_FIELDS)
            wr.writeheader()
            wr.writerows(rows)
    print(summary(rows), end="")


if __name__ == "__main__":
    main()
