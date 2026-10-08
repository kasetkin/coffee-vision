"""ML-5 D23: does ft_s123 follow points through the output it was trained on (multi3) as well as `single`?

Reference set A: ML-2 base-point masks the owner accepted (pretrained L0, box + points, single output).
Reference set B: ML-2 labels accepted in round 2 or 3 (pretrained L0, box + judge points, single output).
For each decoder (pretrained L0, ft_s123) and output (single, multi3):
  replay  box + the stored points -> IoU against the reference; share IoU >= 0.9, share empty
  clicks  start from the box alone, then 3 simulated clicks, each at the distance-transform maximum of the
          largest error region (include on a missed region, exclude on a false one), box always kept.
IoU and error regions at 1024 px on the long side.

    PYTHONPATH=. python analysis/ml5_point_probe/point_probe.py   # ~13 min at 4 threads -> rows.csv
    PYTHONPATH=. python analysis/ml5_point_probe/summary.py       # -> summary.txt
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from coffeecv.config import REPO_ROOT
from coffeecv.dataset import load_rgb_image
from coffeecv.segment_beans import BeanSegmenter, SegParams

ROWS = Path(__file__).with_name("rows.csv")
SMALL = 1024                      # long side for error regions and IoU
THREADS = 4                       # pinned, like seg_labels: the thread count changes mask bits
L0 = "efficientvit_sam/efficientvit_sam_l0.pt"
OUTPUTS = ("single", "multi3")
CLICKS = 3
# ML-2's lists name internet photos under a subfolder that has since been flattened.
OLD_INTERNET, NEW_INTERNET = "ood_positives_internet/internet_beans/", "ood_positives_internet/"


def small(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    s = SMALL / max(h, w)
    return cv2.resize(mask.astype(np.uint8), (round(w * s), round(h * s)),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    return 1.0 if union == 0 else float((a & b).sum() / union)


def items() -> list[tuple]:
    """(set, photo path, reference mask path, include points, exclude points) for every reference mask."""
    out = []
    points = yaml.safe_load((REPO_ROOT / "labels/ml2/base_points.yaml").read_text())["items"]
    by_sha = {e["sha256"]: e for e in points}
    base = REPO_ROOT / "data/seg_masks/base_points"
    for r in csv.DictReader((base / "index.csv").read_text().splitlines()):
        if r["output"] != "single":       # frame-filling photos: the box alone, no points
            continue
        e = by_sha[r["photo_sha256"]]
        out.append(("A", r["path"], base / f"{r['id']}.png", e["include"], e["exclude"]))
    labels_csv = (REPO_ROOT / "data/seg_labels/labels.csv").read_text().splitlines()
    labels = {r["id"]: r for r in csv.DictReader(labels_csv)}
    for k in (2, 3):
        rounds = REPO_ROOT / f"data/seg_labels/r{k}"
        for r in csv.DictReader((rounds / "index.csv").read_text().splitlines()):
            lab = labels.get(r["id"])
            if lab and lab["status"] == "accepted" and lab["round"] == str(k) \
                    and lab["mask_sha256"] == r["mask_sha256"]:
                out.append(("B", r["path"], rounds / f"{r['id']}.png",
                            json.loads(r["include"]), json.loads(r["exclude"])))
    return out


def decode(seg: BeanSegmenter, h: int, w: int, inc: list, exc: list, output: str) -> np.ndarray:
    """The whole-image box plus points (fractions of width/height) on the image already encoded."""
    pr = seg.predictor
    xy = [[fx * (w - 1), fy * (h - 1)] for fx, fy in [*inc, *exc]]
    box = pr.apply_boxes_torch(torch.tensor([[0, 0, w - 1, h - 1]], dtype=torch.float))
    coords = pr.apply_coords_torch(torch.tensor([xy], dtype=torch.float)) if xy else None
    point_labels = torch.tensor([[1] * len(inc) + [0] * len(exc)], dtype=torch.int) if xy else None
    multimask = output != "single"
    k = int(output[-1]) - 1 if multimask else 0
    with torch.inference_mode():
        _, _, low = pr.predict_torch(point_coords=coords, point_labels=point_labels, boxes=box,
                                     multimask_output=multimask, return_logits=True)
    return small(seg._upsample(low[0, k]))


def next_click(pred: np.ndarray, ref: np.ndarray) -> tuple[str, list[float]] | None:
    """("inc" | "exc", [x, y] as fractions) at the deepest point of the largest error region."""
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


def resolve(path: str) -> str:
    return path if (REPO_ROOT / path).is_file() else path.replace(OLD_INTERNET, NEW_INTERNET)


def main() -> None:
    torch.set_num_threads(THREADS)
    segs = {"pretrained_l0": BeanSegmenter(SegParams(mask_select="multi3", weights=L0)),
            "ft_s123": BeanSegmenter(SegParams(mask_select="multi3", weights=L0, decoder="models/seg/ft_s123.pt"))}
    its = [(s, resolve(path), mask, inc, exc) for s, path, mask, inc, exc in items()]
    print(f"{sum(i[0] == 'A' for i in its)} set A, {sum(i[0] == 'B' for i in its)} set B", flush=True)
    missing = [i[1] for i in its if not (REPO_ROOT / i[1]).is_file()]
    print(f"{len(missing)} photos missing, skipped: {missing}", flush=True)
    its = [i for i in its if (REPO_ROOT / i[1]).is_file()]
    rows = []
    t0 = time.perf_counter()
    for n, (s, path, mask_path, inc, exc) in enumerate(its):
        rgb = load_rgb_image(REPO_ROOT / path)
        h, w = rgb.shape[:2]
        ref_full = np.array(Image.open(mask_path)) > 0
        if ref_full.shape != (h, w):
            raise ValueError(f"{path}: reference {ref_full.shape} does not match photo {(h, w)}")
        ref = small(ref_full)
        # Both decoders sit on the same frozen L0 encoder: encode once, share the features.
        segs["pretrained_l0"]._encode(rgb)
        pa, pb = segs["pretrained_l0"].predictor, segs["ft_s123"].predictor
        pb.reset_image()
        pb.original_size, pb.input_size, pb.features = pa.original_size, pa.input_size, pa.features
        pb.is_image_set = True
        for name, seg in segs.items():
            for output in OUTPUTS:
                replay = decode(seg, h, w, inc, exc, output)
                clicks = {"inc": [], "exc": []}
                pred = decode(seg, h, w, [], [], output)
                curve = [iou(pred, ref)]
                for _ in range(CLICKS):
                    if c := next_click(pred, ref):
                        clicks[c[0]].append(c[1])
                    pred = decode(seg, h, w, clicks["inc"], clicks["exc"], output)
                    curve.append(iou(pred, ref))
                rows.append({"set": s, "path": path, "model": name, "output": output,
                             "n_points": len(inc) + len(exc), "replay_iou": iou(replay, ref),
                             "replay_empty": int(replay.sum() == 0),
                             **{f"click{i}": v for i, v in enumerate(curve)}})
        if (n + 1) % 20 == 0:
            print(f"{n + 1}/{len(its)} {time.perf_counter() - t0:.0f}s", flush=True)
    with open(ROWS, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print("done", flush=True)


if __name__ == "__main__":
    main()
