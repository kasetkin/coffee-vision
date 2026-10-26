"""XL0-v5 seeds on the test split, without the review page: where do their masks differ?

    python analysis/xl0v5/seed_compare.py xl0_v5 xl0_v5_s42 xl0_v5_s7 --json analysis/xl0v5/seed_compare.json \
        [--render /tmp/xl0v5/seed_compare] > analysis/xl0v5/seed_compare.txt

Reads each model's test masks from data/seg_masks_xl0v5/<model>/ (index.csv and one PNG per photo).
Negatives: each model's empty masks, its tiny ones (area under params.yaml seg_min_area_frac, serving's no-pile
rule, D18) and the rest, with every negative that some model drew a non-tiny mask on. Positives: for each pair
of models, the IoU of their masks and the largest connected region where they disagree, as a share of the
photo; a photo is listed as differing when that region is at least --min-blob of the photo in some pair. The
owner's earlier verdicts (labels/ml5/*.paired.jsonl) follow a mask by its sha256, so a mask identical to a
judged one shows that verdict. --render writes, for each listed photo, one panel per model (its outline; red:
only this mask, white: another mask but not this one) and, when it is local, a zoom on the largest disagreement.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coffeecv.config import REPO_ROOT  # noqa: E402
from coffeecv.dataset import load_rgb_image  # noqa: E402

MASK_ROOT = REPO_ROOT / "data/seg_masks_xl0v5"
COLOURS = [(0, 255, 0), (0, 160, 255), (255, 200, 0), (255, 0, 255)]


def load_index(model: str) -> dict[str, dict]:
    return {r["id"]: r for r in csv.DictReader(open(MASK_ROOT / model / "index.csv"))}


def load_mask(model: str, item: str) -> np.ndarray:
    return np.array(Image.open(MASK_ROOT / model / f"{item}.png")) > 0


def owner_verdicts() -> dict[str, str]:
    """{mask_sha256: the owner's latest decision on that mask, in any paired session}."""
    out: dict[str, tuple[str, str]] = {}
    for f in sorted((REPO_ROOT / "labels/ml5").glob("*.paired.jsonl")):
        for line in open(f):
            v = json.loads(line)
            if v.get("reviewer") == "owner" and v.get("decision") in ("accept", "decline"):
                if v["mask_sha256"] not in out or v["ts"] > out[v["mask_sha256"]][0]:
                    out[v["mask_sha256"]] = (v["ts"], v["decision"])
    return {sha: d for sha, (_, d) in out.items()}


def disagreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float, tuple[slice, slice] | None]:
    """IoU, the largest disagreeing region's share of the photo, and that region's bounding box."""
    xor = a ^ b
    union = (a | b).sum()
    iou = float((a & b).sum() / union) if union else 1.0
    if not xor.any():
        return iou, 0.0, None
    lab, n = ndimage.label(xor)
    sizes = np.bincount(lab.ravel())[1:]
    j = int(sizes.argmax())
    return iou, float(sizes[j] / xor.size), ndimage.find_objects(lab)[j]


def panel(rgb: np.ndarray, mask: np.ndarray, others: np.ndarray, colour: tuple, title: str, height: int) -> Image.Image:
    """The photo with `mask`'s outline in `colour`; pixels only this mask has in red, pixels it lacks that another
    mask has in white (both half-transparent). `others` is (every model's mask, some model's mask)."""
    size = (max(1, round(rgb.shape[1] * height / rgb.shape[0])), height)
    small = lambda m: np.array(Image.fromarray(m).resize(size, Image.Resampling.NEAREST))  # noqa: E731
    out = np.array(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)).astype(np.float32)
    m, every, some = small(mask), small(others[0]), small(others[1])
    out[m & ~every] = 0.5 * out[m & ~every] + 0.5 * np.array((255, 0, 0))
    out[~m & some] = 0.5 * out[~m & some] + 0.5 * np.array((255, 255, 255))
    out[m & ~ndimage.binary_erosion(m, iterations=2)] = colour
    im = Image.fromarray(out.astype(np.uint8))
    ImageDraw.Draw(im).text((6, 4), title, fill=(255, 255, 0))
    return im


def render(item: str, rgb: np.ndarray, masks: dict[str, np.ndarray], box: tuple[slice, slice] | None,
           labels: dict[str, str], out: Path) -> None:
    every, some = np.logical_and.reduce(list(masks.values())), np.logical_or.reduce(list(masks.values()))
    rows = [[panel(rgb, m, (every, some), COLOURS[i % 4], labels[n], 700) for i, (n, m) in enumerate(masks.items())]]
    if box is not None and (box[0].stop - box[0].start) * (box[1].stop - box[1].start) < 0.25 * every.size:
        h, w = every.shape
        cy, cx = (box[0].start + box[0].stop) // 2, (box[1].start + box[1].stop) // 2
        half = max(box[0].stop - box[0].start, box[1].stop - box[1].start, 300)
        ys, xs = slice(max(0, cy - half), min(h, cy + half)), slice(max(0, cx - half), min(w, cx + half))
        rows.append([panel(rgb[ys, xs], m[ys, xs], (every[ys, xs], some[ys, xs]), COLOURS[i % 4],
                           labels[n] + " (zoom)", 700) for i, (n, m) in enumerate(masks.items())])
    width = max(sum(p.width for p in r) for r in rows)
    sheet = Image.new("RGB", (width, 700 * len(rows)))
    for y, r in enumerate(rows):
        x = 0
        for p in r:
            sheet.paste(p, (x, 700 * y))
            x += p.width
    sheet.save(out, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("models", nargs="+", help="mask directories under data/seg_masks_xl0v5/")
    ap.add_argument("--min-blob", type=float, default=0.001,
                    help="a disagreeing region at least this share of the photo lists the photo (default 0.001)")
    ap.add_argument("--render", type=Path, help="write a comparison sheet per listed photo here")
    ap.add_argument("--json", type=Path, help="write the numbers here")
    a = ap.parse_args()
    tiny = yaml.safe_load(open(REPO_ROOT / "params.yaml"))["seg_min_area_frac"]
    idx = {m: load_index(m) for m in a.models}
    items = sorted(idx[a.models[0]])
    for m in a.models[1:]:
        if sorted(idx[m]) != items:
            sys.exit(f"{m} does not hold the same test photos as {a.models[0]}")
    verdicts = owner_verdicts()
    if a.render:
        a.render.mkdir(parents=True, exist_ok=True)
    rep: dict = {"models": a.models, "min_area_frac": tiny, "min_blob": a.min_blob, "negatives": {}, "positives": {}}

    negs = [i for i in items if idx[a.models[0]][i]["source"] == "negative"]
    print(f"test negatives ({len(negs)}): empty / tiny (area < {tiny:g}) / larger")
    area = {m: {i: float(load_mask(m, i).mean()) for i in negs} for m in a.models}   # index.csv's is rounded
    for m in a.models:
        v = area[m].values()
        c = {"empty": sum(x == 0 for x in v), "tiny": sum(0 < x < tiny for x in v), "larger": sum(x >= tiny for x in v)}
        rep["negatives"][m] = c
        print(f"  {m:14s} {c['empty']:3d} / {c['tiny']:3d} / {c['larger']:3d}")
    flagged = {i: {m: round(area[m][i], 5) for m in a.models} for i in negs
               if any(area[m][i] >= tiny for m in a.models)}
    print(f"negatives where some model drew a non-tiny mask ({len(flagged)}), area per model:")
    for i in sorted(flagged):
        print(f"  {i:90s} " + "  ".join(f"{flagged[i][m]:.3f}" for m in a.models))
    rep["negatives"]["flagged"] = flagged

    pos = [i for i in items if idx[a.models[0]][i]["source"] != "negative"]
    pairs = list(itertools.combinations(a.models, 2))
    print(f"\ntest positives ({len(pos)}): identical masks / differing (largest disagreement >= {a.min_blob:g})")
    listed = []
    for i in pos:
        masks = {m: load_mask(m, i) for m in a.models}
        shas = {m: idx[m][i]["mask_sha256"] for m in a.models}
        row = {"source": idx[a.models[0]][i]["source"], "area": {m: float(idx[m][i]["area_frac"]) for m in a.models},
               "owner": {m: verdicts.get(shas[m]) for m in a.models}, "pairs": {}}
        worst, box = 0.0, None
        for x, y in pairs:
            iou, blob, b = disagreement(masks[x], masks[y])
            row["pairs"][f"{x}|{y}"] = {"iou": round(iou, 4), "blob": round(blob, 5)}
            if blob > worst:
                worst, box = blob, b
        row["worst_blob"] = round(worst, 5)
        row["identical"] = len(set(shas.values())) == 1
        rep["positives"][i] = row
        if worst > 0 and worst >= a.min_blob:
            listed.append(i)
            if a.render:
                labels = {m: f"{m}  area {row['area'][m]:.3f}  owner: {row['owner'][m] or '-'}" for m in a.models}
                render(i, load_rgb_image(REPO_ROOT / idx[a.models[0]][i]["path"]), masks, box, labels,
                       a.render / f"{len(listed):02d}_{i}.jpg")
    same = sum(r["identical"] for r in rep["positives"].values())
    print(f"  {same} identical, {len(pos) - same - len(listed)} differing below the bar, {len(listed)} listed")
    print("listed, largest disagreement first (blob = share of the photo; IoU per pair; owner verdict per model):")
    for i in sorted(listed, key=lambda k: -rep["positives"][k]["worst_blob"]):
        r = rep["positives"][i]
        ious = " ".join(f"{p['iou']:.3f}" for p in r["pairs"].values())
        own = " ".join((r["owner"][m] or "-")[:3] for m in a.models)
        print(f"  {r['worst_blob']:.4f}  iou {ious}  owner {own}  {r['source'][:7]:7s} {i}")
    rep["listed"] = listed
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
