"""ML-34 experiment A: how much of the bean region holds light the tone map changes, or light above white.

    python -I analysis/ml34/a_light_above_white.py /workspace [--jobs 4] [--out tmp/ml34_a]

For every segmenter mask in data/seg_masks_ml19/xl0_v7r_s123 whose original carries a gain map (tmp/ml29_scan/
files.csv, ML-29's scan), the original is decoded as the store decodes it (coffeecv.canonical.decode, STORE_SETTINGS,
gain map applied): linear Rec.2020, 1.0 = standard-range white (203 cd/m2, BT.2408). From that light, before any tone
map: the picture's peak (D33's picture_peak), the EETF's knee for that peak (tone_map.eetf_knee), and inside the bean
mask the share of pixels whose max(R, G, B) is above the knee (the tone map changes them) and above 1.0 (light above
white, what an HDR canonical photo would keep). The same for the whole picture. Nothing is written to the store.

Writes <out>/a_light_above_white.csv (one row a photo) and prints the summary by camera.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def measure(args: tuple[str, str, str, str]) -> dict:
    repo, photo, mask_path, camera = args
    sys.path.insert(0, repo)
    from coffeecv import canonical as cn
    from coffeecv import tone_map as tm
    data = Path(photo).read_bytes()
    src = cn.decode(data, Path(photo).name)
    row = {"photo": photo, "camera": camera, "decoder": src.decoder}
    if not src.tone_map:
        return {**row, "skipped": "no map applied"}
    light = src.pixels                                      # H x W x 3 float32 linear Rec.2020, 1.0 white
    top = light.max(axis=-1)
    peak = tm.picture_peak(top.ravel().copy())
    knee = tm.eetf_knee(peak)
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L").resize((top.shape[1], top.shape[0]), Image.NEAREST)) > 127
    if not mask.any():
        return {**row, "skipped": "empty mask"}
    lum = light @ np.array(tm.BT2100_LUMINANCE, np.float32)
    bean, bean_lum = top[mask], lum[mask]
    return {**row, "skipped": "", "h": top.shape[0], "w": top.shape[1], "bean_frac": float(mask.mean()),
            "peak": peak, "knee": knee, "img_max": float(top.max()),
            "img_above_knee": float((top > knee).mean()), "img_above_white": float((top > 1).mean()),
            "bean_above_knee": float((bean > knee).mean()), "bean_above_white": float((bean > 1).mean()),
            "bean_maxrgb_p50": float(np.percentile(bean, 50)), "bean_maxrgb_p99": float(np.percentile(bean, 99)),
            "bean_maxrgb_max": float(bean.max()),
            "bean_lum_p50_cdm2": float(np.percentile(bean_lum, 50) * tm.REFERENCE_WHITE),
            "bean_lum_p01_cdm2": float(np.percentile(bean_lum, 1) * tm.REFERENCE_WHITE),
            "bean_lum_p99_cdm2": float(np.percentile(bean_lum, 99) * tm.REFERENCE_WHITE)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--masks", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    repo = a.repo.resolve()
    masks = a.masks or repo / "data" / "seg_masks_ml19" / "xl0_v7r_s123"
    out = a.out or repo / "tmp" / "ml34_a"
    out.mkdir(parents=True, exist_ok=True)
    scanned = {"__".join(Path(r["file"].split("dataset/")[1]).with_suffix("").parts): r
               for r in csv.DictReader((repo / "tmp" / "ml29_scan" / "files.csv").open())}
    jobs = []
    for mask_path in sorted(masks.glob("*.png")):
        r = scanned.get(mask_path.stem)
        if r and r["gain_map_detail"]:
            jobs.append((str(repo), r["file"], str(mask_path), f"{r['type']} {r['make']} {r['model']}"))
    if a.limit:
        jobs = jobs[:a.limit]
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, row in enumerate(ex.map(measure, jobs, chunksize=1)):
            rows.append(row)
            if i % 25 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("photo", "camera"), k))
    with (out / "a_light_above_white.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    print("skipped:", dict(collections.Counter(r["skipped"] for r in rows if r["skipped"])))
    by = collections.defaultdict(list)
    for r in rows:
        if not r["skipped"]:
            by[r["camera"]].append(r)
            by["ALL"].append(r)
    cols = ("peak", "knee", "img_above_white", "img_above_knee", "bean_above_white", "bean_above_knee",
            "bean_maxrgb_p50", "bean_maxrgb_p99", "bean_lum_p50_cdm2", "bean_lum_p99_cdm2")
    print("median photo (and worst photo for the shares)")
    print(f"{'camera':34} {'n':>4} " + " ".join(f"{c:>17}" for c in cols))
    for cam, rs in sorted(by.items(), key=lambda kv: (kv[0] == "ALL", kv[0])):
        cells = []
        for c in cols:
            v = np.array([r[c] for r in rs])
            cells.append(f"{np.median(v):8.4f} ({v.max():6.3f})" if "above" in c else f"{np.median(v):17.3f}")
        print(f"{cam[:34]:34} {len(rs):4d} " + " ".join(cells))
    for t in (0.01, 0.05):
        n = sum(r["bean_above_knee"] > t for r in by["ALL"])
        m = sum(r["bean_above_white"] > t for r in by["ALL"])
        print(f"photos with over {t:.0%} of the bean region above the knee: {n}; above white: {m} (of {len(by['ALL'])})")
    print(json.dumps({"n": len(by["ALL"])}))


if __name__ == "__main__":
    main()
