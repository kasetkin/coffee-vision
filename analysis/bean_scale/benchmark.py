"""Run every estimator over the 30 manually-counted crops and score them.

The headline metric is NOT raw accuracy. A method with a large but *constant*
multiplicative bias is fixable with one global constant, which is legitimate; a
method whose bias changes per rig is not fixable without a per-rig constant,
which is exactly what this project refuses to introduce. So the ranking is by
bias consistency across rigs, with raw accuracy reported alongside.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from estimators import METHODS, timed  # noqa: E402

GT = Path(__file__).parent / "ground_truth.json"
SRC = Path("/workspace/data/cropped")


def load_crop(m):
    im = cv2.imread(str(SRC / m["photo"]))
    s = m["crop_side_px"]
    h, w = im.shape[:2]
    cy, cx = h // 2, w // 2
    return im[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2]


def main():
    gt = json.loads(GT.read_text())
    rows = []
    for i, m in enumerate(gt, 1):
        crop = load_crop(m)
        rec = {"rig": m["rig"], "name": m["name"], "gt": m["gt_spacing_px"]}
        for name, fn in METHODS.items():
            v, t, err = timed(fn, crop)
            rec[name] = v
            rec[name + "__ms"] = t * 1000
        rows.append(rec)
        print(f"  {i}/{len(gt)} {m['name']}", flush=True)
    json.dump(rows, open(Path(__file__).parent / "benchmark.json", "w"), indent=1)
    print("wrote benchmark.json")


if __name__ == "__main__":
    main()
