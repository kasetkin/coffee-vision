"""ML-37 D10: the frozen head fit's patch store in memory, by sample type, against ML-8 P6's measured peak.

    cd /workspace && PYTHONPATH=/workspace uv run python analysis/ml37/d10_memory.py

Arithmetic from params.yaml (patch_store_size, the per-class budgets) and the class count of the latest recorded run
(outputs/frozen_head/s7/metrics.json), with ML-8 P6's measurement (docs/ml-0008_plan_data_refresh_retrain.html, P6:
"peak RSS 5.5 GB (the augmented fit; 4.5 GB while the patches are held)") as the base: fit_one_seed holds one split's
patches at a time, the training split's while its views are embedded, and drops them before the head fits.
Writes analysis/ml37/d10_memory.txt.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from coffeecv.config import REPO_ROOT

OUT = Path(__file__).with_suffix(".txt")
HELD_GB, FIT_GB = 4.5, 5.5          # ML-8 P6, uint8 patches (3.6 GB of the 4.5)


def main() -> None:
    p = yaml.safe_load((REPO_ROOT / "params.yaml").read_text())
    classes = len(json.loads((REPO_ROOT / "outputs/frozen_head/s7/metrics.json").read_text())["class_ids"])
    side = p["patch_store_size"]
    n = {s: p[f"{s}_patches_per_class"] * classes for s in ("train", "val", "test")}
    lines = [f"{classes} classes; patches per split {n}; patch_store_size {side}"]
    base = n["train"] * side * side * 3 / 1e9
    for dt in (np.uint8, np.uint16, np.float16, np.float32):
        one = np.zeros((side, side, 3), dt).nbytes
        gb = n["train"] * one / 1e9
        lines.append(f"{np.dtype(dt).name:8s} {one / 1e6:5.2f} MB per patch; training split {gb:5.2f} GB; "
                     f"peak while held ~{HELD_GB - base + gb:5.1f} GB (ML-8 P6's 4.5 with {base:.2f} GB of uint8); "
                     f"the head fits after it {FIT_GB} GB")
    lines.append(f"one view in float32 for albumentations: {side * side * 3 * 4 / 1e6:.1f} MB at {side}, "
                 f"{224 * 224 * 3 * 4 / 1e6:.2f} MB at 224; a batch of 32 views {32 * side * side * 3 * 4 / 1e6:.0f} MB")
    for mp in (11.4, 18.2, 50.1):
        lines.append(f"one crop held while its patches are cut, {mp} MP: uint8 {mp * 3:.0f} MB, uint16/float16 "
                     f"{mp * 6:.0f} MB, float32 {mp * 12:.0f} MB")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
