"""Guard for the Phase 8 augmentation knobs.

The whole Phase 8 method rests on one property: with every augmentation knob at
its default, `build_train_transform` must be a *bit-exact* no-op versus the
Phase 7 pipeline — otherwise the recorded Phase 7 baselines (exp 20/34/35) stop
being valid comparison points and every Phase 8 verdict inherits a silent
confound. This asserts that against the real pre-Phase-8 code, read straight out
of git, plus the converse: that each knob actually does something when enabled.

    python -m coffeecv.check_augmentation [--ref a87c260]
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coffeecv import transforms as new
from coffeecv.config import REPO_ROOT

PHASE7_HEAD = "a87c260"  # "Plan Phase 8" — last commit before any Phase 8 code change
EXPECTED_DEFAULT_STEPS = [
    "RandomRightAngleRotation", "RandomHorizontalFlip", "RandomVerticalFlip",
    "ColorJitter", "Resize", "ToTensor", "Normalize",
]
KNOBS = [
    dict(rotation_degrees=25.0),
    dict(zoom_scale_min=0.7),
    dict(random_erasing_p=0.5),
    dict(perspective_distortion=0.2),
    dict(illum_gradient_strength=0.2),
]


def _load_ref_module(ref: str):
    src = subprocess.check_output(
        ["git", "show", f"{ref}:coffeecv/transforms.py"], cwd=REPO_ROOT
    )
    tmp = Path(tempfile.mkdtemp()) / "ref_transforms.py"
    tmp.write_bytes(src)
    spec = importlib.util.spec_from_file_location("ref_transforms", tmp)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply(transform, patches) -> torch.Tensor:
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    return torch.stack([transform(p) for p in patches])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default=PHASE7_HEAD, help="git ref holding the pre-Phase-8 transforms.py")
    args = p.parse_args()

    ref = _load_ref_module(args.ref)
    rng = np.random.default_rng(0)
    patches = [Image.fromarray(rng.integers(0, 256, (900, 900, 3), dtype=np.uint8)) for _ in range(12)]

    ref_t, new_t = ref.build_train_transform(224, 0.2), new.build_train_transform(224, 0.2)
    steps = [type(s).__name__ for s in new_t.transforms]
    assert steps == EXPECTED_DEFAULT_STEPS, f"default pipeline structure changed: {steps}"
    assert steps == [type(s).__name__ for s in ref_t.transforms], "differs from ref structure"

    baseline = _apply(new_t, patches)
    assert torch.equal(_apply(ref_t, patches), baseline), (
        "default pipeline is NOT a no-op vs {args.ref} — Phase 7 baselines do not transfer"
    )
    print(f"defaults bit-exact vs {args.ref}: OK ({len(patches)} patches)")

    for kw in KNOBS:
        active = new.build_train_transform(224, 0.2, **kw)
        assert not torch.equal(_apply(active, patches), baseline), f"{kw} had no effect"
        print(f"  {list(kw)[0]:<24} enabled -> changes output: OK")

    assert [type(s).__name__ for s in new.build_eval_transform(224).transforms] == \
           [type(s).__name__ for s in ref.build_eval_transform(224).transforms], "eval transform changed"
    print("eval transform unchanged: OK\n\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
