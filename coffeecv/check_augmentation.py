"""Guard for the Phase 8 augmentation knobs.

Two properties Phase 8's method depends on, both asserted here rather than assumed:

1. **Every knob off is a bit-exact no-op** versus the Phase 7 pipeline, read
   straight out of git. Phase 8 compares against exp 20's recorded numbers
   instead of re-deriving them, which is only legitimate if the off state is
   genuinely identical — a stray extra RNG draw would silently shift every
   result and quietly invalidate every verdict.
2. **Rotation jitter invents no pixels.** The whole point of doing the jitter at
   patch-sampling time is that the extra content a rotation needs comes from the
   surrounding source photo, not from a fill colour. That is checked directly:
   run the real crop-rotate-centre-crop path over an all-white source and assert
   not one fill pixel reaches the output.

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
import torchvision.transforms.functional as TF
from PIL import Image

from coffeecv import transforms as new
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import SPLIT_SEED_COMPONENT, MultiPhotoPatchDataset, discover_classes_multi
from coffeecv.geometry import (
    Region,
    compute_valid_region_rect,
    rotated_box_side,
    sample_patch_boxes,
    sample_rotated_patch_boxes,
)

PHASE7_HEAD = "a87c260"  # "Plan Phase 8" — last commit before any Phase 8 code change
EXPECTED_DEFAULT_STEPS = [
    "RandomRightAngleRotation", "RandomHorizontalFlip", "RandomVerticalFlip",
    "ColorJitter", "Resize", "ToTensor", "Normalize",
]
KNOBS = [
    dict(zoom_scale_min=0.7),
    dict(random_erasing_p=0.5),
    dict(illum_gradient_strength=0.2),
]
CROP_SIZE = 900


def _load_ref_module(ref: str):
    src = subprocess.check_output(["git", "show", f"{ref}:coffeecv/transforms.py"], cwd=REPO_ROOT)
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


def check_transforms(ref_name: str) -> None:
    ref = _load_ref_module(ref_name)
    rng = np.random.default_rng(0)
    patches = [Image.fromarray(rng.integers(0, 256, (CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)) for _ in range(12)]

    ref_t, new_t = ref.build_train_transform(224, 0.2), new.build_train_transform(224, 0.2)
    steps = [type(s).__name__ for s in new_t.transforms]
    assert steps == EXPECTED_DEFAULT_STEPS, f"default pipeline structure changed: {steps}"
    assert steps == [type(s).__name__ for s in ref_t.transforms], "differs from ref structure"

    baseline = _apply(new_t, patches)
    assert torch.equal(_apply(ref_t, patches), baseline), (
        f"default pipeline is NOT a no-op vs {ref_name} — Phase 7 baselines do not transfer"
    )
    print(f"transforms: defaults bit-exact vs {ref_name} ({len(patches)} patches)")

    for kw in KNOBS:
        active = new.build_train_transform(224, 0.2, **kw)
        assert not torch.equal(_apply(active, patches), baseline), f"{kw} had no effect"
        print(f"  {list(kw)[0]:<24} enabled -> changes output")

    assert [type(s).__name__ for s in new.build_eval_transform(224).transforms] == \
           [type(s).__name__ for s in ref.build_eval_transform(224).transforms], "eval transform changed"
    print("  eval transform unchanged")


def check_rotation_jitter() -> None:
    """The jittered path must (a) be a no-op at 0 and (b) invent nothing at >0."""
    region = Region(y0=0, y1=1464, x0=0, x1=1016)  # tightest valid region in the dataset

    # Checked where it actually matters — through MultiPhotoPatchDataset, which
    # routes jitter=0 to the original `sample_patch_boxes` with an untouched RNG.
    # (Comparing the two samplers directly would fail for an uninteresting
    # reason: the rotated one draws all its angles before any position, so its
    # RNG stream differs by construction even though the dataset never uses it
    # at 0 degrees.)
    cfg = RunConfig.from_params_yaml()
    cropped_dir, classes_file = cfg.resolve_paths()
    kwargs = dict(
        cropped_dir=cropped_dir, classes_file=classes_file, split="train",
        class_ids=discover_classes_multi(cropped_dir)[:1], seed=42, crop_size=CROP_SIZE,
        resize=224, safety_margin=cfg.safety_margin,
        patches_per_class={"train": 14, "val": 4, "test": 4},
        photos_per_split={"train": 14, "val": 3, "test": 3},
    )
    ds_off = MultiPhotoPatchDataset(rotation_jitter_degrees=0.0, **kwargs)
    assert all(a == 0.0 for *_, a in ds_off._samples), "jitter=0 produced non-zero angles"

    # 14 patches over 14 photos is one patch each, so sample i came from photo i.
    expected = []
    for photo_idx, (class_id, photo_name, _, _) in enumerate(ds_off._samples):
        h, w = ds_off._images[(class_id, photo_name)].shape[:2]
        photo_region = compute_valid_region_rect(h, w, cfg.safety_margin)
        rng = np.random.default_rng([42, 0, photo_idx, SPLIT_SEED_COMPONENT["train"]])
        expected += sample_patch_boxes(rng, photo_region, 1, CROP_SIZE)
    assert [s[2] for s in ds_off._samples] == expected, "jitter=0 changed box placement"
    print("\nrotation jitter: dataset at 0 degrees reproduces the Phase 7 boxes exactly")

    ds_on = MultiPhotoPatchDataset(rotation_jitter_degrees=5.0, **kwargs)
    assert any(a != 0.0 for *_, a in ds_on._samples), "jitter=5 produced no rotation"
    assert ds_on[0][0].shape == ds_off[0][0].shape, "jittered patch changed tensor shape"
    print("  dataset at 5 degrees rotates, and patch tensor shape is unchanged")

    # The real extraction path, over an all-white source: any fill would show up
    # as a pure-black pixel, since TF.rotate fills with 0 by default.
    for jitter in (1.0, 3.0, 5.0, 6.0):
        boxes = sample_rotated_patch_boxes(np.random.default_rng([4, 5, 6]), region, 40, CROP_SIZE, jitter)
        worst = 0
        for box, angle in boxes:
            assert box.y0 >= region.y0 and box.y1 <= region.y1, "box escaped the valid region"
            assert box.x0 >= region.x0 and box.x1 <= region.x1, "box escaped the valid region"
            source = Image.fromarray(np.full((box.height, box.width, 3), 255, np.uint8))
            out = TF.center_crop(
                TF.rotate(source, angle, interpolation=TF.InterpolationMode.BILINEAR),
                [CROP_SIZE, CROP_SIZE],
            )
            arr = np.asarray(out)
            assert arr.shape == (CROP_SIZE, CROP_SIZE, 3), f"wrong patch size {arr.shape}"
            worst = max(worst, int((arr < 250).sum()))
        room = min(region.width, region.height) - rotated_box_side(CROP_SIZE, jitter)
        assert worst == 0, f"+/-{jitter} deg leaked {worst} fill pixels into the patch"
        print(f"  +/-{jitter:>4} deg: 0 fill pixels over 40 patches, {room}px placement room left")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default=PHASE7_HEAD, help="git ref holding the pre-Phase-8 transforms.py")
    args = p.parse_args()
    check_transforms(args.ref)
    check_rotation_jitter()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
