"""A small, fixed set of REAL bean patches: the input to the Meta-equivalence fixture and the tests.

Drawn by coffeecv's own single-rig builder (`xrig_eval.build_xrig_dataset` -- the construction a
fold's cross-rig set uses) from the DVC-tracked crops, through the project's eval transform, so no
check in this package runs on synthetic data. cam_iphone because it is the smallest rig (92 photos,
~12 s to decode); it carries 8 of the 10 classes, and labels here index those 8.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import discover_classes_multi, load_class_labels, resolve_rigs
from coffeecv.xrig_eval import build_xrig_dataset

REFERENCE_RIG = "data/cropped/cam_iphone"
REFERENCE_SEED = 42


def have_reference_data() -> bool:
    return (REPO_ROOT / REFERENCE_RIG).is_dir()


def reference_patches(per_class: int) -> tuple[torch.Tensor, np.ndarray, list[str], dict[str, str]]:
    """(eval-transformed patches [N, 3, 224, 224], labels [N], class_ids, class_labels)."""
    cfg = replace(RunConfig.from_params_yaml(), seed=REFERENCE_SEED)
    rig_dir = REPO_ROOT / REFERENCE_RIG
    classes_file = REPO_ROOT / cfg.classes_file
    class_ids = sorted(discover_classes_multi(rig_dir))
    ds = build_xrig_dataset(cfg, resolve_rigs([rig_dir])[0], class_ids, classes_file, per_class)
    items = [ds[i] for i in range(len(ds))]
    labels = load_class_labels(classes_file)
    return (torch.stack([x for x, _ in items]), np.array([int(y) for _, y in items]), class_ids,
            {c: labels.get(c, c) for c in class_ids})
