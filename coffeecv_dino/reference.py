"""A small, fixed set of REAL bean patches: the input to the Meta-equivalence fixture and the tests.

Drawn by coffeecv's own single-capture-dir builder (`fold_data.build_capture_dataset`: every photo of
one dir, split="all") from the DVC-tracked crops, through the project's eval transform, so no
check in this package runs on synthetic data. cam_iphone because it is the smallest rig (92 photos,
~12 s to decode); it carries 8 of the 10 classes, and labels here index those 8. The segmenter's pool
since ticket ML-3 P2b (the tray heuristic's data/cropped/cam_iphone is retired), with params.yaml's
crop_method, so patches are placed by the D17 rule against each crop's mask. ML-3 adds no iPhone
photos, so the pool stays fixed through the merge.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from coffeecv.class_list import folder_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import discover_classes_multi, resolve_captures
from coffeecv.fold_data import build_capture_dataset

REFERENCE_RIG = "data/segcropped/cam_iphone"
REFERENCE_SEED = 42


def have_reference_data() -> bool:
    return (REPO_ROOT / REFERENCE_RIG).is_dir()


def reference_patches(per_class: int) -> tuple[torch.Tensor, np.ndarray, list[str], dict[str, str]]:
    """(eval-transformed patches [N, 3, 224, 224], labels [N], class_ids, class_labels)."""
    cfg = replace(RunConfig.from_params_yaml(), seed=REFERENCE_SEED)
    rig_dir = REPO_ROOT / REFERENCE_RIG
    # Per folder, whatever classes.txt's format: a class's index seeds its boxes, so the fixture's
    # classes must stay the 8 folder ids the rig carries, in sorted order (ticket ML-3).
    classes = folder_classes(REPO_ROOT / cfg.classes_file).select(sorted(discover_classes_multi(rig_dir)))
    ds = build_capture_dataset(cfg, resolve_captures([rig_dir])[0], classes, per_class)
    items = [ds[i] for i in range(len(ds))]
    return (torch.stack([x for x, _ in items]), np.array([int(y) for _, y in items]), list(classes.keys),
            dict(classes.labels))
