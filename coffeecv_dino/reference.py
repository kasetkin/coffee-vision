"""A small, fixed set of REAL bean patches: the input to the Meta-equivalence fixture and the tests.

Drawn by coffeecv's own single-capture-dir builder (`fold_data.build_capture_dataset`: every photo of
one dir, split="all") from the DVC-tracked crops, through the project's eval transform, so no
check in this package runs on synthetic data. Since ticket ML-8 (D2, plan P5) the photos are a fixed
list from the pool, one per class folder (REFERENCE_PHOTOS, a committed file): the builder runs on a
temporary dir holding links to just those crops and their masks, so drawing them decodes 15 photos, not
the pool, and the patches change only when the list or the crops do. Labels index the listed folders,
sorted. Patches are placed by the D17 rule against each crop's mask (params.yaml's crop_method), and every
listed crop has one. Until ML-8 the reference was every photo of data/segcropped/cam_iphone, the smallest
camera pool (92 photos, 8 of the 10 classes then).
"""
from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from coffeecv.class_list import folder_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import CLASS_DIR_RE, bean_mask_path, resolve_captures
from coffeecv.fold_data import build_capture_dataset

REFERENCE_POOL = "data/segcropped/pool"
REFERENCE_PHOTOS = Path(__file__).with_name("reference_photos.txt")
REFERENCE_SEED = 42


def folder_of(photo: str) -> str:
    """A listed crop's class folder id (`class_001__...` -> "001")."""
    return CLASS_DIR_RE.match(Path(photo).parts[0]).group(1)


def reference_photos() -> list[str]:
    """The listed crops, `<class dir>/<crop file>` under REFERENCE_POOL, one per class folder."""
    lines = [ln.strip() for ln in REFERENCE_PHOTOS.read_text().splitlines()]
    photos = [ln for ln in lines if ln and not ln.startswith("#")]
    folders = [folder_of(p) for p in photos]
    if len(set(folders)) != len(folders):
        raise ValueError(f"{REFERENCE_PHOTOS} lists a class folder twice")
    return photos


def have_reference_data() -> bool:
    pool = REPO_ROOT / REFERENCE_POOL
    return all((pool / p).is_file() and bean_mask_path(pool / p).is_file() for p in reference_photos())


def reference_patches(per_class: int) -> tuple[torch.Tensor, np.ndarray, list[str], dict[str, str]]:
    """(eval-transformed patches [N, 3, 224, 224], labels [N], class_ids, class_labels)."""
    cfg = replace(RunConfig.from_params_yaml(), seed=REFERENCE_SEED)
    pool = REPO_ROOT / REFERENCE_POOL
    photos = reference_photos()
    with tempfile.TemporaryDirectory() as tmp:
        ref_dir = Path(tmp) / "reference"
        for p in photos:
            crop = pool / p
            (ref_dir / crop.parent.name).mkdir(parents=True, exist_ok=True)
            for f in (crop, bean_mask_path(crop)):
                (ref_dir / crop.parent.name / f.name).symlink_to(f)
        # Per folder, whatever classes.txt's format: a class's index seeds its boxes, so the fixture's
        # classes are the listed folder ids in sorted order (ticket ML-3).
        folders = sorted(folder_of(p) for p in photos)
        classes = folder_classes(REPO_ROOT / cfg.classes_file).select(folders)
        ds = build_capture_dataset(cfg, resolve_captures([ref_dir])[0], classes, per_class)
        items = [ds[i] for i in range(len(ds))]
    return (torch.stack([x for x, _ in items]), np.array([int(y) for _, y in items]), list(classes.keys),
            dict(classes.labels))
