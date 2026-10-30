"""A small, fixed set of REAL bean patches: the input to the Meta-equivalence fixture and the tests.

Since ticket ML-13 (D9, plan P7) the patches come from the original photos in dataset/, one per class folder
(REFERENCE_PHOTOS, a committed file), which no re-crop touches: the fixture checks only the backbone against
Meta's code, so nothing upstream of the backbone may move it. Each listed photo is decoded by photo_io (most are
iPhone HEIC), and `per_class` square boxes of PATCH_SIDE px, the stored patch's side, are placed uniformly at random
inside the photo's middle half by a generator keyed on (REFERENCE_SEED, the photo's draw key), then put through
the project's eval transform as a stored patch is. The draw key is the photo's path unless its line names another:
a photo renamed since the fixture was made keeps its old path as its key, so its boxes stay where they were and the
fixture needs no rerun of Meta's code (ticket ML-14 D16). No mask, bean scale, crop or segmenter setting reaches the draw. Labels index the listed folder
ids, sorted (a folder's id is the number its name starts with, so `class_009__Vietnam_Robusta` in dataset/ is
the pool's `class_009__Vietnam`).

Until ML-13 the patches came from the pool's crops of the same photos, placed by the D17 rule against each crop's
mask, so every re-crop changed them and the fixture had to be regenerated; until ML-8 the reference was every photo
of data/segcropped/cam_iphone.
"""
from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coffeecv.class_list import folder_classes
from coffeecv.config import REPO_ROOT, RunConfig
from coffeecv.dataset import CLASS_DIR_RE
from coffeecv.photo_io import load_rgb_image
from coffeecv.transforms import build_eval_transform

REFERENCE_PHOTOS = Path(__file__).with_name("reference_photos.txt")
REFERENCE_SEED = 42
PATCH_SIDE = 448            # params.yaml patch_store_size when the draw moved to the originals (2026-10-27)


def folder_of(photo: str) -> str:
    """A listed photo's class folder id (`dataset/<session>/class_001__.../x.jpg` -> "001")."""
    return CLASS_DIR_RE.match(Path(photo).parent.name).group(1)


def reference_entries(path: Path = REFERENCE_PHOTOS) -> list[tuple[str, str]]:
    """(photo, draw key) per listed line: the photo's repo path under dataset/, one per class folder, and the key
    its boxes are drawn by, the line's second field or else the photo's path (ML-14 D16)."""
    entries = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) > 2:
            raise ValueError(f"{path}: a line is a photo and at most one draw key: {line!r}")
        entries.append((fields[0], fields[-1]))
    folders = [folder_of(p) for p, _ in entries]
    if len(set(folders)) != len(folders):
        raise ValueError(f"{path} lists a class folder twice")
    return entries


def reference_photos() -> list[str]:
    """The listed photos, repo paths under dataset/, one per class folder."""
    return [p for p, _ in reference_entries()]


def have_reference_data() -> bool:
    return all((REPO_ROOT / p).is_file() for p in reference_photos())


def reference_boxes(key: str, h: int, w: int, per_class: int) -> list[tuple[int, int]]:
    """The (y0, x0) corners of a listed photo's PATCH_SIDE boxes, inside its middle half on each axis (the owner
    centres the beans: drawn over the whole frame, 3 of the 15 first boxes held only tray and table), from a
    generator keyed on the seed and the photo's draw key (`reference_entries`), so its boxes depend on nothing else."""
    rng = np.random.default_rng([REFERENCE_SEED, zlib.crc32(key.encode())])
    return [(int(rng.integers(h // 4, 3 * h // 4 - PATCH_SIDE + 1)),
             int(rng.integers(w // 4, 3 * w // 4 - PATCH_SIDE + 1))) for _ in range(per_class)]


def reference_patches(per_class: int) -> tuple[torch.Tensor, np.ndarray, list[str], dict[str, str]]:
    """(eval-transformed patches [N, 3, 224, 224], labels [N], class_ids, class_labels)."""
    cfg = RunConfig.from_params_yaml()
    eval_tf = build_eval_transform(cfg.patch_resize)
    entries = sorted(reference_entries(), key=lambda e: folder_of(e[0]))
    # Per folder, whatever classes.txt's format: the fixture's classes are the listed folder ids, sorted (ML-3).
    classes = folder_classes(REPO_ROOT / cfg.classes_file).select([folder_of(p) for p, _ in entries])
    xs, ys = [], []
    for label, (photo, key) in enumerate(entries):
        rgb = load_rgb_image(REPO_ROOT / photo)
        for y0, x0 in reference_boxes(key, *rgb.shape[:2], per_class):
            xs.append(eval_tf(Image.fromarray(np.ascontiguousarray(rgb[y0:y0 + PATCH_SIDE, x0:x0 + PATCH_SIDE]))))
            ys.append(label)
    return torch.stack(xs), np.array(ys), list(classes.keys), dict(classes.labels)
