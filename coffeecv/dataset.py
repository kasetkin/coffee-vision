"""HEIF loading and patch-based PyTorch Dataset for the coffee bean classes."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image
from torch.utils.data import Dataset

from coffeecv.geometry import (
    Region,
    assert_region_fully_opaque,
    compute_valid_region,
    sample_patch_boxes,
    split_regions,
)

pillow_heif.register_heif_opener()

CLASS_FILENAME_RE = re.compile(r"class=(\d+)\.heif$", re.IGNORECASE)
SPLIT_SEED_COMPONENT = {"train": 0, "val": 1, "test": 2}


def discover_classes(dataset_dir: Path) -> list[str]:
    ids = set()
    for p in Path(dataset_dir).glob("*.heif"):
        m = CLASS_FILENAME_RE.search(p.name)
        if m:
            ids.add(m.group(1))
    if not ids:
        raise FileNotFoundError(f"No 'class=NNN.heif' files found under {dataset_dir}")
    return sorted(ids)


def find_class_file(dataset_dir: Path, class_id: str) -> Path:
    matches = []
    for p in Path(dataset_dir).glob("*.heif"):
        m = CLASS_FILENAME_RE.search(p.name)
        if m and m.group(1) == class_id:
            matches.append(p)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one heif file for class={class_id}, found {len(matches)}")
    return matches[0]


def load_class_labels(classes_file: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    if not Path(classes_file).exists():
        return labels
    for line in Path(classes_file).read_text().splitlines():
        line = line.strip()
        if not line or ";" not in line:
            continue
        cid, label = line.split(";", 1)
        labels[cid.strip()] = label.strip()
    return labels


def get_class_label(class_id: str, labels: dict[str, str]) -> str:
    if class_id not in labels:
        print(f"WARNING: class {class_id} missing from classes.txt, using raw id as label")
        return class_id
    return labels[class_id]


def load_source_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rgb HxWx3 uint8, alpha HxW uint8)."""
    img = Image.open(path)
    arr = np.array(img.convert("RGBA"))
    return arr[:, :, :3], arr[:, :, 3]


class PatchCoffeeDataset(Dataset):
    """Yields (patch_tensor, class_index) pairs cropped from the source photos.

    Patch boxes are precomputed once at construction time from a seeded RNG,
    so a given (seed, class, split) always yields the same boxes regardless
    of DataLoader iteration order.
    """

    def __init__(
        self,
        dataset_dir: Path,
        classes_file: Path,
        split: str,
        class_ids: list[str],
        seed: int,
        crop_size: int,
        resize: int,
        safety_margin: float,
        patches_per_class: dict[str, int],
        transform=None,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.class_ids = class_ids
        self.resize = resize
        self.transform = transform
        self.class_labels = load_class_labels(classes_file)

        n_patches = patches_per_class[split]
        self._images: dict[str, np.ndarray] = {}
        self._samples: list[tuple[str, Region]] = []  # (class_id, box)

        for class_idx, class_id in enumerate(class_ids):
            img_path = find_class_file(dataset_dir, class_id)
            rgb, alpha = load_source_image(img_path)
            h, w = alpha.shape
            valid_region = compute_valid_region(h, w, safety_margin)
            regions = split_regions(valid_region)
            region = regions[split]
            assert_region_fully_opaque(alpha, region)

            rng = np.random.default_rng([seed, class_idx, SPLIT_SEED_COMPONENT[split]])
            boxes = sample_patch_boxes(rng, region, n_patches, crop_size)

            self._images[class_id] = rgb
            self._samples.extend((class_id, box) for box in boxes)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        class_id, box = self._samples[idx]
        rgb = self._images[class_id]
        patch = rgb[box.y0:box.y1, box.x0:box.x1]
        pil_patch = Image.fromarray(patch)
        label = self.class_ids.index(class_id)
        if self.transform is not None:
            tensor = self.transform(pil_patch)
        else:
            pil_patch = pil_patch.resize((self.resize, self.resize), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(pil_patch)).permute(2, 0, 1).float() / 255.0
        return tensor, label
