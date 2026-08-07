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
    compute_valid_region_rect,
    sample_patch_boxes,
    split_regions,
)

pillow_heif.register_heif_opener()

CLASS_FILENAME_RE = re.compile(r"class=(\d+)\.heif$", re.IGNORECASE)
CLASS_DIR_RE = re.compile(r"^class_(\d+)__")
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


# ---- Multi-photo box-rig dataset (2026-08-07__box_pictures_all_classes and later) --------


def discover_classes_multi(dataset_dir: Path) -> list[str]:
    ids = set()
    for p in Path(dataset_dir).iterdir():
        if p.is_dir():
            m = CLASS_DIR_RE.match(p.name)
            if m:
                ids.add(m.group(1))
    if not ids:
        raise FileNotFoundError(f"No 'class_NNN__Label' directories found under {dataset_dir}")
    return sorted(ids)


def find_class_dir(dataset_dir: Path, class_id: str) -> Path:
    matches = [p for p in Path(dataset_dir).iterdir() if p.is_dir() and p.name.startswith(f"class_{class_id}__")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one directory for class={class_id}, found {len(matches)}")
    return matches[0]


def list_cropped_photos(class_dir: Path) -> list[Path]:
    photos = sorted((class_dir / "cropped").glob("*__cropped.jpg"))
    if not photos:
        raise FileNotFoundError(f"No cropped photos found in {class_dir / 'cropped'}")
    return photos


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def split_photos_by_class(
    photos: list[Path], seed: int, class_idx: int, photos_per_split: dict[str, int]
) -> dict[str, list[Path]]:
    """Shuffles (not just slices in filename order) before splitting, since
    filenames encode capture timestamp and photos within one class's shoot
    could still carry a time-correlated drift -- see the lighting-drift
    finding that broke the original per-image crop heuristic across this
    same session. Shuffling avoids reintroducing a train/val/test split that
    quietly correlates with capture order."""
    n_train, n_val, n_test = photos_per_split["train"], photos_per_split["val"], photos_per_split["test"]
    if n_train + n_val + n_test != len(photos):
        raise ValueError(
            f"photos_per_split sums to {n_train + n_val + n_test} but found {len(photos)} photos"
        )
    rng = np.random.default_rng([seed, class_idx, 9999])  # distinct stream from patch-box sampling below
    shuffled = [photos[i] for i in rng.permutation(len(photos))]
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


class MultiPhotoPatchDataset(Dataset):
    """Like PatchCoffeeDataset, but for the box-rig dataset where each class
    has many already-cropped bean-tray photos (one subfolder per class)
    instead of a single circular-lens photo. Train/val/test are split at the
    *photo* level -- disjoint photos per split -- rather than spatial regions
    of one photo, so a split never shares a single photo's lighting/color
    with another split the way the original dataset had to.
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
        photos_per_split: dict[str, int],
        transform=None,
    ):
        assert split in ("train", "val", "test")
        self.split = split
        self.class_ids = class_ids
        self.resize = resize
        self.transform = transform
        self.class_labels = load_class_labels(classes_file)

        n_patches_total = patches_per_class[split]
        self._images: dict[tuple[str, str], np.ndarray] = {}  # (class_id, photo_name) -> rgb
        self._samples: list[tuple[str, str, Region]] = []  # (class_id, photo_name, box)

        for class_idx, class_id in enumerate(class_ids):
            class_dir = find_class_dir(dataset_dir, class_id)
            photos = list_cropped_photos(class_dir)
            split_photos = split_photos_by_class(photos, seed, class_idx, photos_per_split)[split]

            n_photos = len(split_photos)
            base, extra = divmod(n_patches_total, n_photos)
            for photo_idx, photo_path in enumerate(split_photos):
                n_patches = base + (1 if photo_idx < extra else 0)
                rgb = load_rgb_image(photo_path)
                h, w = rgb.shape[:2]
                region = compute_valid_region_rect(h, w, safety_margin)
                rng = np.random.default_rng([seed, class_idx, photo_idx, SPLIT_SEED_COMPONENT[split]])
                boxes = sample_patch_boxes(rng, region, n_patches, crop_size)

                key = (class_id, photo_path.name)
                self._images[key] = rgb
                self._samples.extend((class_id, photo_path.name, box) for box in boxes)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        class_id, photo_name, box = self._samples[idx]
        rgb = self._images[(class_id, photo_name)]
        patch = rgb[box.y0:box.y1, box.x0:box.x1]
        pil_patch = Image.fromarray(patch)
        label = self.class_ids.index(class_id)
        if self.transform is not None:
            tensor = self.transform(pil_patch)
        else:
            pil_patch = pil_patch.resize((self.resize, self.resize), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(pil_patch)).permute(2, 0, 1).float() / 255.0
        return tensor, label
