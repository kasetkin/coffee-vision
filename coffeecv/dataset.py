"""HEIF loading and patch-based PyTorch Dataset for the coffee bean classes."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pillow_heif
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

from coffeecv.bean_scale import estimate_bean_pitch
from coffeecv.geometry import (
    Region,
    assert_jitter_fits,
    assert_region_fully_opaque,
    compute_valid_region,
    compute_valid_region_rect,
    sample_bean_unit_patch_boxes,
    sample_patch_boxes,
    sample_rotated_patch_boxes,
    sample_scaled_patch_boxes,
    split_regions,
)

pillow_heif.register_heif_opener()

CLASS_FILENAME_RE = re.compile(r"class=(\d+)\.heif$", re.IGNORECASE)
CLASS_DIR_RE = re.compile(r"^class_(\d+)__")
# "all" is the held-out rig's whole-rig test split; it gets its own stream
# component so its boxes don't coincide with the in-distribution test split's.
SPLIT_SEED_COMPONENT = {"train": 0, "val": 1, "test": 2, "all": 3}


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


def discover_classes_multi(cropped_dir: Path) -> list[str]:
    ids = set()
    for p in Path(cropped_dir).iterdir():
        if p.is_dir():
            m = CLASS_DIR_RE.match(p.name)
            if m:
                ids.add(m.group(1))
    if not ids:
        raise FileNotFoundError(f"No 'class_NNN__Label' directories found under {cropped_dir}")
    return sorted(ids)


def find_class_dir(cropped_dir: Path, class_id: str) -> Path:
    matches = [p for p in Path(cropped_dir).iterdir() if p.is_dir() and p.name.startswith(f"class_{class_id}__")]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one directory for class={class_id}, found {len(matches)}")
    return matches[0]


def list_cropped_photos(class_dir: Path) -> list[Path]:
    """`class_dir` lives under the *cropped* root (`data/cropped/<session>/`), not
    the raw session directory — the crops are a pipeline output produced by the
    `crop` stage, while the raw photos stay `dvc add`-tracked data. Sibling files
    such as `crop_report.json` are excluded by the `*__cropped.jpg` glob."""
    photos = sorted(class_dir.glob("*__cropped.jpg"))
    if not photos:
        raise FileNotFoundError(
            f"No cropped photos found in {class_dir}. Run the crop stage first: "
            f"`dvc repro crop` (or `python -m coffeecv.crop_session --session <name>`)."
        )
    return photos


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


@dataclass(frozen=True)
class PatchMeta:
    """Where one extracted patch came from."""
    class_id: str
    rig_name: str
    photo_name: str
    box: Region
    angle: float
    side: int  # patch side in *source* pixels before storage resize; varies under scale aug


@dataclass(frozen=True)
class Rig:
    """One capture rig (== one session): a camera + framing + lighting setup.

    The rig is a first-class dimension because cross-rig transfer is the thing
    being measured. Photos are split into train/val/test *within* each rig, so
    every split keeps the intended proportion of every training rig rather than
    letting a pooled shuffle hand one rig most of the val set.
    """
    name: str
    cropped_dir: Path


def resolve_rigs(cropped_dirs: list[Path]) -> list[Rig]:
    """`.../data/cropped/<session>` -> Rig(name=<session>)."""
    rigs = []
    for path in cropped_dirs:
        if not path.is_dir():
            raise FileNotFoundError(
                f"No cropped rig at {path}. Run the crop stage first: `dvc repro crop`."
            )
        rigs.append(Rig(name=path.name, cropped_dir=path))
    if not rigs:
        raise ValueError("At least one rig is required")
    return rigs


def split_photos_by_class(
    photos: list[Path], seed: int, class_idx: int, photos_per_split: dict[str, int],
    rig_idx: int = 0,
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
    # rig_idx keeps each rig's photo shuffle independent; 9999 keeps this stream
    # distinct from patch-box sampling below.
    rng = np.random.default_rng([seed, rig_idx, class_idx, 9999])
    shuffled = [photos[i] for i in rng.permutation(len(photos))]
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:n_train + n_val + n_test],
    }


class MultiPhotoPatchDataset(Dataset):
    """Patch dataset over one or more capture rigs.

    Each class has many already-"cropped" photos per rig (one subfolder per
    class; for frame-filling rigs the crop stage is a byte-copy passthrough).
    Train/val/test are split at the *photo* level -- disjoint photos per split,
    within each rig -- rather than by spatial region of one photo, so a split
    never shares a single photo's lighting/colour with another split.

    Patches are materialised at construction rather than held as whole photos.
    That is forced by rig size: the 2026-08-09 rigs decode to 37 MB and 57 MB per
    photo, so the previous "keep every photo in RAM" approach needed 5.5 GB to
    train on two rigs and 10.4 GB to evaluate on sony_cam, against ~8 GB
    available. Extracting each photo's patches and then dropping the photo caps
    peak memory at one photo plus the patch store. This changes no semantics:
    patch boxes were always fixed at construction, so nothing that varies per
    epoch is being frozen here -- only the photometric transforms vary per epoch,
    and those still run in __getitem__.

    `patch_store_size` is the edge length patches are kept at. None means "keep
    the full crop_size", which is what a single-rig run should use to stay
    comparable with pre-Phase-11 experiments; multi-rig runs set it to bound
    memory. It must stay comfortably above `resize` so downstream zoom
    augmentation crops into real detail instead of upsampling.
    """

    def __init__(
        self,
        rigs: list[Rig],
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
        rotation_jitter_degrees: float = 0.0,
        patch_store_size: int | None = None,
        patch_scale_frac: tuple[float, float] | None = None,
        patch_beans: tuple[float, float] | None = None,
        return_domain_id: bool = False,
    ):
        assert split in ("train", "val", "test", "all")
        self.split = split
        self.rigs = rigs
        # Cross-rig MixStyle (mixstyle_mode="cross_rig") needs a per-sample rig id
        # at train time to restrict the mixing partner to a different rig. Default
        # False keeps __getitem__'s return arity unchanged for every other caller
        # (val/test/xrig loaders, evaluate()) -- no regression risk there.
        self.return_domain_id = return_domain_id
        self._rig_name_to_domain_id = {r.name: i for i, r in enumerate(rigs)}
        self.class_ids = class_ids
        self.resize = resize
        self.crop_size = crop_size
        self.transform = transform
        self.patch_store_size = patch_store_size
        # Scale augmentation applies to *every* split, not just train. That is
        # the opposite of the usual rule, and deliberate: the patch side is what
        # decides how many beans a patch covers, so evaluating at one fixed pixel
        # size would score each rig at a different bean coverage and make the
        # cross-rig number a measurement of magnification rather than of the
        # model. Eval draws are seeded, so they stay deterministic.
        # Bean-unit sizing takes precedence: it is the only mode that is
        # measurable at inference, since it needs no knowledge of how the rig
        # was framed. See coffeecv/bean_scale.py.
        self.patch_beans = patch_beans
        self.n_clamped = 0
        self.pitch_by_photo: dict[str, float] = {}
        self.patch_scale_frac = patch_scale_frac
        if patch_beans is not None and patch_store_size is None:
            raise ValueError("patch_beans requires patch_store_size to be set")
        if patch_scale_frac is not None and patch_store_size is None:
            # Sides then vary from ~170px to ~2275px across rigs; storing them at
            # native size would make memory depend on the draw (a single 2275px
            # patch is 15 MB).
            raise ValueError("patch_scale_frac requires patch_store_size to be set")
        # Train-only, like every other augmentation: val/test stay deterministic.
        self.rotation_jitter_degrees = rotation_jitter_degrees if split == "train" else 0.0

        self.class_labels = load_class_labels(classes_file)

        # Budget is per class *per rig*, so adding a rig adds data rather than
        # diluting the existing rigs' share of a fixed total.
        n_patches_total = patches_per_class[split]
        self._patches: list[np.ndarray] = []
        # Provenance per patch. The photo itself is dropped after extraction, so
        # this is the only remaining record of where a patch came from -- needed
        # by check_augmentation.py and worth having when a patch looks wrong.
        self._meta: list[PatchMeta] = []

        for rig_idx, rig in enumerate(rigs):
            for class_idx, class_id in enumerate(class_ids):
                class_dir = find_class_dir(rig.cropped_dir, class_id)
                photos = list_cropped_photos(class_dir)
                if split == "all":
                    # Held-out rig: every photo is test data, nothing is withheld.
                    selected = photos
                else:
                    selected = split_photos_by_class(
                        photos, seed, class_idx, photos_per_split, rig_idx
                    )[split]

                base, extra = divmod(n_patches_total, len(selected))
                for photo_idx, photo_path in enumerate(selected):
                    n_patches = base + (1 if photo_idx < extra else 0)
                    self._extract_photo(
                        photo_path, n_patches, seed, rig_idx, class_idx, photo_idx,
                        class_id, rig.name, crop_size, safety_margin,
                    )

    def _extract_photo(
        self, photo_path: Path, n_patches: int, seed: int, rig_idx: int, class_idx: int,
        photo_idx: int, class_id: str, rig_name: str, crop_size: int, safety_margin: float,
    ) -> None:
        """Load one photo, cut its patches out, and let the photo go."""
        rgb = load_rgb_image(photo_path)
        h, w = rgb.shape[:2]
        region = compute_valid_region_rect(h, w, safety_margin)
        rng = np.random.default_rng(
            [seed, rig_idx, class_idx, photo_idx, SPLIT_SEED_COMPONENT[self.split]]
        )
        if self.patch_beans is not None:
            # Estimated per photo, never per session: at inference there is only
            # one photo, so a session-level estimate here would train the model
            # on a precision it will not have in the field.
            gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114)
            pitch = estimate_bean_pitch(gray.astype(np.uint8))
            self.pitch_by_photo[photo_path.name] = pitch
            boxes, clamped = sample_bean_unit_patch_boxes(
                rng, region, n_patches, pitch, self.patch_beans[0], self.patch_beans[1],
                self.rotation_jitter_degrees,
            )
            self.n_clamped += clamped
        elif self.patch_scale_frac is not None:
            frac_min, frac_max = self.patch_scale_frac
            boxes = sample_scaled_patch_boxes(
                rng, region, n_patches, frac_min, frac_max, self.rotation_jitter_degrees
            )
        elif self.rotation_jitter_degrees > 0:
            assert_jitter_fits(region, crop_size, self.rotation_jitter_degrees)
            boxes = [
                (box, angle, crop_size)
                for box, angle in sample_rotated_patch_boxes(
                    rng, region, n_patches, crop_size, self.rotation_jitter_degrees
                )
            ]
        else:
            boxes = [
                (box, 0.0, crop_size)
                for box in sample_patch_boxes(rng, region, n_patches, crop_size)
            ]

        for box, angle, side in boxes:
            patch = Image.fromarray(rgb[box.y0:box.y1, box.x0:box.x1])
            if angle:
                # `box` is the bounding box of the rotated crop, so rotating it
                # and centre-cropping back lands entirely on real pixels -- no fill.
                patch = TF.center_crop(
                    TF.rotate(patch, angle, interpolation=TF.InterpolationMode.BILINEAR),
                    [side, side],
                )
            if self.patch_store_size is not None and patch.size[0] != self.patch_store_size:
                patch = patch.resize(
                    (self.patch_store_size, self.patch_store_size), Image.BILINEAR
                )
            self._patches.append(np.asarray(patch, dtype=np.uint8))
            self._meta.append(PatchMeta(class_id, rig_name, photo_path.name, box, angle, side))
        del rgb

    def __len__(self) -> int:
        return len(self._patches)

    def rig_names(self) -> list[str]:
        """Per-sample rig name, for reporting metrics broken down by rig."""
        return [m.rig_name for m in self._meta]

    def __getitem__(self, idx: int):
        pil_patch = Image.fromarray(self._patches[idx])
        label = self.class_ids.index(self._meta[idx].class_id)
        if self.transform is not None:
            tensor = self.transform(pil_patch)
        else:
            pil_patch = pil_patch.resize((self.resize, self.resize), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(pil_patch)).permute(2, 0, 1).float() / 255.0
        if self.return_domain_id:
            domain_id = self._rig_name_to_domain_id[self._meta[idx].rig_name]
            return tensor, label, domain_id
        return tensor, label
