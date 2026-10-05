"""Photo loading and the patch-based PyTorch Dataset for the coffee classes (`coffeecv.class_list`)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pillow_avif  # noqa: F401 -- import alone registers AVIF with Pillow
import pillow_heif
import pillow_jxl  # noqa: F401 -- import alone registers JPEG XL with Pillow
import rawpy
import torch
from PIL import Image
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF

from coffeecv.bean_scale import estimate_bean_pitch
from coffeecv.class_list import ClassList
from coffeecv.geometry import (
    Region,
    assert_jitter_fits,
    compute_valid_region_rect,
    sample_bean_unit_patch_boxes,
    sample_masked_bean_unit_boxes,
    sample_patch_boxes,
    sample_rotated_patch_boxes,
    sample_scaled_patch_boxes,
)

pillow_heif.register_heif_opener()
# AVIF and JPEG XL (pillow_avif/pillow_jxl above) need no explicit register_opener()
# call -- each self-registers with Pillow's `Image.ID` on import, same effect as this
# line has for HEIF, just triggered differently by each library's own __init__.

CLASS_DIR_RE = re.compile(r"^class_(\d+)__")
# "all" is every photo of the dirs passed (build_capture_dataset, the DINOv3
# fixture); it gets its own stream component so its boxes don't coincide with
# the test split's.
SPLIT_SEED_COMPONENT = {"train": 0, "val": 1, "test": 2, "all": 3}


# ---- Multi-photo capture-dir dataset (2026-08-07__box_pictures_all_classes and later) ----


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


def bean_mask_path(cropped_photo: Path) -> Path:
    """The bean-region mask the segcrop stage writes beside a `*__cropped.jpg` (ticket ML-2): same stem,
    `__beanmask.png`, aligned pixel for pixel with the crop. Absent on a D18 fallback photo."""
    return cropped_photo.with_name(cropped_photo.name.replace("__cropped.jpg", "__beanmask.png"))


def load_bean_mask(cropped_photo: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """The bool mask beside a segmenter crop, or None if there is none (a D18 fallback photo). A mask
    whose shape is not the crop's is refused: it would place patches against the wrong pixels."""
    path = bean_mask_path(cropped_photo)
    if not path.exists():
        return None
    mask = np.array(Image.open(path).convert("1"), dtype=bool)
    if mask.shape != tuple(shape):
        raise ValueError(f"{path.name} is {mask.shape}, its crop {tuple(shape)}")
    return mask


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


# Major-brand camera RAW extensions LibRaw (via rawpy) decodes. Unlike every
# other format above, Pillow can't open these at all -- there's no opener to
# register, so load_rgb_image branches explicitly instead.
RAW_EXTENSIONS = {".dng", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def load_rgb_image(path: Path) -> np.ndarray:
    if path.suffix.lower() in RAW_EXTENSIONS:
        try:
            with rawpy.imread(str(path)) as raw:
                # output_bps=8: the rest of the pipeline (PIL.Image.fromarray in
                # patches_for_photo, bean_scale's luma weights) expects uint8,
                # not rawpy's 16-bit default. use_camera_wb: the color the camera
                # actually recorded, not a gray-world guess.
                return raw.postprocess(output_bps=8, use_camera_wb=True)
        except rawpy.LibRawError as exc:
            # Normalized to OSError so this joins the same "can't read this
            # photo" handling every other decode failure already gets
            # (patches_for_photo's try/except, webapp's /preview and
            # /classify) -- rawpy's own exceptions aren't OSError/ValueError.
            # Message is fixed rather than str(exc): LibRaw's own messages
            # come through as a bytes-repr ("b'Input/output error'"), which
            # would look broken surfaced straight to a user.
            raise OSError(f"unreadable RAW file ({type(exc).__name__})") from exc
    return np.array(Image.open(path).convert("RGB"))


@dataclass(frozen=True)
class PatchMeta:
    """Where one extracted patch came from."""
    class_id: str  # the class key the patch is labelled with (ClassList.keys): a country since ML-3
    capture: str  # the capture dir's name (Capture.name)
    photo_name: str
    box: Region
    angle: float
    side: int  # patch side in *source* pixels before storage resize; varies under scale aug
    folder_id: str  # the class folder (coffee) the photo came from; equals class_id for per-folder classes


# The capture dirs a run trains on: every data/cropped/cam_* pool, each built by
# its own merge_cam_* stage in dvc.yaml from one camera's sessions. Formerly
# run_folds.RIGS, the fold rotation; moved here when the fold driver was retired
# (ticket ML-1, 2026-09-29). run_all_rigs.py and fit_frozen_head.py train on
# exactly this list, and params.yaml rests at it.
#
# The order is not cosmetic: a dir's position here is the `capture_idx` that
# seeds its patch boxes (MultiPhotoPatchDataset._extract_photo). It no longer
# decides which photos land in which split -- that is pooled per class and
# order-independent (split_photos_by_class).
#
# cam_iphone carries eight classes (no class_008, no class_010); the other three
# carry all ten.
CAPTURES = [
    "data/cropped/cam_pixel",
    "data/cropped/cam_sony",
    "data/cropped/cam_oneplus",
    "data/cropped/cam_iphone",
]
# Ticket ML-2: the same four pools built from the segmenter's crops (segcrop@<session> -> merge_segcam_*),
# in the same order, so a seed draws the same photo split and the same capture_idx on both sets of pools:
# the photo names are the raw stems on both sides, and Capture.name is the basename.
SEG_CAPTURES = [
    "data/segcropped/cam_pixel",
    "data/segcropped/cam_sony",
    "data/segcropped/cam_oneplus",
    "data/segcropped/cam_iphone",
]


def bean_share_rule(cfg) -> tuple[float, int] | None:
    """(patch_min_bean_share, patch_max_attempts_factor) for a run on the segmenter's pools, where
    MultiPhotoPatchDataset places patches by the D17 rule against each photo's mask; None on the tray
    heuristic's pools, whose crops are all-bean rectangles with no mask."""
    if cfg.crop_method == "segment":
        return cfg.patch_min_bean_share, cfg.patch_max_attempts_factor
    if cfg.crop_method != "tray_heuristic":
        raise ValueError(f"unknown crop_method {cfg.crop_method!r}")
    return None


@dataclass(frozen=True)
class Capture:
    """One capture dir under data/cropped/: a name and where its crops live.

    A capture is a pool of photos defined by camera + date + setup + lighting --
    deliberately not "a camera": a future dir need not map 1:1 to a physical
    camera, even though today's four cam_* dirs each merge one camera's
    sessions. It is not a label and not a unit of evaluation. Since ticket ML-1
    (2026-09-29) no capture is held out, trained on selectively or reported
    separately; photos are pooled across captures and split per class only
    (`split_photos_by_class`, D1(b)). A capture survives as the patch budget's
    unit, as a box-RNG seed component, and as provenance in PatchMeta.
    """
    name: str
    cropped_dir: Path


def resolve_captures(cropped_dirs: list[Path]) -> list[Capture]:
    """`.../data/cropped/<name>` -> Capture(name=<name>)."""
    captures = []
    for path in cropped_dirs:
        if not path.is_dir():
            raise FileNotFoundError(
                f"No cropped capture dir at {path}. Run the crop stage first: `dvc repro crop`."
            )
        captures.append(Capture(name=path.name, cropped_dir=path))
    if not captures:
        raise ValueError("At least one capture dir is required")
    # The name is the capture half of the pooled split's sort key (CapturePhoto),
    # so two dirs sharing a basename would make that key ambiguous.
    names = [c.name for c in captures]
    if len(set(names)) != len(names):
        raise ValueError(f"capture dir names must be unique, got {names}")
    return captures


@dataclass(frozen=True, order=True)
class CapturePhoto:
    """One cropped photo in its class's pool, tagged with the capture dir it came from.

    Ordered (and compared, and hashed) by `(capture, name)` only: that is the global
    key the pooled split sorts on before it shuffles, so which split a photo lands in
    depends on the seed, the class and the set of photos -- never on the order the
    capture dirs happen to be listed in config.
    """
    capture: str  # Capture.name, the capture dir's basename
    name: str     # photo file name
    path: Path = field(compare=False)
    folder_id: str | None = field(default=None, compare=False)  # its class folder (coffee)


def pooled_class_photos(captures: list[Capture], folder_ids) -> tuple[list[CapturePhoto], list[str]]:
    """(every cropped photo of one class -- all of `folder_ids`, its class folders -- across `captures`,
    sorted by the global key; the names of the captures with none of those folders at all).

    A class pools several folders since ML-3 (a country's coffees, `ClassList.folders`). Within one
    capture they share one pool and one patch budget (D6). The split key stays (capture, photo name), so
    a name repeated across the folders makes `split_photos_by_class` refuse the pool.

    A dir lacking a class is normal (cam_iphone has no class_008 or class_010). A class
    directory that exists but holds no crops is not, and still raises from
    `list_cropped_photos` -- that means the crop stage has not run.
    """
    if isinstance(folder_ids, str):
        raise TypeError("pooled_class_photos takes a class's folder ids (ClassList.folders[key]), not one id")
    pool: list[CapturePhoto] = []
    absent: list[str] = []
    for capture in captures:
        found = False
        for folder_id in folder_ids:
            try:
                class_dir = find_class_dir(capture.cropped_dir, folder_id)
            except FileNotFoundError:
                continue
            found = True
            pool.extend(CapturePhoto(capture.name, p.name, p, folder_id) for p in list_cropped_photos(class_dir))
        if not found:
            absent.append(capture.name)
    return sorted(pool), absent


def split_photos_by_class(
    photos: list[CapturePhoto], seed: int, class_idx: int, photo_frac: dict[str, float],
) -> dict[str, list[CapturePhoto]]:
    """Split one class's photos, pooled across every capture dir, into train/val/test.

    Pooled (ticket ML-1, D1(b), 2026-09-29): the split is computed once per class
    over the union of all capture dirs' photos, not once per (capture dir, class) as
    before. No camera is guaranteed a share of val or test any more -- a low-count
    dir can land zero photos in a split for some class, which MultiPhotoPatchDataset
    records and warns about rather than crashing (see its `starved`).

    Order-independent by construction: the pool is sorted by the global
    `(capture, name)` key *before* the seeded shuffle, and the shuffle is seeded on
    `(seed, class_idx)` alone. Reordering `train_capture_dirs` therefore cannot move a
    photo between splits. Each split is returned sorted by the same key, so any
    per-dir subset a caller filters out of it is in a stable order too.

    Shuffles (not just slices in filename order) before splitting, since filenames
    encode capture timestamp and photos within one class's shoot could still carry a
    time-correlated drift -- see the lighting-drift finding that broke the original
    per-image crop heuristic. Shuffling avoids a train/val/test split that quietly
    correlates with capture order.

    Fractional, not a fixed count: val and test are each a fraction of however many
    photos the pool holds, floored at 1 so every class gets real val/test coverage;
    train gets the remainder.
    """
    pool = sorted(photos)
    if len(set(pool)) != len(pool):
        raise ValueError("duplicate (capture, photo name) in the pool -- the split key would be ambiguous")
    n = len(pool)
    n_val = max(1, round(photo_frac["val"] * n))
    n_test = max(1, round(photo_frac["test"] * n))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"{n} photos is too few to split at fractions {photo_frac} "
            f"(train would be {n_train}); need at least 3 photos of a class across all capture dirs"
        )
    # 9999 keeps this stream distinct from patch-box sampling (_extract_photo).
    rng = np.random.default_rng([seed, class_idx, 9999])
    shuffled = [pool[i] for i in rng.permutation(n)]
    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }


def split_census(captures: list[Capture], classes: ClassList, seed: int,
                 photo_frac: dict[str, float]) -> dict[str, dict]:
    """Photo counts of the pooled split, per class, per split, per capture dir -- no
    image is decoded, so this is cheap enough to run anywhere.

    `{class key: {"pooled": n, "absent": [dirs without the class],
                 "train"|"val"|"test": {capture: n_photos, ...}}}`, with every dir that
    has the class present in each split's dict, zero included. A zero there is a dir
    the pooled split starved out of that split for that class: legitimate under D1(b),
    but it should be seen, not discovered later. `coffeecv.split_report` prints this.
    """
    census: dict[str, dict] = {}
    for class_idx, key in enumerate(classes.keys):
        pool, absent = pooled_class_photos(captures, classes.folders[key])
        entry: dict = {"pooled": len(pool), "absent": absent}
        have = [c.name for c in captures if c.name not in absent]
        if pool:
            for split, chosen in split_photos_by_class(pool, seed, class_idx, photo_frac).items():
                entry[split] = {name: sum(1 for p in chosen if p.capture == name) for name in have}
        census[key] = entry
    return census


class MultiPhotoPatchDataset(Dataset):
    """Patch dataset over one or more capture dirs.

    Each class has many already-"cropped" photos per dir (one subfolder per
    class; for frame-filling dirs the crop stage is a byte-copy passthrough).
    Train/val/test are split at the *photo* level -- disjoint photos per split --
    rather than by spatial region of one photo, so a split never shares a single
    photo's lighting/colour with another split. The photo split is pooled across
    every dir per class (`split_photos_by_class`, ticket ML-1 D1(b)); the patch
    budget is still spent per (dir, class), over that dir's photos in the split.

    Patches are materialised at construction rather than held as whole photos.
    That is forced by photo size: the 2026-08-09 sessions decode to 37 MB and 57 MB
    per photo, so the previous "keep every photo in RAM" approach needed 5.5 GB to
    train on two of them and 10.4 GB to evaluate on sony_cam, against ~8 GB
    available. Extracting each photo's patches and then dropping the photo caps
    peak memory at one photo plus the patch store. This changes no semantics:
    patch boxes were always fixed at construction, so nothing that varies per
    epoch is being frozen here -- only the photometric transforms vary per epoch,
    and those still run in __getitem__.

    `patch_store_size` is the edge length patches are kept at. None means "keep
    the full crop_size", which is what a single-dir run should use to stay
    comparable with pre-Phase-11 experiments; multi-dir runs set it to bound
    memory. It must stay comfortably above `resize` so downstream zoom
    augmentation crops into real detail instead of upsampling.
    """

    def __init__(
        self,
        captures: list[Capture],
        classes: ClassList,
        split: str,
        seed: int,
        crop_size: int,
        resize: int,
        safety_margin: float,
        patches_per_class: dict[str, int],
        photo_frac: dict[str, float],
        transform=None,
        rotation_jitter_degrees: float = 0.0,
        patch_store_size: int | None = None,
        patch_scale_frac: tuple[float, float] | None = None,
        patch_beans: tuple[float, float] | None = None,
        pitch_geometry: dict | None = None,
        bean_share_rule: tuple[float, int] | None = None,
    ):
        assert split in ("train", "val", "test", "all")
        self.split = split
        self.captures = captures
        self.classes = classes
        self.class_ids = list(classes.keys)
        self.resize = resize
        self.crop_size = crop_size
        self.transform = transform
        self.patch_store_size = patch_store_size
        # Bean-pitch estimator geometry, from RunConfig via bean_scale.pitch_kwargs.
        # Empty dict = the module defaults, i.e. every run before 2026-09-03.
        self.pitch_geometry = pitch_geometry or {}
        # Scale augmentation applies to *every* split, not just train. That is
        # the opposite of the usual rule, and deliberate: the patch side is what
        # decides how many beans a patch covers, so evaluating at one fixed pixel
        # size would score each capture at a different bean coverage, and a val
        # or test number would partly measure magnification rather than the
        # model. Eval draws are seeded, so they stay deterministic.
        # Bean-unit sizing takes precedence: it is the only mode that is
        # measurable at inference, since it needs no knowledge of how the photo
        # was framed. See coffeecv/bean_scale.py.
        self.patch_beans = patch_beans
        self.n_clamped = 0
        self.pitch_by_photo: dict[str, float] = {}
        self.patch_scale_frac = patch_scale_frac
        # Ticket ML-2 D17/D18 (`bean_share_rule(cfg)`): on the segmenter's pools each photo's patches are
        # placed against its `__beanmask.png`, exactly as infer.patches_for_photo places them against the
        # mask it computes live. A photo with no mask is a D18 fallback and accepts every box.
        self.bean_share_rule = bean_share_rule
        self.n_below_share = 0
        self.fallback_photos: list[str] = []
        if bean_share_rule is not None and patch_beans is None:
            raise ValueError("bean_share_rule (crop_method 'segment') needs bean-unit patch sizing (patch_beans)")
        if patch_beans is not None and patch_store_size is None:
            raise ValueError("patch_beans requires patch_store_size to be set")
        if patch_scale_frac is not None and patch_store_size is None:
            # Sides then vary from ~170px to ~2275px across captures; storing them at
            # native size would make memory depend on the draw (a single 2275px
            # patch is 15 MB).
            raise ValueError("patch_scale_frac requires patch_store_size to be set")
        # Train-only, like every other augmentation: val/test stay deterministic.
        self.rotation_jitter_degrees = rotation_jitter_degrees if split == "train" else 0.0

        self.class_labels = dict(classes.labels)

        # Budget is per class *per capture dir*, so adding a dir adds data rather
        # than diluting the existing dirs' share of a fixed total. Each dir spends
        # its budget over its own photos in this split -- which dir a photo came
        # from no longer decides *which* split it is in (that is pooled), only how
        # the patches are apportioned once it is there.
        n_patches_total = patches_per_class[split]
        self._patches: list[np.ndarray] = []
        # Provenance per patch. The photo itself is dropped after extraction, so
        # this is the only remaining record of where a patch came from -- needed
        # by check_augmentation.py and worth having when a patch looks wrong.
        self._meta: list[PatchMeta] = []
        # (dir, class_id) pairs where the dir has no directory for the class at all
        # (cam_iphone has no class_008 or class_010). Tolerated: the other dirs
        # still supply the class.
        self.absent: list[tuple[str, str]] = []
        # (dir, class_id) pairs where the dir HAS photos of the class but the pooled
        # split put none of them in this split, so the dir contributes no patches
        # to it for that class. Legitimate under D1(b) -- a 10-photo dir pooled with
        # three 20-photo dirs can miss a 15% split by chance -- but warned and kept
        # here, so it is visible in the run log rather than discovered later.
        self.starved: list[tuple[str, str]] = []
        # Classes with no photos in ANY dir passed. A total loss for train/val/test
        # (raised below); tolerated for split="all", where one dir is scored as it is.
        self.missing_classes: list[str] = []

        for class_idx, class_id in enumerate(self.class_ids):
            pool, absent = pooled_class_photos(captures, classes.folders[class_id])
            for name in absent:
                self.absent.append((name, class_id))
                print(f"  WARNING: {name} has no class {class_id} -- it contributes nothing to "
                      f"that class ({split} split)")
            if not pool:
                self.missing_classes.append(class_id)
                continue
            if split == "all":
                # Every photo of every dir passed; nothing is withheld.
                selected = pool
            else:
                selected = split_photos_by_class(pool, seed, class_idx, photo_frac)[split]

            for capture_idx, capture in enumerate(captures):
                if capture.name in absent:
                    continue
                # A sorted list filtered stays sorted: photo_idx below is this
                # photo's rank among the dir's own photos in the split, by name,
                # and _extract_photo's box RNG is keyed on it.
                mine = [p for p in selected if p.capture == capture.name]
                if not mine:
                    self.starved.append((capture.name, class_id))
                    n_have = sum(1 for p in pool if p.capture == capture.name)
                    print(f"  WARNING: {capture.name} has {n_have} photo(s) of class {class_id}, but the "
                          f"pooled split put none in {split} ({len(selected)} of {len(pool)} pooled "
                          f"photos) -- it contributes no {split} patches for this class")
                    continue
                base, extra = divmod(n_patches_total, len(mine))
                for photo_idx, photo in enumerate(mine):
                    n_patches = base + (1 if photo_idx < extra else 0)
                    self._extract_photo(
                        photo.path, n_patches, seed, capture_idx, class_idx, photo_idx,
                        class_id, photo.folder_id, capture.name, crop_size, safety_margin,
                    )

        # For train/val/test a class absent from EVERY dir really is a total loss
        # (the model would never see it), which is different from "absent from some
        # of several dirs" -- that is `absent` above, and fine.
        if split != "all" and self.missing_classes:
            raise ValueError(
                f"class(es) {self.missing_classes} not found in ANY of the capture dirs passed to "
                f"this dataset ({[c.name for c in captures]}) -- every class must exist in at least one "
                f"dir, or the model can never see it at all. A class missing from SOME but not all "
                f"dirs is fine (see `absent`); this is a total loss."
            )

        # Indices into class_ids that this dataset actually has data for -- lets
        # the metrics layer exclude a never-present class from a macro average
        # instead of scoring it as a phantom F1=0 (see compute_split_metrics's
        # macro_labels). Equal to every index for train/val/test.
        self.present_class_idxs = [
            i for i, c in enumerate(self.class_ids) if c not in self.missing_classes
        ]

    def _extract_photo(
        self, photo_path: Path, n_patches: int, seed: int, capture_idx: int, class_idx: int,
        photo_idx: int, class_id: str, folder_id: str, capture_name: str, crop_size: int,
        safety_margin: float,
    ) -> None:
        """Load one photo, cut its patches out, and let the photo go.

        The box RNG key `[seed, capture_idx, class_idx, photo_idx, split]` is unchanged by
        ticket ML-1 (formerly `rig_idx`; same value, new name)."""
        rgb = load_rgb_image(photo_path)
        h, w = rgb.shape[:2]
        region = compute_valid_region_rect(h, w, safety_margin)
        rng = np.random.default_rng(
            [seed, capture_idx, class_idx, photo_idx, SPLIT_SEED_COMPONENT[self.split]]
        )
        if self.patch_beans is not None:
            # Estimated per photo, never per session: at inference there is only
            # one photo, so a session-level estimate here would train the model
            # on a precision it will not have in the field.
            gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114)
            pitch = estimate_bean_pitch(gray.astype(np.uint8), **self.pitch_geometry)
            self.pitch_by_photo[photo_path.name] = pitch
            if self.bean_share_rule is not None:
                mask = load_bean_mask(photo_path, rgb.shape[:2])
                if mask is None:
                    self.fallback_photos.append(photo_path.name)
                boxes, clamped, below = sample_masked_bean_unit_boxes(
                    rng, region, n_patches, pitch, self.patch_beans[0], self.patch_beans[1],
                    mask, *self.bean_share_rule, self.rotation_jitter_degrees,
                )
                self.n_below_share += below
            else:
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
            self._meta.append(PatchMeta(class_id, capture_name, photo_path.name, box, angle, side, folder_id))
        del rgb

    def __len__(self) -> int:
        return len(self._patches)

    def capture_names(self) -> list[str]:
        """Per-sample capture dir name (provenance; no metric is broken down by it)."""
        return [m.capture for m in self._meta]

    def __getitem__(self, idx: int):
        pil_patch = Image.fromarray(self._patches[idx])
        label = self.classes.index(self._meta[idx].class_id)
        if self.transform is not None:
            tensor = self.transform(pil_patch)
        else:
            pil_patch = pil_patch.resize((self.resize, self.resize), Image.BILINEAR)
            tensor = torch.from_numpy(np.array(pil_patch)).permute(2, 0, 1).float() / 255.0
        return tensor, label
