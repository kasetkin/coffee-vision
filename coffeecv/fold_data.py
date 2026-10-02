"""The train/val/test datasets a run trains and scores on, built one way for every caller.

Moved out of `train_baseline.main()` on 2026-09-24, verbatim, so that a second consumer -- the
frozen-backbone fit (`fit_frozen_head`) -- gets byte-identical patches instead of re-typing the
construction. Re-typing it is not a safe shortcut here: `capture_idx`, a capture dir's position in the
config's list, seeds every patch box (`MultiPhotoPatchDataset._extract_photo`), so a caller that builds
each dir on its own, or in another order, silently draws different patches. The 2026-09-21 backbone
screen did exactly that. See docs/dinov3_integration_plan.md §0.1 and §4.1.

Which *photos* land in which split no longer depends on that position: since ticket ML-1 (D1(b),
2026-09-29) the photo split is pooled across dirs per class and seeded on (seed, class) alone
(`dataset.split_photos_by_class`). The leave-one-camera-out cross-rig set this module used to build is
gone with the fold protocol; the names (`build_fold_datasets`, `FoldDatasets`) are kept.
"""
from __future__ import annotations

from dataclasses import dataclass

from coffeecv.bean_scale import pitch_kwargs
from coffeecv.config import RunConfig
from coffeecv.dataset import (Capture, MultiPhotoPatchDataset, bean_share_rule, load_class_labels,
                             resolve_captures)
from coffeecv.transforms import build_eval_transform


@dataclass
class FoldDatasets:
    train: MultiPhotoPatchDataset | None  # None only when the caller asked for other splits (`only`)
    val: MultiPhotoPatchDataset | None
    test: MultiPhotoPatchDataset | None
    captures: list[Capture]
    class_ids: list[str]
    class_labels: dict[str, str]


def build_fold_datasets(cfg: RunConfig, train_transform, eval_transform, *,
                        only: tuple[str, ...] | None = None) -> FoldDatasets:
    """Build train/val/test from `cfg.train_capture_dirs` exactly as `train_baseline` trains on them.

    `train_transform` is applied to the train split only. A frozen-feature caller passes the eval
    transform for both, which changes what `__getitem__` returns but not which boxes are drawn:
    boxes are sampled once, at construction, from seeded generators that never see the transform.

    `only` builds just the named splits ("train", "val", "test") and leaves the rest None. The photo
    split is a pure function of (seed, class, photo pool) and each box draws from a generator keyed on
    (seed, capture_idx, class, photo, split), never from a shared stream, so a split built alone is
    byte-identical to the same split built with the others -- and a caller that needs one split at a
    time holds one split's decoded patches in memory, not three.
    """
    want = set(only) if only is not None else {"train", "val", "test"}
    unknown = want - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"unknown split(s) {sorted(unknown)}; want train, val and/or test")
    capture_dirs, classes_file = cfg.resolve_paths()
    captures = resolve_captures(capture_dirs)
    class_labels = load_class_labels(classes_file)
    class_ids = sorted(class_labels)
    patches_per_class = {
        "train": cfg.train_patches_per_class,
        "val": cfg.val_patches_per_class,
        "test": cfg.test_patches_per_class,
    }
    photo_frac = {
        "train": cfg.train_photo_frac,
        "val": cfg.val_photo_frac,
        "test": cfg.test_photo_frac,
    }

    common_kwargs = dict(
        captures=captures,
        classes_file=classes_file,
        class_ids=class_ids,
        seed=cfg.seed,
        crop_size=cfg.patch_crop_size,
        resize=cfg.patch_resize,
        safety_margin=cfg.safety_margin,
        patches_per_class=patches_per_class,
        photo_frac=photo_frac,
        patch_store_size=cfg.patch_store_size or None,
        patch_scale_frac=(
            (cfg.patch_scale_frac_min, cfg.patch_scale_frac_max)
            if cfg.patch_scale_frac_max > 0 else None
        ),
        patch_beans=(
            (cfg.patch_beans_min, cfg.patch_beans_max)
            if cfg.patch_beans_max > 0 else None
        ),
        pitch_geometry=pitch_kwargs(cfg),
        bean_share_rule=bean_share_rule(cfg),
    )
    train_ds = MultiPhotoPatchDataset(
        split="train", transform=train_transform,
        rotation_jitter_degrees=cfg.rotation_jitter_degrees, **common_kwargs,
    ) if "train" in want else None
    val_ds = (MultiPhotoPatchDataset(split="val", transform=eval_transform, **common_kwargs)
              if "val" in want else None)
    test_ds = (MultiPhotoPatchDataset(split="test", transform=eval_transform, **common_kwargs)
               if "test" in want else None)
    return FoldDatasets(train_ds, val_ds, test_ds, captures, class_ids, class_labels)


def build_capture_dataset(cfg: RunConfig, capture: Capture, class_ids: list[str], classes_file,
                          n_patches: int) -> MultiPhotoPatchDataset:
    """Every photo of one capture dir, `split="all"`, eval transform, `cfg`'s own patch geometry.

    No holdout semantics: this is the generic "draw n_patches per class from all of one dir's photos"
    builder. It feeds the DINOv3 backbone-equivalence fixture (`coffeecv_dino.reference`), whose
    patches must stay byte-identical to the ones `scripts/make_dinov3_fixture.py` recorded -- so the
    construction, the "all" split's seed component and the single-dir `capture_idx` of 0 must not move.
    Moved here on 2026-09-29 (ticket ML-1) from the retired cross-camera evaluator, construction unchanged.
    """
    return MultiPhotoPatchDataset(
        split="all",
        transform=build_eval_transform(cfg.patch_resize),
        captures=[capture],
        classes_file=classes_file,
        class_ids=class_ids,
        seed=cfg.seed,
        crop_size=cfg.patch_crop_size,
        resize=cfg.patch_resize,
        safety_margin=cfg.safety_margin,
        patches_per_class={
            "train": cfg.train_patches_per_class, "val": cfg.val_patches_per_class,
            "test": cfg.test_patches_per_class, "all": n_patches,
        },
        photo_frac={
            "train": cfg.train_photo_frac, "val": cfg.val_photo_frac,
            "test": cfg.test_photo_frac,
        },
        patch_store_size=cfg.patch_store_size or None,
        patch_scale_frac=((cfg.patch_scale_frac_min, cfg.patch_scale_frac_max)
                          if cfg.patch_scale_frac_max > 0 else None),
        patch_beans=((cfg.patch_beans_min, cfg.patch_beans_max)
                     if cfg.patch_beans_max > 0 else None),
        pitch_geometry=pitch_kwargs(cfg),
        bean_share_rule=bean_share_rule(cfg),
    )
