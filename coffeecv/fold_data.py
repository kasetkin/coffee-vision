"""The four datasets one leave-one-rig-out fold trains and scores on, built one way for every caller.

Moved out of `train_baseline.main()` on 2026-09-24, verbatim, so that a second consumer -- the
frozen-backbone screen in `coffeecv_dino` -- gets byte-identical patches instead of re-typing the
construction. Re-typing it is not a safe shortcut here: `rig_idx`, a rig's position in
`cfg.train_rigs`, seeds both the photo split (`split_photos_by_class`) and every patch box
(`MultiPhotoPatchDataset._extract_photo`), so a caller that builds each rig on its own, or in another
order, silently trains on different photos. The 2026-09-21 backbone screen did exactly that. See
docs/dinov3_integration_plan.md §0.1 and §4.1.

The held-out rig is always built alone, `split="all"`, so its `rig_idx` is 0 in every fold.
"""
from __future__ import annotations

from dataclasses import dataclass

from coffeecv.bean_scale import pitch_kwargs
from coffeecv.config import RunConfig
from coffeecv.dataset import MultiPhotoPatchDataset, Rig, load_class_labels, resolve_rigs
from coffeecv.transforms import build_eval_transform


@dataclass
class FoldDatasets:
    train: MultiPhotoPatchDataset | None  # None only when the caller asked for other splits (`only`)
    val: MultiPhotoPatchDataset | None
    test: MultiPhotoPatchDataset | None
    xrig: MultiPhotoPatchDataset | None  # None for an all-rigs run (empty heldout_rig), or not asked for
    train_rigs: list[Rig]
    heldout_rig: Rig | None
    class_ids: list[str]
    class_labels: dict[str, str]


def build_fold_datasets(cfg: RunConfig, train_transform, eval_transform, *,
                        return_domain_id: bool = False,
                        only: tuple[str, ...] | None = None) -> FoldDatasets:
    """Build train/val/test from `cfg.train_rigs` (in that order) and the cross-rig set from
    `cfg.heldout_rig`, exactly as `train_baseline` trains on them.

    `train_transform` is applied to the train split only. A frozen-feature caller passes the eval
    transform for both, which changes what `__getitem__` returns but not which boxes are drawn:
    boxes are sampled once, at construction, from seeded generators that never see the transform.

    `only` builds just the named splits ("train", "val", "test", "xrig") and leaves the rest None. Each
    split draws from generators keyed on (seed, rig_idx, class, photo, split), never from a shared
    stream, so a split built alone is byte-identical to the same split built with the others -- and a
    caller that needs one split at a time holds one split's decoded patches in memory, not four.
    """
    want = set(only) if only is not None else {"train", "val", "test", "xrig"}
    train_rig_dirs, heldout_rig_dir, classes_file = cfg.resolve_paths()
    train_rigs = resolve_rigs(train_rig_dirs)
    heldout_rig = resolve_rigs([heldout_rig_dir])[0] if heldout_rig_dir else None
    class_labels = load_class_labels(classes_file)
    class_ids = sorted(class_labels)
    patches_per_class = {
        "train": cfg.train_patches_per_class,
        "val": cfg.val_patches_per_class,
        "test": cfg.test_patches_per_class,
        "all": cfg.xrig_patches_per_class,
    }
    photo_frac = {
        "train": cfg.train_photo_frac,
        "val": cfg.val_photo_frac,
        "test": cfg.test_photo_frac,
    }

    common_kwargs = dict(
        rigs=train_rigs,
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
    )
    train_ds = MultiPhotoPatchDataset(
        split="train", transform=train_transform,
        rotation_jitter_degrees=cfg.rotation_jitter_degrees,
        return_domain_id=return_domain_id, **common_kwargs,
    ) if "train" in want else None
    val_ds = (MultiPhotoPatchDataset(split="val", transform=eval_transform, **common_kwargs)
              if "val" in want else None)
    test_ds = (MultiPhotoPatchDataset(split="test", transform=eval_transform, **common_kwargs)
               if "test" in want else None)

    # The cross-rig test set: every photo of a rig the model never trained on.
    # This is the headline generalization number; `test_ds` above stays as the
    # in-distribution control, so a change that trades one for the other is
    # visible rather than hidden behind a single metric.
    xrig_ds = (
        MultiPhotoPatchDataset(
            **{**common_kwargs, "rigs": [heldout_rig]},
            split="all", transform=eval_transform,
        )
        if heldout_rig and "xrig" in want else None
    )
    return FoldDatasets(train_ds, val_ds, test_ds, xrig_ds, train_rigs, heldout_rig,
                        class_ids, class_labels)


def build_capture_dataset(cfg: RunConfig, capture: Rig, class_ids: list[str], classes_file,
                          n_patches: int) -> MultiPhotoPatchDataset:
    """Every photo of one capture dir, `split="all"`, eval transform, `cfg`'s own patch geometry.

    No holdout semantics: this is the generic "draw n_patches per class from all of one dir's photos"
    builder. It feeds the DINOv3 backbone-equivalence fixture (`coffeecv_dino.reference`), whose
    patches must stay byte-identical to the ones `scripts/make_dinov3_fixture.py` recorded -- so the
    construction, the "all" split's seed component and the single-dir `rig_idx` of 0 must not move.
    Moved here from `xrig_eval.build_xrig_dataset` on 2026-09-29 (ticket ML-1).
    """
    return MultiPhotoPatchDataset(
        split="all",
        transform=build_eval_transform(cfg.patch_resize),
        rigs=[capture],
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
    )
