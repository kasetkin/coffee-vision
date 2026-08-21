"""Evaluate an already-trained checkpoint on its cross-rig split, optionally with
inference-time adaptation layered on top -- AdaBN, photometric TTA, multi-scale
TTA. No retraining, no `params.yaml` edits, no `dvc repro`: everything here is a
forward-pass-only operation on a checkpoint that already exists.

Fold checkpoints are not archived under `experiments/` --
`archive_experiment.py`'s own docstring: "the 44MB checkpoints stay in DVC, which
is what DVC is for." `outputs/checkpoints/{best,last}.pt` holds whatever the most
recent run produced, so evaluating an older fold means fetching its checkpoint out
of DVC history. This deliberately uses `dvc get . <path> --rev <sha> -o <dest>`
rather than `git checkout <sha> -- dvc.lock && dvc checkout` -- the latter
overwrites the *working tree's* `outputs/checkpoints/`, and at the time this was
written that directory held a file matching no known commit (unarchived,
untracked state of unclear origin). `dvc get` fetches into an arbitrary
destination without touching it.

    python -m coffeecv.xrig_eval --exp-id 60 --adabn
    python -m coffeecv.xrig_eval --exp-id 60 --photometric-tta
    python -m coffeecv.xrig_eval --exp-id 60 --multiscale-tta --multiscale-b 4.0,5.5,7.0
    python -m coffeecv.xrig_eval --checkpoint outputs/checkpoints/best.pt --rig-dir <fresh-scoop dir>

Each flag is independently screened -- report deltas per technique, never a
combined number, so a win from one is never credited to another
(see coffeecv/tta_variants.py's own docstring for why they're not interchangeable).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from coffeecv.bean_scale import estimate_bean_pitch
from coffeecv.config import REPO_ROOT
from coffeecv.dataset import (
    MultiPhotoPatchDataset,
    discover_classes_multi,
    find_class_dir,
    list_cropped_photos,
    load_class_labels,
    load_rgb_image,
    resolve_rigs,
)
from coffeecv.geometry import (
    bean_unit_box_at_center,
    compute_valid_region_rect,
    sample_bean_unit_centers,
    sample_bean_unit_patch_boxes,
)
from coffeecv.infer import config_for_checkpoint, forward_with_embeddings, load_model
from coffeecv.metrics import compute_split_metrics
from coffeecv.transforms import build_eval_transform
from coffeecv.tta_variants import photometric_tta_probs

EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DEFAULT_SCRATCH = REPO_ROOT / "outputs" / "xrig_eval_checkpoints"


def resolve_exp(exp_id: int) -> tuple[Path, str]:
    """(config.json path, commit sha) for an archived leave-one-rig-out fold.

    The commit whose dvc.lock actually contains this fold's checkpoint is the one
    `run_folds.py` creates *after* archiving, with subject "exp{id}: ...". This is
    NOT the same as config.json's own `env.git_commit` -- that field is written by
    `train_baseline.py` before training starts, so it records the *parent* commit,
    which predates this fold's own dvc.lock entry. Verified by hash-matching
    `dvc get --rev <that commit>` output against the checkpoint hash the fold's
    own dvc.lock records.
    """
    matches = sorted(EXPERIMENTS_DIR.glob(f"exp{exp_id}__*"))
    if not matches:
        raise SystemExit(f"No archived experiment exp{exp_id} under {EXPERIMENTS_DIR}")
    config_path = matches[0] / "config.json"
    if not config_path.exists():
        raise SystemExit(f"{matches[0]} has no config.json")

    log = subprocess.check_output(
        ["git", "log", "--all", "--format=%H %s"], cwd=REPO_ROOT
    ).decode()
    prefix = f"exp{exp_id}: "
    for line in log.splitlines():
        commit, _, subject = line.partition(" ")
        if subject.startswith(prefix):
            return config_path, commit
    raise SystemExit(f"No commit with subject '{prefix}...' found in git history")


def fetch_fold_checkpoint(sha: str, exp_id: int, scratch_dir: Path) -> Path:
    dest = scratch_dir / f"exp{exp_id}_{sha[:8]}_best.pt"
    if dest.exists():
        return dest
    scratch_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dvc", "get", ".", "outputs/checkpoints/best.pt", "--rev", sha, "-o", str(dest)],
        cwd=REPO_ROOT, check=True,
    )
    return dest


def reference_xrig_f1(exp_id: int) -> float | None:
    matches = sorted(EXPERIMENTS_DIR.glob(f"exp{exp_id}__*"))
    if not matches:
        return None
    metrics_path = matches[0] / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    return metrics.get("splits", {}).get("test_xrig", {}).get("macro_f1")


def adabn_recalibrate(model: nn.Module, tensors: torch.Tensor, batch_size: int = 32) -> None:
    """Reset BatchNorm running stats and recompute them from `tensors` (no
    labels used) via forward passes in train mode, without touching any weight.

    `momentum=None` switches each BN layer to cumulative-average mode so the
    whole recalibration pass contributes one true average rather than an
    exponential moving average biased toward the last batches seen.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            model(tensors[i:i + batch_size])
    model.eval()


def snapshot_bn_stats(model: nn.Module) -> dict:
    """Save every BatchNorm2d's running_mean/var/num_batches_tracked so they can
    be restored between photos in `run_single_photo_adabn` -- each photo's
    recalibration must start from the checkpoint's *original* statistics, not
    the previous photo's, matching a stateless per-request inference call
    (a real user's photo must not be biased by whatever photo came before it)."""
    return {
        name: (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)
    }


def restore_bn_stats(model: nn.Module, snapshot: dict) -> None:
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            rm, rv, nb = snapshot[name]
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nb)


def build_xrig_dataset(cfg, rig, class_ids, classes_file, n_patches: int):
    """Same construction as `train_baseline.py`'s `xrig_ds` / `build_ood_reference.py`'s
    `ds` -- split="all", eval transform, this fold's own patch geometry."""
    return MultiPhotoPatchDataset(
        split="all",
        transform=build_eval_transform(cfg.patch_resize),
        rigs=[rig],
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
        photos_per_split={
            "train": cfg.train_photos_per_class, "val": cfg.val_photos_per_class,
            "test": cfg.test_photos_per_class,
        },
        patch_store_size=cfg.patch_store_size or None,
        patch_scale_frac=((cfg.patch_scale_frac_min, cfg.patch_scale_frac_max)
                          if cfg.patch_scale_frac_max > 0 else None),
        patch_beans=((cfg.patch_beans_min, cfg.patch_beans_max)
                    if cfg.patch_beans_max > 0 else None),
    )


def _iter_photos_with_progress(rig, class_ids, label: str):
    """Yields (class_idx, class_id, photo_idx, photo_path), printing a flushed
    progress line per photo. Both photo-by-photo loops below run for minutes with
    no output otherwise -- redirected to a file (background runs, `tee`), that
    looks indistinguishable from a hang until the whole thing finishes."""
    per_class = [(cid, list_cropped_photos(find_class_dir(rig.cropped_dir, cid))) for cid in class_ids]
    total = sum(len(photos) for _, photos in per_class)
    done = 0
    for class_idx, (class_id, photos) in enumerate(per_class):
        for photo_idx, photo_path in enumerate(photos):
            done += 1
            print(f"  [{label}] {done}/{total}  class {class_id}  {photo_path.name}", flush=True)
            yield class_idx, class_id, photo_idx, photo_path


def run_photowise(model, head, cfg, rig, class_ids, n_patches_per_photo, seed, dihedral_tta, b_values=None):
    """Iterates every photo in `rig`, drawing exactly `n_patches_per_photo` patches
    per photo at `n_patches_per_photo` *shared centre points* -- either sized by
    the standard continuous log-uniform B draw (`b_values=None`) or by a fixed
    set of scales assigned to those same centres in order (`b_values` given).

    Centres come from `geometry.sample_bean_unit_centers`, seeded only by
    `[seed, class_idx, photo_idx]` -- the same key regardless of `b_values` -- so
    calling this twice (once per arm) reproduces *identical* centre points on
    both calls without the two arms needing to share any runtime state. That
    means a delta between the two calls isolates the scale mechanism alone: not
    patch count (both draw `n_patches_per_photo`, matching an earlier version of
    this comparison that didn't -- see [[project-phase16-screens]]), and not
    which random positions happened to get sampled either.
    """
    label = "multiscale-tta" if b_values else "photowise-baseline"
    eval_transform = build_eval_transform(cfg.patch_resize)
    store_size = cfg.patch_store_size or cfg.patch_crop_size
    all_probs, all_labels = [], []
    for class_idx, class_id, photo_idx, photo_path in _iter_photos_with_progress(rig, class_ids, label):
        seed_key = [seed, class_idx, photo_idx]
        rgb = load_rgb_image(photo_path)
        h, w = rgb.shape[:2]
        region = compute_valid_region_rect(h, w, cfg.safety_margin)
        gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype("uint8")
        pitch = estimate_bean_pitch(gray)

        centers_rng = np.random.default_rng([*seed_key, 100])
        centers = sample_bean_unit_centers(
            centers_rng, region, n_patches_per_photo, pitch, cfg.patch_beans_max)

        if b_values:
            base, extra = divmod(n_patches_per_photo, len(b_values))
            beans = []
            for b_idx, b in enumerate(b_values):
                beans.extend([b] * (base + (1 if b_idx < extra else 0)))
        else:
            beans_rng = np.random.default_rng([*seed_key, 101])
            beans = np.exp(beans_rng.uniform(
                np.log(cfg.patch_beans_min), np.log(cfg.patch_beans_max), size=n_patches_per_photo,
            )).tolist()

        patches = []
        for center, b in zip(centers, beans):
            box = bean_unit_box_at_center(center, b, pitch, region)
            patch = Image.fromarray(rgb[box.y0:box.y1, box.x0:box.x1])
            if patch.size[0] != store_size:
                patch = patch.resize((store_size, store_size), Image.BILINEAR)
            patches.append(patch)

        tensors = torch.stack([eval_transform(p) for p in patches])
        probs, _ = forward_with_embeddings(model, head, tensors, tta=dihedral_tta)
        all_probs.append(probs)
        all_labels.extend([class_idx] * len(probs))
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def run_single_photo_adabn(model, head, cfg, rig, class_ids, n_patches, seed, dihedral_tta):
    """The deployment-realistic version of AdaBN: recalibrate on *one photo's*
    own patches, not the whole rig's.

    The batch `--adabn` path above recalibrates on all 1080 patches from the
    held-out rig's 20 photos, then scores those same patches -- legitimate
    (no labels used) but optimistic: `infer.py` gets one photo per request,
    which yields ~40 correlated patches (same beans, same single-shot lighting),
    not 1080 patches spanning 20 photos' worth of variety. This measures what a
    single stateless inference call could actually achieve: for every photo,
    restore the checkpoint's *original* BN statistics (via
    `snapshot_bn_stats`/`restore_bn_stats`), recalibrate on only that photo's
    patches, score, then move to the next photo starting from the original
    statistics again -- no state carries over between photos, matching how a
    real deployment must treat one user's photo as independent of the last.
    """
    eval_transform = build_eval_transform(cfg.patch_resize)
    snapshot = snapshot_bn_stats(model)
    all_probs, all_labels = [], []
    for class_idx, class_id, photo_idx, photo_path in _iter_photos_with_progress(
        rig, class_ids, "adabn-single-photo"
    ):
        rgb = load_rgb_image(photo_path)
        h, w = rgb.shape[:2]
        region = compute_valid_region_rect(h, w, cfg.safety_margin)
        gray = (rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114).astype("uint8")
        pitch = estimate_bean_pitch(gray)
        rng = np.random.default_rng([seed, class_idx, photo_idx])
        boxes, _clamped = sample_bean_unit_patch_boxes(
            rng, region, n_patches, pitch, cfg.patch_beans_min, cfg.patch_beans_max, 0.0,
        )
        store = cfg.patch_store_size or cfg.patch_crop_size
        patches = []
        for box, _angle, _side in boxes:
            patch = Image.fromarray(rgb[box.y0:box.y1, box.x0:box.x1])
            if patch.size[0] != store:
                patch = patch.resize((store, store), Image.BILINEAR)
            patches.append(patch)
        tensors = torch.stack([eval_transform(p) for p in patches])

        restore_bn_stats(model, snapshot)
        adabn_recalibrate(model, tensors)
        probs, _ = forward_with_embeddings(model, head, tensors, tta=dihedral_tta)

        all_probs.append(probs)
        all_labels.extend([class_idx] * len(probs))
    restore_bn_stats(model, snapshot)  # leave the model as it was found
    return np.concatenate(all_probs, axis=0), np.array(all_labels)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exp-id", type=int, help="archived leave-one-rig-out fold, e.g. 60")
    p.add_argument("--checkpoint", help="explicit checkpoint path, alternative to --exp-id")
    p.add_argument("--config", help="explicit config.json; defaults per infer.py's resolution order")
    p.add_argument("--rig-dir", help="override target rig; defaults to the checkpoint's own heldout_rig")
    p.add_argument("--adabn", action="store_true", help="recalibrate BatchNorm on the target rig's images")
    p.add_argument("--adabn-single-photo", action="store_true",
                   help="deployment-realistic AdaBN: recalibrate per photo (~40 patches), not on the "
                        "whole rig at once (see run_single_photo_adabn's docstring)")
    p.add_argument("--photometric-tta", action="store_true")
    p.add_argument("--photometric-draws", type=int, default=4)
    p.add_argument("--multiscale-tta", action="store_true")
    p.add_argument("--multiscale-b", default="4.0,5.5,7.0")
    p.add_argument("--no-dihedral-tta", action="store_true",
                   help="dihedral TTA is on by default, matching infer.py")
    p.add_argument("--n-patches", type=int, default=None, help="defaults to the fold's own xrig_patches_per_class")
    p.add_argument("--scratch-dir", default=str(DEFAULT_SCRATCH))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out")
    args = p.parse_args()

    if args.exp_id is not None:
        config_path, sha = resolve_exp(args.exp_id)
        print(f"exp{args.exp_id}: commit {sha}")
        ckpt = fetch_fold_checkpoint(sha, args.exp_id, Path(args.scratch_dir))
        cfg, cfg_source = config_for_checkpoint(ckpt, str(config_path))
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        cfg, cfg_source = config_for_checkpoint(ckpt, args.config)
    else:
        raise SystemExit("pass --exp-id or --checkpoint")
    print(f"checkpoint: {ckpt}")
    print(f"config: {cfg_source}")

    train_rig_dirs, heldout_rig_dir, classes_file = cfg.resolve_paths()
    rig_dir = Path(args.rig_dir) if args.rig_dir else heldout_rig_dir
    if rig_dir is None:
        raise SystemExit("this checkpoint's config has no heldout_rig and no --rig-dir was given")
    rig = resolve_rigs([rig_dir])[0]
    # Class-id ordering must come from the *training* rigs, matching how the
    # checkpoint's final layer was fitted -- not from rig_dir, which may be a
    # target the model never trained on (that's the whole point of this script).
    class_ids = discover_classes_multi(resolve_rigs(train_rig_dirs)[0].cropped_dir)
    class_labels = load_class_labels(classes_file)
    print(f"target rig: {rig.name}  ({len(class_ids)} classes)")

    model, head = load_model(ckpt, cfg.model_name, len(class_ids), cfg.dropout)
    n_patches = args.n_patches or cfg.xrig_patches_per_class
    dihedral = not args.no_dihedral_tta

    def score(labels_, probs_) -> dict:
        preds_ = probs_.argmax(axis=1)
        losses_ = -np.log(np.clip(probs_[np.arange(len(labels_)), labels_], 1e-12, 1.0))
        return compute_split_metrics(labels_, preds_, losses_, class_ids, class_labels)

    # Same-TTA-setting baseline, computed on the *unmodified* model before any
    # adaptation runs (critical for AdaBN, which mutates BN running stats in
    # place). This is the fair comparison for judging an adaptation technique --
    # the archived exp reference below has no TTA at all, so diffing an
    # adaptation directly against it would silently credit dihedral TTA's own
    # already-known effect to whichever technique this run is screening.
    ds = build_xrig_dataset(cfg, rig, class_ids, classes_file, n_patches)
    labels = np.array([ds[i][1] for i in range(len(ds))])
    tensors = torch.stack([ds[i][0] for i in range(len(ds))])
    baseline_probs, _ = forward_with_embeddings(model, head, tensors, tta=dihedral)
    baseline_metrics = score(labels, baseline_probs)
    print(f"in-run baseline (dihedral_tta={dihedral}, no adaptation): "
          f"xrig macro_f1={baseline_metrics['macro_f1']:.4f}")

    if args.multiscale_tta:
        b_values = [float(x) for x in args.multiscale_b.split(",")]
        # infer.py's own default, not xrig_patches_per_class (120) -- and NOT the
        # dataset-based baseline_metrics above, which spreads its 120-per-class
        # budget over 20 photos (6/photo). Both arms below must draw the exact
        # same count per photo or the comparison is confounded by patch count
        # again (see fixed_scale_patches' docstring and [[project-phase16-screens]]
        # for the earlier version that got this wrong).
        ms_n_patches = args.n_patches or 40
        print(f"multi-scale TTA: B={b_values}, {ms_n_patches} patches/photo (both arms)")
        fair_probs, fair_labels = run_photowise(
            model, head, cfg, rig, class_ids, ms_n_patches, args.seed, dihedral, b_values=None)
        fair_baseline_metrics = score(fair_labels, fair_probs)
        print(f"photo-wise baseline (same {ms_n_patches}/photo budget, standard log-uniform draw): "
              f"xrig macro_f1={fair_baseline_metrics['macro_f1']:.4f}")
        ms_probs, ms_labels = run_photowise(
            model, head, cfg, rig, class_ids, ms_n_patches, args.seed, dihedral, b_values=b_values)
        metrics = score(ms_labels, ms_probs)
        # Overrides the dataset-based baseline_metrics printed above for the
        # final "vs in-run baseline" line -- the fair, same-budget one is what
        # actually isolates the scale-distribution mechanism.
        baseline_metrics = fair_baseline_metrics
    elif args.adabn:
        print(f"AdaBN: recalibrating BatchNorm on {len(tensors)} unlabeled target-rig patches")
        adabn_recalibrate(model, tensors)
        adapted_probs, _ = forward_with_embeddings(model, head, tensors, tta=dihedral)
        metrics = score(labels, adapted_probs)
    elif args.adabn_single_photo:
        # infer.py's own default, not the whole-rig xrig_patches_per_class (120)
        # used above -- this path is measuring what a single real inference call
        # would actually have available, so it must match that call's patch count.
        sp_n_patches = args.n_patches or 40
        print(f"AdaBN (single-photo): recalibrating per photo ({sp_n_patches} patches each)")
        sp_probs, sp_labels = run_single_photo_adabn(
            model, head, cfg, rig, class_ids, sp_n_patches, args.seed, dihedral)
        metrics = score(sp_labels, sp_probs)
    elif args.photometric_tta:
        patches = [Image.fromarray(ds._patches[i]) for i in range(len(ds))]
        print(f"photometric TTA: {args.photometric_draws} draws, "
              f"jitter_strength={cfg.color_jitter_strength}")
        photo_probs = photometric_tta_probs(
            model, patches, cfg.patch_resize, cfg.color_jitter_strength,
            args.photometric_draws, args.seed,
        )
        adapted_probs = (photo_probs + baseline_probs) / 2 if dihedral else photo_probs
        metrics = score(labels, adapted_probs)
    else:
        metrics = baseline_metrics  # no adaptation flag: just report the baseline

    print(f"\nxrig macro_f1: {metrics['macro_f1']:.4f}   mcc: {metrics['mcc']:.4f}   "
          f"n={metrics['n_samples']}")
    print(f"vs in-run baseline: {metrics['macro_f1'] - baseline_metrics['macro_f1']:+.4f}")

    if args.exp_id is not None:
        ref = reference_xrig_f1(args.exp_id)
        if ref is not None:
            print(f"vs archived reference (exp{args.exp_id}, no TTA at all): "
                  f"{ref:.4f}   delta: {metrics['macro_f1'] - ref:+.4f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": str(ckpt),
            "config_source": cfg_source,
            "rig": rig.name,
            "adabn": args.adabn,
            "adabn_single_photo": args.adabn_single_photo,
            "photometric_tta": args.photometric_tta,
            "multiscale_tta": args.multiscale_tta,
            "multiscale_b": args.multiscale_b if args.multiscale_tta else None,
            "dihedral_tta": not args.no_dihedral_tta,
            "metrics": metrics,
        }, indent=2))
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
