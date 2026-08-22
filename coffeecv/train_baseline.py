"""Baseline training entry point. Invoked directly or via `dvc exp run` (see dvc.yaml)."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from coffeecv.config import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    PLOTS_DIR,
    TENSORBOARD_DIR,
    RunConfig,
    build_env_block,
    config_to_dict,
    set_seed,
)
from coffeecv.dataset import (
    MultiPhotoPatchDataset,
    discover_classes_multi,
    load_class_labels,
    resolve_rigs,
)
from coffeecv.metrics import (
    build_metrics_json,
    build_summary_json,
    compute_split_metrics,
    write_predictions_csv,
)
from coffeecv.model import build_model
from coffeecv.plotting import plot_confusion_matrix, plot_patch_samples, plot_training_curves
from coffeecv.transforms import build_eval_transform, build_train_transform

DEVICE = torch.device("cpu")  # this project trains on a CPU-only devcontainer

# Opt-in override for torch's intraop thread pool, read once at import time (must
# run before any torch op). Unset by default, so every existing/reference run
# keeps torch's own default (physical-core count, not logical/SMT count) and this
# cannot silently change a comparison's results. `"all"` means every logical CPU
# (`os.cpu_count()`) rather than a number hardcoded for one machine, so the same
# invocation is portable across hosts with a different core count.
_threads_env = os.environ.get("COFFEECV_TORCH_THREADS")
if _threads_env == "all":
    torch.set_num_threads(os.cpu_count())
elif _threads_env:
    torch.set_num_threads(int(_threads_env))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the baseline coffee-bean-origin classifier.")
    p.add_argument("--model-name", choices=["mobilenet_v3_small", "resnet18", "efficientnet_b0"], default=None)
    p.add_argument("--freeze-mode", choices=["full", "last_block", "none"], default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.freeze_mode is not None:
        cfg.freeze_mode = args.freeze_mode
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed
    return cfg


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module):
    model.eval()
    all_true, all_pred, all_losses = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            losses = criterion(logits, y)
            preds = logits.argmax(dim=1)
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_losses.append(losses.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_losses)


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer, criterion: nn.Module, mixup_alpha: float = 0.0,
    cross_domain_mixstyle: bool = False,
) -> float:
    """`mixup_alpha` > 0 enables mixup: blend each batch with a shuffled copy of
    itself and take the correspondingly weighted loss against both label sets.
    At 0.0 no RNG is drawn, so the unmixed path is bit-identical to pre-Phase-8.

    `cross_domain_mixstyle=True` means `loader` yields (x, y, domain_id) triples
    (see `MultiPhotoPatchDataset(return_domain_id=True)`) and the model has
    `mixstyle1`/`mixstyle2` submodules expecting `domain_ids` set on them before
    each forward pass -- a forward hook only sees `(module, input, output)`, so
    this is the only way to get per-sample rig identity into it."""
    model.train()
    total_loss, n = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        if cross_domain_mixstyle:
            x, y, domain_id = batch
            domain_id = domain_id.to(DEVICE)
            model.mixstyle1.domain_ids = domain_id
            model.mixstyle2.domain_ids = domain_id
        else:
            x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        if mixup_alpha > 0:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(x.size(0), device=DEVICE)
            x = lam * x + (1.0 - lam) * x[perm]
            logits = model(x)
            loss = (lam * criterion(logits, y) + (1.0 - lam) * criterion(logits, y[perm])).mean()
        else:
            logits = model(x)
            loss = criterion(logits, y).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(RunConfig.from_params_yaml(), args)
    set_seed(cfg.seed)

    train_rig_dirs, heldout_rig_dir, classes_file = cfg.resolve_paths()
    train_rigs = resolve_rigs(train_rig_dirs)
    heldout_rig = resolve_rigs([heldout_rig_dir])[0] if heldout_rig_dir else None
    class_ids = discover_classes_multi(train_rigs[0].cropped_dir)
    class_labels = load_class_labels(classes_file)
    print(f"train rigs: {[r.name for r in train_rigs]}")
    print(f"held-out rig: {heldout_rig.name if heldout_rig else '(none)'}")
    print(f"torch threads: {torch.get_num_threads()} (COFFEECV_TORCH_THREADS={_threads_env!r})")
    patches_per_class = {
        "train": cfg.train_patches_per_class,
        "val": cfg.val_patches_per_class,
        "test": cfg.test_patches_per_class,
        "all": cfg.xrig_patches_per_class,
    }
    photos_per_split = {
        "train": cfg.train_photos_per_class,
        "val": cfg.val_photos_per_class,
        "test": cfg.test_photos_per_class,
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
        photos_per_split=photos_per_split,
        patch_store_size=cfg.patch_store_size or None,
        patch_scale_frac=(
            (cfg.patch_scale_frac_min, cfg.patch_scale_frac_max)
            if cfg.patch_scale_frac_max > 0 else None
        ),
        patch_beans=(
            (cfg.patch_beans_min, cfg.patch_beans_max)
            if cfg.patch_beans_max > 0 else None
        ),
    )
    train_transform = build_train_transform(
        cfg.patch_resize,
        cfg.color_jitter_strength,
        zoom_scale_min=cfg.zoom_scale_min,
        random_erasing_p=cfg.random_erasing_p,
        illum_gradient_strength=cfg.illum_gradient_strength,
        brightness_jitter_strength=cfg.brightness_jitter_strength,
    )
    cross_domain_mixstyle = cfg.mixstyle_p > 0 and cfg.mixstyle_mode == "cross_rig"
    train_ds = MultiPhotoPatchDataset(
        split="train", transform=train_transform,
        rotation_jitter_degrees=cfg.rotation_jitter_degrees,
        return_domain_id=cross_domain_mixstyle, **common_kwargs,
    )
    eval_transform = build_eval_transform(cfg.patch_resize)
    val_ds = MultiPhotoPatchDataset(split="val", transform=eval_transform, **common_kwargs)
    test_ds = MultiPhotoPatchDataset(split="test", transform=eval_transform, **common_kwargs)

    # The cross-rig test set: every photo of a rig the model never trained on.
    # This is the headline generalization number; `test_ds` above stays as the
    # in-distribution control, so a change that trades one for the other is
    # visible rather than hidden behind a single metric.
    xrig_ds = (
        MultiPhotoPatchDataset(
            **{**common_kwargs, "rigs": [heldout_rig]},
            split="all", transform=eval_transform,
        )
        if heldout_rig else None
    )

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    xrig_loader = (
        DataLoader(xrig_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        if xrig_ds else None
    )
    print(f"patches: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
          + (f" xrig={len(xrig_ds)}" if xrig_ds else ""))
    # Clamping means a requested patch did not fit its photo and was shrunk to the
    # frame, flattening the scale distribution. Printed so a run that quietly lost
    # its scale variety shows up in the log, not only in the result.
    for nm, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds), ("xrig", xrig_ds)]:
        if ds is not None and getattr(ds, "n_clamped", 0):
            print(f"  {nm}: {ds.n_clamped}/{len(ds)} patches clamped "
                  f"({ds.n_clamped / len(ds) * 100:.0f}%)")

    model, head_module = build_model(
        cfg.model_name, num_classes=len(class_ids), freeze_mode=cfg.freeze_mode, dropout=cfg.dropout,
        mixstyle_p=cfg.mixstyle_p, mixstyle_alpha=cfg.mixstyle_alpha, mixstyle_mode=cfg.mixstyle_mode,
    )
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(reduction="none", label_smoothing=cfg.label_smoothing)

    head_param_ids = {id(p) for p in head_module.parameters()}
    head_params = [p for p in model.parameters() if p.requires_grad and id(p) in head_param_ids]
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
    param_groups = [{"params": head_params, "lr": cfg.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})

    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(param_groups, weight_decay=cfg.weight_decay, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    OUTPUTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(exist_ok=True)

    # Written at run start so even a crashed run leaves a record.
    config_record = {**config_to_dict(cfg), "env": build_env_block()}
    (OUTPUTS_DIR / "config.json").write_text(json.dumps(config_record, indent=2))

    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR))
    history: list[dict] = []
    best_epoch, best_val_macro_f1 = 0, -1.0
    best_val_metrics, best_val_true, best_val_pred = None, None, None
    epochs_since_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, cfg.mixup_alpha,
            cross_domain_mixstyle=cross_domain_mixstyle,
        )
        val_true, val_pred, val_losses = evaluate(model, val_loader, criterion)
        val_metrics = compute_split_metrics(val_true, val_pred, val_losses, class_ids, class_labels)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss_mean"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mcc": val_metrics["mcc"],
            "lr": current_lr,
        }
        if xrig_loader is not None:
            # Recorded per epoch purely so the val/cross-rig gap is visible as a
            # curve -- it is the whole subject of Phase 11. It is NOT used for
            # checkpoint selection or early stopping: `best_epoch` is chosen on
            # val_macro_f1 alone (see below), so the held-out rig never
            # influences training. Treating it as a selection signal would make
            # the reported transfer number meaningless.
            xr_true, xr_pred, xr_losses = evaluate(model, xrig_loader, criterion)
            xr = compute_split_metrics(xr_true, xr_pred, xr_losses, class_ids, class_labels)
            epoch_row["xrig_macro_f1"] = xr["macro_f1"]
            epoch_row["xrig_loss"] = xr["loss_mean"]
        history.append(epoch_row)
        print(
            f"epoch {epoch}/{cfg.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss_mean']:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}  "
            f"val_mcc={val_metrics['mcc']:.4f}"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_metrics["loss_mean"], epoch)
        writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        writer.add_scalar("val/mcc", val_metrics["mcc"], epoch)
        writer.add_scalar("lr", current_lr, epoch)
        for cid, pc in val_metrics["per_class"].items():
            if pc["loss_mean"] is not None:
                writer.add_scalar(f"val_loss_per_class/{cid}", pc["loss_mean"], epoch)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_val_true, best_val_pred = val_true, val_pred
            torch.save(model.state_dict(), CHECKPOINTS_DIR / "best.pt")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        scheduler.step()

        if epochs_since_improvement >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch}: no val_macro_f1 improvement in "
                  f"{cfg.early_stop_patience} epochs (best={best_val_macro_f1:.4f} at epoch {best_epoch})")
            break

    torch.save(model.state_dict(), CHECKPOINTS_DIR / "last.pt")
    writer.close()

    (OUTPUTS_DIR / "history.json").write_text(json.dumps(history, indent=2))

    # Test split evaluated exactly once, with the best-val checkpoint.
    model.load_state_dict(torch.load(CHECKPOINTS_DIR / "best.pt"))
    test_true, test_pred, test_losses = evaluate(model, test_loader, criterion)
    test_metrics = compute_split_metrics(test_true, test_pred, test_losses, class_ids, class_labels)

    xrig_metrics = None
    if xrig_loader is not None:
        xrig_true, xrig_pred, xrig_losses = evaluate(model, xrig_loader, criterion)
        xrig_metrics = compute_split_metrics(
            xrig_true, xrig_pred, xrig_losses, class_ids, class_labels
        )

    metrics_json = build_metrics_json(
        class_ids, class_labels, epochs_trained=epoch, best_epoch=best_epoch,
        val_metrics=best_val_metrics, test_metrics=test_metrics,
        xrig_metrics=xrig_metrics,
        rigs={
            "train": [r.name for r in train_rigs],
            "heldout": heldout_rig.name if heldout_rig else None,
        },
    )
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics_json, indent=2))
    (OUTPUTS_DIR / "summary.json").write_text(json.dumps(build_summary_json(metrics_json), indent=2))

    write_predictions_csv(OUTPUTS_DIR / "predictions_val.csv", best_val_true, best_val_pred, class_ids)
    write_predictions_csv(OUTPUTS_DIR / "predictions_test.csv", test_true, test_pred, class_ids)
    if xrig_metrics is not None:
        write_predictions_csv(OUTPUTS_DIR / "predictions_xrig.csv", xrig_true, xrig_pred, class_ids)
        plot_confusion_matrix(
            xrig_metrics["confusion_matrix"], class_ids, class_labels,
            PLOTS_DIR / "confusion_matrix_xrig.png",
            f"Cross-rig confusion matrix (held out: {heldout_rig.name})",
        )
    else:
        # A run with no held-out rig (heldout_rig: "") has no cross-rig split, but
        # dvc.yaml declares this file as a plot and DVC fails the whole stage when
        # a declared output is missing -- which is how an all-rigs run lost its
        # pipeline record after training successfully for 27 epochs. Write the
        # header alone: unambiguous next to a metrics.json that has no test_xrig
        # split at all, and it cannot be mistaken for a measurement of zero.
        write_predictions_csv(OUTPUTS_DIR / "predictions_xrig.csv", [], [], class_ids)

    plot_confusion_matrix(
        best_val_metrics["confusion_matrix"], class_ids, class_labels,
        PLOTS_DIR / "confusion_matrix_val.png", f"Val confusion matrix (epoch {best_epoch})",
    )
    plot_confusion_matrix(
        test_metrics["confusion_matrix"], class_ids, class_labels,
        PLOTS_DIR / "confusion_matrix_test.png", "Test confusion matrix",
    )
    plot_training_curves(history, PLOTS_DIR / "training_curves.png")
    plot_patch_samples(val_ds, class_ids, class_labels, PLOTS_DIR / "patch_samples.png")

    print(f"\nDone. best_epoch={best_epoch}  val_macro_f1={best_val_metrics['macro_f1']:.4f}  "
          f"test_macro_f1={test_metrics['macro_f1']:.4f}  test_mcc={test_metrics['mcc']:.4f}")


if __name__ == "__main__":
    main()
