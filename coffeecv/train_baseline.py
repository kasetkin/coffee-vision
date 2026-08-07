"""Baseline training entry point. Invoked directly or via `dvc exp run` (see dvc.yaml)."""
from __future__ import annotations

import argparse
import json

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
from coffeecv.dataset import MultiPhotoPatchDataset, discover_classes_multi, load_class_labels
from coffeecv.metrics import build_metrics_json, compute_split_metrics, write_predictions_csv
from coffeecv.model import build_model
from coffeecv.plotting import plot_confusion_matrix, plot_patch_samples, plot_training_curves
from coffeecv.transforms import build_eval_transform, build_train_transform

DEVICE = torch.device("cpu")  # this project trains on a CPU-only devcontainer


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


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: nn.Module) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
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

    dataset_dir, classes_file = cfg.resolve_paths()
    class_ids = discover_classes_multi(dataset_dir)
    class_labels = load_class_labels(classes_file)
    patches_per_class = {
        "train": cfg.train_patches_per_class,
        "val": cfg.val_patches_per_class,
        "test": cfg.test_patches_per_class,
    }
    photos_per_split = {
        "train": cfg.train_photos_per_class,
        "val": cfg.val_photos_per_class,
        "test": cfg.test_photos_per_class,
    }

    common_kwargs = dict(
        dataset_dir=dataset_dir,
        classes_file=classes_file,
        class_ids=class_ids,
        seed=cfg.seed,
        crop_size=cfg.patch_crop_size,
        resize=cfg.patch_resize,
        safety_margin=cfg.safety_margin,
        patches_per_class=patches_per_class,
        photos_per_split=photos_per_split,
    )
    train_transform = build_train_transform(cfg.patch_resize, cfg.color_jitter_strength)
    train_ds = MultiPhotoPatchDataset(split="train", transform=train_transform, **common_kwargs)
    val_ds = MultiPhotoPatchDataset(split="val", transform=build_eval_transform(cfg.patch_resize), **common_kwargs)
    test_ds = MultiPhotoPatchDataset(split="test", transform=build_eval_transform(cfg.patch_resize), **common_kwargs)

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model, head_module = build_model(
        cfg.model_name, num_classes=len(class_ids), freeze_mode=cfg.freeze_mode, dropout=cfg.dropout
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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_true, val_pred, val_losses = evaluate(model, val_loader, criterion)
        val_metrics = compute_split_metrics(val_true, val_pred, val_losses, class_ids, class_labels)

        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss_mean"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_mcc": val_metrics["mcc"],
            "lr": current_lr,
        })
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

    metrics_json = build_metrics_json(
        class_ids, class_labels, epochs_trained=epoch, best_epoch=best_epoch,
        val_metrics=best_val_metrics, test_metrics=test_metrics,
    )
    (OUTPUTS_DIR / "metrics.json").write_text(json.dumps(metrics_json, indent=2))

    write_predictions_csv(OUTPUTS_DIR / "predictions_val.csv", best_val_true, best_val_pred, class_ids)
    write_predictions_csv(OUTPUTS_DIR / "predictions_test.csv", test_true, test_pred, class_ids)

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
