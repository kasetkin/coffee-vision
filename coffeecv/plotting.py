"""Static matplotlib plots for quick viewing without invoking DVC commands."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"


def plot_confusion_matrix(
    cm: list[list[int]], class_ids: list[str], class_labels: dict[str, str], out_path: Path, title: str
) -> None:
    cm = np.array(cm)
    tick_labels = [f"{cid}\n{class_labels.get(cid, cid)}" for cid in class_ids]

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_ids)))
    ax.set_yticks(range(len(class_ids)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=7,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list[dict], out_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(epochs, train_loss, color=TRAIN_COLOR, label="train loss")
    ax1.plot(epochs, val_loss, color=VAL_COLOR, label="val loss")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Training curves")

    ax2.plot(epochs, val_macro_f1, color=TRAIN_COLOR)
    ax2.set_ylabel("Val macro-F1")
    ax2.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_patch_samples(dataset, class_ids: list[str], class_labels: dict[str, str], out_path: Path, n_per_class: int = 3) -> None:
    fig, axes = plt.subplots(len(class_ids), n_per_class, figsize=(n_per_class * 2.2, len(class_ids) * 2.2))

    by_class: dict[str, list[int]] = {cid: [] for cid in class_ids}
    for idx, sample in enumerate(dataset._samples):
        cid = sample[0]  # (class_id, box) or (class_id, photo_name, box) -- class_id is always first
        if len(by_class[cid]) < n_per_class:
            by_class[cid].append(idx)

    for row, cid in enumerate(class_ids):
        for col in range(n_per_class):
            ax = axes[row, col] if len(class_ids) > 1 else axes[col]
            idxs = by_class[cid]
            if col < len(idxs):
                tensor, _label = dataset[idxs[col]]
                img = tensor.permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(class_labels.get(cid, cid), fontsize=8)
                ax.axis("on")
                ax.set_xticks([])
                ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
