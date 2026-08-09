"""Metric computation: per-class + macro-averaged precision/recall/F1, MCC,
confusion matrix, and a predictions CSV export for DVC's confusion plot template."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


def compute_split_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    losses: np.ndarray,
    class_ids: list[str],
    class_labels: dict[str, str],
) -> dict:
    n_classes = len(class_ids)
    labels_idx = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_idx, average="macro", zero_division=0
    )
    mcc = matthews_corrcoef(y_true, y_pred)
    accuracy = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    per_class = {}
    for i, class_id in enumerate(class_ids):
        class_mask = y_true == i
        class_loss = float(losses[class_mask].mean()) if class_mask.any() else None
        per_class[class_id] = {
            "label": class_labels.get(class_id, class_id),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "loss_mean": class_loss,
        }

    return {
        "n_samples": int(len(y_true)),
        "loss_mean": float(losses.mean()),
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_row_order": class_ids,
    }


def build_metrics_json(
    class_ids: list[str],
    class_labels: dict[str, str],
    epochs_trained: int,
    best_epoch: int,
    val_metrics: dict,
    test_metrics: dict,
) -> dict:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_ids": class_ids,
        "class_labels": {cid: class_labels.get(cid, cid) for cid in class_ids},
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "best_epoch_selection_metric": "val_macro_f1",
        "splits": {"val": val_metrics, "test": test_metrics},
    }


def build_summary_json(metrics_json: dict) -> dict:
    """A deliberately flat, six-number view of a run, for `dvc metrics diff`.

    `metrics.json` nests per-class stats, and DVC flattens every leaf into its own
    column — 140+ of them — which makes `dvc metrics show/diff` unreadable and so
    unused. This is the same data's headline, one level deep, so a diff between two
    commits fits on a screen. The full per-class detail stays in metrics.json and in
    the experiments/ archive; this is for scanning, not for analysis.
    """
    val, test = metrics_json["splits"]["val"], metrics_json["splits"]["test"]
    return {
        "val_macro_f1": round(val["macro_f1"], 4),
        "val_mcc": round(val["mcc"], 4),
        "test_macro_f1": round(test["macro_f1"], 4),
        "test_mcc": round(test["mcc"], 4),
        "best_epoch": metrics_json["best_epoch"],
        "epochs_trained": metrics_json["epochs_trained"],
    }


def write_predictions_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, class_ids: list[str]) -> None:
    """Columns: true_label,pred_label — feeds DVC's built-in `confusion` plot template."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_label", "pred_label"])
        for t, p in zip(y_true, y_pred):
            writer.writerow([class_ids[t], class_ids[p]])
