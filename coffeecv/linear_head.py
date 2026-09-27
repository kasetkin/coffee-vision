"""The depth-0 head: L2 logistic regression on standardised frozen features, exported as nn.Linear.

A frozen backbone needs no training loop: the head is a convex fit with one hyperparameter. C is
chosen on the **val** split's macro-F1 (the depth-0 analogue of the ResNet arm's
checkpoint-on-best-val rule -- selection never sees test or cross-rig data). Ties go to the smaller C,
i.e. the stronger regularisation. See docs/dinov3_integration_plan.md §5.3.

The fitted scaler is folded into the exported Linear, so the head is an ordinary module that
`coffeecv.infer.forward_with_embeddings` and `coffeecv.xrig_eval.run_photowise` accept unchanged.
"""
from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from coffeecv.metrics import compute_split_metrics

C_GRID: tuple[float, ...] = tuple(float(c) for c in np.logspace(-3, 2, 11))   # 1e-3 ... 1e2
MAX_ITER = 3000


@dataclass
class HeadFit:
    linear: nn.Linear
    C: float
    val_macro_f1_by_C: dict[float, float]
    not_converged: list[float] = field(default_factory=list)   # C values that hit MAX_ITER
    fit_seconds: float = 0.0


def export_linear(clf: LogisticRegression, scaler: StandardScaler) -> nn.Linear:
    """logits = W((x - mu) / sd) + b  ==  (W / sd) x + (b - W (mu / sd)): one Linear, no scaler."""
    W = clf.coef_ / scaler.scale_[None, :]
    b = clf.intercept_ - (clf.coef_ * (scaler.mean_ / scaler.scale_)[None, :]).sum(axis=1)
    linear = nn.Linear(W.shape[1], W.shape[0])
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(W).float())
        linear.bias.copy_(torch.from_numpy(b).float())
    return linear


def fit_head(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
             class_ids: list[str], class_labels: dict[str, str],
             C_grid: tuple[float, ...] = C_GRID) -> HeadFit:
    t0 = time.perf_counter()
    n_classes = len(class_ids)
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)
    # Ascending C with warm starts: each fit starts from the previous (more regularised) optimum.
    # The problem is convex, so this only changes how fast lbfgs gets there, not where.
    clf = LogisticRegression(C=C_grid[0], max_iter=MAX_ITER, warm_start=True)
    scores: dict[float, float] = {}
    not_converged: list[float] = []
    best: tuple[float, float, LogisticRegression] | None = None
    dummy_losses = np.zeros(len(y_val))
    for C in sorted(C_grid):
        clf.set_params(C=C)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(Xt, y_train)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            not_converged.append(C)
        # The selection metric is computed by the same function that reports every other macro-F1
        # in this project, not by a second implementation of it.
        f1 = compute_split_metrics(y_val, clf.predict(Xv), dummy_losses, class_ids, class_labels)["macro_f1"]
        scores[C] = f1
        if best is None or f1 > best[0]:          # strict: a tie keeps the smaller C
            best = (f1, C, copy.deepcopy(clf))
    _, C_best, clf_best = best
    if list(clf_best.classes_) != list(range(n_classes)):
        raise ValueError(f"head saw classes {list(clf_best.classes_)}, expected 0..{n_classes - 1} -- "
                         f"a class is missing from every training rig")
    return HeadFit(export_linear(clf_best, scaler), C_best, scores, not_converged,
                   time.perf_counter() - t0)


def fit_head_at(X_train: np.ndarray, y_train: np.ndarray, C: float, n_classes: int) -> tuple[nn.Linear, bool]:
    """One fit at a fixed C -- the shipping fit (plan §9.1), where C comes from the folds rather than
    from this run's own val split. Returns (head, converged). Same solver, scaler and export as
    `fit_head`, so a shipped head is the thing the folds measured."""
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(C=C, max_iter=MAX_ITER)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(scaler.transform(X_train), y_train)
    if list(clf.classes_) != list(range(n_classes)):
        raise ValueError(f"head saw classes {list(clf.classes_)}, expected 0..{n_classes - 1}")
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return export_linear(clf, scaler), converged


@torch.no_grad()
def predict(linear: nn.Linear, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(predicted class, probabilities, logits) for features X."""
    logits = linear(torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)))
    probs = torch.softmax(logits, dim=1).numpy()
    return probs.argmax(axis=1), probs, logits.numpy()


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None))
