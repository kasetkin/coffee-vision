"""Fit the shipping head for a frozen backbone on the pool, and ship it (plan §9.1).

The frozen-backbone counterpart of `run_all_rigs.py`: the photos of `dataset.CAPTURES` (since ticket ML-8 one
pool of every session, no camera or rig) split per class (ticket ML-1), val/test in-distribution only -- they
say nothing about a new camera or a new scoop of beans, and are never comparable to the fold-era cross-camera
numbers.

- **C is a fixed default, 0.1** (`--C` overrides it): the lower median of the leave-one-camera-out
  folds' val picks (exp240-251), the value the first shipped DINOv3 head (`allrigs_dino3b16_s123`, retired) was fitted at, frozen as
  a constant by the owner on 2026-09-30 (ticket ML-1 D3) now that the folds are gone. The val-selected C
  is computed and recorded beside it as a diagnostic only: the head's val macro-F1 is nearly flat across
  the C grid, so a val pick would mostly follow noise.
- **Same patches, same transform, same fit as training.** Datasets come from
  `fold_data.build_fold_datasets` (one split at a time, byte-identical), every patch is embedded through the
  eval transform, and the head is `linear_head.fit_head_at`: the solver, scaler and export at a fixed C.
- **Augmented training patches, in memory only** (ticket ML-8 D4, ADR 0025; params.yaml head_augment, empty
  before ML-8). Beside each training patch, `views` augmented views of it (`HeadAugment`: the `patch` pipeline
  on the stored patch, the resize to patch_resize, the `input` pipeline), each seeded by (run seed, patch,
  view), are drawn from the in-memory patch, embedded in batches and dropped; only their embeddings are kept.
  Val and test are never augmented. Nothing augmented is written. The run fits two heads at the same C and the
  same total sample weight (each of a patch's 1 + views rows weighs 1/(1 + views)), so augmentation is the
  only difference between them: the augmented one is the run's head, the unaugmented one is archived beside it
  (`model_unaugmented.pt`, metrics.json `arms`), ADR 0002's paired print within one run. A ResNet-era
  augmentation knob set in params.yaml is refused here (`frozen_config`): this path applies none of them.
- **Seeds only vary the patch draw** (the head is convex). Each seed is archived; the one to ship is
  chosen on val macro-F1, never test, and shipping is a separate, explicit step.

    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp 270
    python -m coffeecv.fit_frozen_head --smoke --allow-dirty          # tiny budgets, nothing archived
    python -m coffeecv.fit_frozen_head --ship 271 --name dino3b16_segxl0v3_country_aug8_s123 --seed-exps 270 271 272

Every fit reads the segmenter's crops (dataset.CAPTURES) with crop_method "segment", so patches are placed
by the D17 rule against each photo's mask and the checkpoint serves through the segmenter; params.yaml's seg_*
fields name it. Until ticket ML-3 P2b this took `--pools cropped|segcropped` and defaulted to the tray
heuristic's pools, which are retired (Q3). Runs and models are named without `allrigs` since ML-8 (D10).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import socket
import time
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from coffeecv import augment, model_io
from coffeecv.archive_experiment import EXPERIMENTS_DIR, archive
from coffeecv.backbones import SPECS, FrozenBackbone, assert_input_size, build_backbone
from coffeecv.class_list import folder_classes, load_classes
from coffeecv.config import OUTPUTS_DIR, REPO_ROOT, RunConfig, build_env_block, config_to_dict, set_seed
from coffeecv.dino_classifier import is_frozen_model, save_frozen_checkpoint
from coffeecv.dataset import CAPTURES
from coffeecv.fold_data import build_fold_datasets
from coffeecv.infer import classes_path_for
from coffeecv.linear_head import cross_entropy, fit_head, fit_head_at, predict
from coffeecv.metrics import build_metrics_json, compute_split_metrics, write_predictions_csv
from coffeecv.repro_utils import dirty_provenance_paths, stale_crop_stages
from coffeecv.transforms import build_eval_transform

DEFAULT_OUT = OUTPUTS_DIR / "frozen_head"
MODELS_DIR = REPO_ROOT / "models"
# The fold median of exp240-251's val picks, frozen (ML-1 D3, 2026-09-30).
DEFAULT_C = 0.1
DEFAULT_C_RULE = "fixed default: the lower median of exp240-251's per-fold val picks, frozen 2026-09-30 (ML-1 D3)"
EMBED_BATCH = 32
SMOKE_BUDGET = dict(train_patches_per_class=6, val_patches_per_class=3, test_patches_per_class=3)
SHIFT_PATCHES = 16      # --smoke: the training patches each transform alone is applied to (its embedding shift)
HEAD_AUGMENT_KEYS = ("views", "patch", "input")
# ResNet-era augmentation this path never applies (every patch goes through the eval transform); set in
# params.yaml, it would be recorded in the run's config as if it had. Their no-op values. The adopted ResNet
# recipe's color_jitter_strength, random_erasing_p and mixup_alpha are switched off in `frozen_config` instead,
# as before (run_all_rigs.ADOPTED holds them in params.yaml).
RESNET_ERA_KNOBS = {"brightness_jitter_strength": 0.0, "illum_gradient_strength": 0.0, "zoom_scale_min": 1.0,
                    "patch_scale_frac_min": 0.0, "patch_scale_frac_max": 0.0, "rotation_jitter_degrees": 0.0}


def log(msg: str) -> None:
    print(f"[fit_frozen_head {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def frozen_config(backbone: str, seed: int, smoke: bool) -> RunConfig:
    """params.yaml's sampling geometry and head_augment, the pool (dataset.CAPTURES), and the fields that
    describe a frozen backbone with a convex head set to what actually runs (as the screen records its folds).
    Refuses a ResNet-era augmentation knob set in params.yaml (RESNET_ERA_KNOBS). Until ML-8, allrigs_config."""
    base = RunConfig.from_params_yaml()
    on = [f"{k}={getattr(base, k)}" for k, off in RESNET_ERA_KNOBS.items() if getattr(base, k) != off]
    if on:
        raise ValueError(f"params.yaml sets {', '.join(on)}: ResNet-era augmentation the frozen head fit never "
                         "applies (its augmentation is head_augment, ticket ML-8 D4). Set it back to "
                         f"{RESNET_ERA_KNOBS} for a frozen fit.")
    cfg = replace(base, seed=seed, train_capture_dirs=tuple(CAPTURES), crop_method="segment",
                  model_name=backbone, freeze_mode="full", mixstyle_p=0.0, dropout=0.0,
                  color_jitter_strength=0.0, random_erasing_p=0.0, mixup_alpha=0.0)
    return replace(cfg, **SMOKE_BUDGET) if smoke else cfg


def run_slug(backbone: str, readout: str, views: int, seed: int) -> str:
    """An archived run's name (ticket ML-8 D10: no `allrigs`; `aug<views>` when the head is augmented)."""
    return (f"{backbone.replace('dinov3_vit', 'dino3')}_frozen_{readout.replace('_', '')}_segcrop_country"
            f"{f'_aug{views}' if views else ''}_s{seed}")


class HeadAugment:
    """params.yaml head_augment: `views` augmented views of a training patch (ticket ML-8 D4, ADR 0025). A view is
    the `patch` pipeline on the stored patch, the eval transform's resize to `resize` (model_io.eval_resize, the
    one serving's eval_patch calls), then the `input` pipeline on the model's input pixels. Both pipelines are built
    by coffeecv.augment's strict builder and reseeded per view from (run seed, patch index, view), so a view
    is the same whatever was drawn before it, and one seed repeats its stream."""

    def __init__(self, views: int, patch: list[dict], input: list[dict], resize: int):
        if not isinstance(views, int) or isinstance(views, bool) or views < 1:
            raise ValueError(f"head_augment.views must be a positive integer, not {views!r}")
        self.views, self.patch, self.input, self.resize = views, list(patch or ()), list(input or ()), resize
        self._patch = augment.build(self.patch, where="head_augment.patch", seed=0)
        self._input = augment.build(self.input, where="head_augment.input", seed=0)

    @classmethod
    def from_config(cls, cfg: RunConfig) -> "HeadAugment | None":
        block = cfg.head_augment
        if not block:
            return None
        if sorted(block) != sorted(HEAD_AUGMENT_KEYS):
            raise ValueError(f"params.yaml head_augment: expected keys {', '.join(HEAD_AUGMENT_KEYS)}, "
                             f"got {', '.join(sorted(block))}")
        return cls(block["views"], block["patch"], block["input"], cfg.patch_resize)

    def view(self, patch: np.ndarray, seed: int, index: int, k: int) -> np.ndarray:
        """View `k` of training patch `index` (HxWx3 uint8, as stored) -> resize x resize x 3 uint8."""
        s_patch, s_input = np.random.SeedSequence([seed, index, k]).generate_state(2)
        self._patch.set_random_seed(int(s_patch))
        self._input.set_random_seed(int(s_input))
        x = self._patch(image=patch)["image"]
        if x.shape != patch.shape:
            raise ValueError(f"head_augment.patch changed the patch's shape {patch.shape} to {x.shape}")
        x = np.asarray(model_io.eval_resize(Image.fromarray(x), self.resize))
        out = self._input(image=x)["image"]
        if out.shape != (self.resize, self.resize, 3):
            raise ValueError(f"head_augment.input changed the input's shape to {out.shape}")
        return out

    def record(self) -> dict:
        return {"views": self.views, "patch": self.patch, "input": self.input, "albumentations": augment.version()}


def single_transforms(aug: HeadAugment) -> list[tuple[str, HeadAugment]]:
    """Each transform of `aug` alone, always applied: ("patch.<name>" | "input.<name>", a one-view pipeline)."""
    out = []
    for where in ("patch", "input"):
        for t in getattr(aug, where):
            alone = [{**t, "p": 1.0}]
            out.append((f"{where}.{t['transform']}",
                        HeadAugment(1, alone if where == "patch" else [], alone if where == "input" else [], aug.resize)))
    return out


def augmented_rows(X: np.ndarray, X_views: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The augmented arm's training rows: the originals, then each view of every patch (view-major), each row
    weighing 1/(1 + views), so the arm's total sample weight is the unaugmented arm's len(X)."""
    k = X_views.shape[1]
    X_all = np.concatenate([X] + [X_views[:, v] for v in range(k)])
    return X_all, np.concatenate([y] * (k + 1)), np.full(len(X_all), 1.0 / (k + 1))


@torch.no_grad()
def embed(ds, bb: FrozenBackbone, readout: str) -> tuple[np.ndarray, np.ndarray, float]:
    feats, labels, spent = [], [], 0.0
    for start in range(0, len(ds), EMBED_BATCH):
        items = [ds[i] for i in range(start, min(start + EMBED_BATCH, len(ds)))]
        x = torch.stack([it[0] for it in items])
        labels.extend(int(it[1]) for it in items)
        t = time.perf_counter()
        feats.append(bb.features(x)[readout].float().numpy())
        spent += time.perf_counter() - t
    return np.concatenate(feats), np.array(labels), 1000 * spent / max(len(labels), 1)


@torch.no_grad()
def embed_views(patches: list[np.ndarray], indices, bb: FrozenBackbone, readout: str, aug: HeadAugment,
                seed: int) -> tuple[np.ndarray, float]:
    """([len(indices), aug.views, D] embeddings of every view of each patch, backbone ms per view). Views are
    made and embedded EMBED_BATCH at a time and dropped; nothing is written."""
    eval_tf = build_eval_transform(aug.resize)
    jobs = [(row, i, k) for row, i in enumerate(indices) for k in range(aug.views)]
    feats, spent = None, 0.0
    for start in range(0, len(jobs), EMBED_BATCH):
        chunk = jobs[start:start + EMBED_BATCH]
        x = torch.stack([eval_tf(Image.fromarray(aug.view(patches[i], seed, i, k))) for _, i, k in chunk])
        t = time.perf_counter()
        f = bb.features(x)[readout].float().numpy()
        spent += time.perf_counter() - t
        if feats is None:
            feats = np.empty((len(indices), aug.views, f.shape[1]), np.float32)
        for (row, _, k), v in zip(chunk, f):
            feats[row, k] = v
    return feats, 1000 * spent / max(len(jobs), 1)


def transform_shifts(patches: list[np.ndarray], X: np.ndarray, bb: FrozenBackbone, readout: str,
                     aug: HeadAugment, seed: int, n: int = SHIFT_PATCHES) -> dict[str, float]:
    """Per transform of `aug` applied alone: the mean cosine between a view's embedding and its patch's (rows of
    X), over `n` training patches spread over the split. A transform near 1 costs a backbone pass and buys
    nothing (plan P6)."""
    idx = np.linspace(0, len(patches) - 1, min(n, len(patches))).round().astype(int)
    base = X[idx] / np.linalg.norm(X[idx], axis=1, keepdims=True)
    out = {}
    for name, alone in single_transforms(aug):
        V = embed_views(patches, idx, bb, readout, alone, seed)[0][:, 0]
        out[name] = round(float(np.mean(np.sum(base * V / np.linalg.norm(V, axis=1, keepdims=True), axis=1))), 4)
    return out


class Arm(StrEnum):
    """One of an augmented run's two heads (ADR 0025): the run's own, fitted on the views too, and the paired one
    fitted on the patches alone. Its value is the run's metrics.json `arms` key and ship's --arm."""
    AUGMENTED = "augmented"
    UNAUGMENTED = "unaugmented"


@dataclass
class ArmFit:
    """fit_arm's head, its val and test predictions and probabilities, and `record`, what metrics.json keeps."""
    head: object
    preds: dict[str, np.ndarray]
    probs: dict[str, np.ndarray]
    record: dict


def fit_arm(X_train: np.ndarray, y_train: np.ndarray, w: np.ndarray | None, X: dict, y: dict, C: float,
            class_ids: list[str], class_labels: dict[str, str]) -> ArmFit:
    """One head at the fixed C on these training rows (weights `w`), scored on val and test, with the
    diagnostic val-picked C beside it (never used)."""
    t = time.perf_counter()
    head, converged = fit_head_at(X_train, y_train, C, len(class_ids), sample_weight=w)
    fit_s = time.perf_counter() - t
    grid = fit_head(X_train, y_train, X["val"], y["val"], class_ids, class_labels, sample_weight=w)
    splits, preds, probs_by = {}, {}, {}
    for s in ("val", "test"):
        pred, probs, _ = predict(head, X[s])
        preds[s], probs_by[s] = pred, probs
        splits[s] = compute_split_metrics(y[s], pred, cross_entropy(probs, y[s]), class_ids, class_labels)
    return ArmFit(head, preds, probs_by, {
        "splits": splits, "train_rows": int(len(X_train)),
        "total_sample_weight": float(len(X_train) if w is None else w.sum()),
        "C": C, "converged": converged, "fit_seconds": round(fit_s, 1),
        "diagnostic_val_selected_C": grid.C,
        "diagnostic_val_macro_f1_by_C": {str(k): v for k, v in grid.val_macro_f1_by_C.items()}})


def fit_one_seed(bb: FrozenBackbone, readout: str, seed: int, C: float, C_rule: str,
                 out: Path, smoke: bool) -> Path:
    cfg = frozen_config(bb.name, seed, smoke)
    aug = HeadAugment.from_config(cfg)
    assert_input_size(bb, cfg.patch_resize)
    set_seed(seed)
    eval_tf = build_eval_transform(cfg.patch_resize)
    X, y, sizes, ms, seg, meta = {}, {}, {}, {}, {}, {}
    views = shifts = None
    views_s = 0.0
    class_ids = class_labels = captures = None
    for split in ("train", "val", "test"):
        t = time.perf_counter()
        fold = build_fold_datasets(cfg, eval_tf, eval_tf, only=(split,))
        ds = getattr(fold, split)
        built = time.perf_counter() - t
        X[split], y[split], ms[split] = embed(ds, bb, readout)
        sizes[split] = len(ds)
        meta[split] = list(ds.meta)
        class_ids, class_labels, captures = fold.class_ids, fold.class_labels, [c.name for c in fold.captures]
        log(f"s{seed}: {split:5s} {len(ds)} patches, built in {built:.0f}s, embedded at {ms[split]:.1f} ms/img")
        if split == "train" and aug is not None:
            # ADR 0025: the training patches' views, from the patches in memory; only embeddings are kept.
            t = time.perf_counter()
            views, ms["train_views"] = embed_views(ds.patches, range(len(ds)), bb, readout, aug, seed)
            views_s = time.perf_counter() - t
            log(f"s{seed}: train {aug.views} views of each patch embedded in {views_s / 60:.1f} min "
                f"({1000 * views_s / max(views.shape[0] * views.shape[1], 1):.1f} ms/view all in, "
                f"{ms['train_views']:.1f} ms/view in the backbone)")
            if smoke:
                shifts = transform_shifts(ds.patches, X["train"], bb, readout, aug, seed)
                log(f"s{seed}: mean cosine of a view to its patch, each transform alone: {shifts}")
        if ds.bean_share_rule is not None:
            # D17/D18 per split: photos with no mask (whole photo, every box accepted) and patches the
            # rule had to top up below the bean-share bar.
            seg[split] = {"fallback_photos": sorted(ds.fallback_photos), "patches_below_share": ds.n_below_share}
            log(f"s{seed}: {split:5s} {len(ds.fallback_photos)} D18 fallback photos, "
                f"{ds.n_below_share} patches below the D17 bean share")
        del fold, ds
        gc.collect()

    # ADR 0002 within one run: the same C and the same total sample weight, so augmentation is the only change.
    arms = {Arm.UNAUGMENTED: fit_arm(X["train"], y["train"], None, X, y, C, class_ids, class_labels)}
    if aug is not None:
        X_aug, y_aug, w_aug = augmented_rows(X["train"], views, y["train"])
        del views
        gc.collect()
        arms[Arm.AUGMENTED] = fit_arm(X_aug, y_aug, w_aug, X, y, C, class_ids, class_labels)
        del X_aug, y_aug, w_aug
    main_arm = Arm.AUGMENTED if aug is not None else Arm.UNAUGMENTED
    run = arms[main_arm]
    split_metrics = run.record["splits"]
    metrics = build_metrics_json(class_ids, class_labels, epochs_trained=None, best_epoch=None,
                                 val_metrics=split_metrics["val"], test_metrics=split_metrics["test"],
                                 captures=captures)
    metrics["best_epoch_selection_metric"] = f"none: C fixed at {C:g} (no selection here)"
    if aug is not None:
        metrics["arms"] = {name.value: a.record for name, a in arms.items()}

    import timm

    run_dir = out / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_frozen_checkpoint(run_dir / "model.pt", bb, readout, run.head, class_ids, C)
    if aug is not None:
        save_frozen_checkpoint(run_dir / "model_unaugmented.pt", bb, readout, arms[Arm.UNAUGMENTED].head,
                               class_ids, C)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rec = run.record
    (run_dir / "config.json").write_text(json.dumps({
        **config_to_dict(cfg), "class_ids": class_ids,
        # classes_file is a path; its digest tells a later reader whether the file still says what it said.
        "classes_sha256": hashlib.sha256((REPO_ROOT / cfg.classes_file).read_bytes()).hexdigest(),
        "env": build_env_block(),
        "dino": {
            "experiment": ("docs/dinov3_integration_plan.md §9.1: shipping fit of the selected cell; since ticket "
                           "ML-8 on the one pool, with augmented training patches (ADR 0025)"),
            "backbone": bb.name, "readout": readout, "weights": bb.spec.weights,
            "weights_sha256": bb.weights_sha256, "timm_version": timm.__version__,
            "head": "sklearn LogisticRegression (L2, lbfgs) on standardised features at a FIXED C, exported to "
                    "nn.Linear; the SGD-loop fields above (epochs, lr, optimizer, scheduler, ...) are unused",
            "C": C, "C_rule": C_rule, "converged": rec["converged"], "fit_seconds": rec["fit_seconds"],
            "diagnostic_val_selected_C": rec["diagnostic_val_selected_C"],
            "diagnostic_val_macro_f1_by_C": rec["diagnostic_val_macro_f1_by_C"],
            "train_transform": ("eval transform, and head_augment's views of each training patch beside it "
                                "(dino.augment)" if aug is not None else
                                "eval transform (no head_augment: no augmentation)"),
            "augment": None if aug is None else {
                **aug.record(), "head_arm": main_arm.value, "unaugmented_head": "model_unaugmented.pt",
                "sample_weight_per_row": 1.0 / (aug.views + 1), "views_seconds": round(views_s, 1),
                **({"shift_by_transform": shifts, "shift_patches": min(SHIFT_PATCHES, sizes["train"])}
                   if shifts is not None else {})},
            "sizes": sizes, "embed_ms_per_img": {k: round(v, 1) for k, v in ms.items()},
            **({"segment": seg} if seg else {}),
            "smoke": smoke, "host": socket.gethostname(), "torch_threads": torch.get_num_threads(),
        },
    }, indent=2))
    for s in ("val", "test"):
        write_predictions_csv(run_dir / f"predictions_{s}.csv", y[s], run.preds[s], class_ids, meta[s],
                              run.probs[s])
    for name, a in arms.items():
        r = a.record
        log(f"s{seed}: {name.value:11s} C={C:g} ({'converged' if r['converged'] else 'NOT CONVERGED'}, "
            f"{r['fit_seconds']:.0f}s, {r['train_rows']} rows)  val {r['splits']['val']['macro_f1']:.4f}  "
            f"test {r['splits']['test']['macro_f1']:.4f}  (val alone would pick C={r['diagnostic_val_selected_C']:g})")
    return run_dir


def arm_splits(metrics: dict, arm: Arm | None) -> dict:
    """metrics.json's val and test numbers of one head of the run: `arm` (ticket ML-8 P6, metrics `arms`), or None
    for the run's head (`splits`)."""
    if arm is None:
        return metrics["splits"]
    if arm.value not in (metrics.get("arms") or {}):
        raise SystemExit(f"the run has no {arm.value!r} head (its arms: {sorted(metrics.get('arms') or {})})")
    return metrics["arms"][arm.value]["splits"]


def seed_summary(exp: int, seed_exps: list[int], config: dict, arm: Arm | None = None) -> dict:
    """The in-distribution numbers of every seed of one fit (of head `arm`, `arm_splits`), with the mean and
    range: the shipped seed was picked on val, so its own test number is selection-biased (ticket ML-3 P7)."""
    if exp not in seed_exps or len(set(seed_exps)) != len(seed_exps):
        raise SystemExit(f"--seed-exps {seed_exps} must list exp{exp} and each run once")
    runs = {}
    for e in seed_exps:
        [d] = list(EXPERIMENTS_DIR.glob(f"exp{e}__*"))
        c = json.loads((d / "config.json").read_text())
        if c.get("classes_sha256") != config.get("classes_sha256") or c["class_ids"] != config["class_ids"]:
            raise SystemExit(f"exp{e} was not fitted on exp{exp}'s class list")
        runs[c["seed"]] = (d.name, json.loads((d / "metrics.json").read_text()))
    if len(runs) != len(seed_exps):
        raise SystemExit(f"--seed-exps {seed_exps} repeat a seed")
    out = {"runs": {f"s{sd}": name for sd, (name, _) in runs.items()}}
    for split in ("val", "test"):
        v = {f"s{sd}": arm_splits(m, arm)[split]["macro_f1"] for sd, (_, m) in runs.items()}
        out[f"{split}_macro_f1"] = {**v, "mean": float(np.mean(list(v.values()))),
                                    "min": min(v.values()), "max": max(v.values())}
    return out


def ship(exp: int, name: str, seed_exps: list[int] | None = None, arm: Arm | None = None) -> None:
    """experiments/exp<N> + its fitted head -> models/<name>.pt with card and frozen class list. `arm`
    Arm.UNAUGMENTED ships the run's unaugmented head instead of its augmented one (ticket ML-8 P7's option)."""
    [exp_dir] = list(EXPERIMENTS_DIR.glob(f"exp{exp}__*"))
    config = json.loads((exp_dir / "config.json").read_text())
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    augment_record = (config.get("dino") or {}).get("augment")
    if arm is not None and not augment_record:
        raise SystemExit(f"exp{exp} was fitted without head_augment: it has one head; ship it without --arm")
    if arm is Arm.AUGMENTED:
        arm = None                                          # the run's own head
    unaugmented = arm is Arm.UNAUGMENTED
    splits = arm_splits(metrics, arm)
    # A country list (ticket ML-3) has no fold-era numbers, so the card carries every seed's instead.
    per_folder = list(folder_classes(REPO_ROOT / config["classes_file"]).keys) == config["class_ids"]
    if not per_folder and not seed_exps:
        raise SystemExit("a country-class head ships with --seed-exps, every seed of its fit (ticket ML-3 P8)")
    seeds = seed_summary(exp, seed_exps, config, arm) if seed_exps else None
    src = Path(config["dino"]["run_dir"]) / ("model_unaugmented.pt" if unaugmented else "model.pt")
    if not src.exists():
        raise SystemExit(f"{src} is gone -- refit exp{exp} (same seed gives the same head) before shipping")
    dst = MODELS_DIR / f"{name}.pt"
    if dst.exists():
        raise SystemExit(f"{dst} exists; ship under a new name rather than overwrite a shipped model")
    shutil.copy2(src, dst)
    # The class list the head was fitted against, frozen beside it (infer.classes_path_for): classes.txt
    # may grow before the next fit, and this head's width must not follow it.
    live_classes = REPO_ROOT / config["classes_file"]
    if list(load_classes(live_classes).keys) != config["class_ids"]:
        raise SystemExit(f"{live_classes} no longer lists exp{exp}'s classes {config['class_ids']}")
    shutil.copy2(live_classes, classes_path_for(dst))
    training_config = {k: v for k, v in config.items() if k not in ("dino", "env")}
    dino = config["dino"]
    if unaugmented:
        # The run's unaugmented arm: same patches and C, no views; the card says what this head was fitted on.
        training_config["head_augment"] = {}
        dino = {**dino, "augment": None, "shipped_arm": "unaugmented (the run's head is its augmented arm)"}
    card = {
        "name": dst.name,
        "source_experiment": exp_dir.name,
        "model": (f"frozen {dino['backbone']} (models_pretrained/{dino['weights']}, sha256 "
                  f"{dino['weights_sha256'][:16]}..., timm {dino['timm_version']}) -> {dino['readout']} readout -> "
                  f"L2 logistic-regression head at C={dino['C']:g}"
                  + (f", fitted on each training patch and {dino['augment']['views']} augmented views of it "
                     "(ADR 0025; the pipeline in dino.augment)" if dino.get("augment") else "")
                  + ". This .pt holds only the head and the backbone's "
                  "identity; the backbone is loaded from models_pretrained/ and checked against that digest."),
        "why_this_one": None,        # written by the person who ships it
        "inference_defaults": {
            "tta": False,
            "why": ("No dihedral TTA: 8 views x 40 patches is ~46 s per photo for ViT-B/16 on the production VM "
                    "(docs/dinov3_integration_plan.md §6.2), and TTA for the frozen head is unmeasured (Screen D). "
                    "Without it, model time is ~5.8 s per photo at 4 threads, parity with ResNet18 + TTA."),
        },
        "expected_performance": {
            **({"cross_camera_from_selection_folds": ("exp240-251, 3 seeds x 4 camera folds: patch macro-F1 "
                                                      "0.8873 (3 ten-class folds), photo-pooled 0.9489 over 40 "
                                                      "patches, no TTA")} if per_folder else {}),
            "this_run_in_distribution": {s: {k: splits[s][k] for k in ("macro_f1", "mcc", "accuracy")}
                                         for s in ("val", "test")},
            **({"all_seeds_in_distribution": seeds,
                "headline": None} if seeds else {}),   # the fit's comparison printout, written by the person who ships it
            "note": ("in-distribution only -- compare to the folds' test split, never to a cross-camera number" if per_folder else
                     "in-distribution only; country classes (ticket ML-3) are not comparable with any card "
                     "fitted on per-folder classes"),
        },
        **({"ood_guard": ("ADR 0016: the linear probe at a fixed 0.5 after every classification, fitted on the "
                          "segmenter dataset's training and validation negatives (fit_ood_probe) and measured on "
                          "its test split (ood_eval)")}
           if not per_folder else {}),
        "training_config": training_config,
        "dino": dino,
    }
    MODELS_DIR.joinpath(f"{name}.json").write_text(json.dumps(card, indent=1) + "\n")
    log(f"shipped {exp_dir.name} -> {dst.relative_to(REPO_ROOT)} (+ .json card, .classes.txt). Next: "
        f"build_ood_reference and fit_ood_probe on it (plan §9.3), then fill in the card's why_this_one"
        + (" and headline." if seeds else "."))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov3_vitb16", choices=[n for n in SPECS if is_frozen_model(n)])
    p.add_argument("--readout", default="cls_mean")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--start-exp", type=int, help="first experiment id; one per seed, in --seeds order")
    p.add_argument("--C", type=float, default=DEFAULT_C,
                   help=f"the head's inverse regularisation strength (default {DEFAULT_C:g}, ML-1 D3)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoke", action="store_true", help="tiny patch budgets; nothing is archived")
    p.add_argument("--allow-dirty", action="store_true", help="skip the provenance checks (smoke only)")
    p.add_argument("--ship", type=int, metavar="EXP", help="ship an archived run to models/ and exit")
    p.add_argument("--name", help="with --ship: the models/<name>.pt to create")
    p.add_argument("--seed-exps", type=int, nargs="+",
                   help="with --ship: every seed's experiment of the shipped fit, for the card's mean and range")
    p.add_argument("--arm", type=Arm, choices=list(Arm),
                   help="with --ship: which of an augmented run's two heads (default: the run's head, augmented)")
    args = p.parse_args(argv)

    if args.ship is not None:
        if not args.name:
            raise SystemExit("--ship needs --name")
        ship(args.ship, args.name, args.seed_exps, args.arm)
        return 0
    if not args.smoke and args.start_exp is None:
        raise SystemExit("--start-exp is required for a real run (the ids must not collide with another machine's)")
    if not args.allow_dirty:
        dirty = dirty_provenance_paths()
        if dirty:
            raise SystemExit("uncommitted source would make this run unreproducible:\n  " + "\n  ".join(dirty))
        stale = stale_crop_stages(CAPTURES)
        if stale:
            raise SystemExit(f"stale upstream data stages {stale}: the crops on disk are not the tracked ones")
    if args.start_exp is not None:
        taken = [e for e in range(args.start_exp, args.start_exp + len(args.seeds))
                 if list(EXPERIMENTS_DIR.glob(f"exp{e}__*"))]
        if taken:
            raise SystemExit(f"experiment ids {taken} already exist")

    C = args.C
    C_rule = DEFAULT_C_RULE if C == DEFAULT_C else "set on the command line (--C)"
    log(f"{args.backbone}/{args.readout}: C={C:g}, {C_rule}; torch threads {torch.get_num_threads()}")
    bb = build_backbone(args.backbone)
    out = Path(args.out)
    for i, seed in enumerate(args.seeds):
        t = time.perf_counter()
        run_dir = fit_one_seed(bb, args.readout, seed, C, C_rule, out / ("smoke" if args.smoke else ""), args.smoke)
        log(f"s{seed}: done in {(time.perf_counter() - t) / 60:.1f} min")
        if args.smoke:
            continue
        exp = args.start_exp + i
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["dino"]["run_dir"] = str(run_dir.resolve().relative_to(REPO_ROOT))
        cfg_path.write_text(json.dumps(cfg, indent=2))
        views = (cfg.get("head_augment") or {}).get("views", 0)
        archive(str(exp), run_slug(args.backbone, args.readout, views, seed),
                f"plan §9.1 shipping fit: frozen {args.backbone}, readout {args.readout}, "
                f"L2 logistic-regression head at fixed C={C:g}, no TTA; "
                + (f"fitted on {views} augmented views per training patch beside it at equal total sample weight "
                   "(ticket ML-8 D4, ADR 0025; the unaugmented arm in metrics.json arms); " if views else "")
                + "one pool of every session (data/segcropped/pool, ticket ML-8 D2), crop_method segment; "
                "country classes (ticket ML-3, dataset/classes.txt); in-distribution metrics only",
                src_dir=run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
