"""Fit the shipping head for a frozen backbone on every camera, and ship it (plan §9.1).

The frozen-backbone counterpart of `run_all_rigs.py`, and bound by the same rule: **this run chooses
nothing.** With every camera in training there is no cross-camera number left to measure, so the folds
(exp240-251) already made every choice -- backbone, readout, and C -- and this run only applies them to
all the data. Its val/test scores are an in-distribution sanity check ("did anything break?"),
comparable to the folds' in-distribution numbers and to the ResNet18 all-rigs card's, never to a
cross-camera number.

- **C is the folds' median, not this run's val pick.** Twelve folds each chose C on their own val split
  (exp240-251). Picking it again here, on a val split drawn from the same cameras as train, would be a
  thirteenth, weaker vote. The lower median is taken so a tie goes to the stronger regularisation, the
  same tie rule `linear_head.fit_head` uses. The val-selected C is still computed and recorded, as a
  diagnostic only.
- **Same patches, same transform, same fit as the folds.** Datasets come from
  `fold_data.build_fold_datasets` (one split at a time, byte-identical), the train split gets the eval
  transform (frozen arms get no photometric augmentation, as in the screen), and the head is
  `linear_head.fit_head_at`: the fold fit's solver, scaler and export at a fixed C.
- **Seeds only vary the patch draw** (the head is convex). Each seed is archived; the one to ship is
  chosen on val macro-F1, never test, and shipping is a separate, explicit step.

    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp 252
    python -m coffeecv.fit_frozen_head --ship 252 --name allrigs_dino3b16_s42
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import socket
import statistics
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from coffeecv.archive_experiment import EXPERIMENTS_DIR, archive
from coffeecv.backbones import SPECS, FrozenBackbone, assert_input_size, build_backbone
from coffeecv.config import OUTPUTS_DIR, REPO_ROOT, RunConfig, build_env_block, config_to_dict, set_seed
from coffeecv.dino_classifier import is_frozen_model, save_frozen_checkpoint
from coffeecv.dataset import CAPTURES, load_class_labels
from coffeecv.fold_data import build_fold_datasets
from coffeecv.infer import classes_path_for
from coffeecv.linear_head import C_GRID, cross_entropy, fit_head, fit_head_at, predict
from coffeecv.metrics import build_metrics_json, compute_split_metrics, write_predictions_csv
from coffeecv.repro_utils import dirty_provenance_paths, stale_crop_stages
from coffeecv.transforms import build_eval_transform

DEFAULT_OUT = OUTPUTS_DIR / "frozen_allrigs"
MODELS_DIR = REPO_ROOT / "models"
# The folds whose per-fold C picks decide the shipping C: the selected cell of Experiment 1.
SELECTION_EXPS = tuple(range(240, 252))
EMBED_BATCH = 32
SMOKE_BUDGET = dict(train_patches_per_class=6, val_patches_per_class=3, test_patches_per_class=3)


def log(msg: str) -> None:
    print(f"[fit_frozen_head {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def fold_C(backbone: str, readout: str, exps=SELECTION_EXPS) -> tuple[float, list[dict]]:
    """The lower median of the C each selection fold chose on its own val split."""
    picks = []
    for e in exps:
        [d] = list(EXPERIMENTS_DIR.glob(f"exp{e}__*"))
        dino = json.loads((d / "config.json").read_text())["dino"]
        if (dino["backbone"], dino["readout"]) != (backbone, readout):
            raise SystemExit(f"{d.name} is {dino['backbone']}/{dino['readout']}, not {backbone}/{readout}")
        picks.append({"exp": e, "C": float(dino["C"])})
    return statistics.median_low([p["C"] for p in picks]), picks


def allrigs_config(backbone: str, seed: int, smoke: bool) -> RunConfig:
    """params.yaml's sampling geometry, every camera in training, and the fields that describe a frozen
    backbone with a convex head set to what actually runs (as the screen records its folds)."""
    cfg = replace(RunConfig.from_params_yaml(), seed=seed, train_capture_dirs=tuple(CAPTURES), heldout_rig="",
                  model_name=backbone, freeze_mode="full", mixstyle_p=0.0, dropout=0.0,
                  color_jitter_strength=0.0, random_erasing_p=0.0, mixup_alpha=0.0)
    return replace(cfg, **SMOKE_BUDGET) if smoke else cfg


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


def fit_one_seed(bb: FrozenBackbone, readout: str, seed: int, C: float, picks: list[dict],
                 out: Path, smoke: bool) -> Path:
    cfg = allrigs_config(bb.name, seed, smoke)
    assert_input_size(bb, cfg.patch_resize)
    set_seed(seed)
    eval_tf = build_eval_transform(cfg.patch_resize)
    X, y, sizes, ms = {}, {}, {}, {}
    class_ids = class_labels = captures = None
    for split in ("train", "val", "test"):
        t = time.perf_counter()
        fold = build_fold_datasets(cfg, eval_tf, eval_tf, only=(split,))
        ds = getattr(fold, split)
        built = time.perf_counter() - t
        X[split], y[split], ms[split] = embed(ds, bb, readout)
        sizes[split] = len(ds)
        class_ids, class_labels, captures = fold.class_ids, fold.class_labels, [c.name for c in fold.captures]
        log(f"s{seed}: {split:5s} {len(ds)} patches, built in {built:.0f}s, embedded at {ms[split]:.1f} ms/img")
        del fold, ds
        gc.collect()

    t = time.perf_counter()
    head, converged = fit_head_at(X["train"], y["train"], C, len(class_ids))
    fit_s = time.perf_counter() - t
    # Diagnostic only: what this run's own val split would have picked. Never used.
    grid = fit_head(X["train"], y["train"], X["val"], y["val"], class_ids, class_labels)

    split_metrics, preds = {}, {}
    for s in ("val", "test"):
        pred, probs, _ = predict(head, X[s])
        preds[s] = pred
        split_metrics[s] = compute_split_metrics(y[s], pred, cross_entropy(probs, y[s]), class_ids, class_labels)
    metrics = build_metrics_json(class_ids, class_labels, epochs_trained=None, best_epoch=None,
                                 val_metrics=split_metrics["val"], test_metrics=split_metrics["test"],
                                 captures=captures)
    metrics["best_epoch_selection_metric"] = "none: C fixed to the selection folds' median (no selection here)"

    import timm

    run_dir = out / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_frozen_checkpoint(run_dir / "model.pt", bb, readout, head, class_ids, C)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        **config_to_dict(cfg), "class_ids": class_ids, "env": build_env_block(),
        "dino": {
            "experiment": "docs/dinov3_integration_plan.md §9.1: all-cameras shipping fit of the selected cell",
            "backbone": bb.name, "readout": readout, "weights": bb.spec.weights,
            "weights_sha256": bb.weights_sha256, "timm_version": timm.__version__,
            "head": "sklearn LogisticRegression (L2, lbfgs) on standardised features at a FIXED C, exported to "
                    "nn.Linear; the SGD-loop fields above (epochs, lr, optimizer, scheduler, ...) are unused",
            "C": C, "C_rule": f"lower median of the per-fold val picks of exp{SELECTION_EXPS[0]}-{SELECTION_EXPS[-1]}",
            "C_fold_picks": picks, "converged": converged, "fit_seconds": round(fit_s, 1),
            "diagnostic_val_selected_C": grid.C,
            "diagnostic_val_macro_f1_by_C": {str(k): v for k, v in grid.val_macro_f1_by_C.items()},
            "train_transform": "eval transform (frozen arms get no photometric augmentation)",
            "sizes": sizes, "embed_ms_per_img": {k: round(v, 1) for k, v in ms.items()},
            "smoke": smoke, "host": socket.gethostname(), "torch_threads": torch.get_num_threads(),
        },
    }, indent=2))
    for s in ("val", "test"):
        write_predictions_csv(run_dir / f"predictions_{s}.csv", y[s], preds[s], class_ids)
    log(f"s{seed}: C={C:g} ({'converged' if converged else 'NOT CONVERGED'}, {fit_s:.0f}s)  "
        f"val {split_metrics['val']['macro_f1']:.4f}  test {split_metrics['test']['macro_f1']:.4f}  "
        f"(val alone would pick C={grid.C:g})")
    return run_dir


def ship(exp: int, name: str) -> None:
    """experiments/exp<N> + its fitted head -> models/<name>.pt with card and frozen class list."""
    [exp_dir] = list(EXPERIMENTS_DIR.glob(f"exp{exp}__*"))
    config = json.loads((exp_dir / "config.json").read_text())
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    src = Path(config["dino"]["run_dir"]) / "model.pt"
    if not src.exists():
        raise SystemExit(f"{src} is gone -- refit exp{exp} (same seed gives the same head) before shipping")
    dst = MODELS_DIR / f"{name}.pt"
    if dst.exists():
        raise SystemExit(f"{dst} exists; ship under a new name rather than overwrite a shipped model")
    shutil.copy2(src, dst)
    # The class list the head was fitted against, frozen beside it (infer.classes_path_for): classes.txt
    # may grow before the next fit, and this head's width must not follow it.
    live_classes = REPO_ROOT / config["classes_file"]
    if sorted(load_class_labels(live_classes)) != config["class_ids"]:
        raise SystemExit(f"{live_classes} no longer lists exp{exp}'s classes {config['class_ids']}")
    shutil.copy2(live_classes, classes_path_for(dst))
    training_config = {k: v for k, v in config.items() if k not in ("dino", "env")}
    dino = config["dino"]
    card = {
        "name": dst.name,
        "source_experiment": exp_dir.name,
        "model": (f"frozen {dino['backbone']} (models_pretrained/{dino['weights']}, sha256 "
                  f"{dino['weights_sha256'][:16]}..., timm {dino['timm_version']}) -> {dino['readout']} readout -> "
                  f"L2 logistic-regression head at C={dino['C']:g}. This .pt holds only the head and the backbone's "
                  f"identity; the backbone is loaded from models_pretrained/ and checked against that digest."),
        "why_this_one": None,        # written by the person who ships it
        "inference_defaults": {
            "tta": False,
            "why": ("No dihedral TTA: 8 views x 40 patches is ~46 s per photo for ViT-B/16 on the production VM "
                    "(docs/dinov3_integration_plan.md §6.2), and TTA for the frozen head is unmeasured (Screen D). "
                    "Without it, model time is ~5.8 s per photo at 4 threads, parity with ResNet18 + TTA."),
        },
        "expected_performance": {
            "cross_camera_from_selection_folds": ("exp240-251, 3 seeds x 4 camera folds: patch macro-F1 0.8873 "
                                                  "(3 ten-class folds), photo-pooled 0.9489 over 40 patches, no TTA"),
            "this_run_in_distribution": {s: {k: metrics["splits"][s][k] for k in ("macro_f1", "mcc", "accuracy")}
                                         for s in ("val", "test")},
            "note": "in-distribution only -- compare to the folds' test split and to allrigs_cam_s123's card, "
                    "never to a cross-camera number",
        },
        "training_config": training_config,
        "dino": dino,
    }
    MODELS_DIR.joinpath(f"{name}.json").write_text(json.dumps(card, indent=1) + "\n")
    log(f"shipped {exp_dir.name} -> {dst.relative_to(REPO_ROOT)} (+ .json card, .classes.txt). Next: "
        f"build_ood_reference and fit_ood_probe on it (plan §9.3), then fill in the card's why_this_one.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="dinov3_vitb16", choices=[n for n in SPECS if is_frozen_model(n)])
    p.add_argument("--readout", default="cls_mean")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    p.add_argument("--start-exp", type=int, help="first experiment id; one per seed, in --seeds order")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoke", action="store_true", help="tiny patch budgets; nothing is archived")
    p.add_argument("--allow-dirty", action="store_true", help="skip the provenance checks (smoke only)")
    p.add_argument("--ship", type=int, metavar="EXP", help="ship an archived run to models/ and exit")
    p.add_argument("--name", help="with --ship: the models/<name>.pt to create")
    args = p.parse_args()

    if args.ship is not None:
        if not args.name:
            raise SystemExit("--ship needs --name")
        ship(args.ship, args.name)
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

    C, picks = fold_C(args.backbone, args.readout)
    log(f"{args.backbone}/{args.readout}: C={C:g}, the lower median of the fold picks "
        f"{sorted(p['C'] for p in picks)}; torch threads {torch.get_num_threads()}")
    bb = build_backbone(args.backbone)
    out = Path(args.out)
    for i, seed in enumerate(args.seeds):
        t = time.perf_counter()
        run_dir = fit_one_seed(bb, args.readout, seed, C, picks, out / ("smoke" if args.smoke else ""), args.smoke)
        log(f"s{seed}: done in {(time.perf_counter() - t) / 60:.1f} min")
        if args.smoke:
            continue
        exp = args.start_exp + i
        cfg_path = run_dir / "config.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["dino"]["run_dir"] = str(run_dir.resolve().relative_to(REPO_ROOT))
        cfg_path.write_text(json.dumps(cfg, indent=2))
        archive(str(exp), f"allrigs_{args.backbone.replace('dinov3_vit', 'dino3')}_frozen_"
                          f"{args.readout.replace('_', '')}_s{seed}",
                f"plan §9.1 all-cameras shipping fit: frozen {args.backbone}, readout {args.readout}, "
                f"L2 logistic-regression head at the selection folds' median C={C:g}, no TTA; "
                f"in-distribution metrics only", src_dir=run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
