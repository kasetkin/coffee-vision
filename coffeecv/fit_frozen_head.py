"""Fit the shipping head for a frozen backbone on every capture dir, and ship it (plan §9.1).

The frozen-backbone counterpart of `run_all_rigs.py`: photos pooled across `dataset.CAPTURES` and split
per class (ticket ML-1), val/test in-distribution only -- they say nothing about a new camera or a new
scoop of beans, and are never comparable to the fold-era cross-camera numbers.

- **C is a fixed default, 0.1** (`--C` overrides it): the lower median of the leave-one-camera-out
  folds' val picks (exp240-251), the value the first shipped DINOv3 head (`allrigs_dino3b16_s123`, retired) was fitted at, frozen as
  a constant by the owner on 2026-09-30 (ticket ML-1 D3) now that the folds are gone. The val-selected C
  is computed and recorded beside it as a diagnostic only: the head's val macro-F1 is nearly flat across
  the C grid, so a val pick would mostly follow noise.
- **Same patches, same transform, same fit as training.** Datasets come from
  `fold_data.build_fold_datasets` (one split at a time, byte-identical), the train split gets the eval
  transform (frozen arms get no photometric augmentation), and the head is `linear_head.fit_head_at`:
  the solver, scaler and export at a fixed C.
- **Seeds only vary the patch draw** (the head is convex). Each seed is archived; the one to ship is
  chosen on val macro-F1, never test, and shipping is a separate, explicit step.

    python -m coffeecv.fit_frozen_head --seeds 42 123 7 --start-exp 252
    python -m coffeecv.fit_frozen_head --ship 252 --name allrigs_dino3b16_s42
    python -m coffeecv.fit_frozen_head --ship 262 --name allrigs_dino3b16_seg_country_s123 --seed-exps 261 262 263

Every fit reads the segmenter's pools (dataset.CAPTURES) with crop_method "segment", so patches are placed
by the D17 rule against each photo's mask and the checkpoint serves through the segmenter; params.yaml's seg_*
fields name it. Until ticket ML-3 P2b this took `--pools cropped|segcropped` and defaulted to the tray
heuristic's pools, which are retired (Q3).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import socket
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

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

DEFAULT_OUT = OUTPUTS_DIR / "frozen_allrigs"
MODELS_DIR = REPO_ROOT / "models"
# The fold median of exp240-251's val picks, frozen (ML-1 D3, 2026-09-30).
DEFAULT_C = 0.1
DEFAULT_C_RULE = "fixed default: the lower median of exp240-251's per-fold val picks, frozen 2026-09-30 (ML-1 D3)"
EMBED_BATCH = 32
SMOKE_BUDGET = dict(train_patches_per_class=6, val_patches_per_class=3, test_patches_per_class=3)


def log(msg: str) -> None:
    print(f"[fit_frozen_head {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def allrigs_config(backbone: str, seed: int, smoke: bool) -> RunConfig:
    """params.yaml's sampling geometry, every camera in training (the segmenter's pools), and the fields
    that describe a frozen backbone with a convex head set to what actually runs (as the screen records its
    folds)."""
    cfg = replace(RunConfig.from_params_yaml(), seed=seed, train_capture_dirs=tuple(CAPTURES),
                  crop_method="segment",
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


def fit_one_seed(bb: FrozenBackbone, readout: str, seed: int, C: float, C_rule: str,
                 out: Path, smoke: bool) -> Path:
    cfg = allrigs_config(bb.name, seed, smoke)
    assert_input_size(bb, cfg.patch_resize)
    set_seed(seed)
    eval_tf = build_eval_transform(cfg.patch_resize)
    X, y, sizes, ms, seg, meta = {}, {}, {}, {}, {}, {}
    class_ids = class_labels = captures = None
    for split in ("train", "val", "test"):
        t = time.perf_counter()
        fold = build_fold_datasets(cfg, eval_tf, eval_tf, only=(split,))
        ds = getattr(fold, split)
        built = time.perf_counter() - t
        X[split], y[split], ms[split] = embed(ds, bb, readout)
        sizes[split] = len(ds)
        meta[split] = list(ds._meta)
        class_ids, class_labels, captures = fold.class_ids, fold.class_labels, [c.name for c in fold.captures]
        log(f"s{seed}: {split:5s} {len(ds)} patches, built in {built:.0f}s, embedded at {ms[split]:.1f} ms/img")
        if ds.bean_share_rule is not None:
            # D17/D18 per split: photos with no mask (whole photo, every box accepted) and patches the
            # rule had to top up below the bean-share bar.
            seg[split] = {"fallback_photos": sorted(ds.fallback_photos), "patches_below_share": ds.n_below_share}
            log(f"s{seed}: {split:5s} {len(ds.fallback_photos)} D18 fallback photos, "
                f"{ds.n_below_share} patches below the D17 bean share")
        del fold, ds
        gc.collect()

    t = time.perf_counter()
    head, converged = fit_head_at(X["train"], y["train"], C, len(class_ids))
    fit_s = time.perf_counter() - t
    # Diagnostic only: what this run's own val split would have picked. Never used.
    grid = fit_head(X["train"], y["train"], X["val"], y["val"], class_ids, class_labels)

    split_metrics, preds, probs_by = {}, {}, {}
    for s in ("val", "test"):
        pred, probs, _ = predict(head, X[s])
        preds[s], probs_by[s] = pred, probs
        split_metrics[s] = compute_split_metrics(y[s], pred, cross_entropy(probs, y[s]), class_ids, class_labels)
    metrics = build_metrics_json(class_ids, class_labels, epochs_trained=None, best_epoch=None,
                                 val_metrics=split_metrics["val"], test_metrics=split_metrics["test"],
                                 captures=captures)
    metrics["best_epoch_selection_metric"] = f"none: C fixed at {C:g} (no selection here)"

    import timm

    run_dir = out / f"s{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_frozen_checkpoint(run_dir / "model.pt", bb, readout, head, class_ids, C)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps({
        **config_to_dict(cfg), "class_ids": class_ids,
        # classes_file is a path; its digest tells a later reader whether the file still says what it said.
        "classes_sha256": hashlib.sha256((REPO_ROOT / cfg.classes_file).read_bytes()).hexdigest(),
        "env": build_env_block(),
        "dino": {
            "experiment": "docs/dinov3_integration_plan.md §9.1: all-cameras shipping fit of the selected cell",
            "backbone": bb.name, "readout": readout, "weights": bb.spec.weights,
            "weights_sha256": bb.weights_sha256, "timm_version": timm.__version__,
            "head": "sklearn LogisticRegression (L2, lbfgs) on standardised features at a FIXED C, exported to "
                    "nn.Linear; the SGD-loop fields above (epochs, lr, optimizer, scheduler, ...) are unused",
            "C": C, "C_rule": C_rule, "converged": converged, "fit_seconds": round(fit_s, 1),
            "diagnostic_val_selected_C": grid.C,
            "diagnostic_val_macro_f1_by_C": {str(k): v for k, v in grid.val_macro_f1_by_C.items()},
            "train_transform": "eval transform (frozen arms get no photometric augmentation)",
            "sizes": sizes, "embed_ms_per_img": {k: round(v, 1) for k, v in ms.items()},
            **({"segment": seg} if seg else {}),
            "smoke": smoke, "host": socket.gethostname(), "torch_threads": torch.get_num_threads(),
        },
    }, indent=2))
    for s in ("val", "test"):
        write_predictions_csv(run_dir / f"predictions_{s}.csv", y[s], preds[s], class_ids, meta[s], probs_by[s])
    log(f"s{seed}: C={C:g} ({'converged' if converged else 'NOT CONVERGED'}, {fit_s:.0f}s)  "
        f"val {split_metrics['val']['macro_f1']:.4f}  test {split_metrics['test']['macro_f1']:.4f}  "
        f"(val alone would pick C={grid.C:g})")
    return run_dir


def seed_summary(exp: int, seed_exps: list[int], config: dict) -> dict:
    """The in-distribution numbers of every seed of one fit, with the mean and range: the shipped seed was picked
    on val, so its own test number is selection-biased (ticket ML-3 P7)."""
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
        v = {f"s{sd}": m["splits"][split]["macro_f1"] for sd, (_, m) in runs.items()}
        out[f"{split}_macro_f1"] = {**v, "mean": float(np.mean(list(v.values()))),
                                    "min": min(v.values()), "max": max(v.values())}
    return out


def ship(exp: int, name: str, seed_exps: list[int] | None = None) -> None:
    """experiments/exp<N> + its fitted head -> models/<name>.pt with card and frozen class list."""
    [exp_dir] = list(EXPERIMENTS_DIR.glob(f"exp{exp}__*"))
    config = json.loads((exp_dir / "config.json").read_text())
    metrics = json.loads((exp_dir / "metrics.json").read_text())
    # A country list (ticket ML-3) has no fold-era numbers, so the card carries every seed's instead.
    per_folder = list(folder_classes(REPO_ROOT / config["classes_file"]).keys) == config["class_ids"]
    if not per_folder and not seed_exps:
        raise SystemExit("a country-class head ships with --seed-exps, every seed of its fit (ticket ML-3 P8)")
    seeds = seed_summary(exp, seed_exps, config) if seed_exps else None
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
    if list(load_classes(live_classes).keys) != config["class_ids"]:
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
            **({"cross_camera_from_selection_folds": ("exp240-251, 3 seeds x 4 camera folds: patch macro-F1 "
                                                      "0.8873 (3 ten-class folds), photo-pooled 0.9489 over 40 "
                                                      "patches, no TTA")} if per_folder else {}),
            "this_run_in_distribution": {s: {k: metrics["splits"][s][k] for k in ("macro_f1", "mcc", "accuracy")}
                                         for s in ("val", "test")},
            **({"all_seeds_in_distribution": seeds,
                "headline": ("analysis/ml3/printout.txt: patch macro-F1 over the 8 pre-ML-3 countries on "
                             "pre-ML-3 photos, with photo-level scores and bootstrap CIs, and the live model's "
                             "remapped reference")} if seeds else {}),
            "note": ("in-distribution only -- compare to the folds' test split, never to a cross-camera number" if per_folder else
                     "in-distribution only; country classes (ticket ML-3) are not comparable with any card "
                     "fitted on per-folder classes"),
        },
        **({"ood_guard": ("not freshly validated (ticket ML-3 D11): the probe is refitted on this head, but its "
                          "holdout was spent before; --verify on it is a regression check only")}
           if not per_folder else {}),
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
    p.add_argument("--C", type=float, default=DEFAULT_C,
                   help=f"the head's inverse regularisation strength (default {DEFAULT_C:g}, ML-1 D3)")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--smoke", action="store_true", help="tiny patch budgets; nothing is archived")
    p.add_argument("--allow-dirty", action="store_true", help="skip the provenance checks (smoke only)")
    p.add_argument("--ship", type=int, metavar="EXP", help="ship an archived run to models/ and exit")
    p.add_argument("--name", help="with --ship: the models/<name>.pt to create")
    p.add_argument("--seed-exps", type=int, nargs="+",
                   help="with --ship: every seed's experiment of the shipped fit, for the card's mean and range")
    args = p.parse_args()

    if args.ship is not None:
        if not args.name:
            raise SystemExit("--ship needs --name")
        ship(args.ship, args.name, args.seed_exps)
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
        archive(str(exp), f"allrigs_{args.backbone.replace('dinov3_vit', 'dino3')}_frozen_"
                          f"{args.readout.replace('_', '')}_segcrop_country_s{seed}",
                f"plan §9.1 all-cameras shipping fit: frozen {args.backbone}, readout {args.readout}, "
                f"L2 logistic-regression head at fixed C={C:g}, no TTA; "
                "ML-2 segmenter pools (data/segcropped, crop_method segment); "
                "country classes (ticket ML-3, dataset/classes.txt), not comparable with exp258-260; "
                "in-distribution metrics only", src_dir=run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
